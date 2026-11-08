"""
Dense -> MoE "sparse upcycling" on Apple Silicon (MPS).

Phase 1  train a small dense char-level GPT (FFN = Linear -> GELU -> Linear).
Phase 2  convert every FFN into an 8-expert top-2 MoE whose experts are copies of
         the trained FFN (router freshly initialised). Because the top-k gate
         weights are renormalised to sum to 1 and all experts are identical, the
         MoE computes exactly the dense function at conversion time.
Phase 3  continue training the MoE, and -- as a fair control -- continue training
         the dense model from the same checkpoint for the same number of steps.

Usage:  .venv/bin/python train.py | tee logs/train.log
"""
import copy
import json
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ----------------------------------------------------------------------------- config
@dataclass
class Cfg:
    data_path: str = "data/tinystories_valid.txt"
    block: int = 256
    batch: int = 64
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    dropout: float = 0.0
    # phase 1 (dense)
    dense_steps: int = 2500
    dense_lr: float = 1e-3
    # phase 2/3 (continued training, MoE and dense control)
    cont_steps: int = 2000
    cont_lr: float = 6e-4
    n_experts: int = 8
    top_k: int = 2
    aux_coef: float = 0.01
    warmup: int = 100
    eval_every: int = 100
    eval_batches: int = 20


C = Cfg()

# ----------------------------------------------------------------------------- data
text = open(C.data_path, encoding="utf-8").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
VOCAB = len(chars)
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n_split = int(0.95 * len(data))
train_data, val_data = data[:n_split], data[n_split:]


def get_batch(split, gen=None):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - C.block - 1, (C.batch,), generator=gen)
    x = torch.stack([src[i : i + C.block] for i in ix])
    y = torch.stack([src[i + 1 : i + C.block + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


# fixed validation batches so every eval point is directly comparable
_g = torch.Generator().manual_seed(0)
VAL_BATCHES = [get_batch("val", _g) for _ in range(C.eval_batches)]


# ----------------------------------------------------------------------------- model
class Attn(nn.Module):
    def __init__(s):
        super().__init__()
        s.qkv = nn.Linear(C.d_model, 3 * C.d_model)
        s.proj = nn.Linear(C.d_model, C.d_model)

    def forward(s, x):
        B, T, D = x.shape
        q, k, v = s.qkv(x).split(D, dim=2)
        q, k, v = (t.view(B, T, C.n_head, D // C.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.proj(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    """The dense FFN: two Linear layers."""

    def __init__(s):
        super().__init__()
        s.fc = nn.Linear(C.d_model, 4 * C.d_model)
        s.proj = nn.Linear(4 * C.d_model, C.d_model)

    def forward(s, x):
        return s.proj(F.gelu(s.fc(x)))


class MoE(nn.Module):
    """Top-k routed mixture of MLP experts, built by upcycling a trained MLP."""

    def __init__(s, dense: MLP, n_experts: int, top_k: int):
        super().__init__()
        s.top_k = top_k
        s.n_experts = n_experts
        s.experts = nn.ModuleList([copy.deepcopy(dense) for _ in range(n_experts)])
        s.router = nn.Linear(C.d_model, n_experts, bias=False)
        nn.init.normal_(s.router.weight, std=0.02)
        s.aux_loss = torch.tensor(0.0)
        s.last_load = None

    def forward(s, x):
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        probs = F.softmax(s.router(xf), dim=-1)                 # (N, E)
        topv, topi = probs.topk(s.top_k, dim=-1)                 # (N, k)
        gates = topv / topv.sum(-1, keepdim=True)                # renormalise -> exact at init
        # sort (token, slot) assignments by expert -> one host sync per layer
        flat_e = topi.reshape(-1)                                # (N*k,)
        order = flat_e.argsort()
        tok = order // s.top_k
        counts = torch.bincount(flat_e, minlength=s.n_experts).tolist()
        chunks = xf[tok].split(counts)
        ys = []
        for exp, c in zip(s.experts, chunks):
            n = c.shape[0]
            if n == 0:
                continue
            pad = -n % 1024  # bucket shapes so MPS doesn't recompile kernels every step
            ys.append(exp(torch.cat([c, c.new_zeros(pad, D)]))[:n] if pad else exp(c))
        y = torch.cat(ys)
        y = y * gates.reshape(-1)[order, None]
        out = torch.zeros_like(xf).index_add_(0, tok, y)
        # Switch-style load-balancing loss: E * sum_i f_i * P_i
        dispatch = F.one_hot(topi, s.n_experts).sum(1).float()   # (N, E)
        f = dispatch.mean(0) / s.top_k
        P = probs.mean(0)
        s.aux_loss = s.n_experts * (f * P).sum()
        s.last_load = f.detach()
        return out.view(B, T, D)


class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(C.d_model), nn.LayerNorm(C.d_model)
        s.attn = Attn()
        s.ffn = MLP()

    def forward(s, x):
        x = x + s.attn(s.ln1(x))
        return x + s.ffn(s.ln2(x))


class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok = nn.Embedding(VOCAB, C.d_model)
        s.pos = nn.Embedding(C.block, C.d_model)
        s.blocks = nn.ModuleList([Block() for _ in range(C.n_layer)])
        s.ln_f = nn.LayerNorm(C.d_model)
        s.head = nn.Linear(C.d_model, VOCAB, bias=False)
        s.head.weight = s.tok.weight
        s.apply(s._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(s, idx, targets=None):
        T = idx.shape[1]
        x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device))
        for b in s.blocks:
            x = b(x)
        logits = s.head(s.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))
        return logits, loss

    def aux_loss(s):
        return sum((b.ffn.aux_loss for b in s.blocks if isinstance(b.ffn, MoE)), torch.tensor(0.0, device=DEVICE))

    @torch.no_grad()
    def generate(s, idx, n):
        for _ in range(n):
            logits, _ = s(idx[:, -C.block :])
            nxt = torch.multinomial(F.softmax(logits[:, -1] / 0.8, -1), 1)
            idx = torch.cat([idx, nxt], 1)
        return idx


def upcycle(dense: GPT) -> GPT:
    moe = copy.deepcopy(dense)
    for b in moe.blocks:
        b.ffn = MoE(b.ffn, C.n_experts, C.top_k)
    return moe.to(DEVICE)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def active_params(m):
    total = n_params(m)
    for b in m.blocks:
        if isinstance(b.ffn, MoE):
            per = n_params(b.ffn.experts[0])
            total -= per * (b.ffn.n_experts - b.ffn.top_k)
    return total


# ----------------------------------------------------------------------------- train
@torch.no_grad()
def evaluate(model):
    model.eval()
    loss = sum(model(x, y)[1].item() for x, y in VAL_BATCHES) / len(VAL_BATCHES)
    model.train()
    return loss


def lr_at(step, total, peak):
    if step < C.warmup:
        return peak * (step + 1) / C.warmup
    p = (step - C.warmup) / max(1, total - C.warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def train(model, steps, peak_lr, tag, step_offset=0):
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.1)
    hist = {"step": [], "val": [], "train": []}
    gen = torch.Generator().manual_seed(42)  # identical data order for MoE and dense control
    t0, run = time.time(), []
    for step in range(steps + 1):
        if step % C.eval_every == 0 or step == steps:
            v = evaluate(model)
            tr = sum(run) / len(run) if run else float("nan")
            hist["step"].append(step + step_offset)
            hist["val"].append(v)
            hist["train"].append(tr)
            extra = ""
            if isinstance(model.blocks[0].ffn, MoE) and model.blocks[0].ffn.last_load is not None:
                load = model.blocks[0].ffn.last_load.tolist()
                extra = " L0 load=[" + " ".join(f"{l:.2f}" for l in load) + "]"
            print(f"[{tag}] step {step+step_offset:5d}  train {tr:.4f}  val {v:.4f}  "
                  f"{(time.time()-t0):6.1f}s{extra}", flush=True)
            run = []
        if step == steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, peak_lr)
        x, y = get_batch("train", gen)
        _, loss = model(x, y)
        total = loss + C.aux_coef * model.aux_loss()
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run.append(loss.item())
    return hist


def main():
    print(f"device={DEVICE} vocab={VOCAB} train_tokens={len(train_data):,} val_tokens={len(val_data):,}")
    results = {"cfg": asdict(C)}

    # ---- Phase 1: dense
    dense = GPT().to(DEVICE)
    print(f"dense params: {n_params(dense)/1e6:.2f}M")
    results["dense"] = train(dense, C.dense_steps, C.dense_lr, "dense")
    torch.save(dense.state_dict(), "dense.pt")

    # ---- Phase 2: upcycle
    moe = upcycle(dense)
    l_dense, l_moe = evaluate(dense), evaluate(moe)
    print(f"\nUPCYCLE  dense val {l_dense:.6f}  ->  MoE val {l_moe:.6f}  (diff {abs(l_dense-l_moe):.2e})")
    print(f"MoE params total {n_params(moe)/1e6:.2f}M, active/token {active_params(moe)/1e6:.2f}M\n")
    results["upcycle"] = {"dense_val": l_dense, "moe_val": l_moe,
                          "dense_params": n_params(dense), "moe_params": n_params(moe),
                          "moe_active_params": active_params(moe)}

    # ---- Phase 3: continue both arms from the same checkpoint
    results["moe_cont"] = train(moe, C.cont_steps, C.cont_lr, "moe  ", step_offset=C.dense_steps)
    torch.save(moe.state_dict(), "moe.pt")
    results["dense_cont"] = train(dense, C.cont_steps, C.cont_lr, "dense", step_offset=C.dense_steps)

    # expert load per layer at the end
    xb, _ = VAL_BATCHES[0]
    with torch.no_grad():
        moe.eval(); moe(xb)
    results["final_load"] = [b.ffn.last_load.tolist() for b in moe.blocks]

    prompt = torch.tensor([[stoi[c] for c in "Once upon a time"]], device=DEVICE)
    for name, m in (("dense", dense), ("moe", moe)):
        m.eval()
        sample = "".join(itos[i] for i in m.generate(prompt, 300)[0].tolist())
        results[f"sample_{name}"] = sample
        print(f"\n--- {name} sample ---\n{sample}")

    json.dump(results, open("logs/results.json", "w"), indent=1)
    print("\nwrote logs/results.json")


if __name__ == "__main__":
    main()
