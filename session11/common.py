"""Shared pieces for session 11: a small byte-level GPT, the corpus, LR schedules, and one training loop."""
import glob, json, math, os, sysconfig, time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------- data
def load_corpus(max_bytes=8_000_000):
    """Byte-level corpus: CPython's stdlib .py files (tests excluded). Vocab = 256. No download needed."""
    root = sysconfig.get_paths()["stdlib"]
    buf = bytearray()
    for f in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
        if "site-packages" in f or "/test" in f:
            continue
        with open(f, "rb") as fh:
            buf += fh.read()
        if len(buf) >= max_bytes:
            break
    data = torch.frombuffer(buf[:max_bytes], dtype=torch.uint8).long()
    n = int(0.95 * len(data))
    return data[:n], data[n:]


def get_batch(data, B, T, gen, device=DEV):
    ix = torch.randint(len(data) - T - 1, (B,), generator=gen)
    x = torch.stack([data[i : i + T] for i in ix])
    y = torch.stack([data[i + 1 : i + T + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------- model
@dataclass
class Cfg:
    d: int = 256          # width
    n_layer: int = 4
    head_dim: int = 64    # fixed; n_head = d / head_dim, so heads scale with width
    ctx: int = 128
    vocab: int = 256


class Block(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.h = c.d // c.head_dim
        self.ln1, self.ln2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.qkv, self.proj = nn.Linear(c.d, 3 * c.d), nn.Linear(c.d, c.d)
        self.fc, self.out = nn.Linear(c.d, 4 * c.d), nn.Linear(4 * c.d, c.d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    """Standard parametrisation (init std 0.02 everywhere, one LR for all params) - deliberately, since
    the LR sweep asks how the optimum moves with width when nothing compensates for it."""
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.tok, self.pos = nn.Embedding(c.vocab, c.d), nn.Embedding(c.ctx, c.d)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


# ---------------------------------------------------------------- schedules
def lr_at(step, sched, peak, total, warmup, decay_frac=0.2):
    """LR for 0-indexed `step` of a run planned for `total` steps.
    cosine : linear warmup, then cosine from peak to 0 at `total`.
    wsd    : linear warmup, constant at peak, then linear decay to 0 over the last `decay_frac` of `total`.
    constant: linear warmup, then peak forever."""
    if step < warmup:
        return peak * (step + 1) / warmup
    if sched == "constant":
        return peak
    if sched == "cosine":
        p = (step - warmup) / max(1, total - warmup)
        return peak * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    if sched == "wsd":
        decay_start = total - int(round(decay_frac * total))
        if step < decay_start:
            return peak
        return peak * max(0.0, (total - step) / (total - decay_start))
    raise ValueError(sched)


# ---------------------------------------------------------------- one loop for every experiment
_CORPUS = None
def corpus():
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = load_corpus()
    return _CORPUS


_VAL = {}
def val_batches(T, n=16, B=32):
    if T not in _VAL:
        g = torch.Generator().manual_seed(999)
        _VAL[T] = [get_batch(corpus()[1], B, T, g) for _ in range(n)]
    return _VAL[T]


@torch.no_grad()
def evaluate(model):
    model.eval()
    ls = [F.cross_entropy(model(x).view(-1, model.c.vocab), y.view(-1)).item() for x, y in val_batches(model.c.ctx)]
    model.train()
    return sum(ls) / len(ls)


def train(cfg: Cfg, peak_lr, sched="cosine", total=300, stop=None, warmup=30, seed=0, B=32,
          betas=(0.9, 0.95), opt_factory=None, log_ratio=False, eval_at=(), return_model=False):
    """Train for `stop` steps (default `total`) of a schedule planned for `total` steps.
    Same seed -> same init and same batch order, so two runs differ only in what you changed."""
    stop = total if stop is None else stop
    torch.manual_seed(seed)
    model = GPT(cfg).to(DEV)
    opt = (opt_factory(model.parameters(), peak_lr) if opt_factory
           else torch.optim.Adam(model.parameters(), lr=peak_lr, betas=betas, eps=1e-8))
    g = torch.Generator().manual_seed(seed)
    tracked = {n: p for n, p in model.named_parameters() if p.dim() == 2}
    log = dict(loss=[], lr=[], val={}, ratio={n: [] for n in tracked} if log_ratio else None)
    t0 = time.time()
    for s in range(stop):
        lr = lr_at(s, sched, peak_lr, total, warmup)
        for pg in opt.param_groups:
            pg["lr"] = lr
        x, y = get_batch(corpus()[0], B, cfg.ctx, g)
        loss = F.cross_entropy(model(x).view(-1, cfg.vocab), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if log_ratio:
            before = {n: p.detach().clone() for n, p in tracked.items()}
        opt.step()
        if log_ratio:
            for n, p in tracked.items():   # ||dW|| / ||W|| per weight matrix
                log["ratio"][n].append(((p.detach() - before[n]).norm() / before[n].norm()).item())
        log["loss"].append(loss.item()); log["lr"].append(lr)
        if (s + 1) in eval_at:
            log["val"][s + 1] = evaluate(model)
        if not math.isfinite(log["loss"][-1]):
            break   # diverged; caller records it
    log["final_val"] = evaluate(model) if math.isfinite(log["loss"][-1]) else float("nan")
    log["seconds"] = time.time() - t0
    log["cfg"] = asdict(cfg)
    return (log, model, opt, g) if return_model else log   # opt + data RNG let a caller continue the run


class ResultStore:
    """One JSON file of finished runs, keyed by a string. Re-running a script skips finished runs,
    so a killed sweep resumes at run granularity."""
    def __init__(self, path):
        self.path = path
        self.data = json.load(open(path)) if os.path.exists(path) else {}
    def __contains__(self, k):
        return k in self.data
    def __getitem__(self, k):
        return self.data[k]
    def put(self, k, v):
        self.data[k] = v
        json.dump(self.data, open(self.path + ".tmp", "w"))
        os.replace(self.path + ".tmp", self.path)
