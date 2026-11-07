"""Shared pieces: a small byte-level GPT and a local corpus (CPython's stdlib source)."""
import glob, math, os, sysconfig
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- data
def load_corpus(max_bytes=8_000_000):
    """Byte-level corpus: concatenated .py files from the Python stdlib. Vocab = 256."""
    root = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    buf = bytearray()
    for f in files:
        if "site-packages" in f or "/test" in f:
            continue
        with open(f, "rb") as fh:
            buf += fh.read()
        if len(buf) >= max_bytes:
            break
    data = torch.frombuffer(buf[:max_bytes], dtype=torch.uint8).long()
    n = int(0.95 * len(data))
    return data[:n], data[n:]


def get_batch(data, B, T, gen, device):
    ix = torch.randint(len(data) - T - 1, (B,), generator=gen)
    x = torch.stack([data[i : i + T] for i in ix])
    y = torch.stack([data[i + 1 : i + T + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------- model
@dataclass
class Cfg:
    vocab: int = 256
    ctx: int = 256
    d: int = 384
    n_layer: int = 6
    n_head: int = 6


class ShapeLog:
    """Collects (name, shape, meaning) so the step can describe itself."""
    def __init__(self):
        self.rows = []
    def __call__(self, name, t, meaning):
        self.rows.append((name, tuple(t.shape), str(t.dtype).replace("torch.", ""), meaning))
        return t
    def dump(self):
        w = max(len(r[0]) for r in self.rows)
        s = max(len(str(r[1])) for r in self.rows)
        for name, shape, dt, m in self.rows:
            print(f"{name:<{w}}  {str(shape):<{s}}  {dt:<8}  {m}")


class Block(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.h = c.n_head
        self.ln1 = nn.LayerNorm(c.d)
        self.qkv = nn.Linear(c.d, 3 * c.d)
        self.proj = nn.Linear(c.d, c.d)
        self.ln2 = nn.LayerNorm(c.d)
        self.fc = nn.Linear(c.d, 4 * c.d)
        self.out = nn.Linear(4 * c.d, c.d)

    def forward(self, x, log=None):
        B, T, D = x.shape
        H, hd = self.h, D // self.h
        a = self.ln1(x)
        qkv = self.qkv(a)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, H, hd).transpose(1, 2)
        k = k.view(B, T, H, hd).transpose(1, 2)
        v = v.view(B, T, H, hd).transpose(1, 2)
        if log is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Same math as SDPA, written out so the T x T score matrix is visible.
            log("  ln1(x)", a, "B, T, D: x normalised per token over its D features (no shape change)")
            log("  qkv = ln1(x) @ W_qkv", qkv, "B, T, 3D: query, key and value for every token, packed side by side")
            log("  q (after view+transpose)", q, "B, H, T, hd: D split into H heads of hd dims; heads moved ahead of T so each head is its own batch")
            log("  k", k, "B, H, T, hd: keys, same layout as q")
            log("  v", v, "B, H, T, hd: values, same layout as q")
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            log("  scores = q @ k^T / sqrt(hd)", att, "B, H, T_query, T_key: how much each query position attends to each key position")
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~mask, float("-inf")).softmax(-1)
            log("  softmax(causal_mask(scores))", att, "B, H, T_q, T_k: each row is a probability distribution over keys <= that query (causal)")
            y = att @ v
            log("  att @ v", y, "B, H, T, hd: per-head weighted sum of values")
        y = y.transpose(1, 2).reshape(B, T, D)
        if log: log("  heads merged", y, "B, T, D: the H heads concatenated back into D")
        x = x + self.proj(y)
        if log: log("  x + proj(attn)", x, "B, T, D: residual stream after attention")
        h = F.gelu(self.fc(self.ln2(x)))
        if log: log("  gelu(fc(ln2(x)))", h, "B, T, 4D: MLP hidden layer, 4x wider per token")
        x = x + self.out(h)
        if log: log("  x + mlp", x, "B, T, D: residual stream after the MLP; block output")
        return x


class GPT(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.ctx, c.d)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, log=None):
        B, T = idx.shape
        if log: log("idx", idx, "B, T: B sequences of T byte ids (int64), the model input")
        pos = torch.arange(T, device=idx.device)
        te = self.tok(idx)
        pe = self.pos(pos)
        if log:
            log("tok_emb(idx)", te, "B, T, D: each byte id looked up as a D-dim vector")
            log("pos_emb(arange(T))", pe, "T, D: one learned vector per position, broadcast over B")
        x = te + pe
        if log: log("x = tok + pos", x, "B, T, D: the residual stream entering block 0")
        for i, b in enumerate(self.blocks):
            x = b(x, log if (log and i == 0) else None)  # block 0 in detail; the rest have identical shapes
        if log: log(f"after {len(self.blocks)} blocks", x, "B, T, D: residual stream after the last block")
        x = self.ln_f(x)
        logits = self.head(x)
        if log: log("logits = ln_f(x) @ W_head", logits, "B, T, V: unnormalised scores over the next byte, at every position")
        return logits

    def n_matmul_params(self):
        """Parameters that take part in matmuls (all Linear weights). Embedding lookups are gathers, not FLOPs."""
        return sum(m.weight.numel() for m in self.modules() if isinstance(m, nn.Linear))


def flops_per_token(model: GPT, T: int):
    """Training FLOPs per token: 6N for the dense matmuls (2 fwd + 4 bwd) plus 12*L*T*D for attention
    scores and the weighted sum (PaLM appendix B convention, which counts the full T x T, not the causal half)."""
    c = model.c
    return 6 * model.n_matmul_params() + 12 * c.n_layer * T * c.d


def pick_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
