"""
revllm.py - a small GPT plus several *reversible* residual-stream variants.

Everything the notebooks need lives here so that all runs share exactly the
same model, data order, optimizer and measurement code.

Architectures (cfg.arch):
  standard  - pre-LN GPT block:  x = x + Attn(x);  x = x + MLP(x)         (stores all activations)
  coupling  - RevNet / Reformer additive coupling (two streams):
                 y1 = x1 + Attn(x2);  y2 = x2 + MLP(y1)
              exactly invertible:  x2 = y2 - MLP(y1);  x1 = y1 - Attn(x2)
              (this is the "Euler step on a split state" / symplectic-Euler variant)
  midpoint  - explicit midpoint / leapfrog two-step scheme on the full stream:
                 x_{n+1} = x_{n-1} + 2h * f(x_n)
              exactly invertible:  x_{n-1} = x_{n+1} - 2h * f(x_n)
  momentum  - Momentum-ResNet (Sander et al. 2021), velocity form:
                 v' = g*v + (1-g)*f(x);   x' = x + v'
              invertible:  x = x' - v';   v = (v' - (1-g)*f(x)) / g
  euler     - plain residual x' = x + f(x) made "reversible" by inverting it
              with a fixed-point iteration  x <- y - f(x)  (approximate!)

where f(x) = Attn(x) + MLP(x + Attn(x)) is the residual update of one
standard transformer block, so every variant has the same parameter count.

Reversible variants never store per-layer activations: the forward runs under
no_grad, only the final state is kept, and the backward pass reconstructs each
layer's input from its output, recomputes that layer with grad enabled and
back-propagates through it, one layer at a time.  Memory is therefore O(1)
in depth instead of O(depth).
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ----------------------------------------------------------------------------
# device / memory helpers
# ----------------------------------------------------------------------------
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class MemTracker:
    """Peak-memory tracker.

    CUDA has a real allocator peak counter.  MPS does not, so we sample
    ``torch.mps.current_allocated_memory()`` at the points where the peak can
    occur (end of forward, every layer of the backward pass, after the
    optimizer step).  The MPS number is therefore a (tight) lower bound of the
    true allocator peak; ``driver_peak`` additionally reports the Metal driver
    footprint, which includes the allocator cache.
    """

    def __init__(self, device: str):
        self.device = device
        self.reset()

    def reset(self):
        self.peak = 0
        self.driver_peak = 0
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def sample(self):
        if self.device == "mps":
            self.peak = max(self.peak, torch.mps.current_allocated_memory())
            self.driver_peak = max(self.driver_peak, torch.mps.driver_allocated_memory())

    def peak_bytes(self) -> int:
        if self.device == "cuda":
            return torch.cuda.max_memory_allocated()
        return self.peak

    def driver_peak_bytes(self) -> int:
        if self.device == "cuda":
            return torch.cuda.max_memory_reserved()
        return self.driver_peak


MEM: MemTracker | None = None  # set by the trainer; sampled from inside the model


def _mem_sample():
    if MEM is not None:
        MEM.sample()


def _mem_grad_hook(g):
    _mem_sample()
    return g


def _mem_hook(module, inputs, output):
    if MEM is not None and MEM.device == "mps":
        MEM.sample()
        if output.requires_grad:
            output.register_hook(_mem_grad_hook)


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int = 50304  # GPT-2 BPE (50257) padded to a multiple of 64
    seq_len: int = 512
    n_layer: int = 12
    n_head: int = 6
    d_model: int = 384
    arch: str = "standard"  # standard | coupling | midpoint | momentum | euler
    midpoint_h: float = 0.5  # x_{n+1} = x_{n-1} + 2h f(x_n)
    momentum_gamma: float = 0.9
    euler_fp_iters: int = 10
    ce_chunk_tokens: int = 4096  # chunked (checkpointed) LM head + loss
    # MPS materialises the full (B, heads, T, T) attention matrix and MPSGraph rejects
    # tensors with more than INT_MAX elements; split attention over the batch so each
    # call stays below this many score elements (no effect at small batch sizes).
    attn_max_scores: int = 1 << 30


class Attention(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.n_head = c.n_head
        self.max_scores = c.attn_max_scores
        self.ln = nn.LayerNorm(c.d_model)
        self.qkv = nn.Linear(c.d_model, 3 * c.d_model, bias=False)
        self.proj = nn.Linear(c.d_model, c.d_model, bias=False)
        self.proj.is_residual_out = True

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln(x)).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        per_seq = self.n_head * T * T
        if B * per_seq <= self.max_scores:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            step = max(1, self.max_scores // per_seq)
            y = torch.cat([F.scaled_dot_product_attention(q[i:i + step], k[i:i + step], v[i:i + step], is_causal=True)
                           for i in range(0, B, step)])
        _mem_sample()
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.ln = nn.LayerNorm(c.d_model)
        self.fc = nn.Linear(c.d_model, 4 * c.d_model, bias=False)
        self.proj = nn.Linear(4 * c.d_model, c.d_model, bias=False)
        self.proj.is_residual_out = True

    def forward(self, x):
        return self.proj(F.gelu(self.fc(self.ln(x)), approximate="tanh"))


class Block(nn.Module):
    """One transformer block = one Attn + one MLP.  Every arch reuses it, so
    all variants have identical parameter counts; they differ only in how the
    block's outputs are combined into the residual stream(s)."""

    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.attn = Attention(c)
        self.mlp = MLP(c)

    # residual update of a standard block: (x + a + m) - x
    def f(self, x):
        a = self.attn(x)
        return a + self.mlp(x + a)

    # ---- forward step on the (a, b) state; differentiable if grad enabled ----
    def step(self, a, b):
        c = self.c
        if c.arch == "coupling":  # a = x1, b = x2
            y1 = a + self.attn(b)
            y2 = b + self.mlp(y1)
            return y1, y2
        if c.arch == "midpoint":  # a = x_{n-1}, b = x_n
            return b, a + 2 * c.midpoint_h * self.f(b)
        if c.arch == "momentum":  # a = x, b = v
            v = c.momentum_gamma * b + (1 - c.momentum_gamma) * self.f(a)
            return a + v, v
        if c.arch == "euler":  # a = x, b = unused passthrough
            return a + self.f(a), b
        raise ValueError(c.arch)

    # ---- exact / approximate inverse (no grad), used for diagnostics ----
    @torch.no_grad()
    def inverse(self, ya, yb):
        c = self.c
        if c.arch == "coupling":
            x2 = yb - self.mlp(ya)
            x1 = ya - self.attn(x2)
            return x1, x2
        if c.arch == "midpoint":
            return yb - 2 * c.midpoint_h * self.f(ya), ya
        if c.arch == "momentum":
            x = ya - yb
            v = (yb - (1 - c.momentum_gamma) * self.f(x)) / c.momentum_gamma
            return x, v
        if c.arch == "euler":
            x = ya
            for _ in range(c.euler_fp_iters):
                x = ya - self.f(x)
            return x, yb
        raise ValueError(c.arch)

    # ---- fused reconstruct + backward for one layer ----
    def backward_step(self, ya, yb, dya, dyb):
        """Given the layer's outputs (ya, yb) and dL/d(outputs), reconstruct the
        inputs, accumulate parameter grads, and return (xa, xb, dxa, dxb).
        Each variant needs only ONE forward recomputation of its sub-blocks."""
        c = self.c
        if c.arch == "coupling":
            with torch.enable_grad():
                z1 = ya.detach().requires_grad_(True)
                g = self.mlp(z1)
                torch.autograd.backward(g, dyb)
            with torch.no_grad():
                x2 = yb - g
                dz1 = dya + z1.grad
            del g
            with torch.enable_grad():
                z2 = x2.detach().requires_grad_(True)
                fo = self.attn(z2)
                torch.autograd.backward(fo, dz1)
            with torch.no_grad():
                x1 = ya - fo
                dx2 = dyb + z2.grad
            return x1, x2, dz1, dx2

        if c.arch == "midpoint":
            s = 2 * c.midpoint_h
            with torch.enable_grad():
                z = ya.detach().requires_grad_(True)
                fz = self.f(z)
                torch.autograd.backward(fz, s * dyb)
            with torch.no_grad():
                x_prev = yb - s * fz
                dx_prev = dyb
                dx_cur = dya + z.grad
            return x_prev, ya, dx_prev, dx_cur

        if c.arch == "momentum":
            gma = c.momentum_gamma
            with torch.no_grad():
                x = ya - yb
                dv_tot = dyb + dya  # x' = x + v'
            with torch.enable_grad():
                z = x.detach().requires_grad_(True)
                fz = self.f(z)
                torch.autograd.backward(fz, (1 - gma) * dv_tot)
            with torch.no_grad():
                v = (yb - (1 - gma) * fz) / gma
                dx = dya + z.grad
                dv = gma * dv_tot
            return x, v, dx, dv

        if c.arch == "euler":
            x, _ = self.inverse(ya, yb)
            with torch.enable_grad():
                z = x.detach().requires_grad_(True)
                fz = self.f(z)
                torch.autograd.backward(fz, dya)
            with torch.no_grad():
                dx = dya + z.grad
            return x, yb, dx, dyb
        raise ValueError(c.arch)


class _RevStack(torch.autograd.Function):
    """Runs the whole block stack without storing activations."""

    @staticmethod
    def forward(ctx, blocks, amp_ctx, a, b):
        with torch.no_grad(), amp_ctx():
            for blk in blocks:
                a, b = blk.step(a, b)
        a, b = a.detach(), b.detach()
        ctx.blocks, ctx.amp_ctx = blocks, amp_ctx
        ctx.save_for_backward(a, b)
        _mem_sample()
        return a, b

    @staticmethod
    def backward(ctx, da, db):
        a, b = ctx.saved_tensors
        da = torch.zeros_like(a) if da is None else da
        db = torch.zeros_like(b) if db is None else db
        with ctx.amp_ctx():
            for blk in reversed(ctx.blocks):
                a, b, da, db = blk.backward_step(a, b, da, db)
                _mem_sample()
        return None, None, da, db


def _ce_chunk(h, w, t):
    return F.cross_entropy((h @ w.t()).float(), t, reduction="sum")


class GPT(nn.Module):
    def __init__(self, c: ModelConfig):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab_size, c.d_model)
        self.wpe = nn.Embedding(c.seq_len, c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.d_model)
        self.amp_ctx = contextlib.nullcontext  # replaced by the trainer on CUDA
        self.apply(self._init)
        # memory sampling inside every layer: after each Linear/LayerNorm forward and
        # when its output gradient is produced in backward (only active while MEM is set)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.LayerNorm)):
                m.register_forward_hook(_mem_hook)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.c.n_layer) if getattr(m, "is_residual_out", False) else 0.02
            nn.init.normal_(m.weight, 0.0, std)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def num_params(self):
        total = sum(p.numel() for p in self.parameters())
        emb = self.wte.weight.numel() + self.wpe.weight.numel()
        return total, total - emb

    def _initial_state(self, x):
        arch = self.c.arch
        if arch == "standard":
            return x, None
        if arch == "coupling":
            return x, x
        if arch == "midpoint":
            return x, x  # x_{-1} = x_0  -> first step is an Euler step of size 2h
        if arch == "momentum":
            return x, torch.zeros_like(x)
        if arch == "euler":
            return x, torch.zeros(1, device=x.device, dtype=x.dtype)
        raise ValueError(arch)

    def _readout(self, a, b):
        arch = self.c.arch
        if arch == "coupling":
            return 0.5 * (a + b)
        if arch == "midpoint":
            return b
        return a

    def hidden(self, idx, reversible: bool = True):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        if self.c.arch == "standard":
            for blk in self.blocks:
                x = x + blk.attn(x)
                x = x + blk.mlp(x)
                if x.requires_grad:
                    x.register_hook(lambda g: (_mem_sample(), g)[1])
            return self.ln_f(x)
        a, b = self._initial_state(x)
        if reversible and torch.is_grad_enabled():
            a, b = _RevStack.apply(self.blocks, self.amp_ctx, a, b)
        else:  # plain autograd through the same equations (for grad checks / eval)
            for blk in self.blocks:
                a, b = blk.step(a, b)
        return self.ln_f(self._readout(a, b))

    def forward(self, idx, targets, reversible: bool = True):
        h = self.hidden(idx, reversible).reshape(-1, self.c.d_model)
        t = targets.reshape(-1)
        w = self.wte.weight  # tied LM head
        n = h.shape[0]
        step = self.c.ce_chunk_tokens
        total = h.new_zeros((), dtype=torch.float32)
        for i in range(0, n, step):
            if torch.is_grad_enabled():
                total = total + checkpoint(_ce_chunk, h[i : i + step], w, t[i : i + step], use_reentrant=False)
            else:
                total = total + _ce_chunk(h[i : i + step], w, t[i : i + step])
        _mem_sample()
        return total / n


# ----------------------------------------------------------------------------
# diagnostics for reversible variants
# ----------------------------------------------------------------------------
@torch.no_grad()
def reconstruction_error(model: GPT, idx) -> float:
    """Run forward storing every state, then invert from the top and report the
    max relative error of the reconstructed input state."""
    assert model.c.arch != "standard"
    with model.amp_ctx():
        x = model.wte(idx) + model.wpe(torch.arange(idx.shape[1], device=idx.device))
        a, b = model._initial_state(x)
        a0, b0 = a.clone(), b.clone()
        for blk in model.blocks:
            a, b = blk.step(a, b)
        for blk in reversed(model.blocks):
            a, b = blk.inverse(a, b)
    err = (a - a0).abs().max() / a0.abs().max().clamp_min(1e-12)
    return float(err)


def gradient_error(model: GPT, idx, targets) -> float:
    """Relative L2 error between reversible-backward grads and plain-autograd grads."""
    def grads(rev):
        model.zero_grad(set_to_none=True)
        loss = model(idx, targets, reversible=rev)
        loss.backward()
        return [p.grad.detach().clone() for p in model.parameters()]

    g_rev, g_ref = grads(True), grads(False)
    num = sum(float((a - b).pow(2).sum()) for a, b in zip(g_rev, g_ref))
    den = sum(float(b.pow(2).sum()) for b in g_ref)
    model.zero_grad(set_to_none=True)
    return math.sqrt(num / max(den, 1e-30))


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
class TokenData:
    """Non-overlapping (seq_len+1)-token windows in a fixed shuffled order, so
    every run sees exactly the same tokens in the same order, each once."""

    def __init__(self, path: str, seq_len: int, seed: int = 1337):
        self.tok = np.memmap(path, dtype=np.uint16, mode="r")
        self.T = seq_len
        self.n_seq = (len(self.tok) - 1) // seq_len
        self.order = np.random.default_rng(seed).permutation(self.n_seq)

    def batch(self, start_seq: int, B: int, device: str):
        ids = self.order[start_seq : start_seq + B]
        x = np.stack([self.tok[i * self.T : i * self.T + self.T + 1] for i in ids]).astype(np.int64)
        x = torch.from_numpy(x)
        if device == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
        return x[:, :-1], x[:, 1:]


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    name: str = "run"
    batch_size: int = 32  # sequences per optimizer step (no grad accumulation)
    total_tokens: int = 50_000_000
    lr: float = 1e-3
    min_lr_frac: float = 0.1
    warmup_frac: float = 0.02
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every_tokens: int = 5_000_000
    eval_batches: int = 20
    eval_batch_size: int = 16
    seed: int = 0
    amp: bool = True  # bf16 autocast on CUDA; ignored on MPS/CPU (fp32)
    results_dir: str = "results"
    empty_cache_above_tokens: int = 1 << 18  # release cached allocator blocks every step above this batch


def make_model(mc: ModelConfig, device: str, amp: bool = True) -> GPT:
    model = GPT(mc).to(device)
    if device == "cuda" and amp:
        model.amp_ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return model


def make_optimizer(model, tc: TrainConfig, device):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    kw = dict(lr=tc.lr, betas=(0.9, 0.95), eps=1e-8)
    if device == "cuda":
        kw["fused"] = True
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": tc.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], **kw
    )


def lr_at(step, total_steps, tc: TrainConfig):
    warm = max(1, int(tc.warmup_frac * total_steps))
    if step < warm:
        return tc.lr * (step + 1) / warm
    p = (step - warm) / max(1, total_steps - warm)
    return tc.lr * (tc.min_lr_frac + (1 - tc.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate(model, val: TokenData, tc: TrainConfig, device):
    model.eval()
    losses = []
    for i in range(tc.eval_batches):
        x, y = val.batch(i * tc.eval_batch_size, tc.eval_batch_size, device)
        with model.amp_ctx():
            losses.append(float(model(x, y)))
    model.train()
    return sum(losses) / len(losses)


def train_step(model, opt, x, y, tc, lr):
    for g in opt.param_groups:
        g["lr"] = lr
    with model.amp_ctx():
        loss = model(x, y)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
    opt.step()
    opt.zero_grad(set_to_none=True)
    _mem_sample()
    return loss.detach(), gnorm.detach()


def final_train_loss(hist, frac: float = 0.1) -> float:
    """Mean logged training loss over the last `frac` of training tokens."""
    if not hist["loss"]:
        return float("nan")
    cut = hist["tokens"][-1] * (1 - frac)
    tail = [l for t, l in zip(hist["tokens"], hist["loss"]) if t >= cut]
    return float(np.mean(tail))


def train(mc: ModelConfig, tc: TrainConfig, train_bin: str, val_bin: str, device: str | None = None):
    global MEM
    device = device or pick_device()
    torch.manual_seed(tc.seed)
    data, val = TokenData(train_bin, mc.seq_len), TokenData(val_bin, mc.seq_len, seed=0)
    tokens_per_step = tc.batch_size * mc.seq_len
    total_steps = tc.total_tokens // tokens_per_step
    assert total_steps * tc.batch_size <= data.n_seq, "not enough training tokens"

    os.makedirs(tc.results_dir, exist_ok=True)
    logf = open(os.path.join(tc.results_dir, f"{tc.name}.log"), "w")

    def log(msg, **_):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    model = make_model(mc, device, tc.amp)
    opt = make_optimizer(model, tc, device)
    n_total, n_nonemb = model.num_params()
    log(f"[{tc.name}] arch={mc.arch} params={n_total/1e6:.2f}M (non-emb {n_nonemb/1e6:.2f}M) "
          f"device={device} B={tc.batch_size} T={mc.seq_len} tokens/step={tokens_per_step} steps={total_steps}")

    _empty_cache(device)
    MEM = MemTracker(device)
    hist = {"step": [], "tokens": [], "loss": [], "lr": [], "gnorm": [], "tok_s": []}
    evals = {"tokens": [], "val_loss": []}
    next_eval = tc.eval_every_tokens
    train_time = 0.0  # excludes evaluation
    timed_tokens, t_window, tok_window = 0, 0.0, 0
    sync(device)
    t0 = time.perf_counter()
    for step in range(total_steps):
        x, y = data.batch(step * tc.batch_size, tc.batch_size, device)
        loss, gnorm = train_step(model, opt, x, y, tc, lr_at(step, total_steps, tc))
        if tokens_per_step >= tc.empty_cache_above_tokens:
            # huge batches: give every step a clean allocator so fragmentation cannot accumulate
            _empty_cache(device)
        tokens = (step + 1) * tokens_per_step
        if (step + 1) % tc.log_every == 0 or step == total_steps - 1 or step < 3:
            sync(device)
            t1 = time.perf_counter()
            dt = t1 - t0
            train_time += dt
            steps_in_window = (step + 1) - (hist["step"][-1] if hist["step"] else 0)
            tps = steps_in_window * tokens_per_step / dt
            if step >= 3:  # skip warm-up steps from the throughput average
                timed_tokens += steps_in_window * tokens_per_step
                t_window += dt
            l = float(loss)
            hist["step"].append(step + 1); hist["tokens"].append(tokens); hist["loss"].append(l)
            hist["lr"].append(opt.param_groups[0]["lr"]); hist["gnorm"].append(float(gnorm)); hist["tok_s"].append(tps)
            log(f"[{tc.name}] step {step+1:5d}/{total_steps} tok {tokens/1e6:6.2f}M loss {l:.4f} "
                  f"gnorm {float(gnorm):.2f} {tps/1e3:6.1f}k tok/s peak {MEM.peak_bytes()/2**30:.2f}GiB", flush=True)
            if not math.isfinite(l):
                log(f"[{tc.name}] loss diverged, stopping")
                break
            if tokens >= next_eval or step == total_steps - 1:
                vl = evaluate(model, val, tc, device)
                _empty_cache(device)  # release eval-time blocks: avoids fragmentation OOMs at max batch
                evals["tokens"].append(tokens); evals["val_loss"].append(vl)
                log(f"[{tc.name}]   val_loss {vl:.4f}", flush=True)
                next_eval += tc.eval_every_tokens
            sync(device)
            t0 = time.perf_counter()

    res = {
        "name": tc.name,
        "model": asdict(mc),
        "train": asdict(tc),
        "device": device,
        "device_name": torch.cuda.get_device_name() if device == "cuda" else ("Apple " + device.upper()),
        "torch": torch.__version__,
        "params_total": n_total,
        "params_non_embedding": n_nonemb,
        "tokens_per_step": tokens_per_step,
        "steps": len(hist["step"]) and hist["step"][-1],
        "tokens_trained": hist["tokens"][-1] if hist["tokens"] else 0,
        "final_train_loss": final_train_loss(hist),
        "final_val_loss": evals["val_loss"][-1] if evals["val_loss"] else float("nan"),
        "tokens_per_sec": timed_tokens / t_window if t_window else float("nan"),
        "train_time_s": train_time,
        "peak_mem_bytes": MEM.peak_bytes(),
        "driver_peak_mem_bytes": MEM.driver_peak_bytes(),
        "history": hist,
        "evals": evals,
    }
    os.makedirs(tc.results_dir, exist_ok=True)
    with open(os.path.join(tc.results_dir, f"{tc.name}.json"), "w") as f:
        json.dump(res, f, indent=1)
    log(f"[{tc.name}] DONE final_train_loss={res['final_train_loss']:.4f} final_val_loss={res['final_val_loss']:.4f} "
          f"tok/s={res['tokens_per_sec']:.0f} peak={res['peak_mem_bytes']/2**30:.2f}GiB time={train_time/60:.1f}min")
    MEM = None
    logf.close()
    return res, model


# ----------------------------------------------------------------------------
# max-batch search
# ----------------------------------------------------------------------------
def _is_oom(e: BaseException) -> bool:
    s = str(e).lower()
    return ("out of memory" in s or "larger than int_max" in s)


def _empty_cache(device):
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def probe_batch(mc: ModelConfig, B: int, device: str, steps: int = 2, amp: bool = True):
    """Try `steps` full training steps at batch size B.  Returns dict with
    ok / peak memory / tokens per second, never raises on OOM."""
    global MEM
    model = opt = None
    try:
        model = make_model(mc, device, amp)
        opt = make_optimizer(model, TrainConfig(), device)
        MEM = MemTracker(device)
        x = torch.randint(0, 50257, (B, mc.seq_len), device=device)
        tc = TrainConfig()
        train_step(model, opt, x, x, tc, 1e-4)  # warm-up step
        sync(device)
        t0 = time.perf_counter()
        for _ in range(steps):
            loss, _ = train_step(model, opt, x, x, tc, 1e-4)
        float(loss)
        sync(device)
        dt = time.perf_counter() - t0
        return {"B": B, "ok": True, "peak_mem_bytes": MEM.peak_bytes(),
                "driver_peak_mem_bytes": MEM.driver_peak_bytes(),
                "tokens_per_sec": steps * B * mc.seq_len / dt}
    except (RuntimeError, torch.OutOfMemoryError) as e:
        if not _is_oom(e):
            raise
        return {"B": B, "ok": False, "error": str(e).splitlines()[0][:200]}
    finally:
        MEM = None
        del model, opt
        _empty_cache(device)


def find_max_batch(mc: ModelConfig, device: str, start: int = 16, limit: int = 1 << 14, amp: bool = True, log=print):
    """Doubling search followed by a binary search to the nearest multiple of 8."""
    trials = []
    lo, hi, B = 0, None, start
    while B <= limit:
        r = probe_batch(mc, B, device, steps=1, amp=amp); trials.append(r); log(r)
        if r["ok"]:
            lo, B = B, B * 2
        else:
            hi = B
            break
    if hi is None:
        return lo, trials
    while hi - lo > 8:
        mid = (lo + hi) // 2 // 8 * 8
        if mid <= lo:
            break
        r = probe_batch(mc, mid, device, steps=1, amp=amp); trials.append(r); log(r)
        if r["ok"]:
            lo = mid
        else:
            hi = mid
    return lo, trials
