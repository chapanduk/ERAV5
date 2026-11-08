"""Generates the experiment notebooks (plain JSON, no nbformat dependency)."""
import json

REPO_URL = "https://github.com/chapanduk/ERAV5.git"
REPO_BRANCH, REPO_SUBDIR = "session13", "session13"

SETUP = f'''# --- setup: works on Google Colab (CUDA) and locally (Apple MPS / CUDA / CPU) ---
import os, sys
# MPS: make the allocator raise a clean OOM at the GPU's recommended working-set size
# instead of spilling into swap (must be set before torch is imported).
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.9")
if "google.colab" in sys.modules and not os.path.exists("revllm.py"):
    !git clone -q -b {REPO_BRANCH} {REPO_URL} /content/ERAV5
    %cd /content/ERAV5/{REPO_SUBDIR}
    !pip -q install tiktoken pyarrow
if not os.path.exists("data/train.bin"):
    !python prepare_data.py
import json, torch, revllm
from revllm import ModelConfig, TrainConfig, train, pick_device
DEVICE = pick_device()
print("torch", torch.__version__, "| device", DEVICE,
      "|", torch.cuda.get_device_name() if DEVICE == "cuda" else "")
TRAIN_BIN, VAL_BIN = "data/train.bin", "data/val.bin"
GiB = 2**30'''


def nb(cells):
    out = []
    for kind, src in cells:
        c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
        if kind == "code":
            c.update(outputs=[], execution_count=None)
        out.append(c)
    return {
        "cells": out,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(s):
    return ("markdown", s)


def code(s):
    return ("code", s)


notebooks = {}

notebooks["00_prepare_data.ipynb"] = nb([
    md("""# 00 - Data: a 51.5M-token slice of FineWeb-Edu

* Source: `HuggingFaceFW/fineweb-edu`, `sample/10BT/000_00000.parquet` (first shard, 2.15 GB)
* Tokenizer: GPT-2 BPE (`tiktoken`), `<|endoftext|>` appended to every document
* First 1.0M tokens -> `data/val.bin` (held out), next 50.5M tokens -> `data/train.bin`
* Every training run reads non-overlapping 512-token windows in the **same fixed shuffled order**,
  so each token is seen exactly once (single epoch) and all runs see identical data."""),
    code(SETUP),
    code('''import numpy as np, tiktoken
tr = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r"); va = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
print(f"train tokens: {len(tr):,}   val tokens: {len(va):,}")
enc = tiktoken.get_encoding("gpt2")
print("docs in train slice:", int((tr == enc.eot_token).sum()))
print("sample:", repr(enc.decode(tr[:120].tolist())))'''),
])

notebooks["01_baseline.ipynb"] = nb([
    md("""# 01 - Baseline: standard GPT, fixed batch size

Model: 12 layers, d_model 384, 6 heads, context 512, GPT-2 vocab (tied embeddings)
-> **21.3M non-embedding parameters** (40.6M including the tied 50k x 384 embedding table).

Fixed batch: **32 sequences x 512 tokens = 16,384 tokens/step**, 3,051 steps = 50M tokens.
AdamW (0.9, 0.95), lr 1e-3, 2% warm-up, cosine to 10%, wd 0.1, clip 1.0.
The LM head + cross-entropy is computed in checkpointed 4k-token chunks in *every* run, so the
50k-vocab logits never dominate memory and the comparison isolates the transformer blocks."""),
    code(SETUP),
    code('''mc = ModelConfig(arch="standard")
tc = TrainConfig(name="01_baseline_standard_B32", batch_size=32)
res, model = train(mc, tc, TRAIN_BIN, VAL_BIN, DEVICE)'''),
    code('''print(json.dumps({k: res[k] for k in ["final_train_loss", "final_val_loss", "tokens_per_sec",
      "peak_mem_bytes", "driver_peak_mem_bytes", "train_time_s", "params_total", "params_non_embedding"]}, indent=1))
print(f"peak memory: {res['peak_mem_bytes']/GiB:.2f} GiB")'''),
])

notebooks["02_reversible.ipynb"] = nb([
    md("""# 02 - Reversible residual streams: which variant works?

All variants use the *same* blocks (1 attention + 1 MLP per layer, identical parameter count) and only
change how block outputs are combined. In reversible mode the forward pass keeps **no per-layer
activations**; the backward pass reconstructs each layer's input from its output and recomputes it.

| variant | update | inverse |
|---|---|---|
| `coupling` (RevNet / Reformer, "symplectic Euler" on a split state) | y1 = x1 + Attn(x2); y2 = x2 + MLP(y1) | exact: x2 = y2 - MLP(y1); x1 = y1 - Attn(x2) |
| `midpoint` (explicit midpoint / leapfrog) | x_{n+1} = x_{n-1} + 2h f(x_n) | exact: x_{n-1} = x_{n+1} - 2h f(x_n) |
| `momentum` (Momentum ResNet, gamma=0.9) | v' = g v + (1-g) f(x); x' = x + v' | exact, but divides by g |
| `euler` (plain residual) | x' = x + f(x) | **approximate**: fixed-point x <- y - f(x), 10 iters |

Plan: (1) check reconstruction and gradient error vs. plain autograd, (2) 2M-token screening run of
each variant at B=32, (3) train the best one for the full 50M tokens at B=32 (same as the baseline)."""),
    code(SETUP),
    md("## 1. Numerical checks (reconstruction + gradient error)"),
    code('''torch.manual_seed(0)
data = revllm.TokenData(TRAIN_BIN, 512)
x, y = data.batch(0, 4, DEVICE)
checks = {}
for arch in ["coupling", "midpoint", "momentum", "euler"]:
    m = revllm.make_model(ModelConfig(arch=arch), DEVICE)
    checks[arch] = {"recon_err_init": revllm.reconstruction_error(m, x),
                    "grad_err_init": revllm.gradient_error(m, x, y)}
    # stress test: weights scaled up 4x (a proxy for a trained network with larger residual updates)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() >= 2 and "blocks" in n: p.mul_(4.0)
    checks[arch]["recon_err_w4x"] = revllm.reconstruction_error(m, x)
    checks[arch]["grad_err_w4x"] = revllm.gradient_error(m, x, y)
    print(arch, {k: f"{v:.2e}" for k, v in checks[arch].items()})
    del m
os.makedirs("results", exist_ok=True)
json.dump(checks, open("results/02_numerical_checks.json", "w"), indent=1)'''),
    md("## 2. Screening: 2M tokens per variant (B=32)"),
    code('''SCREEN_TOKENS = 2_000_000
screen = {}
for arch in ["standard", "coupling", "midpoint", "momentum", "euler"]:
    r, m = train(ModelConfig(arch=arch),
                 TrainConfig(name=f"02_screen_{arch}", batch_size=32, total_tokens=SCREEN_TOKENS,
                             eval_every_tokens=SCREEN_TOKENS, log_every=50),
                 TRAIN_BIN, VAL_BIN, DEVICE)
    if arch != "standard":  # reconstruction error of the *trained* weights
        r["recon_err_trained"] = revllm.reconstruction_error(m, x)
    screen[arch] = {k: r.get(k) for k in ["final_train_loss", "final_val_loss", "tokens_per_sec",
                                          "peak_mem_bytes", "recon_err_trained"]}
    del m
for a, s in screen.items():
    print(f"{a:9s} val {s['final_val_loss']:.4f}  {s['tokens_per_sec']/1e3:6.1f}k tok/s  "
          f"peak {s['peak_mem_bytes']/GiB:5.2f} GiB  recon_err {s['recon_err_trained']}")
json.dump(screen, open("results/02_screen_summary.json", "w"), indent=1)'''),
    code('''import math
ok = {a: s for a, s in screen.items() if a != "standard" and math.isfinite(s["final_val_loss"])
      and (s["recon_err_trained"] or 1) < 1e-3}
BEST = min(ok, key=lambda a: ok[a]["final_val_loss"])
print("best reversible variant:", BEST)
json.dump({"best": BEST}, open("results/02_best_variant.json", "w"))'''),
    md("## 3. Full 50M-token run with the best reversible variant, B=32 (same as baseline)"),
    code('''res, model = train(ModelConfig(arch=BEST),
                   TrainConfig(name=f"02_reversible_{BEST}_B32", batch_size=32),
                   TRAIN_BIN, VAL_BIN, DEVICE)
print("reconstruction error after training:", revllm.reconstruction_error(model, x))
print("gradient error after training:", revllm.gradient_error(model, x, y))'''),
])

notebooks["03_reversible_max_batch.ipynb"] = nb([
    md("""# 03 - Reversible at the maximum batch size

1. Find the largest batch (in sequences of 512 tokens) that completes a full training step, for the
   standard model and for the best reversible variant (doubling + binary search, granularity 8).
2. Train the reversible model for the same 50M tokens at that batch size.
   With a fixed token budget, a bigger batch means fewer optimizer steps, so the learning rate is scaled
   with sqrt(B/32) (capped at 3e-3) and warm-up is set to 5% of steps.

Peak memory here uses fine-grained sampling (after every Linear/LayerNorm/attention call and on each of
their gradients), which also captures the transient activations *inside* a layer.
On MPS, attention materialises the full (B, heads, T, T) score matrix and MPSGraph rejects tensors with more
than INT_MAX elements (first hit at B=1,536), so attention is split over the batch above 2^30 scores;
the maximum batch is then set by memory, not by that backend limit."""),
    code(SETUP),
    code('''BEST = json.load(open("results/02_best_variant.json"))["best"]
print("reversible variant:", BEST)
SEARCH_JSON = "results/03_max_batch_search.json"
if os.path.exists(SEARCH_JSON):  # reuse a completed search (it takes ~10 min); delete the file to re-run
    search = json.load(open(SEARCH_JSON))
    print("loaded cached search from", SEARCH_JSON)
else:
    search = {}
    for arch in ["standard", BEST]:
        maxB, trials = revllm.find_max_batch(ModelConfig(arch=arch), DEVICE, start=32, limit=8192)
        search[arch] = {"max_batch": maxB, "trials": trials}
    json.dump(search, open(SEARCH_JSON, "w"), indent=1)
for arch, s in search.items():
    print(f"=== {arch}: max batch (1-2 step probe) = {s['max_batch']} sequences = {s['max_batch']*512:,} tokens/step")
    for t in s["trials"]:
        print(f"   B={t['B']:5d} " + (f"ok   peak {t['peak_mem_bytes']/GiB:6.2f} GiB  {t['tokens_per_sec']:8.0f} tok/s" if t["ok"]
              else "FAIL " + t["error"][:90]))'''),
    code('''import math, gc
# The probe max only has to survive 1-2 steps on a fresh allocator. Sustained training fragments the MPS
# allocator, so at ~99% of the limit it can still OOM a few steps in. Start at the probe max and step down
# by 64 sequences on OOM; the first batch that trains for the full 50M tokens is the sustained max batch.
B = search[BEST]["max_batch"]
attempts = []
while True:
    lr = min(3e-3, 1e-3 * math.sqrt(B / 32))
    print(f"training {BEST} at B={B} ({B*512:,} tok/step, {50_000_000//(B*512)} steps), lr={lr:.2e}")
    try:
        res, model = train(ModelConfig(arch=BEST),
                           TrainConfig(name=f"03_reversible_{BEST}_maxB{B}", batch_size=B, lr=lr,
                                       warmup_frac=0.05, log_every=1),
                           TRAIN_BIN, VAL_BIN, DEVICE)
        attempts.append({"B": B, "ok": True}); break
    except (RuntimeError, torch.OutOfMemoryError) as e:
        if not revllm._is_oom(e): raise
        revllm.MEM = None
        msg = str(e).splitlines()[0][:160]
        attempts.append({"B": B, "ok": False, "error": msg}); print("  OOM:", msg)
        gc.collect(); torch.mps.empty_cache() if DEVICE == "mps" else torch.cuda.empty_cache()
        B -= 64
json.dump(attempts, open("results/03_sustained_batch_attempts.json", "w"), indent=1)
print("sustained max batch:", B, attempts)'''),
])

notebooks["04_report.ipynb"] = nb([
    md("# 04 - Results, plots and summary table"),
    code('''import json, glob, os
import matplotlib.pyplot as plt
import revllm
GiB = 2**30
R = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob("results/0*_*.json"))
     if "history" in json.load(open(p))}
for r in R.values():  # recompute with the final metric definition (mean over last 10% of tokens)
    r["final_train_loss"] = revllm.final_train_loss(r["history"])
main = [k for k in R if k.startswith(("01_", "02_reversible", "03_"))]
print("| run | arch | batch (seq x 512) | steps | final train loss | final val loss | tokens/s | peak mem (GiB) | time (min) |")
print("|---|---|---|---|---|---|---|---|---|")
for k in main + [k for k in R if k.startswith("02_screen")]:
    r = R[k]
    print(f"| {k} | {r['model']['arch']} | {r['train']['batch_size']} | {r['steps']} | {r['final_train_loss']:.4f} | "
          f"{r['final_val_loss']:.4f} | {r['tokens_per_sec']:,.0f} | {r['peak_mem_bytes']/GiB:.2f} | {r['train_time_s']/60:.1f} |")'''),
    code('''os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for k in main:
    h, e = R[k]["history"], R[k]["evals"]
    ax[0].plot([t/1e6 for t in h["tokens"]], h["loss"], label=k, lw=1)
    ax[1].plot([t/1e6 for t in e["tokens"]], e["val_loss"], "o-", label=k)
for a, t in zip(ax, ["train loss", "validation loss"]):
    a.set_xlabel("tokens (M)"); a.set_title(t); a.grid(alpha=.3); a.legend(fontsize=8)
ax[0].set_ylim(3, 8)
plt.tight_layout(); plt.savefig("figures/loss_curves.png", dpi=130); plt.show()'''),
    code('''fig, ax = plt.subplots(figsize=(7, 4.5))
for k in [k for k in R if k.startswith("02_screen")]:
    h = R[k]["history"]
    ax.plot([t/1e6 for t in h["tokens"]], h["loss"], label=R[k]["model"]["arch"], lw=1)
ax.set_ylim(4, 9); ax.set_xlabel("tokens (M)"); ax.set_title("2M-token screening of reversible variants (B=32)")
ax.grid(alpha=.3); ax.legend(); plt.tight_layout(); plt.savefig("figures/screening.png", dpi=130); plt.show()'''),
    code('''S = json.load(open("results/03_max_batch_search.json"))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for arch, s in S.items():
    ok = sorted([t for t in s["trials"] if t["ok"]], key=lambda t: t["B"])
    ax[0].plot([t["B"] for t in ok], [t["peak_mem_bytes"]/GiB for t in ok], "o-", label=arch)
    ax[1].plot([t["B"] for t in ok], [t["tokens_per_sec"]/1e3 for t in ok], "o-", label=arch)
ax[0].set_ylabel("peak memory (GiB)"); ax[1].set_ylabel("throughput (k tokens/s)")
for a in ax: a.set_xscale("log", base=2); a.set_xlabel("batch size (sequences of 512)"); a.grid(alpha=.3); a.legend()
plt.tight_layout(); plt.savefig("figures/batch_scaling.png", dpi=130); plt.show()
print({a: s["max_batch"] for a, s in S.items()})'''),
])

import sys

# `python make_notebooks.py [name.ipynb ...]` - regenerate only the named notebooks
# (so executed notebooks with outputs are not overwritten).
for name, content in notebooks.items():
    if len(sys.argv) > 1 and name not in sys.argv[1:]:
        continue
    with open(name, "w") as f:
        json.dump(content, f, indent=1)
    print("wrote", name)
