"""Q4 + Q5: a real loop that logs loss and (pre-clip) grad norm every step, and times itself for MFU.

Runs: 'base' = lr 1e-3, 100-step warmup, cosine, clip 1.0.
      'hot'  = lr 4e-3, no warmup, constant, no clipping -> invites instabilities to look at.
"""
import json, math, os, sys, time
import torch, torch.nn.functional as F
from common import Cfg, GPT, load_corpus, get_batch, pick_device, flops_per_token

dev = pick_device()
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
B, c = 32, Cfg()
T = c.ctx
train, val = load_corpus()
CKPT_DIR, CKPT_EVERY = "ckpt", 100
os.makedirs(CKPT_DIR, exist_ok=True)


def save_ckpt(path, step, model, opt, g, log):
    """Everything needed to continue exactly: weights, AdamW moments + step count, the next step index
    (which also fixes the LR schedule), the data sampler's RNG state, and the logs so far."""
    state = dict(step=step, model=model.state_dict(), opt=opt.state_dict(), data_rng=g.get_state(),
                 torch_rng=torch.get_rng_state(), log=log, steps_total=STEPS)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)   # atomic: a kill mid-save never leaves a half-written checkpoint


RUNS = {"base": dict(lr=1e-3, warmup=100, cosine=True, clip=1.0),
        "hot": dict(lr=4e-3, warmup=0, cosine=False, clip=None)}

out = {}
for name, h in RUNS.items():
    torch.manual_seed(0)
    model = GPT(c).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=h["lr"], betas=(0.9, 0.95), weight_decay=0.1)
    g = torch.Generator().manual_seed(0)
    log = dict(loss=[], gnorm=[], lr=[], dt=[])
    path, start = os.path.join(CKPT_DIR, f"{name}.pt"), 0
    if os.path.exists(path):
        ck = torch.load(path, map_location=dev, weights_only=False)
        assert ck["steps_total"] == STEPS, f"{path} was made for {ck['steps_total']} steps; delete it or match STEPS"
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        g.set_state(ck["data_rng"].cpu()); torch.set_rng_state(ck["torch_rng"].cpu())  # RNG states must stay on CPU
        log, start = ck["log"], ck["step"]
        print(f"{name}: resumed from {path} at step {start}" + (" (already finished)" if start >= STEPS else ""), flush=True)
    for s in range(start, STEPS):
        lr = h["lr"] * min(1.0, (s + 1) / h["warmup"]) if h["warmup"] else h["lr"]
        if h["cosine"] and s >= h["warmup"]:
            lr = h["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - h["warmup"]) / (STEPS - h["warmup"]))))
        for pg in opt.param_groups: pg["lr"] = lr
        torch.mps.synchronize(); t0 = time.perf_counter()
        x, y = get_batch(train, B, T, g, dev)
        loss = F.cross_entropy(model(x).view(-1, c.vocab), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), h["clip"] if h["clip"] else float("inf"))
        opt.step()
        l, gnv = loss.item(), gn.item()          # .item() forces a GPU sync every step: part of the cost
        log["dt"].append(time.perf_counter() - t0)
        log["loss"].append(l); log["gnorm"].append(gnv); log["lr"].append(lr)
        if s % 250 == 0:
            print(f"{name} step {s:5d} loss {l:.4f} gnorm {gnv:.3f} lr {lr:.2e} dt {log['dt'][-1]*1e3:.1f}ms", flush=True)
        if (s + 1) % CKPT_EVERY == 0 or s + 1 == STEPS:
            save_ckpt(path, s + 1, model, opt, g, log)
    out[name] = log

fpt = flops_per_token(model, T)
dts = out["base"]["dt"][50:]
step_s = sorted(dts)[len(dts) // 2]
out["mfu"] = dict(flops_per_token=fpt, tokens_per_step=B * T, median_step_s=step_s,
                  achieved_tflops=fpt * B * T / step_s / 1e12)
print(out["mfu"])
json.dump(out, open("train_results.json", "w"))
