"""Task 2: Adam with and without bias correction.

Without correction, the step at time t is (1 - b1^t) / sqrt(1 - b2^t) times the corrected one (eps aside).
That factor is computed exactly below, then both versions train the same model (same init, same batches)
from step 0 with a constant LR and no warmup, so nothing hides the early steps.
"""
import math, sys
import numpy as np
import torch
from common import Cfg, ResultStore, train, DEV

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
LR, B1 = 1e-3, 0.9
BETA2S, SEEDS = [0.95, 0.999], [0, 1]


class MyAdam(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, bias_correction=True):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))
        self.bc = bias_correction

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            b1, b2 = grp["betas"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"], st["m"], st["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
                st["t"] += 1; t = st["t"]
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                mh = st["m"] / (1 - b1 ** t) if self.bc else st["m"]
                vh = st["v"] / (1 - b2 ** t) if self.bc else st["v"]
                p.add_(-grp["lr"] * mh / (vh.sqrt() + grp["eps"]))


def factor(t, b2, b1=B1):
    """uncorrected step / corrected step at step t (ignoring eps)."""
    return (1 - b1 ** t) / math.sqrt(1 - b2 ** t)


if __name__ == "__main__":
    # 0. my optimizer with correction on must be PyTorch's Adam
    cfg = Cfg(d=256)
    a = train(cfg, LR, sched="constant", total=10, warmup=0, betas=(0.9, 0.999))
    b = train(cfg, LR, sched="constant", total=10, warmup=0,
              opt_factory=lambda ps, lr: MyAdam(ps, lr, (0.9, 0.999), bias_correction=True))
    print("sanity: MyAdam(bias_correction=True) vs torch Adam, max |loss diff| over 10 steps:",
          f"{max(abs(x - y) for x, y in zip(a['loss'], b['loss'])):.1e}")

    # 1. the exact factor
    print("\nuncorrected / corrected step size, (1-b1^t)/sqrt(1-b2^t), b1 = 0.9:")
    for b2 in BETA2S:
        f = [factor(t, b2) for t in range(1, 20001)]
        peak = int(np.argmax(f)) + 1
        within = {tol: next(t for t in range(peak, 20001) if abs(f[t - 1] - 1) < tol) for tol in (0.10, 0.01)}
        print(f"  b2={b2}: t=1 {f[0]:.2f}x, t=20 {f[19]:.2f}x, peak {max(f):.2f}x at t={peak}; "
              f"within 10% of 1 from t={within[0.10]}, within 1% from t={within[0.01]}")

    # 2. train both ways
    store = ResultStore("bias_results.json")
    for b2 in BETA2S:
        for seed in SEEDS:
            for bc in [True, False]:
                key = f"b2={b2}|seed={seed}|bc={bc}"
                if key in store:
                    continue
                log = train(cfg, LR, sched="constant", total=STEPS, warmup=0, seed=seed,
                            opt_factory=lambda ps, lr, b2=b2, bc=bc: MyAdam(ps, lr, (B1, b2), bias_correction=bc))
                store.put(key, dict(loss=log["loss"], final_val=log["final_val"]))
                print(f"  {key}: final val {log['final_val']:.4f}  ({log['seconds']:.0f}s)", flush=True)

    # 3. tune both sides: each (b2, bias_correction) variant at its own best LR, so the comparison isn't
    #    "corrected Adam at a good LR vs uncorrected Adam at a bad one" (or the reverse).
    if len(sys.argv) > 2 and sys.argv[2] == "tune":
        TUNE_STEPS, LRS = 1000, [2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3]
        tune = ResultStore("bias_tune_results.json")
        for b2 in BETA2S:
            for bc in [True, False]:
                for lr in LRS:
                    key = f"b2={b2}|bc={bc}|lr={lr}"
                    if key in tune:
                        continue
                    log = train(cfg, lr, sched="constant", total=TUNE_STEPS, warmup=0, seed=0,
                                opt_factory=lambda ps, lr, b2=b2, bc=bc: MyAdam(ps, lr, (B1, b2), bias_correction=bc))
                    tune.put(key, dict(final_val=log["final_val"], loss=log["loss"]))
                    print(f"  tune {key}: val@{TUNE_STEPS} {log['final_val']:.4f}", flush=True)
        def tune_run(b2, bc, lr):
            key = f"b2={b2}|bc={bc}|lr={lr}"
            if key not in tune:
                log = train(cfg, lr, sched="constant", total=TUNE_STEPS, warmup=0, seed=0,
                            opt_factory=lambda ps, lr, b2=b2, bc=bc: MyAdam(ps, lr, (B1, b2), bias_correction=bc))
                tune.put(key, dict(final_val=log["final_val"], loss=log["loss"]))
                print(f"  tune {key}: val@{TUNE_STEPS} {log['final_val']:.4f}", flush=True)
            return tune[key]["final_val"]

        for b2 in BETA2S:
            for bc in [True, False]:
                lrs = list(LRS)
                for _ in range(4):   # if the optimum sits on the grid's lower edge, extend down by 2x until bracketed
                    vals = {lr: tune_run(b2, bc, lr) for lr in lrs}
                    if min(vals, key=lambda k: vals[k] if math.isfinite(vals[k]) else 9e9) != lrs[0]:
                        break
                    lrs = [lrs[0] / 2] + lrs
                best = min(vals, key=lambda k: vals[k] if math.isfinite(vals[k]) else 9e9)
                print(f"  b2={b2} bias_correction={bc!s:<5}: best lr {best:g} -> val {vals[best]:.4f}   all: "
                      + "  ".join(f"{k:g}:{v:.3f}" for k, v in vals.items()))

        # confirm each variant at its own best LR on more seeds, so "within noise" is measured, not assumed
        print("\n  at each variant's own best LR, seeds 0-2 (val @ %d):" % TUNE_STEPS)
        for b2 in BETA2S:
            for bc in [True, False]:
                cands = {float(k.split("lr=")[1]): v["final_val"] for k, v in tune.data.items()
                         if k.startswith(f"b2={b2}|bc={bc}|lr=")}
                lr = min(cands, key=cands.get)
                vals = [cands[lr]]
                for seed in [1, 2]:
                    key = f"b2={b2}|bc={bc}|lr={lr}|seed={seed}"
                    if key not in tune:
                        log = train(cfg, lr, sched="constant", total=TUNE_STEPS, warmup=0, seed=seed,
                                    opt_factory=lambda ps, lr, b2=b2, bc=bc: MyAdam(ps, lr, (B1, b2), bias_correction=bc))
                        tune.put(key, dict(final_val=log["final_val"]))
                    vals.append(tune[key]["final_val"])
                m = sum(vals) / 3; sd = (sum((v - m) ** 2 for v in vals) / 2) ** 0.5
                print(f"  b2={b2} bias_correction={bc!s:<5} lr {lr:g}: mean {m:.4f}  sd {sd:.4f}  seeds {[round(v, 4) for v in vals]}")
