"""Task 5: sweep the peak LR at widths 256, 512 and 1024 (4 layers, head_dim 64, standard parametrisation),
300-step cosine with 30-step warmup, scored on val loss at step 300. 2 seeds per point.
If a width's best LR is at a grid edge, the grid is extended by 2x on that side until it is bracketed.
Resumable: finished runs are stored in lr_sweep_results.json and skipped.
"""
import math, sys
from common import Cfg, ResultStore, train

WIDTHS = [256, 512, 1024]
GRID = [1.25e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2]
SEEDS = [0, 1, 2]   # 3rd seed added after the 2-seed curves showed a shared dip at 1e-3 (same batch order at every width)
# The factor-2 grid put every width's best at 1e-3 with a sharp cliff to its right, so the optimum's
# movement with width is smaller than one grid cell. Refine to quarter-octave spacing around it.
REFINE = [7.07e-4, 8.41e-4, 1.19e-3, 1.41e-3]
store = ResultStore("lr_sweep_results.json")


def point(d, lr, seed):
    key = f"d={d}|lr={lr:g}|seed={seed}"
    if key not in store:
        log = train(Cfg(d=d), lr, sched="cosine", total=300, warmup=30, seed=seed)
        store.put(key, dict(final_val=log["final_val"], loss=log["loss"][::10], seconds=log["seconds"]))
        print(f"  {key}: val {log['final_val']:.4f} ({log['seconds']:.0f}s)", flush=True)
    return store[key]["final_val"]


def curve(d, lrs):
    out = {}
    for lr in lrs:
        v = [point(d, lr, s) for s in SEEDS]
        v = [x if math.isfinite(x) else float("nan") for x in v]
        out[lr] = sum(v) / len(v)
    return out


if __name__ == "__main__":
    for d in WIDTHS:
        lrs = list(GRID)
        for _ in range(4):
            c = curve(d, lrs)
            best = min(c, key=lambda k: c[k] if math.isfinite(c[k]) else 9e9)
            if best == lrs[0]:
                lrs = [lrs[0] / 2] + lrs
            elif best == lrs[-1]:
                lrs = lrs + [lrs[-1] * 2]
            else:
                break
        c.update(curve(d, [lr for lr in REFINE if lr not in c]))
        best = min(c, key=lambda k: c[k] if math.isfinite(c[k]) else 9e9)
        print(f"d={d}: " + "  ".join(f"{lr:g}:{v:.4f}" for lr, v in sorted(c.items())) + f"   grid best {best:g}", flush=True)
