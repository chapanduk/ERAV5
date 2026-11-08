"""Task 3: per-layer update-to-weight ratio ||dW|| / ||W|| at every step, under linear warmup then constant LR.

Repeated for three warmup lengths. If warmup is what drives the ratio, the step where the ratio stops
moving should move with the warmup length.
"""
import numpy as np
from common import Cfg, ResultStore, train

PEAK, STEPS, WARMUPS = 2e-3, 600, [50, 100, 200]
REF = 0   # no warmup: the counterfactual each warmup run is compared against
store = ResultStore("ratio_results.json")
for w in [REF] + WARMUPS:
    key = f"warmup={w}"
    if key not in store:
        log = train(Cfg(d=256), PEAK, sched="constant", total=STEPS, warmup=w, seed=0, log_ratio=True)
        store.put(key, dict(ratio=log["ratio"], lr=log["lr"], loss=log["loss"]))
        print(f"{key} done ({log['seconds']:.0f}s)", flush=True)
if "warmup=0|seed=1" not in store:   # noise reference: same no-warmup run, different init + batch order
    log = train(Cfg(d=256), PEAK, sched="constant", total=STEPS, warmup=0, seed=1, log_ratio=True)
    store.put("warmup=0|seed=1", dict(ratio=log["ratio"], lr=log["lr"], loss=log["loss"]))
    print("noise reference done", flush=True)


def rolling_median(v, k=21):
    v = np.asarray(v); h = k // 2
    return np.array([np.median(v[max(0, i - h): i + h + 1]) for i in range(len(v))])


def seed_noise(n, start=100, k=41):
    """How far two no-warmup runs that differ only in seed sit apart for this layer (median relative gap,
    steps >= start). Warmup 'still changing' the ratio has to mean a bigger gap than this."""
    a = rolling_median(store["warmup=0"]["ratio"][n], k); b = rolling_median(store["warmup=0|seed=1"]["ratio"][n], k)
    return float(np.median(np.abs(a / b - 1)[start:]))


def washout_step(r_w, r_ref, tol=0.10, k=41):
    """First step after which the smoothed ratio with warmup stays within `tol` (relative) of the smoothed
    ratio of the identical run without warmup, i.e. warmup no longer changes it. The ratio keeps drifting
    on its own after warmup (weights grow, gradients get noisier), so 'stopped moving' is the wrong test;
    'stopped differing from no-warmup' is the right one."""
    a, b = rolling_median(r_w, k), rolling_median(r_ref, k)
    off = np.where(np.abs(a / b - 1) > tol)[0]
    return int(off[-1] + 2) if len(off) else 1


if __name__ == "__main__":
    for w in WARMUPS:
        R = store[f"warmup={w}"]["ratio"]
        print(f"\n== warmup {w} steps, then constant lr {PEAK} ==")
        R0 = store[f"warmup={REF}"]["ratio"]
        print(f"{'layer':<22} {'ratio@1':>9} {'@warmup end':>12} {'no-warmup @same':>16} {'tol (2x seed gap)':>17} {'warmup stops changing it at':>28}")
        steps = []
        for n, r in R.items():
            sm, s0 = rolling_median(r), rolling_median(R0[n])
            tol = max(0.05, 2 * seed_noise(n))     # 2x the seed-to-seed gap, floor 5%
            s = washout_step(r, R0[n], tol); steps.append(s)
            print(f"{n:<22} {r[0]:>9.2e} {sm[w - 1]:>12.2e} {s0[w - 1]:>16.2e} {tol:>17.1%} {('never (<=600)' if s > STEPS else s):>28}")
        done = [x for x in steps if x <= STEPS]
        q = np.percentile(done, [25, 50, 75]).astype(int) if done else [0, 0, 0]
        print(f"{len(done)}/{len(steps)} matrices rejoin the no-warmup curve; among them median step {q[1]} "
              f"(IQR {q[0]}-{q[2]}) = {q[1] - w:+d} steps after warmup ends at {w}. "
              f"Never rejoin: {[n for n, x in zip(R, steps) if x > STEPS]}")
