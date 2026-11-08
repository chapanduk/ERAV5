"""Figure for task 2 (reads bias_results.json and bias_tune_results.json)."""
import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bias_correction import factor, BETA2S
from style import COLORS, INK2, finish, label_end, setup

setup()
R, Tn = json.load(open("bias_results.json")), json.load(open("bias_tune_results.json"))
col = {0.95: COLORS[0], 0.999: COLORS[1]}
ls = {True: "-", False: (0, (4, 2))}
lab = lambda b2, bc: f"b2={b2}, {'corrected' if bc else 'NOT corrected'}"
fig, ax = plt.subplots(1, 4, figsize=(20, 4.8))

a = ax[0]
t = np.arange(1, 10001)
for b2 in BETA2S:
    f = np.array([factor(i, b2) for i in t])
    a.plot(t, f, color=col[b2], lw=2, label=f"b2={b2}")
    t1 = int(next(i for i in range(int(np.argmax(f)), len(f)) if abs(f[i] - 1) < 0.01)) + 1
    a.axvline(t1, color=col[b2], lw=1, ls=(0, (2, 3)))
    a.annotate(f"within 1% from t={t1}", (t1, 1.0), xytext=(4, 30 if b2 == 0.95 else 60),
               textcoords="offset points", fontsize=8, color="#0b0b0b")
a.axhline(1, color=INK2, lw=1)
a.set_xscale("log"); a.set_yscale("log")
a.set_title("uncorrected step / corrected step = (1-b1^t)/sqrt(1-b2^t)", loc="left", fontsize=9.5)
a.set_xlabel("step t"); a.set_ylabel("step-size multiplier (b1 = 0.9)"); a.legend(frameon=False, fontsize=8)

a = ax[1]
for b2 in BETA2S:
    for bc in [True, False]:
        y = R[f"b2={b2}|seed=0|bc={bc}"]["loss"][:20]
        a.plot(range(1, 21), y, color=col[b2], ls=ls[bc], lw=2, marker="o", ms=3, label=lab(b2, bc))
a.set_xticks([1, 5, 10, 15, 20])
a.set_title("first 20 steps, both ways (lr 1e-3, no warmup, seed 0)", loc="left", fontsize=9.5)
a.set_xlabel("step"); a.set_ylabel("train loss"); a.legend(frameon=False, fontsize=8)

a = ax[2]
sm = lambda v, k=50: np.convolve(v, np.ones(k) / k, "valid")
for b2 in BETA2S:
    for bc in [True, False]:
        y = np.mean([sm(R[f"b2={b2}|seed={s}|bc={bc}"]["loss"]) for s in (0, 1)], 0)
        a.plot(np.arange(len(y)) + 25, y, color=col[b2], ls=ls[bc], lw=2, label=lab(b2, bc))
        if b2 == 0.999 and not bc:   # the other three end within 0.07 of each other; their labels would collide
            label_end(a, len(y) + 25, y[-1], f"{y[-1]:.2f}")
a.set_ylim(0.9, 3.6)
a.set_title("2000 steps at the same untuned lr 1e-3 (mean of 2 seeds)", loc="left", fontsize=9.5)
a.set_xlabel("step"); a.set_ylabel("train loss, 50-step mean"); a.legend(frameon=False, fontsize=8)

a = ax[3]
for b2 in BETA2S:
    for bc in [True, False]:
        pts = sorted((float(k.split("lr=")[1]), v["final_val"]) for k, v in Tn.items()
                     if k.startswith(f"b2={b2}|bc={bc}|lr=") and "seed" not in k)
        x, y = zip(*pts)
        a.plot(x, y, color=col[b2], ls=ls[bc], lw=2, marker="o", ms=4, label=lab(b2, bc))
        i = int(np.argmin(y)); a.plot([x[i]], [y[i]], "o", ms=9, mfc="none", mec="#0b0b0b", mew=1.5)
a.set_xscale("log")
a.set_title("tune both sides: val @1000 vs lr (circled = best)", loc="left", fontsize=9.5)
a.set_xlabel("peak lr (constant, no warmup)"); a.set_ylabel("val loss"); a.legend(frameon=False, fontsize=8)
finish(fig, ax, "bias_correction.png")
print("wrote bias_correction.png")
