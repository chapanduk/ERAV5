"""Figure for task 4 (reads schedule_results.json written by schedules.py)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import lr_at
from schedules import LRS, TUNE_SEEDS, FINAL_SEEDS, TOTAL, STOP, WARMUP, COOLDOWN
from style import COLORS, INK, INK2, finish, label_end, setup

setup()
S = json.load(open("schedule_results.json"))
col = {"cosine": COLORS[0], "wsd": COLORS[1]}
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

a = ax[0]
t = np.arange(TOTAL)
for s in ["cosine", "wsd"]:
    y = np.array([lr_at(i, s, 1.0, TOTAL, WARMUP) for i in t])
    a.plot(t[:STOP], y[:STOP], color=col[s], lw=2, label=s)
    a.plot(t[STOP - 1:], y[STOP - 1:], color=col[s], lw=1.5, ls=(0, (2, 2)))   # planned, never run
    label_end(a, STOP, y[STOP - 1], f"{y[STOP - 1]:.2f} x peak", dy=8 if s == "wsd" else -8)
a.axvspan(STOP, TOTAL, color="#e6e5e1", alpha=0.6)
a.text(STOP + 5, 0.5, "planned,\nnever run", color=INK2, fontsize=8)
a.set_title(f"both planned for {TOTAL} steps, both stopped at {STOP}", loc="left", fontsize=9.5)
a.set_xlabel("step"); a.set_ylabel("lr / peak lr"); a.legend(frameon=False, fontsize=8)

a = ax[1]
best = {}
for s in ["cosine", "wsd"]:
    y = [np.mean([S[f"{s}|lr={lr}|seed={sd}"]["final_val"] for sd in TUNE_SEEDS]) for lr in LRS]
    a.plot(LRS, y, color=col[s], lw=2, marker="o", ms=4, label=s)
    i = int(np.argmin(y)); best[s] = LRS[i]
    a.plot([LRS[i]], [y[i]], "o", ms=10, mfc="none", mec=INK, mew=1.5)
a.set_xscale("log"); a.set_ylim(2.35, 3.0)
a.set_title(f"tune both sides: val @{STOP} vs peak lr (mean of {len(TUNE_SEEDS)} seeds)", loc="left", fontsize=9.5)
a.set_xlabel("peak lr"); a.set_ylabel(f"val loss at step {STOP}"); a.legend(frameon=False, fontsize=8)

a = ax[2]
for s in ["cosine", "wsd"]:
    runs = [S[f"{s}|lr={best[s]}|seed={sd}"]["val"] for sd in FINAL_SEEDS]
    steps = sorted(int(k) for k in runs[0])
    m = np.array([np.mean([r[str(k)] for r in runs]) for k in steps])
    lo = np.array([min(r[str(k)] for r in runs) for k in steps]); hi = np.array([max(r[str(k)] for r in runs) for k in steps])
    a.fill_between(steps, lo, hi, color=col[s], alpha=0.15)
    a.plot(steps, m, color=col[s], lw=2, marker="o", ms=4, label=f"{s} @ lr {best[s]:g} (mean, min-max of {len(FINAL_SEEDS)} seeds)")
    label_end(a, steps[-1], m[-1], f"{m[-1]:.3f}", dy=-8 if s == "wsd" else 8)
cd = [S[f"wsd+cooldown{COOLDOWN}|lr={best['wsd']}|seed={sd}"]["after"] for sd in FINAL_SEEDS]
a.plot([STOP + COOLDOWN], [np.mean(cd)], "D", ms=7, color=col["wsd"], mec=INK)
a.annotate(f"extra: WSD@200 + {COOLDOWN}-step\ncooldown = {np.mean(cd):.3f}", (STOP + COOLDOWN, np.mean(cd)),
           xytext=(-115, -32), textcoords="offset points", fontsize=8, color=INK)
a.set_ylim(2.2, 3.2); a.set_xlim(20, STOP + COOLDOWN + 15)
a.set_title("val loss on the way to the stop", loc="left", fontsize=9.5)
a.set_xlabel("step"); a.set_ylabel("val loss"); a.legend(frameon=False, fontsize=8)
finish(fig, ax, "schedules.png")
print("wrote schedules.png")
