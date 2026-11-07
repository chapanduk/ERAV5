import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("accum_results.json"))
BLUE, ORANGE, INK, INK2, GRID, BG = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
COL = {"correct": BLUE, "wrong": ORANGE}
LAB = {"correct": "correct: token-weighted", "wrong": "wrong: mean of micro-batch means"}
DASH = {"0": "-", "1": (0, (4, 2))}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": BG, "figure.facecolor": BG})
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))


def smooth(v, k=25):
    v = np.asarray(v); return np.convolve(v, np.ones(k) / k, mode="valid")


# (a) held-out loss, both runs, both seeds
a = ax[0]
for seed, runs in R.items():
    for mode, log in runs.items():
        a.plot(log["val_step"], log["val"], color=COL[mode], ls=DASH[seed], lw=2,
               label=f"{LAB[mode]} (seed {seed})")
        a.annotate(f"{log['val'][-1]:.3f}", (log["val_step"][-1], log["val"][-1]), xytext=(4, 6 if mode == "wrong" else -6),
                   textcoords="offset points", color=INK, fontsize=9, va="center")
a.set_ylim(1.6, 3.0)
a.set_title("Held-out loss (same token-weighted metric for both)", color=INK, loc="left")
a.set_xlabel("optimizer step"); a.set_ylabel("val loss, nats/byte, T=256")
a.legend(frameon=False, fontsize=8)

# (b) the wrong run's own logged number vs the true loss of the same batches
b = ax[1]
for seed, runs in R.items():
    w = runs["wrong"]
    d = smooth(np.array(w["reported"]) - np.array(w["true"]))
    b.plot(np.arange(len(d)) + 12, d, color=ORANGE, ls=DASH[seed], lw=2, label=f"seed {seed}")
b.axhline(0, color=INK2, lw=1)
b.set_title("Wrong run: logged loss minus true loss of the same batch", color=INK, loc="left")
b.set_xlabel("optimizer step"); b.set_ylabel("nats/byte (25-step mean)")
b.legend(frameon=False, fontsize=8)

# (c) where the damage lands: per-position val loss gap at the end of training
c = ax[2]
for seed, runs in R.items():
    gap = np.array(runs["wrong"]["per_pos"]) - np.array(runs["correct"]["per_pos"])
    c.plot(np.arange(248) + 4, smooth(gap, 9), color=ORANGE, ls=DASH[seed], lw=2, label=f"seed {seed}")
c.axhline(0, color=INK2, lw=1)
c.set_title("Final val loss, wrong minus correct, by position", color=INK, loc="left")
c.set_xlabel("position in sequence (context available)"); c.set_ylabel("nats/byte (>0 = wrong is worse)")
c.legend(frameon=False, fontsize=8)

for x in ax:
    x.grid(color=GRID, lw=0.8); x.set_axisbelow(True)
    for s in ["top", "right"]: x.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig("accum.png", dpi=130)

for seed, runs in R.items():
    w, k = runs["wrong"], runs["correct"]
    gap = np.array(w["per_pos"]) - np.array(k["per_pos"])
    print(f"seed {seed}: final val correct {k['val'][-1]:.4f} wrong {w['val'][-1]:.4f} gap {w['val'][-1]-k['val'][-1]:+.4f} | "
          f"gap pos<16 {gap[:16].mean():+.4f}  pos>=128 {gap[128:].mean():+.4f} | "
          f"logged-true mean {np.mean(np.array(w['reported'])-np.array(w['true'])):+.4f}")
    # how many val checkpoints is wrong behind
    print("   val gap at each 100 steps:", [round(wv - kv, 3) for i, (wv, kv) in enumerate(zip(w["val"], k["val"])) if i % 4 == 0])
