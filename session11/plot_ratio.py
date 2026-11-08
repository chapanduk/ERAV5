"""Figure + summary for task 3 (reads ratio_results.json written by update_ratio.py)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from update_ratio import rolling_median, store
from style import COLORS, finish, label_end, setup

setup()
R = store.data
names = list(R["warmup=0"]["ratio"])
WS = [0, 50, 100, 200]
col = dict(zip(WS, COLORS))


def gap(a, b, k=41):
    return np.median([np.abs(rolling_median(R[a]["ratio"][n], k) / rolling_median(R[b]["ratio"][n], k) - 1)
                      for n in names], 0)


fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
for a, (n, title) in zip(ax[:3], [("blocks.0.qkv.weight", "attention qkv (block 0): peak follows warmup end"),
                                   ("blocks.1.fc.weight", "MLP fc (block 1): peak ~step 250 whatever the warmup"),
                                   ("head.weight", "output head: falls while LR is still rising")]):
    for w in WS:
        r = rolling_median(R[f"warmup={w}"]["ratio"][n])
        a.plot(np.arange(1, len(r) + 1), r, color=col[w], lw=2, label="no warmup" if w == 0 else f"warmup {w}")
        if w:
            a.axvline(w, color=col[w], lw=1, ls=(0, (2, 3)))
        if a is ax[0]:   # in the other two panels the lines end on top of each other; the legend carries identity
            label_end(a, len(r), r[-1], "none" if w == 0 else str(w))
    a.set_yscale("log"); a.set_title(title, loc="left", fontsize=9.5)
    a.set_xlabel("step"); a.set_ylabel("||dW|| / ||W||  (21-step median)")
for a in ax[:3]:
    a.legend(frameon=False, fontsize=8)

a = ax[3]
noise = gap("warmup=0|seed=1", "warmup=0")
a.fill_between(np.arange(1, 601), 0, noise, color="#e6e5e1", label="seed-to-seed gap (noise)")
for w in WS[1:]:
    g = gap(f"warmup={w}", "warmup=0")
    a.plot(np.arange(1, 601), g, color=col[w], lw=2, label=f"warmup {w} vs none")
    label_end(a, 600, g[-1], str(w))
a.set_ylim(0, 0.6)
a.set_title("median over 19 matrices of |ratio / ratio(no warmup) - 1|", loc="left", fontsize=9.5)
a.set_xlabel("step"); a.set_ylabel("relative gap"); a.legend(frameon=False, fontsize=8)
finish(fig, ax, "update_ratio.png")

nz = float(np.median(noise[100:]))
print(f"seed-to-seed gap, median over steps 100-600: {nz:.3f}")
for w in WS[1:]:
    g = gap(f"warmup={w}", "warmup=0")
    pk = {n: int(np.argmax(rolling_median(R[f"warmup={w}"]["ratio"][n]))) + 1 for n in names}
    grp = lambda key: [pk[n] for n in names if key in n]
    print(f"warmup {w:>3}: gap vs no-warmup at warmup end {g[w - 1]:.3f}, step 300 {g[299]:.3f}, step 600 {g[-1]:.3f} "
          f"({g[-1] / nz:.1f}x noise) | ratio peak step: qkv {grp('qkv')}, fc {grp('.fc')}, head {pk['head.weight']}, "
          f"tok {pk['tok.weight']}")
