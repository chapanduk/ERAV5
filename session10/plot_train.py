import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("train_results_uninterrupted.json"))  # the run REPORT numbers come from
BLUE, ORANGE, INK, INK2, GRID, BG = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": BG, "figure.facecolor": BG})


def rmed(v, k=50):
    v = np.asarray(v); return np.array([np.median(v[max(0, i - k + 1): i + 1]) for i in range(len(v))])


L, G = np.array(R["hot"]["loss"]), np.array(R["hot"]["gnorm"])
Lm, Gm = rmed(L), rmed(G)
base_g = np.median(G[350:450])
g_step = next(s for s in range(450, len(G)) if Gm[s] > 2 * base_g)
loss_floor = Lm[400:g_step + 1].min()   # best loss reached before the grad norm moved
floor_at = 400 + int(Lm[400:g_step + 1].argmin())
l_step = next(s for s in range(g_step, len(L)) if Lm[s] > loss_floor + 0.03)
g_turn = 200 + int(Gm[200:g_step + 1].argmin())
print(f"TURNING POINTS: grad-norm median bottoms at step {g_turn} ({Gm[g_turn]:.3f}); loss median bottoms at step {floor_at} ({loss_floor:.3f}); lead {floor_at - g_turn} steps")
print(f"   between them: grad-norm median {Gm[g_turn]:.3f} -> {Gm[floor_at]:.3f} ({Gm[floor_at]/Gm[g_turn]:.2f}x), loss median {Lm[g_turn]:.3f} -> {loss_floor:.3f} (still falling)")
print(f"grad-norm baseline (steps 350-450) {base_g:.3f}; rolling median first > 2x at step {g_step}")
print(f"loss rolling-median floor {loss_floor:.3f} at step {floor_at}; first > floor+0.03 at step {l_step}")
print(f"lead: {l_step - g_step} steps.  loss rolling median at {g_step}: {Lm[g_step]:.3f}, at end: {Lm[-1]:.3f}")
print(f"isolated spike: step 430 gnorm {G[430]:.2f} ({G[430]/base_g:.1f}x baseline), loss {L[430]:.3f} vs per-step std {L[380:480].std():.3f}")

fig, ax = plt.subplots(2, 2, figsize=(14, 7), sharex="col")
for col, name in enumerate(["base", "hot"]):
    l, g = np.array(R[name]["loss"]), np.array(R[name]["gnorm"])
    s = np.arange(len(l))
    ax[0, col].plot(s, l, color=BLUE, lw=0.6, alpha=0.35)
    ax[0, col].plot(s, rmed(l), color=BLUE, lw=2)
    ax[1, col].plot(s, g, color=ORANGE, lw=0.6, alpha=0.35)
    ax[1, col].plot(s, rmed(g), color=ORANGE, lw=2)
    ax[1, col].set_yscale("log")
    ax[0, col].set_ylabel("train loss (nats/byte)"); ax[1, col].set_ylabel("grad norm, pre-clip (log)")
    ax[1, col].set_xlabel("step")
ax[0, 0].set_title("base: lr 1e-3, warmup, cosine, clip 1.0  (thin = per step, thick = 50-step median)", color=INK, loc="left", fontsize=10)
ax[0, 1].set_title("hot: lr 4e-3 constant, no warmup, no clipping", color=INK, loc="left", fontsize=10)
ax[0, 0].set_ylim(0.6, 3.5); ax[0, 1].set_ylim(2.2, 3.0)
for a in ax[:, 1]:
    a.axvline(g_turn, color=INK2, lw=1, ls=(0, (4, 2)))
    a.axvline(floor_at, color=INK2, lw=1, ls=(0, (1, 2)))
ax[1, 1].annotate(f"step {g_turn}: grad-norm median\nbottoms out and turns up", (g_turn, Gm[g_turn]), xytext=(25, 95),
                  textcoords="offset points", color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2))
ax[0, 1].annotate(f"step {floor_at}: loss median bottoms out,\n{floor_at - g_turn} steps later", (floor_at, loss_floor), xytext=(30, -60),
                  textcoords="offset points", color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2))
for a in ax.flat:
    a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for sp in ["top", "right"]: a.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig("gradnorm.png", dpi=130)
