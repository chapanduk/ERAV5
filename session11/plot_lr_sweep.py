"""Figure + extrapolation for task 5 (reads lr_sweep_results.json written by lr_sweep.py)."""
import itertools, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lr_sweep import WIDTHS, SEEDS
from style import COLORS, INK, INK2, finish, label_end, setup

setup()
R = json.load(open("lr_sweep_results.json"))
col = dict(zip(WIDTHS, COLORS))
DIVERGED = 3.5   # plotted ceiling for runs that diverged or ended worse than this


def table(d, seeds=SEEDS):
    lrs = sorted({float(k.split("|")[1][3:]) for k in R if k.startswith(f"d={d}|")})
    out = {}
    for lr in lrs:
        v = [R.get(f"d={d}|lr={lr:g}|seed={s}", {}).get("final_val", float("nan")) for s in seeds]
        v = [x if math.isfinite(x) else DIVERGED for x in v]
        out[lr] = float(np.mean(v))
    return out


def refined_min(t, octaves=1.0):
    """Minimum of a quadratic in log2(lr) least-squares fitted to every point within `octaves` of the grid
    minimum. A 3-point parabola is hijacked by one noisy point when neighbours differ by ~noise; a fit
    over the whole window averages that out."""
    lrs = sorted(t); y = [t[l] for l in lrs]; i = int(np.argmin(y))
    if i in (0, len(lrs) - 1):
        return lrs[i], y[i], True
    win = [l for l in lrs if abs(math.log2(l / lrs[i])) <= octaves + 1e-9]
    x = np.log2(win); a, b, c = np.polyfit(x, [t[l] for l in win], 2)
    xm = -b / (2 * a)
    if a <= 0 or not (x.min() <= xm <= x.max()):
        return lrs[i], y[i], False
    return 2 ** xm, a * xm ** 2 + b * xm + c, False


def within_noise(d):
    """LRs whose mean loss is within one standard error (of the difference) of the best LR's mean."""
    pts = {}
    for k, v in R.items():
        if k.startswith(f"d={d}|"):
            lr = float(k.split("|")[1][3:]); pts.setdefault(lr, []).append(v["final_val"] if math.isfinite(v["final_val"]) else DIVERGED)
    stat = {lr: (np.mean(v), np.std(v, ddof=1) / math.sqrt(len(v))) for lr, v in pts.items() if len(v) > 1}
    b = min(stat, key=lambda l: stat[l][0])
    ok = [lr for lr, (m, se) in stat.items() if m - stat[b][0] <= math.hypot(se, stat[b][1])]
    return min(ok), max(ok), stat


def fit(points):
    """log lr* = log C + alpha * log width"""
    x, y = np.log(np.array([p[0] for p in points])), np.log(np.array([p[1] for p in points]))
    alpha, logc = np.polyfit(x, y, 1)
    return alpha, logc


mins = {d: refined_min(table(d)) for d in WIDTHS}
alpha, logc = fit([(d, mins[d][0]) for d in WIDTHS])
pred = math.exp(logc + alpha * math.log(4096))

# uncertainty: refit using each seed's own minimum, every combination of one seed per width
per_seed = {d: [refined_min(table(d, [s]))[0] for s in SEEDS] for d in WIDTHS}
preds, alphas = [], []
for combo in itertools.product(*[per_seed[d] for d in WIDTHS]):
    a_, c_ = fit(list(zip(WIDTHS, combo)))
    alphas.append(a_); preds.append(math.exp(c_ + a_ * math.log(4096)))
# leave-one-width-out: how much does the prediction depend on any single width?
loo = {}
for drop in WIDTHS:
    keep = [(d, mins[d][0]) for d in WIDTHS if d != drop]
    a_, c_ = fit(keep); loo[drop] = (a_, math.exp(c_ + a_ * math.log(4096)))

fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[1.35, 1]))
a = ax[0]
for d in WIDTHS:
    t = table(d); lrs = sorted(t)
    for s in SEEDS:
        ts = table(d, [s]); a.plot(lrs, [ts[l] for l in lrs], color=col[d], lw=0.8, alpha=0.35)
    a.plot(lrs, [t[l] for l in lrs], color=col[d], lw=2, marker="o", ms=4, label=f"width {d}")
    lo, hi, _ = within_noise(d)
    a.plot([lo, hi], [ym_band := min(t.values()) - 0.03 - 0.025 * WIDTHS.index(d)] * 2, color=col[d], lw=5,
           solid_capstyle="butt", alpha=0.6)
    lm, ym, edge = mins[d]
    a.plot([lm], [ym], marker="v", ms=11, color=col[d], mec=INK, mew=1.2, ls="none")
    a.annotate(f"{lm:.2e}" + (" (edge!)" if edge else ""), (lm, ym), xytext=(-14, 0), textcoords="offset points",
               ha="right", va="center", fontsize=8.5, color=INK)
a.set_xscale("log"); a.set_ylim(1.95, 3.0)
a.set_title(f"val loss after 300 steps vs peak lr\nthick = mean of {len(SEEDS)} seeds, thin = each seed; "
            "triangle = fitted minimum; bar = LRs within 1 SE of the best", loc="left", fontsize=9)
a.set_xlabel("peak lr (300-step cosine, 30-step warmup)"); a.set_ylabel("val loss @300"); a.legend(frameon=False)

a = ax[1]
w = np.array([200, 6000])
a.plot(w, np.exp(logc + alpha * np.log(w)), color=INK2, lw=1.5, ls=(0, (4, 2)),
       label=f"fit: lr* ~ width^{alpha:.2f}")
for d in WIDTHS:
    a.plot([d], [mins[d][0]], "v", ms=11, color=col[d], mec=INK, mew=1.2)
    for p in per_seed[d]:
        a.plot([d], [p], "o", ms=4, color=col[d], alpha=0.5)
a.errorbar([4096], [pred], yerr=[[pred - min(preds)], [max(preds) - pred]], fmt="*", ms=14, color=INK,
           capsize=5, label=f"width 4096: {pred:.1e}  (seed range {min(preds):.1e}-{max(preds):.1e})")
a.set_xscale("log", base=2); a.set_yscale("log")
a.set_xticks([256, 512, 1024, 2048, 4096]); a.set_xticklabels(["256", "512", "1024", "2048", "4096"])
a.set_title("best lr vs width, extrapolated to 4096\ntriangle = fitted minimum, dots = each seed's own minimum", loc="left", fontsize=9)
a.set_xlabel("width"); a.set_ylabel("best peak lr"); a.legend(frameon=False, fontsize=8)
finish(fig, ax, "lr_sweep.png")

for d in WIDTHS:
    t = table(d)
    print(f"width {d:>4}: " + "  ".join(f"{l:g}:{v:.3f}" for l, v in sorted(t.items())))
    print(f"            grid best {min(t, key=t.get):g}; refined minimum {mins[d][0]:.3e} (val {mins[d][1]:.4f})"
          f"{'  AT GRID EDGE' if mins[d][2] else ''}; per-seed minima {[f'{p:.2e}' for p in per_seed[d]]}")
print()
for d in WIDTHS:
    lo, hi, stat = within_noise(d)
    best = min(stat, key=lambda l: stat[l][0])
    print(f"width {d:>4}: LRs within 1 SE of the best: {lo:.2e} - {hi:.2e}; typical per-point SE "
          f"{np.median([se for _, se in stat.values()]):.3f}; cliff: loss(2e-3) - loss(best) = "
          f"{stat[2e-3][0] - stat[best][0]:+.3f}, loss(4e-3) - loss(best) = {stat[4e-3][0] - stat[best][0]:+.3f}")
print(f"\nfit: lr* = {math.exp(logc):.3g} * width^{alpha:.3f}")
print(f"width 4096 prediction: {pred:.2e}; refits with per-seed minima give {min(preds):.2e} - {max(preds):.2e} "
      f"(exponent {min(alphas):.2f} to {max(alphas):.2f})")
for drop, (a_, p_) in loo.items():
    print(f"  leave out width {drop}: exponent {a_:.2f}, prediction {p_:.2e}")
print(f"1/width scaling from 1024 would give {mins[1024][0] / 4:.2e}; 1/sqrt(width) would give {mins[1024][0] / 2:.2e}")
