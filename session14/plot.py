"""Plot validation loss: dense pretraining, then MoE vs dense-control continuation."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open("logs/results.json"))
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})

for a in (ax, ax2):
    a.plot(r["dense"]["step"], r["dense"]["val"], color="#555555", lw=2, label="dense (phase 1)")
    a.plot(r["dense_cont"]["step"], r["dense_cont"]["val"], color="#1f77b4", lw=2,
           label="dense, continued (control)")
    a.plot(r["moe_cont"]["step"], r["moe_cont"]["val"], color="#d62728", lw=2,
           label=f"MoE upcycled ({r['cfg']['n_experts']}e top-{r['cfg']['top_k']}), continued")
    a.axvline(r["cfg"]["dense_steps"], color="k", ls=":", lw=1)
    a.set_xlabel("training step")
    a.set_ylabel("validation loss (nats/char)")
    a.grid(alpha=0.3)

ax.set_title("Dense → MoE upcycling (char-level TinyStories, Apple M4 Max / MPS)")
ax.text(r["cfg"]["dense_steps"], ax.get_ylim()[1] * 0.97, "  convert to MoE", va="top", fontsize=9)
ax.legend()

# zoom on continuation
lo = r["cfg"]["dense_steps"]
ax2.set_xlim(lo - 200, r["moe_cont"]["step"][-1] + 50)
vals = r["moe_cont"]["val"] + r["dense_cont"]["val"]
ax2.set_ylim(min(vals) - 0.02, max(vals) + 0.02)
ax2.set_title("Zoom: after conversion")
for name, color in (("moe_cont", "#d62728"), ("dense_cont", "#1f77b4")):
    s, v = r[name]["step"][-1], r[name]["val"][-1]
    ax2.annotate(f"{v:.4f}", (s, v), textcoords="offset points", xytext=(-40, 6), color=color)

plt.tight_layout()
plt.savefig("loss_curves.png", dpi=130)
print("wrote loss_curves.png")
