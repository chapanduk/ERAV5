"""One chart style for every figure (palette validated with the dataviz skill's validator, light mode)."""
import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # blue, orange, aqua, yellow - fixed order
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


def setup():
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.facecolor": BG, "figure.facecolor": BG, "axes.titlecolor": INK})


def label_end(ax, x, y, text, dy=0):
    """Direct label at a line's end, in text ink (aqua/yellow are below 3:1 on the surface, so identity
    must not rely on colour alone)."""
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", color=INK, fontsize=8, va="center")


def finish(fig, axes, path):
    for a in getattr(axes, "flat", axes):
        a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
