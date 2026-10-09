"""Fig. 1: the spatial residual correction as one pipeline that forks only where IDW and the MLP
differ.

    python figures/make_lstm_method.py   # -> lstm/fig01_method.{png,pdf,svg}

Steps 1, 2 and 4 are shared and drawn once. Step 3 turns the donors' scaled errors into one daily
correction delta, in the target's own units, either (a) as a fixed inverse-distance-weighted mean scaled by lambda, or (b) with
an MLP. Writing both as delta lets step 4 be a single formula. The neighbourhood sketches and the
hydrograph come from make_method_idw.py, so the figure matches the standalone diagrams.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

matplotlib.rcParams["svg.fonttype"] = "path"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src", "corrections", "mlp"))
import make_method_idw as idw  # noqa: E402
from model import HIDDEN  # noqa: E402
from palette import CORRECTED, GRID, INK, LEARNED, MUTE, RAW  # noqa: E402
from palette import figure_variant  # noqa: E402

SUFFIX, OUT_DIR = figure_variant()
SCALED = SUFFIX != "_noscale"


FIG_W, FIG_H = 10.0, 4.0
M, G, Y0 = 0.22, 0.34, 0.12
H = FIG_H - 2 * Y0
W1, W2, W3, W4 = 1.72, 1.80, 3.22, 1.80
SPLIT = 0.22                     # vertical gap between the two step-3 boxes
H3 = (H - SPLIT) / 2
X1 = M
X2 = X1 + W1 + G
X3 = X2 + W2 + G
X4 = X3 + W3 + G

TITLE_FS, EQ_FS, NOTE_FS = 10.0, 11.5, 7.8


def box(ax, x, y, w, h, title, badge, accent=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                                linewidth=1.6 if accent else 1.0, edgecolor=accent or MUTE,
                                facecolor="white", zorder=2))
    idw.track(ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top", fontsize=TITLE_FS,
                      color=accent or INK, fontweight="bold", zorder=4), x + 0.05, x + w - 0.05)
    ax.add_patch(Circle((x + 0.02, y + h - 0.02), 0.15, facecolor=accent or INK,
                        edgecolor="white", linewidth=1.1, zorder=6))
    ax.text(x + 0.02, y + h - 0.02, badge, ha="center", va="center",
            fontsize=8.5 if len(badge) == 1 else 7.2, color="white", fontweight="bold", zorder=7)


def text(ax, x, y, s, lo, hi, size=EQ_FS, color=INK, **kw):
    idw.track(ax.text(x, y, s, ha=kw.pop("ha", "center"), va=kw.pop("va", "center"),
                      fontsize=size, color=color, zorder=4, linespacing=1.4, **kw), lo, hi)


def sketch_rect(x, y, w, top):
    """A square sketch area centred in the space between the box bottom and `top`."""
    side = min(w - 0.2, top - y - 0.14)
    return (x + (w - side) / 2, y + 0.1 + (top - y - 0.14 - side) / 2, side, side)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, linewidth=1.2,
                                 color=MUTE, shrinkA=0, shrinkB=0, zorder=5))


def network(ax, rect):
    """Inputs named in words, two hidden layers, one output: the day's correction."""
    x0, y0, w, h = rect
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                linewidth=0, facecolor=GRID, alpha=0.75, zorder=2.5))
    inputs = ((r"donor errors $r_{1..K}$", RAW), (r"distances $d_{1..K}$", RAW),
              ("day of year", INK),
              (r"own flow $q^{sim}_i/s_i$" if SCALED else r"predicted flow $q^{sim}_i$", LEARNED))
    layers = [[(0.40, v) for v in (0.86, 0.62, 0.38, 0.14)],
              [(0.56, v) for v in (0.80, 0.60, 0.40, 0.20)],
              [(0.70, v) for v in (0.80, 0.60, 0.40, 0.20)],
              [(0.86, 0.50)]]
    for a, b in zip(layers, layers[1:]):
        for ua, va in a:
            for ub, vb in b:
                ax.plot(*zip(idw.pt(rect, ua, va), idw.pt(rect, ub, vb)), color=MUTE, lw=0.4,
                        alpha=0.45, zorder=3)
    for (u, v), (lab, col) in zip(layers[0], inputs):
        ax.scatter(*idw.pt(rect, u, v), s=26, color=col, edgecolor="white", linewidth=0.6,
                   zorder=5)
        text(ax, *idw.pt(rect, u - 0.03, v), lab, x0, x0 + w, size=idw.LAB_FS - 0.4, color=col,
             ha="right")
    for layer in layers[1:3]:
        for u, v in layer:
            ax.scatter(*idw.pt(rect, u, v), s=20, color="white", edgecolor=MUTE, linewidth=0.7,
                       zorder=5)
    ax.scatter(*idw.pt(rect, 0.86, 0.50), s=60, color=LEARNED, edgecolor="white", linewidth=0.6,
               zorder=6)
    text(ax, *idw.pt(rect, 0.955, 0.50), r"$\delta_i$", x0, x0 + w, size=idw.LAB_FS + 1,
         color=LEARNED)
    text(ax, *idw.pt(rect, 0.63, 0.0), f"2 hidden layers × {HIDDEN} units", x0, x0 + w,
         size=6.4, color=MUTE, va="bottom")


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    top = Y0 + H

    # 1. LSTM prediction, shared.
    box(ax, X1, Y0, W1, H, "LSTM prediction", "1", accent=RAW)
    text(ax, X1 + W1 / 2, top - 0.72, r"$\mathrm{LSTM}\rightarrow q^{sim}_i,\, q^{sim}_j$",
         X1 + 0.05, X1 + W1 - 0.05, size=10.5)
    text(ax, X1 + W1 / 2, top - 1.15, "one model predicts\nevery basin", X1 + 0.05,
         X1 + W1 - 0.05, size=NOTE_FS, color=MUTE, va="top")
    idw.sketch(ax, sketch_rect(X1, Y0, W1, top - 1.75), 0)

    # 2. Donor errors, shared.
    box(ax, X2, Y0, W2, H, "Donor errors", "2")
    text(ax, X2 + W2 / 2, top - 0.78, r"$r_j=\dfrac{q^{sim}_j-q^{obs}_j}{s_j}$" if SCALED
         else r"$r_j=q^{sim}_j-q^{obs}_j$",
         X2 + 0.05, X2 + W2 - 0.05)
    text(ax, X2 + W2 / 2, top - 1.25, "same-day error at the K\nnearest gauges, in units\nof each one's flow SD $s$" if SCALED
         else "same-day error at the K\nnearest gauges, in mm/day", X2 + 0.05,
         X2 + W2 - 0.05, size=NOTE_FS, color=MUTE, va="top")
    idw.sketch(ax, sketch_rect(X2, Y0, W2, top - 1.85), 2)

    # 3a. IDW: fixed distance weights. Equations left, weight sketch right.
    ya = Y0 + H3 + SPLIT
    box(ax, X3, ya, W3, H3, "(a) IDW: fixed distance weights", "3a", accent=CORRECTED)
    left_hi = X3 + 1.78
    text(ax, X3 + 0.92, ya + H3 - 0.74, r"$w_{ij}=\dfrac{d_{ij}^{-p}}{\sum_{j'} d_{ij'}^{-p}}$",
         X3 + 0.05, left_hi)
    text(ax, X3 + 0.92, ya + H3 - 1.26, r"$\delta_i=\lambda\sum_j w_{ij}\,r_j$",
         X3 + 0.05, left_hi)
    text(ax, X3 + 0.92, ya + 0.07, r"$K$, $p$, $\lambda$ by cross-validation", X3 + 0.05,
         left_hi, size=NOTE_FS - 0.4, color=MUTE, va="bottom")
    idw.sketch(ax, (left_hi + 0.04, ya + 0.1, X3 + W3 - left_hi - 0.14, H3 - 0.48), 1)

    # 3b. MLP: the correction learned from the same donors.
    box(ax, X3, Y0, W3, H3, "(b) MLP: learned correction", "3b", accent=LEARNED)
    text(ax, X3 + W3 / 2, Y0 + H3 - 0.58,
         r"$\delta_i=f_\theta\,(r_{1..K},\,d_{1..K},\,\mathrm{day},\,q^{sim}_i/s_i)$" if SCALED
         else r"$\delta_i=f_\theta\,(r_{1..K},\,d_{1..K},\,\mathrm{day},\,q^{sim}_i)$",
         X3 + 0.05, X3 + W3 - 0.05, size=10.5)
    network(ax, (X3 + 0.1, Y0 + 0.08, W3 - 0.2, H3 - 0.84))

    # 4. Correct the target, shared: both corrections are a daily correction delta in s_i units.
    box(ax, X4, Y0, W4, H, "Correct the target", "4")
    text(ax, X4 + W4 / 2, top - 0.70, r"$q^{corr}_i=\max(0,$", X4 + 0.05,
         X4 + W4 - 0.05, size=10.5)
    text(ax, X4 + W4 / 2, top - 1.02, r"$q^{sim}_i-s_i\,\delta_i)$" if SCALED
         else r"$q^{sim}_i-\delta_i)$", X4 + 0.05,
         X4 + W4 - 0.05, size=10.5)
    text(ax, X4 + W4 / 2, top - 1.38, "the same step for\nboth corrections", X4 + 0.05,
         X4 + W4 - 0.05, size=NOTE_FS, color=MUTE, va="top")
    idw.hydrograph(ax, sketch_rect(X4, Y0, W4, top - 1.95), accent=INK)

    mid = Y0 + H / 2
    arrow(ax, (X1 + W1, mid), (X2, mid))
    for y in (ya + H3 / 2, Y0 + H3 / 2):
        arrow(ax, (X2 + W2, mid), (X3, y))
        arrow(ax, (X3 + W3, y), (X4, mid))

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    bad = idw.check_fit(fig, ax)
    os.makedirs(f"{HERE}/{OUT_DIR}", exist_ok=True)
    stem = f"{HERE}/{OUT_DIR}/fig01_method"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=300, facecolor="white")
    print(f"-> {stem}.png/.pdf/.svg" + (f"   ({bad} overflowing)" if bad else "   (fit ok)"))


if __name__ == "__main__":
    main()
