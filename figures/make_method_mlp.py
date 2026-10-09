"""Method diagram: the four steps of the learned MLP neighbour-residual correction.

    python figures/make_method_mlp.py    # -> lstm/method_mlp.png, lstm/method_mlp.svg

The same neighbourhood and hydrograph as make_method_idw.py, so the two read side by side. Step 2
is today's error at each of the K nearest gauges; step 3, drawn wider because it is the method,
is a small network that turns those errors, the donors' distances, the season and the target's
own predicted flow into today's correction. The band above says how the network is trained and K
chosen. K, the seeds and the layer width are imported from src/corrections/mlp/model.py so they
cannot drift from the code.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src", "corrections", "mlp"))
from make_method_idw import (BOX_H, CELL_GAP, CELL_H, CELL_W, CV_H, CV_Y, DONORS,  # noqa: E402
                             FIG_H, FIG_W, GAP, LAB_FS, MARGIN, Q_LAB, R_LAB, SPLIT_GAP,
                             TARGET, Y0, add_box, basins, check_fit, hydrograph, label_donors,
                             pt, track)
from make_method_idw import STEPS as IDW_STEPS  # noqa: E402
from model import HIDDEN, K_GRID, SEEDS  # noqa: E402
from palette import GRID, INK, LEARNED, MUTE, RAW  # noqa: E402
from palette import figure_variant  # noqa: E402

SUFFIX, OUT_DIR = figure_variant()
SCALED = SUFFIX != "_noscale"


# Steps 1, 2 and 4 share a width; step 3 takes what is left.
NARROW_W = 1.95
WIDE_W = FIG_W - 2 * MARGIN - 3 * GAP - 3 * NARROW_W
WIDTHS = (NARROW_W, NARROW_W, WIDE_W, NARROW_W)

STEPS = (
    IDW_STEPS[0],
    dict(title="Nearest donors' errors", accent=None, eq_size=11.5,
         eq=r"$r_j=\dfrac{q^{sim}_j-q^{obs}_j}{s_j}$" if SCALED else r"$r_j=q^{sim}_j-q^{obs}_j$",
         note="how wrong the LSTM is today\nat the K nearest gauges"),
    dict(title="Learned correction", accent=LEARNED, eq_size=11,
         eq=r"$\delta_i=f_\theta\,(r_{1..K},\;d_{1..K},\;\mathrm{season},\;q^{sim}_i/s_i)$" if SCALED
         else r"$\delta_i=f_\theta\,(r_{1..K},\;d_{1..K},\;\mathrm{season},\;q^{sim}_i)$",
         note="a small network turns the donors' errors, the season and\n"
              "the target's predicted flow into today's correction"),
    dict(title="Correct the target", accent=LEARNED, eq_size=10.5,
         eq=(r"$q^{corr}_i=\max(0,$", r"$q^{sim}_i-s_i\,\delta_i)$" if SCALED
             else r"$q^{sim}_i-\delta_i)$"),
         note="shift the prediction by\nthat many of its own SDs" if SCALED
         else "shift the prediction\nby that many mm/day"),
)


def network(ax, rect):
    """The four inputs named in words, two hidden layers, one output: today's correction."""
    x0, y0, w, h = rect
    inputs = ((r"donor errors  $r_1 \ldots r_K$", RAW),
              (r"donor distances  $d_1 \ldots d_K$", RAW),
              ("season (day of year)", INK),
              (r"target's own flow  $q^{sim}_i/s_i$" if SCALED
               else r"target's predicted flow  $q^{sim}_i$", LEARNED))
    layers = [[(0.48, v) for v in (0.86, 0.62, 0.38, 0.14)],
              [(0.62, v) for v in (0.80, 0.60, 0.40, 0.20)],
              [(0.75, v) for v in (0.80, 0.60, 0.40, 0.20)],
              [(0.88, 0.50)]]
    for a, b in zip(layers, layers[1:]):
        for ua, va in a:
            for ub, vb in b:
                ax.plot(*zip(pt(rect, ua, va), pt(rect, ub, vb)), color=MUTE, lw=0.4,
                        alpha=0.45, zorder=3)
    for (u, v), (lab, col) in zip(layers[0], inputs):
        ax.scatter(*pt(rect, u, v), s=34, color=col, edgecolor="white", linewidth=0.7, zorder=5)
        track(ax.text(*pt(rect, u - 0.025, v), lab, fontsize=LAB_FS - 0.3, color=col,
                      ha="right", va="center", zorder=7), x0, x0 + w)
    for layer in layers[1:3]:
        for u, v in layer:
            ax.scatter(*pt(rect, u, v), s=26, color="white", edgecolor=MUTE, linewidth=0.8,
                       zorder=5)
    ax.scatter(*pt(rect, 0.88, 0.50), s=80, color=LEARNED, edgecolor="white", linewidth=0.7,
               zorder=6)
    track(ax.text(*pt(rect, 0.88, 0.38), "today's\ncorrection $\\delta_i$", fontsize=LAB_FS - 0.3,
                  color=LEARNED, ha="center", va="top", linespacing=1.3, zorder=7), x0, x0 + w)
    track(ax.text(*pt(rect, 0.68, 0.02), f"2 hidden layers × {HIDDEN} units", fontsize=6.6,
                  color=MUTE, ha="center", va="bottom", zorder=7), x0, x0 + w)


def sketch(ax, rect, i):
    x0, y0, w, h = rect
    # Above the step box's own white fill (zorder 2), or it is painted over and vanishes.
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                linewidth=0, facecolor=GRID, alpha=0.75, zorder=2.5))
    if i == 2:
        network(ax, rect)
        return
    if i == 3:
        hydrograph(ax, rect, accent=LEARNED)
        return

    tx, ty = basins(ax, rect, mark_donors=i != 0, accent=LEARNED)
    if i == 0:
        label_donors(ax, rect, Q_LAB)
        ax.text(*pt(rect, TARGET[0] - 0.08, TARGET[1] - 0.02), r"$q_i$", fontsize=LAB_FS,
                color=LEARNED, ha="right", va="center", zorder=7)
    else:
        for u, v in DONORS:
            ax.plot([tx, pt(rect, u, v)[0]], [ty, pt(rect, u, v)[1]], color=MUTE, lw=0.9,
                    alpha=0.6, solid_capstyle="round", zorder=3)
        label_donors(ax, rect, R_LAB)


def train_band(ax):
    """How the network is trained and K chosen: the held-out fold apart, five gauged folds to
    train on, one gauged fold held back to choose K."""
    x0, w = MARGIN, FIG_W - 2 * MARGIN
    ax.add_patch(FancyBboxPatch((x0, CV_Y), w, CV_H,
                                boxstyle="round,pad=0,rounding_size=0.08", linewidth=1.0,
                                edgecolor=MUTE, facecolor=GRID, alpha=0.45, zorder=2))
    cy = CV_Y + CV_H - 0.10 - CELL_H / 2
    cell_y = cy - CELL_H / 2

    target_x = x0 + 0.16
    train_x = target_x + CELL_W + SPLIT_GAP
    train_end = train_x + 5 * CELL_W + 4 * CELL_GAP
    val_x = train_end + SPLIT_GAP
    cells = [(target_x, LEARNED, "-")] + [(train_x + n * (CELL_W + CELL_GAP), RAW, "-")
                                          for n in range(5)] + [(val_x, RAW, "--")]
    for cx, colour, ls in cells:
        ax.add_patch(FancyBboxPatch((cx, cell_y), CELL_W, CELL_H,
                                    boxstyle="round,pad=0,rounding_size=0.03", linewidth=1.1,
                                    linestyle=ls, edgecolor=colour, facecolor=colour,
                                    alpha=0.30 if ls == "-" else 0.12, zorder=4))

    lab_y = cell_y - 0.05
    arr_x0 = val_x + CELL_W + 0.15
    for cx, text, colour, lo, hi in (
            (target_x + CELL_W / 2, "ungauged\n(target)", LEARNED, x0, train_x),
            ((train_x + train_end) / 2, "gauged:\ntrain", RAW, train_x, train_end),
            (val_x + CELL_W / 2, "gauged:\nchoose K", RAW, train_end, arr_x0)):
        track(ax.text(cx, lab_y, text, fontsize=7.4, color=colour, ha="center", va="top",
                      linespacing=1.25, zorder=5), lo, hi)

    arr_x1 = arr_x0 + 0.43
    ax.add_patch(FancyArrowPatch((arr_x0, cy), (arr_x1, cy), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.2, color=MUTE, zorder=5))
    text_x = arr_x1 + 0.16
    track(ax.text(text_x, cy,
                  f"Train $f_\\theta$ on gauged folds (loss NSE*, {SEEDS} seeds averaged); "
                  f"choose $K$ ({K_GRID[0]}–{K_GRID[-1]}) on a held-back gauged fold",
                  fontsize=8.4, color=INK, ha="left", va="center", zorder=5),
          text_x, x0 + w - 0.12)


def draw_steps(ax, y=Y0):
    """The four step boxes and the arrows between them, with their bottom edge at `y`."""
    x = MARGIN
    for i, (step, w) in enumerate(zip(STEPS, WIDTHS)):
        if i:
            ax.add_patch(FancyArrowPatch((x - GAP, y + BOX_H * 0.75), (x, y + BOX_H * 0.75),
                                         arrowstyle="-|>", mutation_scale=11, linewidth=1.2,
                                         color=MUTE, zorder=5))
        add_box(ax, x, y, step, i + 1, draw=sketch, w=w)
        x += w + GAP


def main(out=None):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    track(ax.text(MARGIN, FIG_H - 0.11,
                  "Correcting a target ungauged basin with a learned function of donor errors",
                  fontsize=13, color=INK, fontweight="bold", ha="left", va="top"),
          MARGIN, FIG_W - MARGIN)

    train_band(ax)
    draw_steps(ax)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    bad = check_fit(fig, ax)
    os.makedirs(f"{HERE}/{OUT_DIR}", exist_ok=True)
    stem = out or f"{HERE}/{OUT_DIR}/method_mlp"
    fig.savefig(f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(f"{stem}.svg", facecolor="white")
    print(f"-> {stem}.png, {stem}.svg" + (f"   ({bad} overflowing)" if bad else "   (fit ok)"))


if __name__ == "__main__":
    main()
