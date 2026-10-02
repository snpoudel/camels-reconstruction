"""Method diagram: the five steps of the learned MLP neighbour-residual correction.

    python figures/make_method_mlp.py    # -> lstm/method_mlp.png, lstm/method_mlp.svg

The same layout, neighbourhood and hydrograph as make_method_idw.py, so the two read side by side:
steps 1 and 3 are shared, step 2 picks the K nearest gauges without weighting them, and step 4
replaces IDW's fixed weighted mean with a small network that turns today's donor errors, their
distances, the season and the target's own predicted flow into today's correction. The band
above says how the network is trained and K chosen. K, the passes, the seeds and the layer
width are imported from src/corrections/mlp/model.py so they cannot drift from the code.
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
from make_method_idw import (BOX_H, BOX_W, CELL_GAP, CELL_H, CELL_W, CV_H, CV_Y,  # noqa: E402
                             DONORS, FIG_H, FIG_W, GAP, LAB_FS, MARGIN, OFF, Q_LAB, R_LAB,
                             SPLIT_GAP, TARGET, Y0, add_box, basins, check_fit, hydrograph,
                             label_donors, pt, track)
from make_method_idw import STEPS as IDW_STEPS  # noqa: E402
from model import EPOCHS, HIDDEN, K_GRID, SEEDS  # noqa: E402
from palette import GRID, INK, LEARNED, MUTE, RAW  # noqa: E402

D_LAB = (r"$d_{i1}$", r"$d_{i2}$", r"$d_{iK}$")

STEPS = (
    IDW_STEPS[0],
    dict(title="Nearest donors", accent=None, eq_size=11,
         eq=r"$d_{i1}\leq d_{i2}\leq\cdots\leq d_{iK}$",
         note="the K nearest gauges,\nnearest first, unweighted"),
    IDW_STEPS[2],
    dict(title="Learned correction", accent=LEARNED, eq_size=9.5,
         eq=(r"$\delta_i=f_\theta\,(r_{1..K},\,d_{1..K},$", r"$\mathrm{season},\,q^{sim}_i)$"),
         note="a small network turns\nthem into today's factor"),
    dict(title="Correct the target", accent=LEARNED, eq_size=10.5,
         eq=(r"$q^{corr}_i=(q^{sim}_i+\epsilon)$", r"$\times\; e^{\delta_i}-\epsilon$"),
         note="scale the prediction\nby that factor"),
)


def network(ax, rect):
    """Four input groups, two hidden layers, one output: the correction delta_i."""
    x0, y0, w, h = rect
    inputs = ((r"$r_j$", RAW), (r"$d_j$", RAW), ("season", INK), (r"$q^{sim}_i$", LEARNED))
    layers = [[(0.30, v) for v in (0.85, 0.62, 0.39, 0.16)],
              [(0.52, v) for v in (0.80, 0.60, 0.40, 0.20)],
              [(0.72, v) for v in (0.80, 0.60, 0.40, 0.20)],
              [(0.92, 0.50)]]
    for a, b in zip(layers, layers[1:]):
        for ua, va in a:
            for ub, vb in b:
                ax.plot(*zip(pt(rect, ua, va), pt(rect, ub, vb)), color=MUTE, lw=0.4,
                        alpha=0.45, zorder=3)
    for (u, v), (lab, col) in zip(layers[0], inputs):
        ax.scatter(*pt(rect, u, v), s=34, color=col, edgecolor="white", linewidth=0.7, zorder=5)
        track(ax.text(*pt(rect, u - 0.06, v), lab, fontsize=LAB_FS - 0.8, color=col,
                      ha="right", va="center", zorder=7), x0, x0 + w)
    for layer in layers[1:3]:
        for u, v in layer:
            ax.scatter(*pt(rect, u, v), s=26, color="white", edgecolor=MUTE, linewidth=0.8,
                       zorder=5)
    ax.scatter(*pt(rect, 0.92, 0.50), s=70, color=LEARNED, edgecolor="white", linewidth=0.7,
               zorder=6)
    ax.text(*pt(rect, 0.92, 0.36), r"$\delta_i$", fontsize=LAB_FS + 0.6, color=LEARNED,
            ha="center", va="top", zorder=7)
    track(ax.text(*pt(rect, 0.62, 0.04), f"2 × {HIDDEN} units", fontsize=6.6, color=MUTE,
                  ha="center", va="bottom", zorder=7), x0, x0 + w)


def sketch(ax, rect, i):
    x0, y0, w, h = rect
    # Above the step box's own white fill (zorder 2), or it is painted over and vanishes.
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                linewidth=0, facecolor=GRID, alpha=0.75, zorder=2.5))
    if i == 3:
        network(ax, rect)
        return
    if i == 4:
        hydrograph(ax, rect, accent=LEARNED)
        return

    tx, ty = basins(ax, rect, mark_donors=i != 0, accent=LEARNED)
    if i == 0:
        label_donors(ax, rect, Q_LAB)
        ax.text(*pt(rect, TARGET[0] - 0.08, TARGET[1] - 0.02), r"$q_i$", fontsize=LAB_FS,
                color=LEARNED, ha="right", va="center", zorder=7)
    elif i == 1:
        # Equal widths: unlike IDW, distance is an input here, not a weight fixed in advance.
        for (u, v), lab, (_, _, _, at) in zip(DONORS, D_LAB, OFF):
            ax.plot([tx, pt(rect, u, v)[0]], [ty, pt(rect, u, v)[1]], color=MUTE, lw=1.2,
                    alpha=0.7, solid_capstyle="round", zorder=3)
            ax.text(*pt(rect, TARGET[0] + at * (u - TARGET[0]), TARGET[1] + at * (v - TARGET[1])),
                    lab, fontsize=LAB_FS, color=INK, ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.92))
    elif i == 2:
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
            ((train_x + train_end) / 2, "gauged: train\n(targets, donors)", RAW, train_x,
             train_end),
            (val_x + CELL_W / 2, "gauged:\nchoose K", RAW, train_end, arr_x0)):
        track(ax.text(cx, lab_y, text, fontsize=7.4, color=colour, ha="center", va="top",
                      linespacing=1.25, zorder=5), lo, hi)

    arr_x1 = arr_x0 + 0.43
    ax.add_patch(FancyArrowPatch((arr_x0, cy), (arr_x1, cy), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.2, color=MUTE, zorder=5))
    text_x = arr_x1 + 0.16
    track(ax.text(text_x, cy - 0.02,
                  f"Train $f_\\theta$ on the gauged folds: loss $1-$KGE, {EPOCHS} passes, "
                  f"{SEEDS} seeds averaged\n"
                  f"Choose $K$ (donor gauges, {K_GRID[0]}–{K_GRID[-1]}) on the held-back gauged "
                  f"fold",
                  fontsize=8.4, color=INK, ha="left", va="center", linespacing=1.45, zorder=5),
          text_x, x0 + w - 0.12)


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

    xs = [MARGIN + i * (BOX_W + GAP) for i in range(5)]
    for i, step in enumerate(STEPS):
        add_box(ax, xs[i], Y0, step, i + 1, draw=sketch)
        if i:
            ax.add_patch(FancyArrowPatch((xs[i - 1] + BOX_W, Y0 + BOX_H * 0.75),
                                         (xs[i], Y0 + BOX_H * 0.75), arrowstyle="-|>",
                                         mutation_scale=11, linewidth=1.2, color=MUTE, zorder=5))

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    bad = check_fit(fig, ax)
    os.makedirs(f"{HERE}/lstm", exist_ok=True)
    stem = out or f"{HERE}/lstm/method_mlp"
    fig.savefig(f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(f"{stem}.svg", facecolor="white")
    print(f"-> {stem}.png, {stem}.svg" + (f"   ({bad} overflowing)" if bad else "   (fit ok)"))


if __name__ == "__main__":
    main()
