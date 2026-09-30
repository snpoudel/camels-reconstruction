"""Overall and flow-band percent bias, before and after the correction.

    python figures/make_boxplot.py --model lstm                  # -> metrics_boxplots_lstm.png
    python figures/make_boxplot.py --cell ungauged --model lstm  # one cell

Four panels: overall bias, then bias on the top 10%, middle 30-90% and bottom 30% of flows. The
bands are cut on the observed flow (Yilmaz et al., 2008). When the model has both cells, each
panel shows one pair of boxes per cell. The y axis is scaled to the whiskers because percent
bias on near-zero-flow streams runs to five figures; the red count under a box is how many of
its basins fall outside the shown range.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import read_metrics  # noqa: E402
from palette import CELL_TITLE, GRID, INK, LABEL, MUTE, PANEL, TITLE  # noqa: E402
from palette import available_cells, model_label, panel_letter, series_pair, style  # noqa: E402

PANELS = [
    ("pbias", "overall bias (%)", True),
    ("bias_high", "bias, top 10% of flows (%)", True),
    ("bias_mid", "bias, middle 30-90% of flows (%)", True),
    ("bias_low", "bias, bottom 30% of flows (%)", True),
]

PAIR_GAP, GROUP_GAP, WIDTH = 0.42, 1.15, 0.38
OFF_COUNT_COLOR = "#c1272d"


def main(cells, model, prefix):
    SERIES = series_pair(model)
    dfs = {cell: read_metrics(cell, model) for cell in cells}
    by_cell = {cell: {name: dfs[cell][dfs[cell].variant == name] for name in SERIES}
               for cell in cells}

    combined = len(cells) > 1
    if combined:
        positions = [g * GROUP_GAP + i * PAIR_GAP for g in range(len(cells))
                     for i in range(len(SERIES))]
        width = WIDTH
        group_centers = [np.mean(positions[g * len(SERIES):(g + 1) * len(SERIES)])
                         for g in range(len(cells))]
    else:
        positions = list(range(len(SERIES)))
        width = 0.52

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.0), sharex=True)
    rng = np.random.default_rng(0)  # reproducible jitter

    for letter, ((r, c), (col, title, centred)) in enumerate(
            zip([(0, 0), (0, 1), (1, 0), (1, 1)], PANELS)):
        ax = axes[r][c]
        vals, colors = [], []
        for cell in cells:
            for name, color in SERIES.items():
                vals.append(by_cell[cell][name][col].dropna().to_numpy())
                colors.append(color)

        bp = ax.boxplot(vals, positions=positions, widths=width, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK, lw=1.6),
                        whiskerprops=dict(color=MUTE, lw=1.0),
                        capprops=dict(color=MUTE, lw=1.0))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.70)
            patch.set_edgecolor("white")
            patch.set_linewidth(1.6)
            patch.set_zorder(3)
        for element in ("medians", "whiskers", "caps"):
            for line in bp[element]:
                line.set_zorder(3)
        if centred:
            ax.axhline(0, color=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=1)
        for x, (v, color) in zip(positions, zip(vals, colors)):
            ax.scatter(x + rng.uniform(-0.16, 0.16, len(v)), v, s=5, color=color, alpha=0.30,
                       linewidths=0, zorder=2)

        ax.set_title(f"{panel_letter(letter)} {title}", color=INK, fontsize=PANEL, loc="left",
                     pad=6)
        style(ax)
        wy = np.concatenate([w.get_ydata() for w in bp["whiskers"]])
        lo, hi = (min(wy.min(), 0.0), max(wy.max(), 0.0)) if centred else (wy.min(), wy.max())
        pad = (hi - lo) * 0.16 + 1e-9
        ax.set_ylim(lo - pad, hi + pad)

        for i, (x, v) in enumerate(zip(positions, vals)):
            ax.annotate(f"{np.median(v):.2f}", (x, np.median(v)), ha="center", va="center",
                        fontsize=LABEL - 2, color=INK, zorder=4,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
            n_off = int((v < lo - pad).sum() + (v > hi + pad).sum())
            if n_off:
                cap_y = bp["caps"][2 * i].get_ydata()[0]
                ax.annotate(f"({n_off})", (x, cap_y), xytext=(0, -3),
                            textcoords="offset points", ha="center", va="top",
                            fontsize=LABEL - 2.5, color=OFF_COUNT_COLOR, zorder=4)

        if combined:
            ax.set_xticks(group_centers)
            if r == 1:
                ax.set_xticklabels([CELL_TITLE[cl].split()[0] for cl in cells],
                                   fontsize=LABEL - 1, color=INK)
        else:
            ax.set_xticks(positions)
            if r == 1:
                ax.set_xticklabels(list(SERIES), fontsize=LABEL - 1, color=INK)

    fig.suptitle(f"{model_label(model)}: bias before and after the correction",
                 fontsize=TITLE, color=INK, x=0.012, ha="left", y=0.995)

    legend_room = 0.0
    if combined:
        # With both cells the x ticks name the cell, so the series need a figure-level legend.
        handles = [mpatches.Patch(facecolor=c, alpha=0.70, edgecolor="white", label=n)
                   for n, c in SERIES.items()]
        legend_room = 0.05
        fig.legend(handles=handles, loc="lower center", ncol=len(SERIES),
                   bbox_to_anchor=(0.5, 0.015), fontsize=LABEL - 1, labelcolor=INK,
                   frameon=True, facecolor=GRID, edgecolor="none", framealpha=0.6)

    fig.tight_layout()
    fig.subplots_adjust(top=0.94, bottom=0.06 + legend_room, hspace=0.22)
    out = f"{HERE}/{prefix}metrics_boxplots_{model}.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"-> {out}")
    for cell in cells:
        for col, title, _ in PANELS:
            a, b = (by_cell[cell][n][col].median() for n in SERIES)
            print(f"  {cell:<9} {title:<34} {a:+8.3f} -> {b:+8.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=list(CELL_TITLE), help="force one cell; default: every "
                    "cell this model has")
    ap.add_argument("--model", default="lstm")
    a = ap.parse_args()
    cells = [a.cell] if a.cell else available_cells(a.model)
    if not cells:
        raise SystemExit(f"no scored metrics for model '{a.model}' -- run figures/evaluate.py "
                         f"--model {a.model} first")
    main(cells, a.model, f"{a.cell}_" if a.cell else "")
