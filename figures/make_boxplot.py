"""Where the correction acts: four bias measures, before and after.

    python figures/make_boxplot.py --model lstm   # -> figures/metrics_boxplots_lstm.png
    python figures/make_boxplot.py --model hbv     # gauged only -> metrics_boxplots_hbv.png
    python figures/make_boxplot.py --cell ungauged --model lstm   # force one cell

Overall skill lives in the CDF figure (NSE and KGE); this one is about water balance -- overall
percent bias, then bias split by how high the river actually was that day: top 10%,
middle 30-90%, bottom 30%. Four panels, one per bias measure -- that grid doesn't grow with the
number of cells (unlike the CDF/change-map/hyperparameter figures): instead, when the model has
both a gauged and an ungauged cell, each panel's two boxes (raw, corrected) become two close
pairs of boxes, one pair per cell, with a gap between the pairs.

The flow bands are cut on the OBSERVED flow (Yilmaz et al. 2008), so "high-flow bias" is the
model's error on the days the river really was high, not on the days it predicted high. For the
bias panels zero is the target and the dashed line marks it, so a box straddling zero more
tightly is better regardless of which side it sits on.

Each box is drawn over its own basins as a faint jittered scatter, so it summarises what is
actually visible underneath rather than standing in for it. The axis is scaled to the drawn
whiskers, not the full data range -- percent bias on near-zero-flow streams runs to five figures
and would flatten every box. A box's median sits inside it; the red count under its lower
whisker is how many of that box's basins (dots) fall outside the shown range.
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
from evaluate import fold_settings, read_info, read_metrics  # noqa: E402
from palette import CELL_BLURB, CELL_TITLE, GRID, INK, LABEL, MUTE, PANEL  # noqa: E402
from palette import TITLE, available_cells, caption, model_label, panel_letter  # noqa: E402
from palette import series_pair, settings_note, style  # noqa: E402

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
    rng = np.random.default_rng(0)  # fixed, so the jitter is reproducible

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
            patch.set_edgecolor("white")  # 2px surface gap between adjacent fills
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
        # Scale to the whiskers, not the data range.
        wy = np.concatenate([w.get_ydata() for w in bp["whiskers"]])
        lo, hi = (min(wy.min(), 0.0), max(wy.max(), 0.0)) if centred else (wy.min(), wy.max())
        pad = (hi - lo) * 0.16 + 1e-9
        ax.set_ylim(lo - pad, hi + pad)

        # Median only inside the box; how many of its basins fall outside the shown axis goes
        # under the lower whisker instead, in red, so the box itself stays uncluttered.
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
            if r == 1:  # shared x axis: labels only on the bottom row
                ax.set_xticklabels([CELL_TITLE[cl].split()[0] for cl in cells],
                                   fontsize=LABEL - 1, color=INK)
        else:
            ax.set_xticks(positions)
            if r == 1:
                ax.set_xticklabels(list(SERIES), fontsize=LABEL - 1, color=INK)

    fig.suptitle(f"{model_label(model)}: bias before and after the correction",
                fontsize=TITLE, color=INK, x=0.012, ha="left", y=0.995)

    blurbs = []
    for cell in cells:
        info = read_info(cell, model)
        blurbs.append(f"{CELL_TITLE[cell]} ({info['n_basins']} basins, {CELL_BLURB[cell]}, "
                      f"{info['n_folds']} folds, {info['n_seeds']} seeds, test "
                      f"{info['test_period']}, "
                      f"{settings_note(fold_settings(dfs[cell], model_label(model)))})")
    bottom = caption(fig, f"{'; '.join(blurbs)}, chosen by cross-validation that never reads "
                          f"the basin being corrected.  Boxes: quartiles and median; whiskers "
                          f"1.5 IQR, outliers omitted; dots are individual basins, jittered.  A "
                          f"red count under a box's lower whisker is how many of that box's "
                          f"basins (dots) fall outside the shown axis range.", 145)

    legend_room = 0.0
    if combined:
        # Above the caption, not below it, and clear of the bottom row's shared x tick labels
        # -- boxed in a faint grey panel so it reads as one legend for the whole figure rather
        # than another line of caption text.
        handles = [mpatches.Patch(facecolor=c, alpha=0.70, edgecolor="white", label=n)
                  for n, c in SERIES.items()]
        legend_room = 0.135
        fig.legend(handles=handles, loc="lower center", ncol=len(SERIES),
                  bbox_to_anchor=(0.5, bottom + 0.015), fontsize=LABEL - 1, labelcolor=INK,
                  frameon=True, facecolor=GRID, edgecolor="none", framealpha=0.6)

    top = 0.94
    fig.tight_layout()
    fig.subplots_adjust(top=top, bottom=bottom + legend_room, hspace=0.22)
    out = f"{HERE}/{prefix}metrics_boxplots_{model}.png"
    fig.savefig(out, dpi=200, facecolor="white")
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
