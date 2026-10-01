"""Maps of the change in NSE and KGE at every basin, corrected minus uncorrected.

    python figures/make_change_map.py --model lstm                  # -> change_map_lstm.png
    python figures/make_change_map.py --cell ungauged --model lstm  # one cell

One row per cell the model has, NSE and KGE side by side. All panels share one colour scale,
capped at palette.CAP_PERCENTILE of |change| pooled across panels, so a shade means the same
change everywhere; basins beyond the cap show at full colour.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)
from evaluate import read_metrics  # noqa: E402
from palette import CAP_PERCENTILE, CELL_TITLE, DIVERGING, INK, LABEL, MUTE, PANEL  # noqa: E402
from palette import TITLE, available_cells, draw_us_outline, model_label  # noqa: E402
from palette import panel_letter, series_pair, style  # noqa: E402
from shared import BASINS_CSV  # noqa: E402

PANELS = [("nse", "NSE, Nash-Sutcliffe efficiency"), ("kge", "KGE, Kling-Gupta efficiency")]


def main(cells, model, prefix):
    raw, corr = list(series_pair(model))
    loc = pd.read_csv(BASINS_CSV, dtype={"gauge_id": str}).set_index("gauge_id")

    gains = {}
    for cell in cells:
        df = read_metrics(cell, model)
        for metric, _ in PANELS:
            piv = df.pivot_table(index="basin", columns="variant", values=metric)
            gains[cell, metric] = (piv[corr] - piv[raw]).dropna()
    cap = np.percentile(np.abs(np.concatenate([g.to_numpy() for g in gains.values()])),
                        CAP_PERCENTILE)
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-cap, vmax=cap)

    height = 6.9 if len(cells) == 2 else 4.0
    fig, axes = plt.subplots(len(cells), 2, figsize=(9, height), squeeze=False,
                             sharex=True, sharey=True)

    letter, lines = 0, []
    for row, cell in enumerate(cells):
        for col, (metric, label) in enumerate(PANELS):
            ax, gain = axes[row][col], gains[cell, metric]
            lat = loc.loc[gain.index, "gauge_lat"].to_numpy()
            lon = loc.loc[gain.index, "gauge_lon"].to_numpy()
            g = gain.to_numpy()

            draw_us_outline(ax)
            order = np.argsort(np.abs(g))  # biggest changes drawn on top
            sc = ax.scatter(lon[order], lat[order], c=g[order], cmap=DIVERGING, norm=norm, s=11,
                            edgecolor=MUTE, linewidth=0.25, zorder=3)

            ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
            if row == len(cells) - 1:
                ax.set_xlabel("longitude (°E)", color=INK, fontsize=LABEL)
            if col == 0:
                ax.set_ylabel("latitude (°N)", color=INK, fontsize=LABEL)
            style(ax, grid=False)
            metric_name = label.split(",")[0]
            improved = int((g > 0).sum())
            ax.set_title(f"{panel_letter(letter)} {CELL_TITLE[cell]} — {metric_name} "
                         f"({100 * improved / len(g):.0f}% improved)",
                         color=INK, fontsize=PANEL, loc="left", pad=6)
            letter += 1

            lines.append(f"{CELL_TITLE[cell]} {metric_name}: median {np.median(g):+.2f}, "
                         f"{improved}/{len(g)} improved, "
                         f"{int((np.abs(g) > cap).sum())} beyond the cap")

    fig.suptitle(f"{model_label(model)}: where the correction changes skill",
                 fontsize=TITLE, color=INK, x=0.012, ha="left", y=0.995)
    # Margins in inches, set by hand: tight_layout sizes the slots before the fixed map aspect
    # shrinks them, and let the titles collide with tick marks.
    fig.subplots_adjust(left=0.075, right=0.99, top=1 - 0.6 / height, bottom=1.15 / height,
                        wspace=0.08, hspace=0.3)
    cax = fig.add_axes((0.3, 0.36 / height, 0.4, 0.1 / height))
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("change in score, corrected minus uncorrected", color=INK,
                 fontsize=LABEL - 1.5)
    cb.ax.tick_params(colors=MUTE, labelsize=LABEL - 2.5, length=2)
    cb.outline.set_visible(False)
    out = f"{HERE}/{prefix}change_map_{model}.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"-> {out}   shared cap {cap:.2f}")
    for line in lines:
        print(f"   {line}")


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
