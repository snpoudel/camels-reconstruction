"""Where the correction helps and where it hurts, geographically.

    python figures/make_change_map.py --model lstm   # -> figures/change_map_lstm.png
    python figures/make_change_map.py --model hbv     # gauged only -> figures/change_map_hbv.png
    python figures/make_change_map.py --cell ungauged --model lstm   # force one cell

Maps, NSE on top row and KGE below (Nash-Sutcliffe and Kling-Gupta efficiency, the project's
two headline skill scores), one column per cell the model has (gauged, ungauged, or both -- see
`palette.available_cells`). Every basin plotted at its gauge location, coloured by how much that
one score changed for that one basin -- corrected minus uncorrected. This is a different job
from every other figure here: those compare two named series (identity, categorical colour);
this shows one continuous quantity with a sign (polarity, diverging colour), so it earns its own
two-hue-plus-neutral palette rather than a generic diverging map.

The two hues are the project's own blue and vermillion -- the same pair every other figure uses
for "LSTM" and "LSTM + correction" -- through a neutral grey at zero. Vermillion already means
"the correction" on every other page in this set, so a vermillion basin here reads as "the
correction helped here" instead of introducing an unrelated hue for the same idea.

Two things matter more than the hue choice for actually being able to read this map, and both
were wrong in an earlier version:

  the colour cap    Most basins change by only a few hundredths -- the middle 80% span roughly
                    +/-0.15-0.2 -- while a handful of near-zero-flow basins swing by 0.5 or more
                    on one bad day. Capping the colour range at the true extreme (or even its
                    98th percentile) spends most of the ramp on those few basins and leaves
                    every ordinary basin sitting in the pale, low-contrast middle third of the
                    ramp, which is what "hard to read" meant. Capping at the 85th percentile of
                    |change| instead means the typical basin actually reaches saturated colour;
                    the (now more numerous) basins beyond the cap are still shown at full
                    colour, the colourbar's arrows say so, and the caption gives the count. Each
                    panel gets its own cap: gauged and ungauged basins have different change
                    distributions, so one shared cap would under- or over-saturate one of them.

  the neutral point   A pale, near-white grey at zero (this project's GRID) reads as "empty"
                    rather than "measured, but unchanged", and blends into the page and the
                    axis gridlines. The zero point uses a visibly grey, not a barely-there one.
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
from evaluate import read_info, read_metrics  # noqa: E402
from palette import CELL_TITLE, DIVERGING, INK, LABEL, MUTE, PANEL  # noqa: E402
from palette import TITLE, available_cells, caption, draw_us_outline, model_label  # noqa: E402
from palette import panel_letter, series_pair, style  # noqa: E402
from residual_postprocessing import BASINS_CSV  # noqa: E402

# The 85th percentile of |change|, not the 98th: see module docstring. ~15% of basins land
# beyond it and show at the pole colour; that trade is what makes the typical basin readable.
CAP_PERCENTILE = 85

PANELS = [("nse", "NSE, Nash-Sutcliffe efficiency"), ("kge", "KGE, Kling-Gupta efficiency")]


def main(cells, model, prefix):
    raw, corr = list(series_pair(model))  # e.g. "LSTM", "LSTM + correction", in that order
    loc = pd.read_csv(BASINS_CSV, dtype={"gauge_id": str}).set_index("gauge_id")

    fig, axes = plt.subplots(len(cells), 2, figsize=(9, 8.5 if len(cells) == 2 else 4.6),
                             squeeze=False)

    letter, lines = 0, []
    for row, cell in enumerate(cells):
        df = read_metrics(cell, model)
        for col, (metric, label) in enumerate(PANELS):
            ax = axes[row][col]
            piv = df.pivot_table(index="basin", columns="variant", values=metric)
            gain = (piv[corr] - piv[raw]).dropna()
            lat = loc.loc[gain.index, "gauge_lat"].to_numpy()
            lon = loc.loc[gain.index, "gauge_lon"].to_numpy()
            g = gain.to_numpy()

            cap = np.percentile(np.abs(g), CAP_PERCENTILE)
            norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-cap, vmax=cap)

            draw_us_outline(ax)
            order = np.argsort(np.abs(g))  # draw the biggest changes last, on top
            sc = ax.scatter(lon[order], lat[order], c=g[order], cmap=DIVERGING, norm=norm, s=11,
                            edgecolor=MUTE, linewidth=0.25, zorder=3)

            ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
            ax.set_xlabel("longitude (°E)", color=INK, fontsize=LABEL)
            ax.set_ylabel("latitude (°N)", color=INK, fontsize=LABEL)
            style(ax, grid=False)
            metric_name = label.split(",")[0]
            ax.set_title(f"{panel_letter(letter)} {CELL_TITLE[cell]} — {metric_name}",
                        color=INK, fontsize=PANEL, loc="left", pad=6)
            letter += 1

            # Horizontal and thin, under the map rather than the tall default beside it -- a
            # map's colour range is a minor readout next to the geography, not a co-equal panel.
            # pad is generous because it has to clear the map's own x tick labels and axis
            # label, which sit below the spine in that same shrunk margin.
            cb = fig.colorbar(sc, ax=ax, orientation="horizontal", location="bottom",
                              extend="both", pad=0.32, fraction=0.045, aspect=32, shrink=0.85)
            cb.set_label(f"change in {metric_name}", color=INK, fontsize=LABEL - 1.5)
            cb.ax.tick_params(colors=MUTE, labelsize=LABEL - 2.5, length=2)
            cb.outline.set_visible(False)

            n_worse = int((g < 0).sum())
            n_off = int((np.abs(g) > cap).sum())
            lines.append(f"{CELL_TITLE[cell]} {metric_name}: median {np.median(g):+.2f}, "
                         f"{len(gain) - n_worse}/{len(gain)} improved, cap {cap:.2f}, "
                         f"{n_off} beyond it")

    fig.suptitle(f"{model_label(model)}: where the correction changes skill",
                fontsize=TITLE, color=INK, x=0.012, ha="left", y=0.995)
    bottom = caption(fig, f"Coloured by each score, corrected minus uncorrected.  Colour capped "
                          f"per panel at the {CAP_PERCENTILE}th percentile of |change| so the "
                          f"typical basin reaches saturated colour; basins beyond it are shown "
                          f"at full colour.  {'  '.join(lines)}.", 104)
    top = 0.94 if len(cells) == 2 else 0.88
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.subplots_adjust(top=top, hspace=0.55, wspace=0.2)
    out = f"{HERE}/{prefix}change_map_{model}.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}   " + "   ".join(lines))


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
