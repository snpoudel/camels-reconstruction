"""Where the correction helps and where it hurts, geographically.

    python figures/make_change_map.py --cell ungauged
    python figures/make_change_map.py --cell gauged

Two maps, NSE on top and KGE below (Nash-Sutcliffe and Kling-Gupta efficiency, the project's
two headline skill scores). Every basin plotted at its gauge location, coloured by how much that
one score changed for that one basin -- corrected minus uncorrected. This is a different job
from every other figure here: those compare two named series (identity, categorical colour);
this shows one continuous quantity with a sign (polarity, diverging colour), so it earns its own
two-hue-plus-neutral palette rather than a generic diverging map.

The two hues are the project's own BLUE and ORANGE -- the same pair every other figure uses for
"LSTM" and "LSTM + correction" -- through a neutral grey at zero. Orange already means "the
correction" on every other page in this set, so an orange basin here reads as "the correction
helped here" instead of introducing an unrelated hue for the same idea.

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
                    colour, the colourbar's arrows say so, and the caption gives the count.

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
from palette import BLUE, CELL_TITLE, INK, LABEL, MUTE, ORANGE, SERIES, TITLE  # noqa: E402
from palette import caption, draw_us_outline, style  # noqa: E402
from residual_postprocessing import BASINS_CSV  # noqa: E402

# A visibly grey zero, not this project's near-white GRID (that reads as "empty" on a map, and
# a marker filled with it is nearly invisible against the page).
DIVERGE_ZERO = "#b6bec4"
CHANGE_CMAP = mcolors.LinearSegmentedColormap.from_list("change", [BLUE, DIVERGE_ZERO, ORANGE])

# The 85th percentile of |change|, not the 98th: see module docstring. ~15% of basins land
# beyond it and show at the pole colour; that trade is what makes the typical basin readable.
CAP_PERCENTILE = 85

PANELS = [("nse", "NSE, Nash-Sutcliffe efficiency"), ("kge", "KGE, Kling-Gupta efficiency")]


def main(cell):
    df = read_metrics(cell)
    info = read_info(cell)
    raw, corr = list(SERIES)  # "LSTM", "LSTM + correction", in that order

    loc = pd.read_csv(BASINS_CSV, dtype={"gauge_id": str}).set_index("gauge_id")

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 9.4))
    lines, n_basins = [], None
    for ax, (col, label) in zip(axes, PANELS):
        piv = df.pivot_table(index="basin", columns="variant", values=col)
        gain = (piv[corr] - piv[raw]).dropna()
        n_basins = len(gain)
        lat = loc.loc[gain.index, "gauge_lat"].to_numpy()
        lon = loc.loc[gain.index, "gauge_lon"].to_numpy()
        g = gain.to_numpy()

        cap = np.percentile(np.abs(g), CAP_PERCENTILE)
        norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-cap, vmax=cap)

        draw_us_outline(ax)
        order = np.argsort(np.abs(g))  # draw the biggest changes last, on top
        sc = ax.scatter(lon[order], lat[order], c=g[order], cmap=CHANGE_CMAP, norm=norm, s=36,
                        edgecolor=MUTE, linewidth=0.5, zorder=3)

        ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
        ax.set_xlabel("longitude (°E)", color=INK, fontsize=LABEL)
        ax.set_ylabel("latitude (°N)", color=INK, fontsize=LABEL)
        style(ax)

        cb = fig.colorbar(sc, ax=ax, extend="both", pad=0.02, fraction=0.046)
        cb.set_label(f"change in {label.split(',')[0]}", color=INK, fontsize=LABEL - 1)
        cb.ax.tick_params(colors=MUTE, labelsize=LABEL - 2)
        cb.outline.set_visible(False)

        n_worse = int((g < 0).sum())
        n_off = int((np.abs(g) > cap).sum())
        lines.append(f"{label.split(',')[0]}: median {np.median(g):+.3f}, "
                     f"{len(gain) - n_worse}/{len(gain)} improved, cap {cap:.2f}, "
                     f"{n_off} beyond it")

    axes[0].set_title(f"{CELL_TITLE[cell]}: where the correction changes skill",
                      color=INK, fontsize=TITLE, loc="left", pad=8)

    bottom = caption(fig, f"{n_basins} basins, "
                          f"test period {info['test_period']}, coloured by each score, corrected "
                          f"minus uncorrected.  Colour capped per panel at the {CAP_PERCENTILE}th "
                          f"percentile of |change| so the typical basin reaches saturated "
                          f"colour; basins beyond it are shown at full colour.  "
                          f"{'  '.join(lines)}.", 104)
    fig.tight_layout(rect=(0, bottom, 1, 0.97))
    out = f"{HERE}/{cell}_change_map.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}   " + "   ".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=list(CELL_TITLE), default="ungauged")
    main(ap.parse_args().cell)
