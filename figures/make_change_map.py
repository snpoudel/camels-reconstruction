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
"LSTM" and "LSTM + correction" -- through a neutral grey at zero, rather than a stock diverging
colormap (e.g. RdBu). Two reasons: the pair is already validated at far higher contrast than a
typical diverging map (normal-vision dE 33.6, CVD dE 24.5 -- see palette.py), and orange already
means "the correction" on every other page in this set, so an orange basin here reads as "the
correction helped here" instead of introducing an unrelated hue for the same idea.

The colour range is capped at the 98th percentile of |change| per panel rather than the true
extreme, so one or two outlier basins do not wash out every other point to near-neutral; basins
beyond the cap still show at full colour (the colourbar's arrows say so) and are counted in the
caption.
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
from palette import BLUE, CELL_TITLE, GRID, INK, LABEL, MUTE, ORANGE, SERIES, TITLE  # noqa: E402
from palette import caption, style  # noqa: E402
from residual_postprocessing import BASINS_CSV  # noqa: E402

# Blue (worse) -> the project's grid grey (no change) -> orange (better), the same identity
# colours every other figure uses for "LSTM" and "LSTM + correction".
CHANGE_CMAP = mcolors.LinearSegmentedColormap.from_list("change", [BLUE, GRID, ORANGE])

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

        cap = np.percentile(np.abs(g), 98)
        norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-cap, vmax=cap)
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
                          f"minus uncorrected.  Colour capped per panel at the 98th percentile "
                          f"of |change|; basins beyond it are shown at full colour.  "
                          f"{'  '.join(lines)}.", 104)
    fig.tight_layout(rect=(0, bottom, 1, 0.97))
    out = f"{HERE}/{cell}_change_map.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}   " + "   ".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=list(CELL_TITLE), default="ungauged")
    main(ap.parse_args().cell)
