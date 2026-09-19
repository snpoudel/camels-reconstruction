"""Where the correction helps and where it hurts, geographically.

    python figures/make_change_map.py --cell ungauged
    python figures/make_change_map.py --cell gauged

Every basin plotted at its gauge location, coloured by how much KGE (Kling-Gupta efficiency)
changed for that one basin -- corrected minus uncorrected. This is a different job from every
other figure here: those compare two named series (identity, categorical colour); this shows one
continuous quantity with a sign (polarity, diverging colour), so it earns its own two-hue-plus-
neutral palette rather than reusing the project's blue/orange series identity, which would read
as "this dot is the LSTM" instead of "this dot improved".

Diverging, blue<->red through a neutral grey at zero (matplotlib's RdBu, the ColorBrewer
diverging ramp): blue basins improved, red basins got worse, pale basins were barely touched.
The colour range is capped at the 98th percentile of |change| rather than the true extreme, so
one or two outlier basins do not wash out every other point to near-white; baskets beyond the
cap still show at full colour (the colourbar's arrows say so) and are listed by count in the
caption.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)
from evaluate import read_info, read_metrics  # noqa: E402
from palette import CELL_TITLE, GRID, INK, LABEL, MUTE, SERIES, TITLE  # noqa: E402
from palette import caption, style  # noqa: E402
from residual_postprocessing import BASINS_CSV  # noqa: E402


def main(cell):
    df = read_metrics(cell)
    info = read_info(cell)
    raw, corr = list(SERIES)  # "LSTM", "LSTM + correction", in that order

    piv = df.pivot_table(index="basin", columns="variant", values="kge")
    gain = (piv[corr] - piv[raw]).dropna()

    loc = pd.read_csv(BASINS_CSV, dtype={"gauge_id": str}).set_index("gauge_id")
    lat = loc.loc[gain.index, "gauge_lat"].to_numpy()
    lon = loc.loc[gain.index, "gauge_lon"].to_numpy()
    g = gain.to_numpy()

    # Symmetric range capped at the 98th percentile of |change| -- a few outlier basins (a
    # near-zero-flow stream where one bad day swings KGE wildly) would otherwise stretch the
    # scale so far that every ordinary basin looks pale grey.
    cap = np.percentile(np.abs(g), 98)
    norm = matplotlib.colors.TwoSlopeNorm(vcenter=0.0, vmin=-cap, vmax=cap)

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    order = np.argsort(np.abs(g))  # draw the biggest changes last, on top
    # A near-zero-change basin fills near-white, so a white edge would make it disappear
    # entirely against the page; MUTE gives every marker a visible outline regardless of fill.
    sc = ax.scatter(lon[order], lat[order], c=g[order], cmap="RdBu", norm=norm, s=36,
                    edgecolor=MUTE, linewidth=0.5, zorder=3)

    ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
    ax.set_xlabel("longitude (°E)", color=INK, fontsize=LABEL)
    ax.set_ylabel("latitude (°N)", color=INK, fontsize=LABEL)
    ax.set_title(f"{CELL_TITLE[cell]}: where the correction changes KGE",
                 color=INK, fontsize=TITLE, loc="left", pad=8)
    style(ax)

    cb = fig.colorbar(sc, ax=ax, extend="both", pad=0.02, fraction=0.046)
    cb.set_label("change in KGE (corrected minus LSTM)", color=INK, fontsize=LABEL - 1)
    cb.ax.tick_params(colors=MUTE, labelsize=LABEL - 2)
    cb.outline.set_visible(False)

    n_worse = int((g < 0).sum())
    n_off = int((np.abs(g) > cap).sum())
    bottom = caption(fig, f"{len(gain)} basins, test period {info['test_period']}, coloured by "
                          f"KGE, corrected minus uncorrected.  Median change "
                          f"{np.median(g):+.3f}; {len(gain) - n_worse} of {len(gain)} basins "
                          f"improved.  Colour capped at the 98th percentile of |change| "
                          f"({cap:.2f}); {n_off} basins exceed it and are shown at full colour.",
                     108)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    out = f"{HERE}/{cell}_change_map.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}   median change {np.median(g):+.4f}, {len(gain) - n_worse}/{len(gain)} improved")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=list(CELL_TITLE), default="ungauged")
    main(ap.parse_args().cell)
