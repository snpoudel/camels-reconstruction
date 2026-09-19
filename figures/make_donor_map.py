"""Which gauges the correction actually borrows from, for the ungauged cell.

    python figures/make_donor_map.py

Every held-out basin joined to the K gauges it actually borrows from, each line's opacity its
inverse-distance weight, so the reader can see without being told that the neighbourhood is
local. One fold's network is drawn -- all seven overlaid would be an unreadable mat of lines,
and every fold's geometry is the same procedure over a different seventh of the basins.

The gauged cell isn't drawn here: there every basin draws from the same all-other-basins pool,
so a per-fold donor map doesn't show anything the ungauged one doesn't already -- one picture is
enough.

Colour: palette slots 0 and 1 (blue = donor gauges, vermillion = the basins being corrected).
The donor links are neutral ink, not a third series, and both point series are named in the
legend and drawn at different marker sizes, so nothing rests on colour alone.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)
from evaluate import fold_settings, read_info, read_metrics  # noqa: E402
from palette import CELL_TITLE, CORRECTED, INK, LABEL, MUTE, RAW, TITLE  # noqa: E402
from palette import caption, draw_us_outline, style  # noqa: E402
from residual_postprocessing import (BASINS_CSV, basin_ids, distance_matrix,  # noqa: E402
                                     fold_labels, idw_weights)

CELL = "ungauged"


def main():
    df = read_metrics(CELL)
    info = read_info(CELL)
    # Pick the fold whose K is closest to 5: few enough donor lines to read individually, many
    # enough to look like a real neighbourhood rather than a single nearest-neighbour lookup.
    settings = fold_settings(df)
    fold, k, lam, p = min(settings, key=lambda s: (abs(s[1] - 5), s[0]))

    labels, dist = fold_labels(), distance_matrix()
    target = np.flatnonzero(labels == fold)
    donor = np.flatnonzero(labels != fold)          # the fold's own gauged basins
    d = dist[np.ix_(target, donor)]

    loc = pd.read_csv(BASINS_CSV, dtype={"gauge_id": str}).set_index("gauge_id")
    lat = loc.loc[basin_ids(), "gauge_lat"].to_numpy()
    lon = loc.loc[basin_ids(), "gauge_lon"].to_numpy()

    W = idw_weights(d, k, p)
    segs, weights = [], []
    for r, i in enumerate(target):
        for c in np.flatnonzero(W[r]):
            segs.append([(lon[i], lat[i]), (lon[donor[c]], lat[donor[c]])])
            weights.append(W[r, c])
    w = np.asarray(weights)

    # Shaped to the CONUS aspect the `set_aspect` below enforces, so the axes fill the
    # figure instead of floating in a band of white.
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    draw_us_outline(ax)
    # Thicker and darker than the default MUTE grid ink, so the donor links read clearly
    # against the basin points rather than disappearing into the background. The alpha floor
    # matters more than the ceiling: it is the weakest links that were vanishing, and they are
    # the ones that show how far the neighbourhood actually reaches.
    ax.add_collection(LineCollection(segs, colors=INK, linewidths=1.35, zorder=2,
                                     alpha=0.40 + 0.50 * (w / w.max())))
    ax.scatter(lon[donor], lat[donor], s=16, color=RAW, alpha=0.75, lw=0.5,
              edgecolor="white", zorder=3, label=f"donor gauges ({len(donor)})")
    ax.scatter(lon[target], lat[target], s=34, color=CORRECTED, lw=0.8, zorder=4,
              edgecolor="white", label=f"basins being corrected ({len(target)})")

    ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
    ax.set_xlabel("longitude (°E)", color=INK, fontsize=LABEL)
    ax.set_ylabel("latitude (°N)", color=INK, fontsize=LABEL)
    ax.set_title(f"{CELL_TITLE[CELL]}: where the borrowed residuals come from (K = {k})",
                 color=INK, fontsize=TITLE, loc="left", pad=8)
    ax.legend(frameon=False, loc="lower left", fontsize=LABEL - 1.5, labelcolor=INK,
              handletextpad=0.4, borderpad=0.2, markerscale=2.2)
    style(ax, grid=False)

    bottom = caption(fig, f"One of the {info['n_folds']} folds shown alone for clarity -- all "
                          f"{info['n_folds']} overlaid would be an unreadable mat of lines, and "
                          f"each fold draws from its own training-basin pool.  {len(target)} of "
                          f"{info['n_basins']} basins, each joined to the K = {k} nearest of "
                          f"that fold's {len(donor)} training basins, line opacity proportional "
                          f"to the inverse-distance weight (p = {p:g}).  Test period "
                          f"{info['test_period']}.",
                     118)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    out = f"{HERE}/{CELL}_donor_map.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}   {len(target)} basins, {len(segs)} donor links")


if __name__ == "__main__":
    main()
