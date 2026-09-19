"""Which gauges the correction actually borrows from.

    python figures/make_donor_map.py --cell ungauged
    python figures/make_donor_map.py --cell gauged

Every corrected basin joined to the K gauges it actually borrows from, each line's opacity its
inverse-distance weight, so the reader can see without being told that the neighbourhood is
local. One fold's network is drawn -- all seven overlaid would be an unreadable mat of lines,
and every fold's geometry is the same procedure over a different seventh of the basins.

Colour: palette slots 0 and 1 (blue = donor gauges, orange = the basins being corrected),
validated in palette.py. The donor links are neutral ink, not a third series, and both point
series are named in the legend, so nothing rests on colour alone.
"""
import argparse
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


def main(cell):
    df = read_metrics(cell)
    info = read_info(cell)
    # Pick the fold/group whose K is closest to 5: few enough donor lines to read individually,
    # many enough to look like a real neighbourhood rather than a single nearest-neighbour lookup.
    settings = fold_settings(df)
    fold, k, lam, p = min(settings, key=lambda s: (abs(s[1] - 5), s[0]))

    labels, dist = fold_labels(), distance_matrix()
    target = np.flatnonzero(labels == fold)
    if cell == "ungauged":
        donor = np.flatnonzero(labels != fold)          # the fold's own gauged basins
        d = dist[np.ix_(target, donor)]
    else:
        donor = np.arange(len(labels))                  # every other basin, self masked
        d = dist[target].copy()
        d[np.arange(len(target)), target] = np.inf

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
    # against the basin points rather than disappearing into the background.
    ax.add_collection(LineCollection(segs, colors=INK, linewidths=1.1, zorder=2,
                                     alpha=0.20 + 0.55 * (w / w.max())))
    # In the gauged cell every basin is a donor for every other, so the blue points are not
    # "the donor pool" -- they are simply the basins not being corrected in this group.
    shown = np.setdiff1d(donor, target)
    ax.scatter(lon[shown], lat[shown], s=16, color=RAW, alpha=0.75, lw=0.5,
              edgecolor="white", zorder=3,
              label=(f"donor gauges ({len(shown)})" if cell == "ungauged"
                     else f"other basins ({len(shown)})"))
    ax.scatter(lon[target], lat[target], s=34, color=CORRECTED, lw=0.8, zorder=4,
              edgecolor="white", label=f"basins being corrected ({len(target)})")

    ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
    ax.set_xlabel("longitude (°E)", color=INK, fontsize=LABEL)
    ax.set_ylabel("latitude (°N)", color=INK, fontsize=LABEL)
    ax.set_title(f"{CELL_TITLE[cell]}: where the borrowed residuals come from (K = {k})",
                 color=INK, fontsize=TITLE, loc="left", pad=8)
    ax.legend(frameon=False, loc="lower left", fontsize=LABEL - 1.5, labelcolor=INK,
              handletextpad=0.4, borderpad=0.2, markerscale=2.2)
    style(ax)

    group = "Fold" if cell == "ungauged" else "Group"
    pool = (f"that fold's {len(donor)} training basins" if cell == "ungauged"
            else f"all {len(donor) - 1} other basins, its own record excluded")
    bottom = caption(fig, f"{group} {fold} of {info['n_folds']}: {len(target)} of "
                          f"{info['n_basins']} basins, each joined to the K = {k} nearest of "
                          f"{pool}, line opacity proportional to the inverse-distance weight "
                          f"(p = {p:g}).  Test period {info['test_period']}.  The other "
                          f"{group.lower()}s repeat this over a different seventh of the basins.",
                     118)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    out = f"{HERE}/{cell}_donor_map.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}   {len(target)} basins, {len(segs)} donor links")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=list(CELL_TITLE), default="ungauged")
    main(ap.parse_args().cell)
