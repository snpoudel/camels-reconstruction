"""Ungauged cell, one fold: which gauges each corrected basin borrows from, and how its NSE changed.

    python figures/make_donor_map.py                  # -> ungauged_donor_map_lstm.png
    python figures/make_donor_map.py --model hybrid   # -> ungauged_donor_map_hybrid.png
    python figures/make_donor_map.py --mark-no-skill

(a) Each held-out basin joined to its K donors, line opacity by inverse-distance weight. Only
one fold is drawn; all seven overlaid would be unreadable.
(b) The same basins coloured by change in NSE, on the change map's colour scale.

`--mark-no-skill` draws basins with NSE below 0 both before and after as hollow rings and drops
them from the "% improved" count: a gain that stays below 0 is still worse than the mean flow.
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
from matplotlib.collections import LineCollection  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src", "corrections", "idw"))
sys.path.insert(0, HERE)
from correction import idw_weights  # noqa: E402
from evaluate import fold_settings, folds_csv_for, read_metrics  # noqa: E402
from palette import CAP_PERCENTILE, CELL_TITLE, CORRECTED, DIVERGING, INK, LABEL, MUTE  # noqa: E402
from palette import PANEL, RAW, TITLE, draw_us_outline, model_label, panel_letter  # noqa: E402
from palette import series_pair, style  # noqa: E402
from shared import BASINS_CSV, basin_ids, distance_matrix, fold_labels  # noqa: E402

CELL = "ungauged"


def main(model, mark_no_skill):
    df = read_metrics(CELL, model)
    # The fold whose K is closest to 5: few enough lines to read, enough to look like a
    # neighbourhood rather than a nearest-neighbour lookup.
    settings = fold_settings(df, model_label(model))
    fold, k, lam, p = min(settings, key=lambda s: (abs(s[1] - 5), s[0]))

    labels, dist = fold_labels(folds_csv=folds_csv_for(model)), distance_matrix()
    target = np.flatnonzero(labels == fold)
    donor = np.flatnonzero(labels != fold)
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

    raw, corr = list(series_pair(model))
    piv = df.pivot_table(index="basin", columns="variant", values="nse")
    before, after = (piv[v].reindex(np.asarray(basin_ids())[target]).to_numpy()
                     for v in (raw, corr))
    gain = after - before
    no_skill = (before < 0) & (after < 0) if mark_no_skill else np.zeros(len(gain), bool)
    improved = int(((gain > 0) & ~no_skill).sum())

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8.0, 10.2), sharex=True, sharey=True)
    draw_us_outline(ax)
    # A high alpha floor: the weakest links are the ones that show how far the neighbourhood
    # reaches, and they vanish first.
    ax.add_collection(LineCollection(segs, colors=INK, linewidths=1.35, zorder=2,
                                     alpha=0.40 + 0.50 * (w / w.max())))
    ax.scatter(lon[donor], lat[donor], s=16, color=RAW, alpha=0.75, lw=0.5,
               edgecolor="white", zorder=3, label=f"donor gauges ({len(donor)})")
    ax.scatter(lon[target], lat[target], s=34, color=CORRECTED, lw=0.8, zorder=4,
               edgecolor="white", label=f"basins being corrected ({len(target)})")

    ax.set_title(f"{panel_letter(0)} donor gauges each corrected basin borrows from",
                 color=INK, fontsize=PANEL, loc="left", pad=6)
    ax.legend(frameon=False, loc="lower left", fontsize=LABEL - 1.5, labelcolor=INK,
              handletextpad=0.4, borderpad=0.2, markerscale=2.2)

    # Cap over every basin, so --mark-no-skill never recolours the others.
    cap = np.nanpercentile(np.abs(gain), CAP_PERCENTILE)
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-cap, vmax=cap)
    draw_us_outline(bx)
    order = [i for i in np.argsort(np.abs(gain)) if not no_skill[i]]  # biggest changes on top
    sc = bx.scatter(lon[target][order], lat[target][order], c=gain[order], cmap=DIVERGING,
                    norm=norm, s=34, edgecolor=MUTE, linewidth=0.4, zorder=3)
    if no_skill.any():
        bx.scatter(lon[target][no_skill], lat[target][no_skill], s=34, facecolor="none",
                   edgecolor=INK, linewidth=1.1, zorder=4,
                   label=f"NSE below 0 before and after ({int(no_skill.sum())})")
        bx.legend(frameon=False, loc="lower left", fontsize=LABEL - 1.5, labelcolor=INK,
                  handletextpad=0.4, borderpad=0.2, markerscale=1.6)
    bx.set_title(f"{panel_letter(1)} change in NSE at the basins being corrected "
                 f"({100 * improved / len(gain):.0f}% of sites improved)",
                 color=INK, fontsize=PANEL, loc="left", pad=6)
    bx.set_xlabel("longitude (°E)", color=INK, fontsize=LABEL)

    for a in (ax, bx):
        a.set_aspect(1 / np.cos(np.radians(lat.mean())))
        a.set_ylabel("latitude (°N)", color=INK, fontsize=LABEL)
        style(a, grid=False)

    fig.suptitle(f"{CELL_TITLE[CELL]}: where the borrowed residuals come from, and what they "
                 f"change ({model_label(model)}, K = {k})", fontsize=TITLE, color=INK,
                 x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.07, 1, 0.985))
    cax = fig.add_axes((0.25, 0.045, 0.5, 0.012))
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("change in NSE, corrected minus uncorrected", color=INK, fontsize=LABEL - 1.5)
    cb.ax.tick_params(colors=MUTE, labelsize=LABEL - 2.5, length=2)
    cb.outline.set_visible(False)
    out = f"{HERE}/{CELL}_donor_map_{model}.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"-> {out}   {len(target)} basins, {len(segs)} donor links, "
          f"{improved} improved, median change {np.nanmedian(gain):+.3f}, "
          f"cap {cap:.2f}, {int(no_skill.sum())} marked no-skill")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm")
    ap.add_argument("--mark-no-skill", action="store_true", help="set apart basins whose NSE "
                    "is below 0 both before and after the correction")
    a = ap.parse_args()
    main(a.model, a.mark_no_skill)
