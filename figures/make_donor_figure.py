"""Which donors the correction actually uses.

    python figures/make_donor_figure.py                  # -> recon_ungauged_donor_map_nse.png
    python figures/make_donor_figure.py --metric kge      # the KGE-maximising selection instead
    python figures/make_donor_figure.py --fold 0          # the gauged-reconstruction pilot instead

Every held-out basin joined to the K gauged basins it actually borrows from, each line's
opacity its IDW weight, so the reader can see without being told that the neighbourhood is
local. (K, lambda, p) come straight from `src.correct.choose()` -- no intermediate CSV.

Colour: `figures/palette.py` slots 0 and 1 (#2a78d6 blue = gauged donors and the uncorrected
baseline, #eb6834 orange = the correction), validated there -- normal-vision dE 33.6, protan
24.5, deutan 31.6, all past the >=15 / >=8 thresholds. Only those two categorical colours are
ever simultaneous; the donor links are neutral ink, not a third series, and every series is
legend- or label-identified so nothing rests on colour alone.
"""
import argparse
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from figures.palette import CORRECTED, GRID, INK, MUTE, RAW  # noqa: E402
from src.correct import (SCORERS, choose, default_target_group, distances,  # noqa: E402
                         find_fold, idw_weights, load_all, reconstruct_gauged)


def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelsize=9, length=3)


def draw_network(ax, runs_dir, k, pw, loc_csv="data/camels531.csv", f_override=None):
    """The donor links of whichever fold is available first, over the CAMELS gauge positions.

    `f_override`, when given, is a gauged-reconstruction pilot dict (from
    `src.correct.reconstruct_gauged`) instead of the real PUB fold -- its `pool`/`held` are the
    donor/"learn" set and the reconstructed group, not the real gauged/ungauged split.
    """
    if f_override is not None:
        f = f_override
    else:
        folds = load_all(runs_dir)
        if not folds:
            raise SystemExit(f"no vanilla_fold*_seed*/ runs under {runs_dir}/")
        f = folds[0]
    D, basins = distances(loc_csv=loc_csv)
    loc = pd.read_csv(loc_csv, dtype={"gauge_id": str}).set_index("gauge_id")
    lat = loc.loc[basins, "gauge_lat"].to_numpy()
    lon = loc.loc[basins, "gauge_lon"].to_numpy()
    pool, held = f["pool"], f["held"]

    W = idw_weights(f["d_held"], k, pw)
    segs, alphas = [], []
    for r, i in enumerate(held):
        for c in np.flatnonzero(W[r]):
            segs.append([(lon[i], lat[i]), (lon[pool[c]], lat[pool[c]])])
            alphas.append(W[r, c])
    a = np.asarray(alphas)
    ax.add_collection(LineCollection(segs, colors=MUTE, linewidths=0.5,
                                     alpha=0.10 + 0.45 * (a / a.max()), zorder=2))
    held_label = ("reconstructed basins" if f_override is not None else "held-out basins")
    ax.scatter(lon[pool], lat[pool], s=5, color=RAW, alpha=0.55, lw=0, zorder=3,
               label=f"gauged donors ({len(pool)})")
    ax.scatter(lon[held], lat[held], s=13, color=CORRECTED, lw=0.4, zorder=4,
               edgecolor="white", label=f"{held_label} ({len(held)})")
    ax.set_aspect(1 / np.cos(np.radians(np.mean(lat))))
    ax.set_xlabel("longitude", color=INK, fontsize=10)
    ax.set_ylabel("latitude", color=INK, fontsize=10)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5, labelcolor=INK,
              handletextpad=0.4, borderpad=0.2)
    target_group = f["target_group"] if f_override is not None else None
    return f["fold"], target_group


def main(runs_dir="runs", out=None, f_override=None, pilot=False, metric="nse"):
    if f_override is not None:
        k, p = f_override["k"], f_override["p"]
        n, nf = len(f_override["held"]), 1
    else:
        folds = load_all(runs_dir)
        if not folds:
            raise SystemExit(f"no vanilla_fold*_seed*/ runs under {runs_dir}/")
        picked, _ = choose(folds, mode="inner", metric=metric)
        k, _, p = picked[folds[0]["fold"]]
        n, nf = sum(len(f["held"]) for f in folds), len(folds)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    fold, target_group = draw_network(ax, runs_dir, k, p, f_override=f_override)
    style(ax)

    title = (f"Reconstruction in training basins using donor gauges (K = {k})" if pilot
             else f"Reconstruction in held-out basins using donor gauges (K = {k})")
    fold_txt = f"fold {fold}" if nf == 1 else f"all {nf} folds"
    if pilot:
        fold_txt += f", group {target_group}"
    basin_word = "reconstructed (gauged-but-discontinued)" if pilot else "held-out"
    caption = (f"{n} {basin_word} basins, {fold_txt}. K and p held at the values the pipeline "
              f"selected inside the training basins; the map shows fold {fold}'s network only.")
    if pilot:
        caption += (" Single-(fold, group) pilot: donors are ~5/6 of that fold's training "
                    "pool, not the full pool a real deployment or the PUB cell has.")
    fig.text(0.01, 0.015, textwrap.fill(caption, 110), fontsize=8.5, color=MUTE, va="bottom")
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.19, top=0.86)
    fig.text(0.10, 0.97, title, color=INK, fontsize=11.5, va="top")
    out_stem = "recon_gauged_donor_map" if pilot else "recon_ungauged_donor_map"
    out = out or f"{HERE}/{out_stem}_{metric}.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--metric", choices=list(SCORERS), default="nse",
                    help="which selection objective chose K, lambda and p")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fold", type=int, default=None,
                    help="build the gauged-reconstruction pilot for this outer fold instead of "
                    "the real PUB folds")
    ap.add_argument("--target-group", type=int, default=None,
                    help="which inner group goes dark (default: smallest available)")
    a = ap.parse_args()

    if a.fold is None:
        main(a.runs, a.out, metric=a.metric)
    else:
        f = find_fold(load_all(a.runs), a.fold)
        target_group = a.target_group if a.target_group is not None else default_target_group(f)
        pilot_f, _ = reconstruct_gauged(f, target_group, metric=a.metric)
        main(a.runs, a.out, f_override=pilot_f, pilot=True, metric=a.metric)
