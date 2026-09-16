"""Which donors the correction uses, and the distance at which it stops paying.

    python figures/make_donor_figure.py                  # -> figures/donor_range.png
    python figures/make_donor_figure.py --arch donor     # parent branch's 7 folds, 531 basins

Left: the donor network itself, for one fold. Every held-out basin is joined to the K gauged
basins it actually borrows from, each line's weight drawn as its opacity, so the reader can see
without being told that the neighbourhood is local. The inset gives the three donor-distance
distributions behind that impression.

Right: what happens when the donors are pushed away. Every donor nearer than the floor on the
x-axis is deleted and the correction re-run, so each basin is its own control and basin
difficulty cannot masquerade as a distance effect. The curve crosses zero at the annotated
break-even; past it the correction is adding noise, not information. The dashed reference is
the real gauge network, where nothing is deleted.

The strip above that panel carries the donor-distance distributions on the same axis, which is
the point of putting them there: the reader can see directly how much of the real network sits
inside the working range and how much of it does not.

Both panels read `runs/donor_geometry.csv` and `runs/donor_range.csv` from
`analysis/donor_geometry.py`, which is where the arithmetic and the caveats live.

Colour: `figures/palette.py` slots 0 and 1 (#2a78d6 blue = gauged donors and the uncorrected
baseline, #eb6834 orange = the correction), validated there -- normal-vision dE 33.6, protan
24.5, deutan 31.6, all past the >=15 / >=8 thresholds. Only those two categorical colours are
ever simultaneous; the donor links and reference lines are neutral ink, not a third series, and
every series is legend- or label-identified so nothing rests on colour alone.
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
from analysis.donor_geometry import break_even  # noqa: E402
from figures.palette import CORRECTED, GRID, INK, MUTE, RAW  # noqa: E402
from src.correct import distances, idw_weights, load_all  # noqa: E402


def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelsize=9, length=3)


def draw_network(ax, runs_dir, arch, k, pw, loc_csv="data/camels531.csv"):
    """The donor links of whichever fold is available first, over the CAMELS gauge positions."""
    folds = load_all(runs_dir, arch=arch)
    if not folds:
        raise SystemExit(f"no {arch}_fold*_seed*/ runs under {runs_dir}/")
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
    ax.scatter(lon[pool], lat[pool], s=5, color=RAW, alpha=0.55, lw=0, zorder=3,
               label=f"gauged donors ({len(pool)})")
    ax.scatter(lon[held], lat[held], s=13, color=CORRECTED, lw=0.4, zorder=4,
               edgecolor="white", label=f"held-out basins ({len(held)})")
    ax.set_aspect(1 / np.cos(np.radians(np.mean(lat))))
    ax.set_xlabel("longitude", color=INK, fontsize=10)
    ax.set_ylabel("latitude", color=INK, fontsize=10)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5, labelcolor=INK,
              handletextpad=0.4, borderpad=0.2)
    return f["fold"], f"Donor links actually used, fold {f['fold']} (K = {k}, p = {pw:g})"


def draw_distance_strip(ax, geo):
    """Donor-distance distributions, sharing the panel's x-axis so they can be read against it."""
    ins = ax.inset_axes((0, 1.04, 1, 0.17), sharex=ax)
    cols = [("d_kth", "K-th"), ("d_eff", "weighted mean"), ("d_near", "nearest")]
    ins.boxplot([geo[c].to_numpy() for c, _ in cols], orientation="horizontal", widths=0.5,
                showfliers=False, medianprops=dict(color=CORRECTED, lw=1.5),
                boxprops=dict(color=MUTE, lw=0.8), whiskerprops=dict(color=MUTE, lw=0.8),
                capprops=dict(color=MUTE, lw=0.8))
    ins.set_yticks(range(1, len(cols) + 1), [lab for _, lab in cols], fontsize=8, color=INK)
    ins.tick_params(axis="x", labelbottom=False, length=0)
    ins.tick_params(axis="y", colors=MUTE, length=0)
    for side in ("top", "right", "bottom"):
        ins.spines[side].set_visible(False)
    ins.spines["left"].set_color(MUTE)
    ins.text(0.998, 1.02, "donor distances in the real network", transform=ins.transAxes,
             ha="right", va="bottom", fontsize=8.5, color=MUTE)
    return ins


def draw_range(ax, curve):
    """Median gain against the donor-distance floor, with the per-fold spread behind it."""
    kept = curve[curve.corrected]
    per_fold = kept.groupby(["fold", "floor"]).gain.median().unstack(0)
    pooled = kept.groupby("floor").gain.median()
    real, x = pooled.loc[0], pooled.index.to_numpy(float)
    m = x > 0

    for col in per_fold.columns:
        v = per_fold[col]
        ax.plot(v.index.to_numpy(float)[m], v.to_numpy()[m], color=CORRECTED, lw=0.9,
                alpha=0.30, zorder=2)
    ax.axhline(0, color=INK, lw=1.0, zorder=3)
    ax.axhline(real, color=MUTE, lw=1.0, ls=(0, (4, 3)), zorder=3)
    ax.plot(x[m], pooled.to_numpy()[m], color=CORRECTED, lw=2.4, zorder=5,
            marker="o", ms=4, markeredgecolor="white", markeredgewidth=0.8,
            label="median over all held-out basins")

    ax.set_xscale("log")
    ax.set_xlim(6, 4200)
    ax.annotate(f"real gauge network  {real:+.4f}", (0.02, real),
                xycoords=("axes fraction", "data"), fontsize=8.5, color=MUTE,
                va="bottom", ha="left")
    be = break_even(curve)
    if be:
        ax.axvline(be, color=INK, lw=0.9, ls=(0, (2, 2)), zorder=4)
        ax.annotate(f"break-even\n{be:.0f} km", xy=(be, 0), xytext=(be * 1.22, real * 0.62),
                    fontsize=9, color=INK, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.8))
    ax.set_xlabel("every donor nearer than this deleted (km)", color=INK, fontsize=10)
    ax.set_ylabel("change in NSE vs the uncorrected LSTM", color=INK, fontsize=10)
    ax.legend(frameon=False, loc="lower left", fontsize=8.5, labelcolor=INK)
    return real, be, "What the correction is worth as its donors recede"


def main(runs_dir="runs", arch="vanilla", out=None,
         geometry_csv="runs/donor_geometry.csv", range_csv="runs/donor_range.csv"):
    for path in (geometry_csv, range_csv):
        if not os.path.exists(path):
            raise SystemExit(f"{path} missing -- run analysis/donor_geometry.py "
                             f"--arch {arch} first")
    geo = pd.read_csv(geometry_csv, dtype={"basin": str})
    curve = pd.read_csv(range_csv, dtype={"basin": str})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6),
                                   gridspec_kw=dict(width_ratios=(1.3, 1.0), wspace=0.22))
    one = geo[geo.fold == geo.fold.min()].iloc[0]
    fold, t1 = draw_network(ax1, runs_dir, arch, int(one.k), float(one.p))
    style(ax1)
    real, be, t2 = draw_range(ax2, curve)
    style(ax2)
    strip = draw_distance_strip(ax2, geo)
    if be:
        strip.axvline(be, color=INK, lw=0.9, ls=(0, (2, 2)), zorder=4)

    n, nf = geo.basin.nunique(), geo.fold.nunique()
    fold_txt = f"fold {fold}" if nf == 1 else f"all {nf} folds"
    fig.text(0.008, 0.012, textwrap.fill(
        f"{n} held-out basins, {fold_txt}. K, $\\lambda$ and p held at the values the pipeline "
        f"selected inside the training basins; the correction is never re-tuned as donors are "
        f"removed, because a deployed one could not be. Basins left with fewer than K donors "
        f"above a floor are not corrected, and are excluded at that floor only.", 168),
        fontsize=8.5, color=MUTE, va="bottom")
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.20, top=0.80, wspace=0.20)
    # Both panel titles on one baseline: ax2 carries the distance strip above it, so neither
    # can use its own axes title without colliding with the other's.
    for ax, title in ((ax1, t1), (ax2, t2)):
        fig.text(ax.get_position().x0, 0.945, title, color=INK, fontsize=11, va="bottom")
    out = out or f"{HERE}/donor_range.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}   real network {real:+.4f}, "
          + (f"break-even {be:.0f} km" if be else "never crosses zero"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--arch", default="vanilla")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.runs, a.arch, a.out)
