"""Why a neighbour can speak for a basin it is not: the LSTM's error is regional.

    python figures/make_coherence_figure.py   # -> figures/recon_ungauged_coherence.png

This figure is the motivation, drawn before the method is introduced. It makes one claim in
three steps, and the claim is falsifiable at every step.

(a) The error is not white noise. On a given day the model runs high across one region and low
    across another, in patches hundreds of km wide. The day drawn is the one whose spatial
    structure is the MEDIAN over the test period -- a typical day, deliberately not the most
    flattering one.

(b) That coherence decays with distance, and dies. Every pair of gauged basins contributes one
    point: the correlation between their same-day log residuals, against the distance between
    them. The correction's independently-measured break-even distance is drawn on top; it lands
    where coherence has decayed to near-nothing, which is the figure's real result -- the
    correction pays out to exactly the distance the error stays shared, and no further.

(c) Not every kind of error is regional, and that is why the method corrects one thing only.
    Each bar asks the same question -- how well do a basin's 20 nearest gauges predict this
    property of its error? Basin flashiness is the control: a genuinely regional physical
    property, to show the question CAN be answered yes. The day-to-day error is nearly as
    regional, so the method borrows it. The dispersion deficit is not, so no amount of borrowing
    fixes variability -- tested directly, and it costs NSE.

Colour: `figures/palette.py` slots 0 and 1 as a diverging pair through white for the map, which
inherits that pair's validated separation (normal dE 33.6, CVD >= 24.5). In (c) colour repeats
what position and the labels already say, so nothing rests on it alone.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from analysis.donor_geometry import break_even  # noqa: E402
from figures.palette import CORRECTED, GRID, INK, MUTE, RAW, TEAL  # noqa: E402
from src.correct import EPS, load_all  # noqa: E402

EDGES = np.array([0, 25, 50, 100, 150, 200, 300, 400, 600, 800, 1200, 2000, 5000])
N_NEAR = 20


def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelsize=9, length=3)


def pairwise_coherence(res, mask):
    """Pairwise-complete correlation between every pair of basins' daily log residuals."""
    r0, n = np.where(mask, res, 0.0), mask.astype(np.float64)
    cnt = np.maximum(n @ n.T, 1)
    mu = (r0 @ n.T) / cnt
    cov = (r0 @ r0.T) / cnt - mu * mu.T
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    return cov / np.outer(sd, sd)


def neighbour_predicts(x, near):
    """Correlation between each basin's value and the mean of its N_NEAR nearest gauges'."""
    ok = np.isfinite(x)
    pred = np.array([np.mean(x[near[i][ok[near[i]]]]) if ok[near[i]].any() else np.nan
                     for i in range(len(x))])
    g = ok & np.isfinite(pred)
    return float(np.corrcoef(x[g], pred[g])[0, 1])


def typical_day(res, mask, near):
    """The day whose error field is median-spatially-structured: a representative day, not a
    flattering one. Structure = how well a basin's error is predicted by its neighbours', that
    day, across basins."""
    X = np.where(mask, res, 0.0)
    Y = np.stack([X[near[i]].mean(0) for i in range(len(X))])
    reported = mask.sum(0)
    xs, ys = X - X.mean(0), Y - Y.mean(0)
    c = (xs * ys).sum(0) / np.maximum(np.sqrt((xs ** 2).sum(0) * (ys ** 2).sum(0)), 1e-12)
    c[reported < 0.8 * len(X)] = np.nan
    return int(np.nanargmin(np.abs(c - np.nanmedian(c)))), float(np.nanmedian(c))


def main(runs_dir="runs", fold=0, out=None, range_csv="runs/recon_ungauged_donor_range.csv",
         loc_csv="data/camels531.csv", cache_dir="data/cache"):
    f = next(x for x in load_all(runs_dir) if x["fold"] == fold)
    pool = f["pool"]
    sim, obs = f["pred"][pool], f["obs"][pool]
    res = np.log(np.clip(sim, 0, None) + EPS) - np.log(obs + EPS)
    mask = np.isfinite(res)
    D = f["D"][np.ix_(pool, pool)]

    Dn = D.copy()
    np.fill_diagonal(Dn, np.inf)
    near = np.argsort(Dn, axis=1)[:, :N_NEAR]

    C = pairwise_coherence(res, mask)
    iu = np.triu_indices(len(pool), 1)
    cc, dd = C[iu], D[iu]

    # explicit margins, not tight_layout: the map's fixed aspect makes tight_layout give up
    fig = plt.figure(figsize=(15.5, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.0], wspace=0.30,
                          left=0.045, right=0.995, top=0.88, bottom=0.235)

    # --- (a) one day's error field ------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    day, med_struct = typical_day(res, mask, near)
    loc = pd.read_csv(loc_csv, dtype={"gauge_id": str}).set_index("gauge_id")
    cache_basins = json.load(open(f"{cache_dir}/meta.json"))["basins"]
    pool_ids = [cache_basins[i] for i in pool]
    lat = loc.loc[pool_ids, "gauge_lat"].to_numpy()
    lon = loc.loc[pool_ids, "gauge_lon"].to_numpy()

    v = res[:, day]
    ok = np.isfinite(v)
    lim = np.nanpercentile(np.abs(v[ok]), 95)
    cmap = LinearSegmentedColormap.from_list("resid", [RAW, "#f7f7f7", CORRECTED])
    sc = ax.scatter(lon[ok], lat[ok], c=np.clip(v[ok], -lim, lim), cmap=cmap,
                    vmin=-lim, vmax=lim, s=26, linewidths=0.4, edgecolors="white", zorder=3)
    # inside the axes, over the empty Pacific corner: keeps the colourbar out of the layout,
    # which otherwise squeezes the map to make room for it
    cax = ax.inset_axes([0.02, 0.13, 0.32, 0.035])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=[-lim, 0, lim])
    cb.ax.set_xticklabels(["model ran low", "0", "ran high"])
    cb.ax.tick_params(colors=MUTE, labelsize=7.5, length=2, pad=1)
    cb.outline.set_visible(False)
    ax.set_aspect(1.25)
    ax.set_xlabel("longitude", color=INK, fontsize=10, labelpad=2)
    ax.set_ylabel("latitude", color=INK, fontsize=10)
    ax.set_title("(a)  the error is regional, not random", color=INK, fontsize=11,
                 loc="left", pad=8)
    ax.text(1.0, 1.012, "a typical day: median spatial structure of the test period",
            transform=ax.transAxes, fontsize=8, color=MUTE, ha="right")
    style(ax)

    # --- (b) coherence vs distance ------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    mid, med, q25, q75 = [], [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = (dd >= lo) & (dd < hi)
        if s.sum() > 20:
            mid.append(np.sqrt(max(lo, 5) * hi))
            med.append(np.median(cc[s]))
            q25.append(np.percentile(cc[s], 25))
            q75.append(np.percentile(cc[s], 75))
    ax.fill_between(mid, q25, q75, color=CORRECTED, alpha=0.16, lw=0, zorder=2)
    ax.plot(mid, med, color=CORRECTED, lw=2, zorder=4)
    ax.plot(mid, med, "o", ms=5, color=CORRECTED, markeredgecolor="white",
            markeredgewidth=1.2, zorder=5)
    ax.axhline(0, color=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=1)

    be = None
    if os.path.exists(range_csv):
        be = break_even(pd.read_csv(range_csv, dtype={"basin": str}))
    if be:
        ax.axvline(be, color=INK, lw=1.2, ls=(0, (5, 3)), zorder=6)
        ax.text(be * 1.08, 0.52, f"the correction stops\npaying at {be:.0f} km",
                fontsize=8.5, color=INK, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("distance between two basins  (km)", color=INK, fontsize=10)
    ax.set_ylabel("correlation of their same-day errors", color=INK, fontsize=10)
    ax.set_title("(b)  and it decays with distance", color=INK, fontsize=11, loc="left", pad=8)
    ax.text(0.015, 0.04, f"{len(cc):,} basin pairs; band is the interquartile range",
            transform=ax.transAxes, fontsize=8, color=MUTE)
    style(ax)

    # --- (c) what transfers, and what does not ------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    mm = np.isfinite(obs)
    disp, bias, flash = [], [], []
    for i in range(len(pool)):
        s, o = sim[i][mm[i]], obs[i][mm[i]]
        disp.append(o.std() / s.std() if s.std() > 0 else np.nan)
        bias.append(np.log((s.mean() + EPS) / (o.mean() + EPS)))
        flash.append(np.log(o + EPS).std())
    anom = res - np.array([res[i][mask[i]].mean() for i in range(len(pool))])[:, None]
    a0 = np.where(mask, anom, 0.0)
    wn = np.zeros_like(D)
    np.put_along_axis(wn, near, 1.0 / N_NEAR, 1)
    donor_anom = wn @ a0
    daily = np.nanmedian([np.corrcoef(donor_anom[i][mask[i]], anom[i][mask[i]])[0, 1]
                          for i in range(len(pool)) if mask[i].sum() > 100])

    bars = [("basin flashiness\n(a real regional property)", neighbour_predicts(np.array(flash), near), TEAL),
            ("day-to-day error\n(what the method borrows)", daily, CORRECTED),
            ("standing volume bias", neighbour_predicts(np.array(bias), near), CORRECTED),
            ("dispersion deficit\n(tested: does not transfer)", neighbour_predicts(np.array(disp), near), MUTE)]
    y = np.arange(len(bars))[::-1]
    ax.barh(y, [b[1] for b in bars], color=[b[2] for b in bars], height=0.62,
            alpha=0.85, edgecolor="white", linewidth=1.4, zorder=3)
    for yi, (_, val, _) in zip(y, bars):
        ax.annotate(f"{val:.2f}", (val, yi), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([b[0] for b in bars], fontsize=8.5, color=INK)
    ax.set_xlim(0, 0.75)
    ax.set_xlabel(f"how well the {N_NEAR} nearest gauges predict it", color=INK, fontsize=10)
    ax.set_title("(c)  but only some kinds of error are", color=INK, fontsize=11,
                 loc="left", pad=8)
    style(ax)
    ax.grid(False, axis="y")

    caption = (
        f"Fold {fold}'s {len(pool)} gauged basins -- the information the correction actually "
        f"has. (a) and (b) use the seed-mean hydrograph's log residual against measured flow.\n"
        f"In (c) the flashiness, volume-bias and dispersion bars correlate one number per basin "
        f"against its {N_NEAR} nearest gauges' mean; the day-to-day bar correlates the daily "
        f"series against theirs, then takes the median basin.")
    fig.text(0.008, 0.022, caption, fontsize=8.5, color=MUTE, linespacing=1.5)
    out = out or f"{HERE}/recon_ungauged_coherence.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}")
    print(f"   typical day index {day}, median spatial structure {med_struct:.3f}")
    print(f"   coherence <25 km {med[0]:+.3f}  ->  1200-2000 km {med[-2]:+.3f}")
    print(f"   bars: " + ", ".join(f"{b[0].splitlines()[0]} {b[1]:.3f}" for b in bars))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.runs, a.fold, a.out)
