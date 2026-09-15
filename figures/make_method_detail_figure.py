"""The correction worked through, stage by stage, with real numbers at every step.

    python figures/make_method_detail_figure.py   # -> figures/method_detail.png

`make_method_figure.py` shows the idea. This one shows the arithmetic, which is the part a
methods section has to be able to stand on. Four stages, following `src/correct.py` line for
line:

  (a) the K nearest gauged basins to one ungauged target, with their distances
  (b) turn distance into weight, inverse-distance, normalised to sum to one
  (c) at each of those gauges, today, how far the model sits from the measurement, in logs
  (d) apply a fraction lambda of the weighted mean to the target's own prediction

Every value printed is read from the fold-0 run and the CAMELS record. The basin and the day
are chosen in code, not hardcoded, so the figure cannot drift from the data behind it, and the
basin is the one whose improvement is *closest to the median* for the fold rather than the one
that improves most -- see `pick_example`.

Two things here read the target basin's observations, and neither is part of the method:
choosing which basin to draw, and the greyed "observed" bar in (d) that lets a reader judge
whether the move helped. `src/correct.py` reads neither; `analysis/leak_audit.py` tests that.

Colour: `figures/palette.py` -- slot 0 blue for the uncorrected model, slot 1 orange for the
correction and the target, ink and mute for text and rules. Validated in that module
(normal dE 33.6, protan 24.5, deutan 31.6).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from figures.palette import CORRECTED, GRID, INK, MUTE, RAW  # noqa: E402
from src.correct import EPS, K, LAMBDA, correct, distances, load_fold  # noqa: E402

MONO = {"family": "DejaVu Sans Mono"}


def blank(ax, title, subtitle=None):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0, 1.0, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=11, color=INK)
    if subtitle:
        ax.text(1.0, 1.0, subtitle, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, color=MUTE)


def main(out=None):
    got = load_fold(0)
    if got is None:
        raise SystemExit("needs runs/vanilla_fold0_seed*/")
    pred, t_idx, gauged, n_seeds = got
    D, basins = distances()
    Q = np.load("data/cache/Q.npy")
    dates = pd.to_datetime(np.load("data/cache/dates.npy")[t_idx])
    obs = Q[:, t_idx]
    pool, held = np.flatnonzero(gauged), np.flatnonzero(~gauged)
    loc = pd.read_csv("data/camels531.csv", dtype={"gauge_id": str}).set_index("gauge_id")

    sys.path.insert(0, HERE)
    from make_method_figure import pick_example
    row, gain = pick_example(pred, obs, held, pool, D)
    i = held[row]
    corrected_i = correct(pred[held], pred[pool], obs[pool], D[np.ix_(held, pool)])[row]

    d_i = D[i, pool]
    nb = np.argsort(d_i)[:K]
    donors, dist = pool[nb], d_i[nb]
    inv = 1.0 / np.maximum(dist, 1.0)
    w = inv / inv.sum()

    res_all = np.log(np.clip(pred[donors], 0, None) + EPS) - np.log(obs[donors] + EPS)
    Rt = np.nansum(w[:, None] * np.nan_to_num(res_all), 0) / np.nansum(
        w[:, None] * np.isfinite(res_all), 0)
    search = slice(900, 1050)
    day = 900 + int(np.nanargmax(np.abs(corrected_i[search] - np.clip(pred[i][search], 0, None))))
    r_day, R = res_all[:, day], Rt[day]
    reported = np.isfinite(r_day)

    fig = plt.figure(figsize=(13.4, 7.6))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.14,
                          left=0.035, right=0.975, top=0.905, bottom=0.07)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axc, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # ---------- (a) the neighbourhood -----------------------------------------
    axa.scatter(loc.loc[basins, "gauge_lon"], loc.loc[basins, "gauge_lat"], s=5,
                color=GRID, zorder=2, linewidths=0)
    tlon, tlat = loc.loc[basins[i], "gauge_lon"], loc.loc[basins[i], "gauge_lat"]
    for j, dj in zip(donors, dist):
        jl, jt = loc.loc[basins[j], "gauge_lon"], loc.loc[basins[j], "gauge_lat"]
        axa.plot([tlon, jl], [tlat, jt], color=MUTE, lw=0.9, alpha=0.6, zorder=3)
        axa.annotate(f"{dj:.0f}", ((tlon + jl) / 2, (tlat + jt) / 2), fontsize=7.5,
                     color=MUTE, ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    axa.scatter(loc.loc[[basins[j] for j in donors], "gauge_lon"],
                loc.loc[[basins[j] for j in donors], "gauge_lat"], s=52, color=RAW,
                edgecolor="white", linewidth=1.2, zorder=4, label=f"K = {K} nearest gauges")
    axa.scatter([tlon], [tlat], s=210, marker="*", color=CORRECTED, edgecolor="white",
                linewidth=1.2, zorder=5, label=f"ungauged target {basins[i]}")
    dlon = loc.loc[[basins[j] for j in donors], "gauge_lon"].to_numpy()
    dlat = loc.loc[[basins[j] for j in donors], "gauge_lat"].to_numpy()
    cx, cy = np.r_[dlon, tlon], np.r_[dlat, tlat]
    half = max(np.ptp(cx), np.ptp(cy)) * 0.80 + 0.3
    axa.set_xlim(cx.mean() - half * 1.35, cx.mean() + half * 1.35)
    axa.set_ylim(cy.mean() - half, cy.mean() + half)
    axa.set_xticks([]); axa.set_yticks([])
    for s in axa.spines.values():
        s.set_color(GRID)
    axa.set_title("(a)  the K nearest gauged basins", color=INK, fontsize=11, loc="left", pad=8)
    axa.text(0.99, 0.02, "labels: great-circle distance, km", transform=axa.transAxes,
             ha="right", va="bottom", fontsize=8, color=MUTE)
    axa.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)

    # ---------- (b) distance -> weight -----------------------------------------
    blank(axb, "(b)  distance $\\rightarrow$ weight", "inverse distance, normalised to sum to 1")
    axb.text(0.0, 0.95, r"$w_j \;=\; \dfrac{1/d_j}{\sum_{j'} 1/d_{j'}}$",
             fontsize=13, color=INK, va="top")
    y0, dy = 0.66, 0.070
    axb.text(0.00, y0, f"{'gauge':<10}{'d_j (km)':>10}{'1/d_j':>10}{'w_j':>9}",
             fontsize=9, color=MUTE, va="top", fontdict=MONO)
    axb.plot([0, 0.62], [y0 - 0.030] * 2, color=GRID, lw=1.0, transform=axb.transAxes)
    for n, (j, dj, iv, wj) in enumerate(zip(donors, dist, inv, w)):
        axb.text(0.00, y0 - 0.065 - n * dy,
                 f"{basins[j]:<10}{dj:>10.1f}{iv:>10.4f}{wj:>9.3f}",
                 fontsize=9, color=INK, va="top", fontdict=MONO)
        axb.barh(y0 - 0.083 - n * dy, wj * 0.28, left=0.68, height=0.040,
                 color=RAW, alpha=0.8, transform=axb.transAxes, zorder=3)
    yb = y0 - 0.065 - K * dy
    axb.plot([0, 0.62], [yb + 0.022] * 2, color=GRID, lw=1.0, transform=axb.transAxes)
    axb.text(0.00, yb - 0.006, f"{'sum':<10}{'':>10}{inv.sum():>10.4f}{w.sum():>9.3f}",
             fontsize=9, color=MUTE, va="top", fontdict=MONO)

    # ---------- (c) the model's error at each gauge, today ---------------------
    blank(axc, "(c)  how wrong the model is at each gauge",
          f"on {dates[day]:%-d %b %Y}")
    axc.text(0.0, 0.95, r"$r_j = \log(q^{sim}_j + \epsilon) - \log(q^{obs}_j + \epsilon)$",
             fontsize=13, color=INK, va="top")
    y0, dy = 0.68, 0.070
    axc.text(0.00, y0, f"{'gauge':<10}{'q_sim':>9}{'q_obs':>9}{'r_j':>9}{'w_j r_j':>10}",
             fontsize=9, color=MUTE, va="top", fontdict=MONO)
    axc.plot([0, 0.80], [y0 - 0.030] * 2, color=GRID, lw=1.0, transform=axc.transAxes)
    for n, j in enumerate(donors):
        o = obs[j][day]
        if np.isfinite(o):
            txt = (f"{basins[j]:<10}{pred[j][day]:>9.2f}{o:>9.2f}"
                   f"{r_day[n]:>+9.3f}{w[n] * r_day[n]:>+10.3f}")
            col = INK
        else:
            txt = f"{basins[j]:<10}{pred[j][day]:>9.2f}{'--':>9}{'dropped':>9}{'':>10}"
            col = MUTE
        axc.text(0.00, y0 - 0.065 - n * dy, txt, fontsize=9, color=col, va="top", fontdict=MONO)
    yb = y0 - 0.065 - K * dy
    axc.plot([0, 0.80], [yb + 0.022] * 2, color=GRID, lw=1.0, transform=axc.transAxes)
    axc.text(0.00, yb - 0.006,
             f"{'R = sum':<10}{'':>9}{'':>9}{'':>9}{R:>+10.3f}",
             fontsize=9.5, color=INK, va="top", fontdict=MONO)
    note = ("every gauge reported" if reported.all()
            else f"{int((~reported).sum())} gauge(s) silent: dropped, weights renormalised")
    axc.text(0.0, yb - 0.085, f"{note}.   $\\epsilon$ = {EPS} mm/day",
             fontsize=8.5, color=MUTE, va="top")

    # ---------- (d) apply it ----------------------------------------------------
    blank(axd, "(d)  correct the ungauged basin", f"basin {basins[i]}, same day")
    axd.text(0.0, 0.95, r"$q^{corr} = (q^{sim} + \epsilon)\; e^{-\lambda R} - \epsilon$",
             fontsize=13, color=INK, va="top")
    qs, qc = float(np.clip(pred[i][day], 0, None)), float(corrected_i[day])
    mult = float(np.exp(-LAMBDA * R))
    lines = [
        (f"q_sim    = {qs:8.3f} mm/day", INK),
        (f"R        = {R:+8.3f}", INK),
        (f"lambda   = {LAMBDA:8.2f}", INK),
        (f"exp(-lR) = {mult:8.3f}   the multiplier", INK),
        (f"q_corr   = {qc:8.3f} mm/day", INK),
        (f"observed = {obs[i][day]:8.3f} mm/day", MUTE),
    ]
    for n, (ln, col) in enumerate(lines):
        axd.text(0.0, 0.70 - n * 0.078, ln, fontsize=10, color=col, va="top", fontdict=MONO)
    axd.text(0.0, 0.70 - len(lines) * 0.078 - 0.010,
             "observed is never used -- shown only\nso the move can be judged",
             fontsize=8.5, color=MUTE, va="top", linespacing=1.5)

    bx, by, bh = 0.70, 0.36, 0.30
    ov = float(obs[i][day])
    scale = bh / max(qs, qc, ov, 1e-6)
    for n, (val, color, lab) in enumerate(((qs, RAW, "q_sim"), (qc, CORRECTED, "q_corr"),
                                           (ov, MUTE, "observed"))):
        axd.bar(bx + n * 0.11, val * scale, width=0.075, bottom=by, color=color,
                alpha=0.85 if n < 2 else 0.35, zorder=3)
        axd.text(bx + n * 0.11, by + val * scale + 0.015, f"{val:.2f}", ha="center",
                 fontsize=8.5, color=INK)
        axd.text(bx + n * 0.11, by - 0.045, lab, ha="center", fontsize=8.5, color=MUTE)
    axd.plot([bx - 0.055, bx + 0.275], [by, by], color=MUTE, lw=0.9, zorder=2)

    fig.suptitle("Neighbour-residual correction, worked through on one basin and one day",
                 fontsize=12.5, color=INK, x=0.035, ha="left", y=0.965)
    fig.text(0.035, 0.018,
             f"Fold 0, {n_seeds}-seed mean hydrograph.  The target basin's own discharge is "
             f"never read: the weights in (b) use distance only, and (c) reads the gauges' "
             f"records, not the target's.",
             fontsize=8.5, color=MUTE)
    out = out or f"{HERE}/method_detail.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}  basin {basins[i]}, {dates[day]:%Y-%m-%d}, R={R:+.3f}, "
          f"multiplier {mult:.3f}, {qs:.2f} -> {qc:.2f} (obs {ov:.2f})")


if __name__ == "__main__":
    main()
