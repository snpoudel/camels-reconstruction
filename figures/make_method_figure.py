"""What the neighbour-residual correction does, on one real basin and one real day.

    python figures/make_method_figure.py    # -> figures/method.png

Three panels, left to right, following the arithmetic in `src/correct.py`:

  (a) pick the K nearest gauged basins to the ungauged target; the weight is 1/distance
  (b) at each of them, today, compare the model against the gauge -- that is the log-residual;
      the weighted mean R says how wrong the model is across this neighbourhood right now
  (c) push the target's own hydrograph by a fraction lambda of that, and nothing else changes

Every number drawn is taken from the actual fold-0 run and the actual CAMELS record; nothing
is illustrative. The basin and the day are chosen in code (see `pick_example`) rather than
hardcoded, so the figure cannot drift away from the data behind it.

Colour: `figures/palette.py` -- slot 0 blue for the uncorrected model, slot 1 orange for the
correction and for the target basin, ink for observations. Validated in that module
(normal dE 33.6, protan 24.5, deutan 31.6). Observations are drawn in ink rather than a third
hue, so the two model series are the only colour pair in play.
"""
import json
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


def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTE); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelsize=8.5, length=3)


def pick_example(pred, obs, held, pool, D):
    """A held-out basin whose improvement is *typical*, and the day the correction moves most.

    Chosen in code, not hardcoded. Among basins whose nearest gauge sits at an ordinary
    distance (the interquartile range), take the one whose NSE gain is closest to the median
    gain. Picking the best-improved basin instead would put a +0.24 basin in a figure whose
    fold-wide median gain is +0.04, which is the kind of illustration a reader is right to
    distrust.

    Note this selection reads the target basins' observations -- that is legitimate for
    choosing what to *draw*, and is not part of the method; `src/correct.py` never does it.
    `analysis/leak_audit.py` tests the method, not this function.
    """
    corrected = correct(pred[held], pred[pool], obs[pool], D[np.ix_(held, pool)])
    def nse(s, o):
        m = np.isfinite(o)
        return 1 - ((np.clip(s, 0, None)[m] - o[m]) ** 2).sum() / ((o[m] - o[m].mean()) ** 2).sum()
    gain = np.array([nse(corrected[i], obs[held][i]) - nse(pred[held][i], obs[held][i])
                     for i in range(len(held))])
    dnn = D[np.ix_(held, pool)].min(1)
    ok = (dnn > np.percentile(dnn, 25)) & (dnn < np.percentile(dnn, 75))
    target = np.median(gain)
    row = int(np.argmin(np.where(ok, np.abs(gain - target), np.inf)))
    return row, gain[row]


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

    row, gain = pick_example(pred, obs, held, pool, D)
    i = held[row]
    d_i = D[i, pool]
    nb = np.argsort(d_i)[:K]
    donors, dist = pool[nb], d_i[nb]
    w = 1.0 / np.maximum(dist, 1.0); w = w / w.sum()

    corrected_i = correct(pred[held], pred[pool], obs[pool], D[np.ix_(held, pool)])[row]
    res_all = (np.log(np.clip(pred[donors], 0, None) + EPS) - np.log(obs[donors] + EPS))
    R_all = np.nansum(w[:, None] * np.nan_to_num(res_all), 0) / np.nansum(
        w[:, None] * np.isfinite(res_all), 0)

    # The window, then the day inside it where the correction actually moves the hydrograph
    # in mm/day -- picking the largest |R| instead lands on a day when the river is near zero
    # and the correction, though large in ratio, is invisible.
    search = slice(900, 1050)
    effect = np.abs(corrected_i[search] - np.clip(pred[i][search], 0, None))
    day = 900 + int(np.nanargmax(effect))
    # Plot a short window centred on that day. A season-long window compresses the gap
    # between the two model lines into the line width, which is the one thing this panel
    # has to show.
    half_win = 32
    win = slice(day - half_win, day + half_win)

    fig = plt.figure(figsize=(13.0, 5.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.55], wspace=0.28,
                          left=0.045, right=0.985, top=0.86, bottom=0.17)
    axm, axr, axh = (fig.add_subplot(gs[0, j]) for j in range(3))

    # ---- (a) which gauges, and how much each counts ---------------------------
    axm.scatter(loc.loc[basins, "gauge_lon"], loc.loc[basins, "gauge_lat"], s=4,
                color=GRID, zorder=2, linewidths=0)
    tlon, tlat = loc.loc[basins[i], "gauge_lon"], loc.loc[basins[i], "gauge_lat"]
    for j, wj in zip(donors, w):
        axm.plot([tlon, loc.loc[basins[j], "gauge_lon"]], [tlat, loc.loc[basins[j], "gauge_lat"]],
                 color=MUTE, lw=0.6 + 7 * wj, alpha=0.65, zorder=3, solid_capstyle="round")
    axm.scatter(loc.loc[[basins[j] for j in donors], "gauge_lon"],
                loc.loc[[basins[j] for j in donors], "gauge_lat"], s=42, color=RAW,
                edgecolor="white", linewidth=1.2, zorder=4, label=f"{K} nearest gauges")
    axm.scatter([tlon], [tlat], s=170, marker="*", color=CORRECTED, edgecolor="white",
                linewidth=1.2, zorder=5, label="ungauged target")
    dlon = loc.loc[[basins[j] for j in donors], "gauge_lon"].to_numpy()
    dlat = loc.loc[[basins[j] for j in donors], "gauge_lat"].to_numpy()
    cx, cy = np.r_[dlon, tlon], np.r_[dlat, tlat]
    half = max(np.ptp(cx), np.ptp(cy)) * 0.85 + 0.35
    axm.set_xlim(cx.mean() - half * 1.25, cx.mean() + half * 1.25)
    axm.set_ylim(cy.mean() - half, cy.mean() + half)
    axm.set_title("(a)  pick the K nearest gauges", color=INK, fontsize=10.5, loc="left", pad=8)
    axm.text(0.5, 0.015, "line width $\\propto$ weight $1/d$", transform=axm.transAxes,
             ha="center", va="bottom", fontsize=8.5, color=MUTE)
    axm.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    axm.set_xticks([]); axm.set_yticks([])
    for s in axm.spines.values():
        s.set_color(GRID)

    # ---- (b) today's error at each gauge --------------------------------------
    r_day = res_all[:, day]
    order = np.argsort(w)
    ypos = np.arange(K)
    axr.barh(ypos, r_day[order], color=np.where(r_day[order] > 0, RAW, CORRECTED),
             height=0.62, zorder=3)
    axr.axvline(0, color=MUTE, lw=0.9, zorder=2)
    axr.axvline(R_all[day], color=INK, lw=1.8, ls=(0, (4, 2)), zorder=4)
    axr.annotate(f"weighted mean\nR = {R_all[day]:+.2f}", (R_all[day], 0.4),
                 xytext=(7 if R_all[day] < 0 else -7, 0), textcoords="offset points",
                 ha="left" if R_all[day] < 0 else "right", va="center", fontsize=9, color=INK,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
    axr.set_yticks(ypos)
    axr.set_yticklabels([f"w={w[o]:.2f}" for o in order], fontsize=8)
    axr.set_xlabel("log(model) $-$ log(gauge), that day", color=INK, fontsize=9.5)
    axr.set_title("(b)  how wrong the model is there", color=INK, fontsize=10.5,
                  loc="left", pad=8)
    axr.text(0.98, 0.985, f"{dates[day]:%-d %b %Y}", transform=axr.transAxes,
             ha="right", va="top", fontsize=9, color=INK)
    axr.text(0.98, 0.915, "positive = model ran high", transform=axr.transAxes,
             ha="right", va="top", fontsize=8.5, color=MUTE)
    style(axr)

    # ---- (c) apply it to the target -------------------------------------------
    t = dates[win]
    axh.fill_between(t, 0, obs[i][win], color=INK, alpha=0.10, zorder=2)
    axh.plot(t, obs[i][win], color=INK, lw=1.3, zorder=5, label="observed (never used)")
    axh.plot(t, np.clip(pred[i][win], 0, None), color=RAW, lw=1.5, zorder=4, label="LSTM")
    axh.plot(t, corrected_i[win], color=CORRECTED, lw=1.8, zorder=6, label="LSTM + correction")
    axh.axvline(dates[day], color=MUTE, lw=0.9, ls=(0, (3, 3)), zorder=3)
    axh.set_ylabel("flow  (mm/day)", color=INK, fontsize=9.5)
    axh.set_title(f"(c)  nudge the target by $\\lambda$R", color=INK, fontsize=10.5,
                  loc="left", pad=8)
    axh.text(0.985, 0.985,
             f"basin {basins[i]}\nNSE {gain:+.3f} -- the median gain for this fold",
             transform=axh.transAxes, ha="right", va="top", fontsize=8.5, color=MUTE,
             linespacing=1.5)
    axh.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    axh.annotate(f"{np.exp(-LAMBDA * R_all[day]):.2f}x here",
                 (dates[day], corrected_i[day]), xytext=(14, 14), textcoords="offset points",
                 fontsize=8.5, color=INK,
                 arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.8))
    style(axh)
    axh.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(byweekday=0, interval=2))
    axh.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%-d %b"))

    fig.text(0.008, 0.025,
             f"$q_{{corr}} = (q_{{sim}} + \\epsilon)\\,e^{{-\\lambda R}} - \\epsilon$   with   "
             f"K = {K},  $\\lambda$ = {LAMBDA},  $\\epsilon$ = {EPS} mm/day.   "
             f"The target basin's own discharge is never read; the weights use distance only.   "
             f"Fold 0, {n_seeds}-seed mean hydrograph.",
             fontsize=8.5, color=MUTE)
    out = out or f"{HERE}/method.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}  (basin {basins[i]}, {dates[day]:%Y-%m-%d}, R={R_all[day]:+.3f}, gain {gain:+.3f})")


if __name__ == "__main__":
    main()
