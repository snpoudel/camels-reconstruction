"""Cross-validation for the correction's three constants: K, lambda, and the IDW power p.

    python analysis/tune.py                              # first, writes ..._tuning_nse.csv
    python figures/make_hyperparam_figure.py              # -> recon_ungauged_hyperparameters_nse.png
    python figures/make_hyperparam_figure.py --metric kge # the KGE-maximising surface instead

The correction has three free numbers: K, how many gauges to consult; lambda, how much of
their error to believe; and p, how steeply weight falls off with distance (weight ~ 1/d^p).
All three are chosen by cross-validation **inside each fold's training basins** -- the
training basins keep their original fold labels, so each of six groups is scored using donors
drawn only from the other five. No held-out basin appears anywhere on this figure, and none is
read to choose anything; `analysis/leak_audit.py` check [6] tests that by destroying every
held-out observation and confirming the choice does not move.

One line per fold. Each is a profile through that fold's own CV optimum: each panel sweeps one
constant with the other two held at that fold's best, and the marker sits on the optimum
itself -- specifically the simplest (K, lambda, p) within `TOLERANCE` of the true maximum (see
`src/correct.py:pick`), the same rule `choose()` actually selects with, not the raw argmax.
Reading across the lines is the point -- the optima cluster in a narrow region, and the curves
are flat around them, which is exactly why that simplification is safe to make.

Folds are replicates of the same procedure, not different things being compared, so they share
one hue rather than getting a categorical palette; identity comes from the count in the
caption, and no line needs to be told apart from another. Colour: `figures/palette.py` slot 1
(#eb6834), with ink and mute for rules and text.
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from figures.palette import CORRECTED, GRID, INK, MUTE  # noqa: E402
from src.correct import SCORERS, TOLERANCE, pick_row  # noqa: E402


def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTE); ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelsize=9, length=3)


def main(tuning="runs/recon_ungauged_tuning_nse.csv", out=None, pilot=False, metric="nse"):
    df = pd.read_csv(tuning)
    if "set" in df.columns:
        df = df.query("set == 'cv'")
    folds = sorted(df.fold.unique())
    n_basins = int(df.n_basins.iloc[0])
    metric_name = metric.upper()

    # Both K and p are sampled unevenly (see src/correct.py:K_GRID, P_GRID), so plotting either
    # by value packs the grid unevenly. Plotting by rank instead gives every step an
    # equal-width slot; the tick labels still show the real value.
    k_grid = sorted(df.k.unique())
    k_pos = {k: i for i, k in enumerate(k_grid)}
    p_grid = sorted(df.p.unique())
    p_pos = {pw: i for i, pw in enumerate(p_grid)}

    fig, (axk, axl, axp) = plt.subplots(1, 3, figsize=(8.0, 4.1), sharey=True)
    single = len(folds) == 1
    lw = 2.2 if single else 1.5
    alpha = 1.0 if single else 0.75

    opt_k, opt_l, opt_p = [], [], []
    for fold in folds:
        d = df.query("fold == @fold")
        best = pick_row(d, TOLERANCE)
        opt_k.append(int(best.k)); opt_l.append(float(best.lam)); opt_p.append(float(best.p))

        sub = d.query("lam == @best.lam and p == @best.p").sort_values("k")
        axk.plot([k_pos[k] for k in sub.k], sub.score, color=CORRECTED, lw=lw,
                 alpha=alpha, zorder=3)
        axk.plot([k_pos[best.k]], [best.score], "o", ms=7, color=CORRECTED,
                 markeredgecolor="white", markeredgewidth=1.5, zorder=5)

        sub = d.query("k == @best.k and p == @best.p").sort_values("lam")
        axl.plot(sub.lam, sub.score, color=CORRECTED, lw=lw, alpha=alpha, zorder=3)
        axl.plot([best.lam], [best.score], "o", ms=7, color=CORRECTED,
                 markeredgecolor="white", markeredgewidth=1.5, zorder=5)

        sub = d.query("k == @best.k and lam == @best.lam").sort_values("p")
        axp.plot([p_pos[pw] for pw in sub.p], sub.score, color=CORRECTED, lw=lw,
                 alpha=alpha, zorder=3)
        axp.plot([p_pos[best.p]], [best.score], "o", ms=7, color=CORRECTED,
                 markeredgecolor="white", markeredgewidth=1.5, zorder=5)

    # Too many K values to label every tick in a narrow panel -- thin to ~6 evenly spaced labels.
    step = max(1, -(-len(k_grid) // 6))
    axk.set_xticks(range(len(k_grid)))
    axk.set_xticklabels([str(k) if i % step == 0 else "" for i, k in enumerate(k_grid)])
    axk.set_xlabel("K, gauges consulted", color=INK, fontsize=10)
    axk.set_title("(a)  gauges consulted", color=INK, fontsize=10.5, loc="left", pad=8)
    axl.set_xlabel("$\\lambda$, fraction of error applied", color=INK, fontsize=10)
    axl.set_title("(b)  error trusted", color=INK, fontsize=10.5, loc="left", pad=8)
    axp.set_xticks(range(len(p_grid)))
    axp.set_xticklabels(p_grid)
    axp.set_xlabel("p, IDW power", color=INK, fontsize=10)
    axp.set_title("(c)  distance weighting", color=INK, fontsize=10.5, loc="left", pad=8)

    line_label = "per-fold profile" if not single else "profile"
    for ax in (axk, axl, axp):
        ax.plot([], [], color=CORRECTED, lw=lw, alpha=alpha, label=line_label)
        ax.plot([], [], "o", ms=7, color=CORRECTED, markeredgecolor="white", markeredgewidth=1.5,
                label="fold optimum")
        ax.legend(frameon=False, loc="lower right", fontsize=8, labelcolor=INK)
        # NSE and KGE surfaces don't necessarily share a range -- pad around the data rather
        # than assume NSE's usual [0.5, 1.03].
        lo, hi = min(df.score.min(), 0.5), max(df.score.max(), 1.0)
        ax.set_ylim(lo - 0.03, hi + 0.03)
        style(ax)
    axl.tick_params(labelleft=False)
    axp.tick_params(labelleft=False)
    fig.supylabel(f"median {metric_name} over cross-validation groups", color=INK, fontsize=9.5)

    basin_word = "reconstructed-group" if pilot else "training"
    fig.suptitle(f"Cross-validation for K, $\\lambda$, p  ({basin_word} basins, max {metric_name})",
                 fontsize=11.5, color=INK, x=0.012, ha="left", y=0.975)
    krange = f"K = {opt_k[0]}" if len(set(opt_k)) == 1 else f"K in [{min(opt_k)}, {max(opt_k)}]"
    lrange = (f"$\\lambda$ = {opt_l[0]:g}" if len(set(opt_l)) == 1
              else f"$\\lambda$ in [{min(opt_l):g}, {max(opt_l):g}]")
    prange = (f"p = {opt_p[0]:g}" if len(set(opt_p)) == 1
              else f"p in [{min(opt_p):g}, {max(opt_p):g}]")
    caption = (f"{len(folds)} fold{'s' if len(folds) > 1 else ''}, {n_basins} {basin_word} "
              f"basins each, scored leave-one-group-out.  Each line holds the other two "
              f"constants at that fold's optimum.  Optima: {krange}, {lrange}, {prange}.  "
              f"$\\lambda$ = 0 is the uncorrected model.  No held-out basin is used here.")
    if pilot:
        caption += ("  Single-(fold, group) pilot: this is the gauged-reconstruction cell "
                    "(OOS-time), not the PUB cell -- donors are ~5/6 of the fold's training "
                    "pool, not the full pool a real deployment or the PUB cell has.")
    wrapped = textwrap.wrap(caption, 108)
    fig.text(0.012, 0.015, "\n".join(wrapped), fontsize=8, color=MUTE, va="bottom")
    bottom = 0.07 + 0.028 * len(wrapped)
    fig.tight_layout(rect=(0.02, bottom, 1, 0.90))
    out_stem = "recon_gauged_hyperparameters" if pilot else "recon_ungauged_hyperparameters"
    out = out or f"{HERE}/{out_stem}_{metric}.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}   optima: K {opt_k}, lambda {opt_l}, p {opt_p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning", default=None)
    ap.add_argument("--metric", choices=list(SCORERS), default="nse",
                    help="which selection objective the tuning surface was built from")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pilot", action="store_true",
                    help="tuning CSV is the gauged-reconstruction pilot's, not the PUB one")
    a = ap.parse_args()
    stem = "recon_gauged_tuning" if a.pilot else "recon_ungauged_tuning"
    main(a.tuning or f"runs/{stem}_{a.metric}.csv", a.out, pilot=a.pilot, metric=a.metric)
