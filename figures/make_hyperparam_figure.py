"""Cross-validation for the correction's three constants: K, lambda, and the IDW power p.

    python analysis/tune.py --out runs/recon_ungauged_tuning.csv     # first
    python figures/make_hyperparam_figure.py                         # -> figures/recon_ungauged_hyperparameters.png

The correction has three free numbers: K, how many gauges to consult; lambda, how much of
their error to believe; and p, how steeply weight falls off with distance (weight ~ 1/d^p).
All three are chosen by cross-validation **inside each fold's training basins** -- the
training basins keep their original fold labels, so each of six groups is scored using donors
drawn only from the other five. No held-out basin appears anywhere on this figure, and none is
read to choose anything; `analysis/leak_audit.py` check [6] tests that by destroying every
held-out observation and confirming the choice does not move.

One line per fold. Each is a profile through that fold's own CV optimum: each panel sweeps one
constant with the other two held at that fold's best, and the marker sits on the optimum
itself. Reading across the lines is the point -- the optima cluster in a narrow region, and the
curves are flat around them, so the exact triple a paper quotes matters much less than that it
lies in that region.

Folds are replicates of the same procedure, not different things being compared, so they share
one hue rather than getting a categorical palette; identity comes from the count in the
caption, and no line needs to be told apart from another. Colour: `figures/palette.py` slot 1
(#eb6834), with ink and mute for rules and text.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from figures.palette import CORRECTED, GRID, INK, MUTE  # noqa: E402


def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTE); ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelsize=9, length=3)


def main(tuning="runs/recon_ungauged_tuning.csv", out=None, pilot=False):
    df = pd.read_csv(tuning)
    if "set" in df.columns:
        df = df.query("set == 'cv'")
    folds = sorted(df.fold.unique())
    n_basins = int(df.n_basins.iloc[0])

    # Both K and p are sampled unevenly (see src/correct.py:K_GRID, P_GRID), so plotting either
    # by value packs the grid unevenly. Plotting by rank instead gives every step an
    # equal-width slot; the tick labels still show the real value.
    k_grid = sorted(df.k.unique())
    k_pos = {k: i for i, k in enumerate(k_grid)}
    p_grid = sorted(df.p.unique())
    p_pos = {pw: i for i, pw in enumerate(p_grid)}

    fig, (axk, axl, axp) = plt.subplots(1, 3, figsize=(15.5, 4.6))
    single = len(folds) == 1
    lw = 2.2 if single else 1.5
    alpha = 1.0 if single else 0.75

    opt_k, opt_l, opt_p = [], [], []
    for fold in folds:
        d = df.query("fold == @fold")
        best = d.loc[d.median_nse.idxmax()]
        opt_k.append(int(best.k)); opt_l.append(float(best.lam)); opt_p.append(float(best.p))

        sub = d.query("lam == @best.lam and p == @best.p").sort_values("k")
        axk.plot([k_pos[k] for k in sub.k], sub.median_nse, color=CORRECTED, lw=lw,
                 alpha=alpha, zorder=3)
        axk.plot([k_pos[best.k]], [best.median_nse], "o", ms=7, color=CORRECTED,
                 markeredgecolor="white", markeredgewidth=1.5, zorder=5)

        sub = d.query("k == @best.k and p == @best.p").sort_values("lam")
        axl.plot(sub.lam, sub.median_nse, color=CORRECTED, lw=lw, alpha=alpha, zorder=3)
        axl.plot([best.lam], [best.median_nse], "o", ms=7, color=CORRECTED,
                 markeredgecolor="white", markeredgewidth=1.5, zorder=5)

        sub = d.query("k == @best.k and lam == @best.lam").sort_values("p")
        axp.plot([p_pos[pw] for pw in sub.p], sub.median_nse, color=CORRECTED, lw=lw,
                 alpha=alpha, zorder=3)
        axp.plot([p_pos[best.p]], [best.median_nse], "o", ms=7, color=CORRECTED,
                 markeredgecolor="white", markeredgewidth=1.5, zorder=5)

    axk.set_xticks(range(len(k_grid)))
    axk.set_xticklabels(k_grid)
    axk.set_xlabel("K, number of neighbouring gauges", color=INK, fontsize=10)
    axk.set_title("(a)  number of gauges consulted", color=INK, fontsize=11, loc="left", pad=8)
    axl.set_xlabel("$\\lambda$, fraction of the neighbours' error applied", color=INK, fontsize=10)
    axl.set_title("(b)  how much of their error to believe", color=INK, fontsize=11,
                  loc="left", pad=8)
    axp.set_xticks(range(len(p_grid)))
    axp.set_xticklabels(p_grid)
    axp.set_xlabel("p, IDW power  (weight $\\propto 1/d^p$)", color=INK, fontsize=10)
    axp.set_title("(c)  how steeply weight favours the nearest gauges", color=INK, fontsize=11,
                  loc="left", pad=8)

    lines = ("one line per fold" if not single else "one fold")
    for ax, held in ((axk, "$\\lambda$ and p at each fold's optimum"),
                     (axl, "K and p at each fold's optimum"),
                     (axp, "K and $\\lambda$ at each fold's optimum")):
        ax.set_ylabel("median NSE over cross-validation groups", color=INK, fontsize=10)
        ax.plot([], [], color=CORRECTED, lw=lw, alpha=alpha,
                label=f"{lines}, {held}")
        ax.plot([], [], "o", ms=7, color=CORRECTED, markeredgecolor="white", markeredgewidth=1.5,
                label="that fold's cross-validation optimum")
        ax.legend(frameon=False, loc="lower right", fontsize=8.5, labelcolor=INK)
        ax.set_ylim(0.5, 1.03)
        style(ax)

    basin_word = "reconstructed-group" if pilot else "training"
    fig.suptitle(f"Cross-validation for K, $\\lambda$, and the IDW power p, within the "
                f"{basin_word} basins",
                 fontsize=12.5, color=INK, x=0.012, ha="left", y=0.975)
    krange = f"K = {opt_k[0]}" if len(set(opt_k)) == 1 else f"K in [{min(opt_k)}, {max(opt_k)}]"
    lrange = (f"$\\lambda$ = {opt_l[0]:g}" if len(set(opt_l)) == 1
              else f"$\\lambda$ in [{min(opt_l):g}, {max(opt_l):g}]")
    prange = (f"p = {opt_p[0]:g}" if len(set(opt_p)) == 1
              else f"p in [{min(opt_p):g}, {max(opt_p):g}]")
    caption = (f"{len(folds)} fold{'s' if len(folds) > 1 else ''}, {n_basins} {basin_word} "
              f"basins each, scored leave-one-group-out.  Optima: {krange}, {lrange}, {prange}.  "
              f"$\\lambda$ = 0 is the uncorrected model.  No held-out basin is used here.")
    if pilot:
        caption += ("  Single-(fold, group) pilot: this is the gauged-reconstruction cell "
                    "(OOS-time), not the PUB cell -- donors are ~5/6 of the fold's training "
                    "pool, not the full pool a real deployment or the PUB cell has.")
    fig.text(0.012, 0.015, caption, fontsize=8.5, color=MUTE)
    fig.tight_layout(rect=(0, 0.045, 1, 0.945))
    out = out or (f"{HERE}/recon_gauged_hyperparameters.png" if pilot
                 else f"{HERE}/recon_ungauged_hyperparameters.png")
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}   optima: K {opt_k}, lambda {opt_l}, p {opt_p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuning", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pilot", action="store_true",
                    help="tuning CSV is the gauged-reconstruction pilot's, not the PUB one")
    a = ap.parse_args()
    main(a.tuning or (f"runs/recon_gauged_tuning.csv" if a.pilot else "runs/recon_ungauged_tuning.csv"),
         a.out, pilot=a.pilot)
