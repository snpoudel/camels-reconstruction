"""Cross-validation for the correction's two constants, K and lambda.

    python analysis/tune.py --out runs/tuning.csv     # first
    python figures/make_hyperparam_figure.py          # -> figures/hyperparameters.png

The correction has two free numbers: K, how many gauges to consult, and lambda, how much of
their error to believe. Both are chosen by cross-validation **inside each fold's training
basins** -- the training basins keep their original fold labels, so each of six groups is
scored using donors drawn only from the other five. No held-out basin appears anywhere on this
figure, and none is read to choose anything; `analysis/leak_audit.py` check [6] tests that by
destroying every held-out observation and confirming the choice does not move.

One line per fold. Each is a profile through that fold's own CV optimum: the left panel sweeps
K with lambda held at that fold's best, the right sweeps lambda with K held at that fold's
best, and the marker sits on the optimum itself. Reading across the lines is the point -- the
optima cluster in a narrow region, and the curves are flat around them, so the exact pair a
paper quotes matters much less than that it lies in that region.

Folds are replicates of the same procedure, not different things being compared, so they share
one hue rather than getting a categorical palette; identity comes from the count in the
caption, and no line needs to be told apart from another. Colour: `figures/palette.py` slot 1
(#eb6834), with ink and mute for rules and text.
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
from figures.palette import CORRECTED, GRID, INK, MUTE  # noqa: E402


def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTE); ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelsize=9, length=3)


def main(tuning="runs/tuning.csv", out=None):
    df = pd.read_csv(tuning)
    if "set" in df.columns:
        df = df.query("set == 'cv'")
    folds = sorted(df.fold.unique())
    n_basins = int(df.n_basins.iloc[0])

    fig, (axk, axl) = plt.subplots(1, 2, figsize=(11, 4.6))
    single = len(folds) == 1
    lw = 2.2 if single else 1.5
    alpha = 1.0 if single else 0.75

    opt_k, opt_l = [], []
    for fold in folds:
        d = df.query("fold == @fold")
        best = d.loc[d.median_nse.idxmax()]
        opt_k.append(int(best.k)); opt_l.append(float(best.lam))

        sub = d.query("lam == @best.lam").sort_values("k")
        axk.plot(sub.k, sub.median_nse, color=CORRECTED, lw=lw, alpha=alpha, zorder=3)
        axk.plot([best.k], [best.median_nse], "o", ms=7, color=CORRECTED,
                 markeredgecolor="white", markeredgewidth=1.5, zorder=5)

        sub = d.query("k == @best.k").sort_values("lam")
        axl.plot(sub.lam, sub.median_nse, color=CORRECTED, lw=lw, alpha=alpha, zorder=3)
        axl.plot([best.lam], [best.median_nse], "o", ms=7, color=CORRECTED,
                 markeredgecolor="white", markeredgewidth=1.5, zorder=5)

    axk.set_xscale("log")
    axk.set_xticks(sorted(df.k.unique()))
    axk.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axk.set_xlabel("K, number of neighbouring gauges", color=INK, fontsize=10)
    axk.set_title("(a)  number of gauges consulted", color=INK, fontsize=11, loc="left", pad=8)
    axl.set_xlabel("$\\lambda$, fraction of the neighbours' error applied", color=INK, fontsize=10)
    axl.set_title("(b)  how much of their error to believe", color=INK, fontsize=11,
                  loc="left", pad=8)

    lines = ("one line per fold" if not single else "one fold")
    for ax, held in ((axk, f"$\\lambda$ at each fold's optimum"),
                     (axl, "K at each fold's optimum")):
        ax.set_ylabel("median NSE over cross-validation groups", color=INK, fontsize=10)
        ax.plot([], [], color=CORRECTED, lw=lw, alpha=alpha,
                label=f"{lines}, {held}")
        ax.plot([], [], "o", ms=7, color=CORRECTED, markeredgecolor="white", markeredgewidth=1.5,
                label="that fold's cross-validation optimum")
        ax.legend(frameon=False, loc="lower right", fontsize=8.5, labelcolor=INK)
        style(ax)

    fig.suptitle("Cross-validation for K and $\\lambda$, within the training basins",
                 fontsize=12.5, color=INK, x=0.012, ha="left", y=0.975)
    krange = f"K = {opt_k[0]}" if len(set(opt_k)) == 1 else f"K in [{min(opt_k)}, {max(opt_k)}]"
    lrange = (f"$\\lambda$ = {opt_l[0]:g}" if len(set(opt_l)) == 1
              else f"$\\lambda$ in [{min(opt_l):g}, {max(opt_l):g}]")
    fig.text(0.012, 0.015,
             f"{len(folds)} fold{'s' if len(folds) > 1 else ''}, {n_basins} training basins each, "
             f"scored leave-one-group-out.  Optima: {krange}, {lrange}.  "
             f"$\\lambda$ = 0 is the uncorrected model.  No held-out basin is used here.",
             fontsize=8.5, color=MUTE)
    fig.tight_layout(rect=(0, 0.045, 1, 0.945))
    out = out or f"{HERE}/hyperparameters.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}   optima: K {opt_k}, lambda {opt_l}")


if __name__ == "__main__":
    main()
