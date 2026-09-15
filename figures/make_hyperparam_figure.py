"""How much the two correction constants actually matter.

    python analysis/tune.py --out runs/tuning.csv     # first
    python figures/make_hyperparam_figure.py          # -> figures/hyperparameters.png

The correction has two free numbers: K, how many gauges to consult, and lambda, how much of
their error to believe. Both are selected (`src/correct.py:choose`) -- the search is seconds,
so there is no reason not to. This figure exists to show that the selection is cheap insurance
rather than a load-bearing step: a wide region of the grid scores nearly the same.

Each panel sweeps one constant. The bold line holds the other at the **selected** value; the
thin grey lines hold it at every other value on the grid, so the spread between them *is* the
sensitivity to the constant that is not on the x-axis. The filled marker is where selection
landed; the hollow one is the best cell on this surface, which is an oracle -- it is chosen by
looking at the answer and is not reachable by any honest rule. The gap between them is what
selection costs. The shaded band marks everything within 0.01 NSE of that oracle. The dashed
line is the uncorrected model, which is also exactly what the correction returns at lambda = 0,
and the left panel is on a log x-axis because K is sampled geometrically.

Colour: `figures/palette.py` slot 1 (#eb6834) for the shipped setting, ink and mute for
everything else -- there are no competing categories here, so no categorical pair is in play.
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
from src.correct import choose, load_all  # noqa: E402

TOL = 0.01  # "as good as the best", in NSE


def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTE); ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelsize=9, length=3)


def main(tuning="runs/tuning.csv", out=None, runs_dir="runs"):
    df = pd.read_csv(tuning)
    folds = sorted(df.fold.unique())
    # Ask the pipeline what it actually selected, rather than restating a constant here.
    picked, note = choose(load_all(runs_dir))
    K_SHIP, LAM_SHIP = picked[folds[0]] if len(set(picked.values())) == 1 else picked[folds[0]]
    varies = len(set(picked.values())) > 1
    # Pool folds by averaging each fold's median, so no fold dominates by basin count.
    g = df.groupby(["k", "lam"]).agg(nse=("median_nse", "mean"),
                                     base=("base_nse", "mean")).reset_index()
    base = g.base.mean()
    best = g.loc[g.nse.idxmax()]
    ship = g.query("k == @K_SHIP and lam == @LAM_SHIP").nse.iloc[0]
    sel_label = "selected" if "fixed" not in note else "fixed fallback"

    fig, (axk, axl) = plt.subplots(1, 2, figsize=(11, 4.4))

    # ---- panel A: sweep K ------------------------------------------------------
    for lam, sub in g.groupby("lam"):
        if lam == LAM_SHIP or lam == 0:
            continue
        axk.plot(sub.k, sub.nse, color=MUTE, lw=0.8, alpha=0.35, zorder=2)
    sub = g.query("lam == @LAM_SHIP").sort_values("k")
    axk.plot(sub.k, sub.nse, color=CORRECTED, lw=2.4, zorder=4,
             label=f"$\\lambda$ = {LAM_SHIP} ({sel_label})")
    axk.plot([K_SHIP], [ship], "o", ms=9, color=CORRECTED, markeredgecolor="white",
             markeredgewidth=1.8, zorder=6)
    axk.plot([best.k], [best.nse], "o", ms=9, mfc="none", color=INK, mew=1.4, zorder=6)
    axk.annotate(f"K = {K_SHIP}\n{ship:.3f}", (K_SHIP, ship), xytext=(10, -26),
                 textcoords="offset points", fontsize=9, color=INK)
    axk.set_xscale("log")
    axk.set_xticks(sorted(g.k.unique()))
    axk.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axk.set_xlabel("K, number of neighbouring gauges", color=INK, fontsize=10)
    axk.set_title("Sensitivity to K", color=INK, fontsize=11, loc="left", pad=8)

    # ---- panel B: sweep lambda -------------------------------------------------
    for k, sub2 in g.groupby("k"):
        if k == K_SHIP:
            continue
        axl.plot(sub2.lam, sub2.nse, color=MUTE, lw=0.8, alpha=0.35, zorder=2)
    sub2 = g.query("k == @K_SHIP").sort_values("lam")
    axl.plot(sub2.lam, sub2.nse, color=CORRECTED, lw=2.4, zorder=4,
             label=f"K = {K_SHIP} ({sel_label})")
    axl.plot([LAM_SHIP], [ship], "o", ms=9, color=CORRECTED, markeredgecolor="white",
             markeredgewidth=1.8, zorder=6)
    axl.plot([best.lam], [best.nse], "o", ms=9, mfc="none", color=INK, mew=1.4, zorder=6)
    axl.annotate(f"$\\lambda$ = {LAM_SHIP}\n{ship:.3f}", (LAM_SHIP, ship), xytext=(10, -26),
                 textcoords="offset points", fontsize=9, color=INK)
    axl.set_xlabel("$\\lambda$, fraction of the neighbours' error applied", color=INK, fontsize=10)
    axl.set_title("Sensitivity to $\\lambda$", color=INK, fontsize=11, loc="left", pad=8)

    for ax, thin in ((axk, "every other $\\lambda$"), (axl, "every other K")):
        ax.axhspan(best.nse - TOL, best.nse, color=CORRECTED, alpha=0.10, zorder=1)
        ax.axhline(base, color=INK, lw=1.0, ls=(0, (5, 3)), zorder=3)
        ax.set_ylabel("median NSE, held-out basins", color=INK, fontsize=10)
        ax.set_ylim(base - 0.055, best.nse + 0.018)
        ax.plot([], [], color=MUTE, lw=0.8, alpha=0.6, label=thin)
        ax.plot([], [], color=INK, lw=1.0, ls=(0, (5, 3)), label=f"uncorrected  {base:.3f}")
        ax.plot([], [], "o", ms=8, color=CORRECTED, markeredgecolor="white", markeredgewidth=1.6,
                label=f"{sel_label}  {ship:.3f}")
        ax.plot([], [], "o", ms=8, mfc="none", color=INK, mew=1.4,
                label=f"best cell (oracle)  {best.nse:.3f}")
        ax.plot([], [], color=CORRECTED, lw=6, alpha=0.18, label=f"within {TOL:g} of oracle")
        ax.legend(frameon=False, loc="lower right", fontsize=8.5, labelcolor=INK)
        style(ax)

    within = (g.nse > best.nse - TOL).mean()
    plateau = g.query("k >= 3 and k <= 20 and lam >= 0.3 and lam <= 0.7")
    fold_txt = "fold " + ", ".join(str(f) for f in folds) if len(folds) < 7 else "all 7 folds"
    fig.text(0.008, 0.015,
             f"{int(df.n_basins.iloc[0])} held-out basins, {fold_txt}.  "
             f"Grid: {g.k.nunique()} values of K x {g.lam.nunique()} of $\\lambda$ = {len(g)} cells.  "
             f"Over K in [3,20] and $\\lambda$ in [0.3,0.7] the range is "
             f"{plateau.nse.min():.3f}-{plateau.nse.max():.3f}, against a gain of "
             f"{best.nse - base:.3f} over uncorrected.\n"
             + note[0].upper() + note[1:] + ("." if not varies else "; per-fold values differ."),
             fontsize=8.5, color=MUTE, linespacing=1.6)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    out = out or f"{HERE}/hyperparameters.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}")
    print(f"   uncorrected {base:.4f} | {sel_label} K={K_SHIP},lam={LAM_SHIP} {ship:.4f} | "
          f"oracle best K={int(best.k)},lam={best.lam:.2f} {best.nse:.4f}")
    print(f"   plateau (K 3-20, lam 0.3-0.7): {plateau.nse.min():.4f}-{plateau.nse.max():.4f}")


if __name__ == "__main__":
    main()
