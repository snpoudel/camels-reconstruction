"""Error metrics before and after the neighbour correction, six panels.

    python figures/make_boxplot_figure.py   # -> figures/metrics_boxplots.png

NSE alone hides where a correction acts. These six say it: overall skill (KGE), magnitude of
error (RMSE), overall water balance (percent bias), and then bias split by how high the river
actually was that day -- top 10%, middle 30-90%, bottom 30%.

The flow bands are cut on the OBSERVED flow (Yilmaz et al. 2008), so "high-flow bias" is the
model's error on the days the river really was high, not on the days it predicted high.

For the bias panels zero is the target and the dashed line marks it, so a box that straddles
zero more tightly is better regardless of which side it sits on. For KGE higher is better;
for RMSE lower is better. Each panel says which way is good, because the direction changes
between them and a reader should not have to remember.

Colour: `figures/palette.py` slots 0 and 1, validated there (normal dE 33.6, CVD >= 24.5).
Boxes are also separated in position and labelled on the x-axis, so identity never rests on
colour alone.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from figures.palette import CORRECTED, GRID, INK, MUTE, RAW  # noqa: E402
from src.correct import K, LAMBDA, apply_all  # noqa: E402
from src.evaluate import metrics  # noqa: E402

PANELS = [
    ("kge", "KGE", "higher is better", False),
    ("rmse", "RMSE  (mm/day)", "lower is better", False),
    ("pbias", "overall bias  (%)", "zero is better", True),
    ("bias_high", "bias, top 10% of flows  (%)", "zero is better", True),
    ("bias_mid", "bias, middle 30-90%  (%)", "zero is better", True),
    ("bias_low", "bias, bottom 30%  (%)", "zero is better", True),
]


def collect(runs_dir="runs"):
    raw, cor, folds = [], [], []
    for f in apply_all(runs_dir=runs_dir):
        raw.append(metrics(f["raw"], f["obs"]))
        cor.append(metrics(f["corrected"], f["obs"]))
        folds.append(f["fold"])
    if not raw:
        raise SystemExit("no runs found under runs/")
    import pandas as pd
    return pd.concat(raw, ignore_index=True), pd.concat(cor, ignore_index=True), folds


def main(runs_dir="runs", out=None):
    raw, cor, folds = collect(runs_dir)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6))

    for ax, (col, title, direction, centred) in zip(axes.ravel(), PANELS):
        a = raw[col].to_numpy(); b = cor[col].to_numpy()
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        bp = ax.boxplot([a, b], positions=[0, 1], widths=0.52, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK, lw=1.6),
                        whiskerprops=dict(color=MUTE, lw=1.0),
                        capprops=dict(color=MUTE, lw=1.0))
        for patch, color in zip(bp["boxes"], (RAW, CORRECTED)):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("white")      # 2px surface gap between adjacent fills
            patch.set_linewidth(1.6)
        if centred:
            ax.axhline(0, color=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=1)
        # medians printed: the boxes are close on several panels and the reader should not
        # have to estimate the difference off the ink.
        for x, v in ((0, np.median(a)), (1, np.median(b))):
            ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 0), textcoords="offset points",
                        ha="center", va="center", fontsize=8.5, color=INK,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["LSTM", "+ correction"], fontsize=9, color=INK)
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=6)
        ax.text(1.0, 1.015, direction, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=MUTE)
        ax.grid(True, axis="y", color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTE); ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=MUTE, labelsize=8.5, length=3)
        # Scale to the drawn whiskers, not the data range: percent bias on the lowest flows
        # runs to five figures for a handful of ephemeral streams, and letting those set the
        # axis flattens every box in the panel into a line.
        wy = np.concatenate([w.get_ydata() for w in bp["whiskers"]])
        lo, hi = wy.min(), wy.max()
        if centred:
            lo, hi = min(lo, 0.0), max(hi, 0.0)
        pad = (hi - lo) * 0.16 + 1e-9
        ax.set_ylim(lo - pad, hi + pad)
        n_off = int((a < lo - pad).sum() + (a > hi + pad).sum()
                    + (b < lo - pad).sum() + (b > hi + pad).sum())
        if n_off:
            ax.text(0.02, 0.04, f"{n_off} basin-values beyond axis", transform=ax.transAxes,
                    fontsize=7.5, color=MUTE, ha="left", va="bottom")

    fold_txt = "fold " + ", ".join(str(f) for f in folds) if len(folds) < 7 else "all 7 folds"
    fig.text(0.008, 0.012,
             f"{len(raw)} held-out basins, {fold_txt}, WY2005-2014.  Boxes: quartiles and "
             f"median; whiskers 1.5 IQR, outliers omitted.  K={K}, $\\lambda$={LAMBDA}.",
             fontsize=8.5, color=MUTE)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = out or f"{HERE}/metrics_boxplots.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}")
    for col, title, _, _ in PANELS:
        print(f"  {title:<32} {raw[col].median():+8.3f} -> {cor[col].median():+8.3f}")


if __name__ == "__main__":
    main()
