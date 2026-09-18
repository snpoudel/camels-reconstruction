"""Error metrics before and after the neighbour correction, six panels.

    python figures/make_boxplot_figure.py                # selected on NSE (default)
    python figures/make_boxplot_figure.py --metric kge    # selected on KGE instead

`--metric` picks which selection objective (`src/correct.py`'s `SCORERS`) chose K, lambda and
p; the panels themselves (including the KGE one) are unaffected by which objective picked the
settings, so this figure is how the two selections compare across every panel at once.

NSE alone hides where a correction acts. These six say it: overall skill (KGE), magnitude of
error (RMSE), overall water balance (percent bias), and then bias split by how high the river
actually was that day -- top 10%, middle 30-90%, bottom 30%.

The flow bands are cut on the OBSERVED flow (Yilmaz et al. 2008), so "high-flow bias" is the
model's error on the days the river really was high, not on the days it predicted high.

For the bias panels zero is the target and the dashed line marks it, so a box that straddles
zero more tightly is better regardless of which side it sits on.

Each box is drawn over its own basins as a faint jittered scatter, so the box summarises what
is actually visible underneath it rather than standing in for it. The axis is scaled to the
drawn whiskers, not the full data range, so a handful of extreme basins (percent bias on
near-zero-flow streams can run to five figures) do not flatten every other box in the panel;
the bracketed count under each box is how many of that box's basin-values fall outside the
shown range.

Colour: `figures/palette.py` slots 0 and 1, validated there (normal dE 33.6, CVD >= 24.5).
Boxes are also separated in position and labelled on the x-axis, so identity never rests on
colour alone.
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
from figures.palette import CORRECTED, GRID, INK, MUTE, RAW  # noqa: E402
from src.config import PERIODS  # noqa: E402
from src.correct import SCORERS  # noqa: E402

PANELS = [
    ("kge", "KGE", False),
    ("rmse", "RMSE  (mm/day)", False),
    ("pbias", "overall bias  (%)", True),
    ("bias_high", "bias, top 10% of flows  (%)", True),
    ("bias_mid", "bias, middle 30-90%  (%)", True),
    ("bias_low", "bias, bottom 30%  (%)", True),
]


def test_period_label(runs_dir):
    import glob, json
    paths = sorted(glob.glob(f"{runs_dir}/vanilla_fold*_seed*/config.json"))
    period = json.load(open(paths[0]))["period"] if paths else "complete"
    (_, _), (t0, t1) = PERIODS[period]
    return f"{t0[:4]}-{t1[:4]}"


def collect(metrics_csv):
    """Read the metrics `src/correct.py` (or `analysis/gauged_reconstruction.py`) already wrote,
    rather than repeating their K x lambda x p grid search here."""
    df = pd.read_csv(metrics_csv)
    raw = df.query("variant == 'vanilla'")
    cor = df.query("variant == 'vanilla + correction'")
    folds = sorted(df.fold.unique())
    settings = [tuple(raw.query("fold == @f")[["k", "lam", "p"]].iloc[0]) for f in folds]
    return raw, cor, folds, settings


def main(metrics_csv, out=None, runs_dir="runs", pilot=False, selected_on="nse"):
    raw, cor, folds, settings = collect(metrics_csv)
    period_label = test_period_label(runs_dir)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6))
    rng = np.random.default_rng(0)

    for ax, (col, title, centred) in zip(axes.ravel(), PANELS):
        a = raw[col].to_numpy(); b = cor[col].to_numpy()
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        bp = ax.boxplot([a, b], positions=[0, 1], widths=0.52, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK, lw=1.6),
                        whiskerprops=dict(color=MUTE, lw=1.0),
                        capprops=dict(color=MUTE, lw=1.0))
        for patch, color in zip(bp["boxes"], (RAW, CORRECTED)):
            patch.set_facecolor(color)
            patch.set_alpha(0.70)
            patch.set_edgecolor("white")      # 2px surface gap between adjacent fills
            patch.set_linewidth(1.6)
            patch.set_zorder(3)
        for element in ("medians", "whiskers", "caps"):
            for line in bp[element]:
                line.set_zorder(3)
        if centred:
            ax.axhline(0, color=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=1)
        # Faint jittered points behind each box: the box summarises what is actually there
        # rather than standing in for it. Fixed seed so the figure is reproducible.
        for x, vals, color in ((0, a, RAW), (1, b, CORRECTED)):
            jitter = rng.uniform(-0.16, 0.16, size=len(vals))
            ax.scatter(x + jitter, vals, s=5, color=color, alpha=0.30, linewidths=0, zorder=2)
        # medians printed: the boxes are close on several panels and the reader should not
        # have to estimate the difference off the ink.
        for x, v in ((0, np.median(a)), (1, np.median(b))):
            ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 0), textcoords="offset points",
                        ha="center", va="center", fontsize=8.5, color=INK, zorder=4,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=6)
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
        n_off_a = int((a < lo - pad).sum() + (a > hi + pad).sum())
        n_off_b = int((b < lo - pad).sum() + (b > hi + pad).sum())
        label = lambda base, n: f"{base}\n({n})" if n else base  # noqa: E731
        ax.set_xticks([0, 1])
        ax.set_xticklabels([label("LSTM", n_off_a), label("+ correction", n_off_b)],
                           fontsize=9, color=INK)

    fold_txt = "fold " + ", ".join(str(f) for f in folds) if len(folds) < 7 else "all 7 folds"
    basin_word = "reconstructed (gauged-but-discontinued)" if pilot else "held-out"
    settings_txt = ("; ".join(f"fold {fd}: K={k}, $\\lambda$={lam:g}, p={pw:g}"
                              for fd, (k, lam, pw) in zip(folds, settings))
                    if len(folds) < 4 else "K, $\\lambda$, and p selected per fold")
    caption = (f"{len(raw)} {basin_word} basins, {fold_txt}, test period {period_label}.  "
              f"Boxes: quartiles and median; whiskers 1.5 IQR, outliers omitted; dots are "
              f"individual basins, jittered.  (n) under a box is how many of its basin-values "
              f"fall beyond the shown axis.  {settings_txt}, chosen to maximise "
              f"{selected_on.upper()} inside the training basins only.")
    if pilot:
        caption += ("  Single-(fold, group) pilot: donors are ~5/6 of that fold's training "
                    "pool, not the full pool a real deployment or the PUB cell has.")
    fig.text(0.008, 0.012, textwrap.fill(caption, 145), fontsize=8, color=MUTE, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = out or f"{HERE}/recon_ungauged_metrics_boxplots.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}")
    for col, title, _ in PANELS:
        print(f"  {title:<32} {raw[col].median():+8.3f} -> {cor[col].median():+8.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--metric", choices=list(SCORERS), default="nse",
                    help="which selection objective chose K, lambda and p")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pilot", action="store_true",
                    help="metrics CSV is the gauged-reconstruction pilot's, not the PUB one")
    a = ap.parse_args()

    stem = "recon_gauged_metrics" if a.pilot else "recon_ungauged_metrics"
    out_stem = "recon_gauged_metrics_boxplots" if a.pilot else "recon_ungauged_metrics_boxplots"
    metrics_csv = a.metrics or f"{a.runs}/{stem}_{a.metric}.csv"
    out = a.out or f"{HERE}/{out_stem}_{a.metric}.png"
    main(metrics_csv, out, a.runs, pilot=a.pilot, selected_on=a.metric)
