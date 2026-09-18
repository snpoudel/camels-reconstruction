"""NSE distribution, the LSTM alone vs the LSTM plus the neighbour correction.

    python figures/make_cdf_figure.py    # -> figures/recon_ungauged_cdf_nse.png

The empirical CDF across held-out (ungauged) basins. Each basin contributes one point -- the
score of its seed-mean hydrograph -- so a curve lying to the right is winning at every quantile
it is right of, not merely on average.

Colour: `figures/palette.py` slots 0 and 1 (#2a78d6 blue, #eb6834 orange), validated there --
normal-vision dE 33.6, protan 24.5, deutan 31.6, all past the >=15 / >=8 thresholds. Identity
never rests on colour alone: both series are legend-identified and separated in position.
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
from figures.palette import CORRECTED, GRID, INK, MUTE, RAW  # noqa: E402
from src.config import PERIODS  # noqa: E402


def test_period_label(runs_dir):
    """Human-readable test window, read from an actual run rather than assumed."""
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
    return raw.nse.to_numpy(), cor.nse.to_numpy(), folds, settings


def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelsize=9, length=3)


def main(metrics_csv, out=None, runs_dir="runs", pilot=False):
    raw, cor, folds, settings = collect(metrics_csv)
    period_label = test_period_label(runs_dir)
    n = len(raw)
    fig, ax1 = plt.subplots(figsize=(6.2, 4.3))

    # --- CDF -------------------------------------------------------------------
    for vals, color, label in ((raw, RAW, "LSTM"), (cor, CORRECTED, "LSTM + correction")):
        v = np.sort(vals[np.isfinite(vals)])
        ax1.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, lw=2,
                 label=f"{label}   median {np.median(v):.3f}", zorder=3)
    for vals, color in ((raw, RAW), (cor, CORRECTED)):
        ax1.plot([np.nanmedian(vals)], [0.5], "o", ms=7, color=color,
                 markeredgecolor="white", markeredgewidth=1.6, zorder=4)
    ax1.axhline(0.5, color=MUTE, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax1.set_xlim(-0.2, 1.0)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("NSE", color=INK, fontsize=10)
    ax1.set_ylabel("fraction of basins below", color=INK, fontsize=10)
    ax1.set_title("Distribution across held-out basins", color=INK, fontsize=11,
                  loc="left", pad=8)
    ax1.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
    style(ax1)

    fold_txt = "fold " + ", ".join(str(f) for f in folds) if len(folds) < 7 else "all 7 folds"
    basin_word = "reconstructed (gauged-but-discontinued)" if pilot else "held-out"
    settings_txt = ("; ".join(f"fold {fd}: K={k}, $\\lambda$={lam:g}, p={pw:g}"
                              for fd, (k, lam, pw) in zip(folds, settings))
                    if len(folds) < 4 else "K, $\\lambda$, and p selected per fold")
    caption = (f"{n} {basin_word} basins, {fold_txt}, test period {period_label}.  Correction: "
              f"{settings_txt}, chosen inside the training basins only.")
    if pilot:
        caption += ("  Single-(fold, group) pilot: donors are ~5/6 of that fold's training "
                    "pool, not the full pool a real deployment or the PUB cell has.")
    fig.text(0.008, 0.015, caption, fontsize=8.5, color=MUTE)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out = out or f"{HERE}/recon_ungauged_cdf_nse.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}   n={n}, median {np.nanmedian(raw):.4f} -> {np.nanmedian(cor):.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pilot", action="store_true",
                    help="metrics CSV is the gauged-reconstruction pilot's "
                    "(runs/recon_gauged_metrics.csv), not the PUB one")
    a = ap.parse_args()

    metrics_csv = a.metrics or (f"{a.runs}/recon_gauged_metrics.csv" if a.pilot
                                else f"{a.runs}/recon_ungauged_metrics.csv")
    out = a.out or (f"{HERE}/recon_gauged_cdf_nse.png" if a.pilot
                    else f"{HERE}/recon_ungauged_cdf_nse.png")
    main(metrics_csv, out, a.runs, pilot=a.pilot)
