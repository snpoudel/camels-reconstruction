"""NSE distribution, the LSTM alone vs the LSTM plus the neighbour correction.

    python figures/make_cdf_figure.py    # -> figures/cdf_nse.png

Left: the empirical CDF across held-out (ungauged) basins. Each basin contributes one point --
the score of its seed-mean hydrograph -- so a curve lying to the right is winning at every
quantile it is right of, not merely on average.

Right: the same basins paired. The correction is applied to each basin's own hydrograph, so
raw and corrected are matched and the honest summary is the per-basin change, not the shift
between two independent distributions. The count above and below zero is printed on the panel
because a median gain says nothing about how often the method hurts.

Colour: `figures/palette.py` slots 0 and 1 (#2a78d6 blue, #eb6834 orange), validated there --
normal-vision dE 33.6, protan 24.5, deutan 31.6, all past the >=15 / >=8 thresholds. Identity
never rests on colour alone: both series are legend-identified and separated in position.
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
from src.config import PERIODS  # noqa: E402
from src.correct import apply_all  # noqa: E402
from src.evaluate import metrics  # noqa: E402


def test_period_label(runs_dir):
    """Human-readable test window, read from an actual run rather than assumed."""
    import glob, json
    paths = sorted(glob.glob(f"{runs_dir}/vanilla_fold*_seed*/config.json"))
    period = json.load(open(paths[0]))["period"] if paths else "complete"
    (_, _), (t0, t1) = PERIODS[period]
    return f"{t0[:4]}-{t1[:4]}"


def collect(runs_dir="runs"):
    got, note = apply_all(runs_dir=runs_dir)
    if not got:
        raise SystemExit(note)
    raw, cor, folds, settings = [], [], [], []
    for f in got:
        obs = f["obs"][f["held"]]
        raw.append(metrics(f["raw"], obs).nse.to_numpy())
        cor.append(metrics(f["corrected"], obs).nse.to_numpy())
        folds.append(f["fold"])
        settings.append((f["k"], f["lam"], f["p"]))
    return np.concatenate(raw), np.concatenate(cor), folds, settings


def style(ax):
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelsize=9, length=3)


def main(runs_dir="runs", out=None):
    raw, cor, folds, settings = collect(runs_dir)
    period_label = test_period_label(runs_dir)
    n = len(raw)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))

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

    # --- paired change ----------------------------------------------------------
    d = cor - raw
    d = d[np.isfinite(d)]
    order = np.argsort(d)
    up, down = (d > 0).sum(), (d < 0).sum()
    ax2.bar(np.arange(len(d)), d[order], width=1.0,
            color=np.where(d[order] > 0, CORRECTED, RAW), zorder=3)
    ax2.axhline(0, color=MUTE, lw=0.9, zorder=2)
    ax2.set_xlim(-1, len(d))
    # A percentile bound, not the max: a single-seed hydrograph can have one basin whose
    # ratio blows up (near-zero observed flow), and letting that basin set the axis crushes
    # the other 530 into an unreadable band near zero.
    lim = np.percentile(np.abs(d), 99) * 1.25
    n_off = int((np.abs(d) > lim).sum())
    ax2.set_ylim(-lim, lim)
    ax2.set_xlabel("basins, ranked by change", color=INK, fontsize=10)
    ax2.set_ylabel("change in NSE", color=INK, fontsize=10)
    ax2.set_title("Per-basin change, same basins paired", color=INK, fontsize=11,
                  loc="left", pad=8)
    ax2.text(0.97, 0.93, f"improved  {up}/{len(d)}  ({up / len(d):.0%})", transform=ax2.transAxes,
             ha="right", va="top", fontsize=9.5, color=INK)
    ax2.text(0.97, 0.05, f"worsened  {down}/{len(d)}", transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=9.5, color=MUTE)
    if n_off:
        ax2.text(0.02, 0.05, f"{n_off} basin(s) beyond axis", transform=ax2.transAxes,
                 ha="left", va="bottom", fontsize=8, color=MUTE)
    style(ax2)

    fold_txt = "fold " + ", ".join(str(f) for f in folds) if len(folds) < 7 else "all 7 folds"
    fig.text(0.008, 0.015,
             f"{n} held-out basins, {fold_txt}, test period {period_label}.  Correction: "
             + ("; ".join(f"fold {fd}: K={k}, $\\lambda$={lam:g}, p={pw:g}"
                          for fd, (k, lam, pw) in zip(folds, settings))
                if len(folds) < 4 else "K, $\\lambda$, and p selected per fold")
             + ", chosen inside the training basins only.",
             fontsize=8.5, color=MUTE)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out = out or f"{HERE}/cdf_nse.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}   n={n}, median {np.nanmedian(raw):.4f} -> {np.nanmedian(cor):.4f}")


if __name__ == "__main__":
    main()
