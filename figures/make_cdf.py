"""How the skill of every basin shifts when the correction is applied.

    python figures/make_cdf.py --cell ungauged    # -> figures/ungauged_cdf.png
    python figures/make_cdf.py --cell gauged      # -> figures/gauged_cdf.png

Two empirical cumulative distributions, NSE on top and KGE below, across basins. Each basin
contributes one point -- the score of its seed-mean hydrograph -- so a curve lying to the right
is winning at every quantile it is right of, not merely on average.

The settings on the page were chosen to maximise KGE (`figures/evaluate.py`'s default), so the
NSE panel is a deliberate cross-metric view and the caption says so. The two panels disagree by
more than their axes alone: the correction improves KGE's correlation and bias terms but
slightly worsens its variability term, which NSE does not charge for, so the NSE gain runs
larger than the KGE gain.

Colour: palette slots 0 and 1, validated in palette.py (normal-vision dE 33.6, protan 24.5,
deutan 31.6, all past the >= 15 / >= 8 thresholds). Identity never rests on colour alone: both
series are named in the legend and separated in position.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import fold_settings, read_info, read_metrics  # noqa: E402
from palette import CELL_BLURB, CELL_TITLE, INK, LABEL, MUTE, SERIES, TITLE  # noqa: E402
from palette import caption, settings_note, style  # noqa: E402

PANELS = [("nse", "NSE, Nash-Sutcliffe efficiency"), ("kge", "KGE, Kling-Gupta efficiency")]


def main(cell):
    df = read_metrics(cell)
    info = read_info(cell)

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 7.6), sharex=True)
    for ax, (col, xlabel) in zip(axes, PANELS):
        for name, color in SERIES.items():
            v = np.sort(df.loc[df.variant == name, col].dropna().to_numpy())
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, lw=2, zorder=3,
                    label=f"{name}   median {np.median(v):.2f}")
            ax.plot([np.median(v)], [0.5], "o", ms=7, color=color, markeredgecolor="white",
                    markeredgewidth=1.6, zorder=4)
        ax.axhline(0.5, color=MUTE, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel, color=INK, fontsize=LABEL)
        ax.set_ylabel("fraction of basins at or below", color=INK, fontsize=LABEL)
        ax.legend(frameon=False, loc="upper left", fontsize=LABEL - 1, labelcolor=INK)
        style(ax)

    axes[0].set_title(f"{CELL_TITLE[cell]}: skill before and after the correction",
                      color=INK, fontsize=TITLE, loc="left", pad=8)
    bottom = caption(fig, f"{info['n_basins']} basins, {CELL_BLURB[cell]}, "
                          f"{info['n_folds']} folds, {info['n_seeds']} seeds, test period "
                          f"{info['test_period']}.  {settings_note(fold_settings(df))}, chosen by "
                          f"cross-validation to maximise KGE, never reading the basin being "
                          f"corrected.  Top: NSE.  Bottom: KGE.", 96)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    out = f"{HERE}/{cell}_cdf.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=list(CELL_TITLE), default="ungauged")
    main(ap.parse_args().cell)
