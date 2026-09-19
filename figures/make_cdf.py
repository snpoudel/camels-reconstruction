"""How the skill of every basin shifts when the correction is applied.

    python figures/make_cdf.py --cell ungauged    # -> figures/ungauged_cdf.png
    python figures/make_cdf.py --cell gauged      # -> figures/gauged_cdf.png

The empirical cumulative distribution of KGE across basins. Each basin contributes one point --
the score of its seed-mean hydrograph -- so a curve lying to the right is winning at every
quantile it is right of, not merely on average.

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


def main(cell):
    df = read_metrics(cell)
    info = read_info(cell)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for name, color in SERIES.items():
        v = np.sort(df.loc[df.variant == name, "kge"].dropna().to_numpy())
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, lw=2, zorder=3,
                label=f"{name}   median {np.median(v):.3f}")
        ax.plot([np.median(v)], [0.5], "o", ms=7, color=color, markeredgecolor="white",
                markeredgewidth=1.6, zorder=4)
    ax.axhline(0.5, color=MUTE, lw=0.8, ls=(0, (4, 3)), zorder=1)

    ax.set_xlim(-0.2, 1.0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("KGE, Kling-Gupta efficiency", color=INK, fontsize=LABEL)
    ax.set_ylabel("fraction of basins at or below", color=INK, fontsize=LABEL)
    ax.set_title(f"{CELL_TITLE[cell]}: skill before and after the correction",
                 color=INK, fontsize=TITLE, loc="left", pad=8)
    ax.legend(frameon=False, loc="upper left", fontsize=LABEL - 1, labelcolor=INK)
    style(ax)

    bottom = caption(fig, f"{info['n_basins']} basins, {CELL_BLURB[cell]}, "
                          f"{info['n_folds']} folds, {info['n_seeds']} seeds, test period "
                          f"{info['test_period']}.  {settings_note(fold_settings(df))}, chosen by "
                          f"cross-validation that never reads the basin being corrected.", 96)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    out = f"{HERE}/{cell}_cdf.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=list(CELL_TITLE), default="ungauged")
    main(ap.parse_args().cell)
