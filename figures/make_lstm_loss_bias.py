"""Percent bias before and after the IDW correction, in the basins where NSE falls.

    python figures/make_lstm_loss_bias.py   # -> lstm/loss_bias.{png,pdf}

Not in the paper, so not numbered; see experiments_log.md, 2026-10-09. One row per cell, one
column per flow band, styled like Figs. S1-S2: boxes over the basins whose NSE is lower after
IDW than before, each y axis scaled to the whiskers, the red count under a box being how many
of its basins fall outside the shown range.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_lstm_boxplots import OFF_COUNT_COLOR, series  # noqa: E402
from palette import CORRECTED, INK, LABEL, MUTE, PANEL, RAW, panel_letter, style  # noqa: E402
from palette import figure_variant  # noqa: E402

SUFFIX, OUT_DIR = figure_variant()


CELLS = {"ungauged": "ungauged", "gauged": "discontinued"}
BANDS = [("pbias", "bias"), ("bias_high", "high-flow bias"), ("bias_mid", "medium-flow bias"),
         ("bias_low", "low-flow bias")]
SERIES = {"LSTM": RAW, "LSTM + IDW": CORRECTED}


def main():
    rng = np.random.default_rng(0)  # reproducible jitter
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 4.4))
    for row, (cell, cell_title) in enumerate(CELLS.items()):
        data = series(cell)
        lose = data["LSTM + IDW"].nse < data["LSTM"].nse
        for col, (band, title) in enumerate(BANDS):
            ax = axes[row, col]
            vals = [data[name][band][lose].dropna().to_numpy() for name in SERIES]
            bp = ax.boxplot(vals, positions=range(2), widths=0.52, patch_artist=True,
                            showfliers=False, medianprops=dict(color=INK, lw=1.6),
                            whiskerprops=dict(color=MUTE, lw=1.0), capprops=dict(color=MUTE, lw=1.0))
            for patch, color in zip(bp["boxes"], SERIES.values()):
                patch.set(facecolor=color, alpha=0.70, edgecolor="white", linewidth=1.6, zorder=3)
            for element in ("medians", "whiskers", "caps"):
                for line in bp[element]:
                    line.set_zorder(3)
            ax.axhline(0, color=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=1)
            for x, (v, color) in enumerate(zip(vals, SERIES.values())):
                ax.scatter(x + rng.uniform(-0.16, 0.16, len(v)), v, s=3, color=color, alpha=0.35,
                           linewidths=0, zorder=2)

            wy = np.concatenate([w.get_ydata() for w in bp["whiskers"]])
            lo, hi = min(wy.min(), 0.0), max(wy.max(), 0.0)
            pad = (hi - lo) * 0.16
            ax.set_ylim(lo - pad, hi + pad)
            for x, v in enumerate(vals):
                ax.annotate(f"{np.median(v):.1f}", (x, np.median(v)), ha="center", va="center",
                            fontsize=LABEL - 3, color=INK, zorder=4,
                            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
                n_off = int((v < lo - pad).sum() + (v > hi + pad).sum())
                if n_off:
                    ax.annotate(f"({n_off})", (x, bp["caps"][2 * x].get_ydata()[0]), xytext=(0, -3),
                                textcoords="offset points", ha="center", va="top",
                                fontsize=LABEL - 3.5, color=OFF_COUNT_COLOR, zorder=4)
            ax.set_title(f"{panel_letter(4 * row + col)} {title}", color=INK, fontsize=PANEL - 2,
                         loc="left", pad=4)
            style(ax)
            ax.set_xticks(range(2))
            ax.set_xticklabels(["LSTM", "+ IDW"], fontsize=LABEL - 2, color=INK)
            ax.tick_params(axis="y", labelsize=LABEL - 2.5)
            if col == 0:
                ax.set_ylabel(f"{cell_title} ({lose.sum()} basins)\npercent bias (%)", color=INK,
                              fontsize=LABEL - 2)
            print(f"  {cell:<9} {title:<17} n={lose.sum()}  LSTM {np.median(vals[0]):+6.1f}  "
                  f"IDW {np.median(vals[1]):+6.1f}")

    fig.tight_layout(pad=0.4, h_pad=1.0, w_pad=0.8)
    os.makedirs(f"{HERE}/{OUT_DIR}", exist_ok=True)
    stem = f"{HERE}/{OUT_DIR}/loss_bias"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, facecolor="white")
    print(f"-> {stem}.png/.pdf")


if __name__ == "__main__":
    main()
