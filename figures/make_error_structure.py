"""Correlation of daily residuals between basin pairs against gauge distance, for three models.

    python figures/make_error_structure.py    # -> gauged_error_structure.png

One panel per model (regional LSTM, regional differentiable HBV, local HBV), all from the gauged
cell. Each dot is one basin pair; the line is a LOESS fit. Built from residuals and gauge
coordinates alone, so it shows a property of each model's errors, not of the correction.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import read_error_structure  # noqa: E402
from palette import INK, LABEL, MUTE, NEUTRAL, RAW, TITLE, style  # noqa: E402

CELL = "gauged"

PANELS = [
    ("lstm", "Regional LSTM (Kratzert et al., 2019b)"),
    ("hybrid", "Regional differentiable HBV (Poudel & Steinschneider, 2026)"),
    ("hbv", "Local HBV (Seibert et al., 2018)"),
]

# Cut above the cloud's true floor so the decay stays legible; only a handful of pairs fall below.
Y_FLOOR = -0.4


def main():
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(6.5, 8), sharex=True)

    for ax, (model, title) in zip(axes, PANELS):
        pairs, fit = read_error_structure(model, CELL)

        ax.scatter(pairs.dist_km, pairs.r, s=5, color=NEUTRAL, alpha=0.10, linewidths=0,
                   zorder=2, rasterized=True, label="one basin pair")
        ax.plot(fit.dist_km, fit.r_fit, color=RAW, lw=2.4, zorder=4, label="LOESS fit")
        ax.axhline(0, color=MUTE, lw=0.8, ls=(0, (4, 3)), zorder=3)

        ax.set_xlim(0, fit.dist_km.max())
        ax.set_ylim(Y_FLOOR, 1)
        ax.set_ylabel("correlation of\ndaily residuals", color=INK, fontsize=LABEL)
        ax.set_title(title, color=INK, fontsize=TITLE, loc="left", pad=8)
        style(ax)

    axes[-1].set_xlabel("distance between gauge pair (km)", color=INK, fontsize=LABEL)
    leg = axes[0].legend(frameon=False, loc="upper right", fontsize=LABEL - 1, labelcolor=INK,
                         handletextpad=0.6, borderpad=0.2)
    leg.legend_handles[0].set_alpha(0.8)  # the cloud's own alpha is unreadable at legend size
    leg.legend_handles[0].set_sizes([26])

    fig.tight_layout()
    out = f"{HERE}/{CELL}_error_structure.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
