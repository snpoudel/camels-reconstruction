"""Fig. 2: correlation of the LSTM's daily residuals between basin pairs against gauge distance.

    python figures/make_lstm_error_structure.py   # -> figures/lstm/fig02_error_structure.{png,svg}
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import read_error_structure  # noqa: E402
from palette import INK, MUTE, NEUTRAL, RAW, style  # noqa: E402
from palette import figure_variant  # noqa: E402

SUFFIX, OUT_DIR = figure_variant()


FS = 8


def main():
    pairs, fit = read_error_structure("lstm", "gauged")
    fig, ax = plt.subplots(figsize=(4.0, 2.85))
    ax.scatter(pairs.dist_km, pairs.r, s=2, color=NEUTRAL, alpha=0.08, linewidths=0,
               rasterized=True, zorder=2)
    ax.plot(fit.dist_km, fit.r_fit, color=RAW, lw=1.8, zorder=4,
            label="daily residuals (LOESS fitted)")
    ax.axhline(0, color=MUTE, lw=0.7, ls=(0, (2, 2)), zorder=3)
    ax.set_xlim(0, 1000)
    # Cut above the cloud's floor so the decay stays legible; only a few pairs fall below.
    ax.set_ylim(-0.4, 1)
    ax.set_xlabel("distance between gauges (km)", color=INK, fontsize=FS)
    ax.set_ylabel("residual correlation", color=INK, fontsize=FS)
    style(ax)
    ax.tick_params(labelsize=FS - 1)
    ax.legend(frameon=False, loc="upper right", fontsize=FS - 1, labelcolor=INK, handlelength=2.2)
    fig.tight_layout(pad=0.3)
    os.makedirs(f"{HERE}/{OUT_DIR}", exist_ok=True)
    stem = f"{HERE}/{OUT_DIR}/fig02_error_structure"
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=300, facecolor="white")
    print(f"-> {stem}.png/.svg")


if __name__ == "__main__":
    main()
