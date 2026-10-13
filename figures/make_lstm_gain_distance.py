"""Fig. S2: NSE gain from the IDW correction against distance to the nearest donor gauge.

    python figures/make_lstm_gain_distance.py   # -> lstm/figS2_gain_distance.{png,svg}

One panel per cell side by side, ungauged on the left, each dot a basin. Ungauged donors are the training basins
of the target's fold; discontinued-gauge donors are all other basins.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)
from make_lstm_boxplots import series  # noqa: E402
from palette import INK, MUTE, NEUTRAL, panel_letter, style  # noqa: E402
from palette import figure_variant  # noqa: E402
from shared import basin_ids, distance_matrix  # noqa: E402

SUFFIX, OUT_DIR = figure_variant()

FS = 8
CELLS = {"ungauged": "Ungauged", "gauged": "Discontinued"}
YLIM = (-0.1, 0.6)
XMAX = 330


def main():
    dist = distance_matrix()
    np.fill_diagonal(dist, np.inf)
    ids = basin_ids()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    for letter, (ax, (cell, title)) in enumerate(zip(axes, CELLS.items())):
        data = series(cell)
        lstm, idw = data["LSTM"].loc[ids], data["LSTM + IDW"].loc[ids]
        gain = (idw.nse - lstm.nse).to_numpy()
        fold = lstm.fold.to_numpy()
        allowed = fold[None, :] != fold[:, None] if cell == "ungauged" else np.ones_like(dist, bool)
        near = np.where(allowed, dist, np.inf).min(1)
        ax.scatter(near, np.clip(gain, *YLIM), s=5, color=NEUTRAL, alpha=0.5, linewidths=0,
                   zorder=2, clip_on=False)
        ax.axhline(0, color=MUTE, lw=0.7, ls=(0, (2, 2)), zorder=3)
        ax.set_xlim(0, XMAX)
        ax.set_ylim(*YLIM)
        ax.set_title(f"{panel_letter(letter)} {title}", color=INK, fontsize=FS + 0.5, loc="left",
                     pad=3)
        style(ax)
        ax.tick_params(labelsize=FS - 1)
        ax.set_xlabel("distance to nearest donor gauge (km)", color=INK, fontsize=FS)
        print(f"  {cell:<9} {(gain < YLIM[0]).sum()} basins drawn at the lower limit")
    axes[0].set_ylabel("change in NSE after IDW", color=INK, fontsize=FS)

    fig.tight_layout(pad=0.3, w_pad=1.5)
    os.makedirs(f"{HERE}/{OUT_DIR}", exist_ok=True)
    stem = f"{HERE}/{OUT_DIR}/figS2_gain_distance"
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=300, facecolor="white")
    print(f"-> {stem}.png/.svg")


if __name__ == "__main__":
    main()
