"""Fig. S3: NSE gain from the IDW correction against distance to the nearest donor gauge.

    python figures/make_lstm_gain_distance.py   # -> lstm/figS3_gain_distance.{png,pdf}

Drawn like Fig. 2: one panel per cell, ungauged on top, each dot a basin, with the LOESS fit
through them. Ungauged donors are the training basins of the target's fold; discontinued-gauge
donors are all other basins.
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
from evaluate import _loess  # noqa: E402
from make_lstm_boxplots import series  # noqa: E402
from palette import CORRECTED, INK, MUTE, NEUTRAL, panel_letter, style  # noqa: E402
from palette import figure_variant  # noqa: E402
from shared import basin_ids, distance_matrix  # noqa: E402

SUFFIX, OUT_DIR = figure_variant()

FS = 8
CELLS = {"ungauged": "Ungauged", "gauged": "Discontinued"}
YLIM = (-0.3, 0.6)
XMAX = 330
# Fewer than a dozen basins lie beyond 150 km, too few for the fit to mean anything there.
FIT_MAX_KM = 150


def main():
    dist = distance_matrix()
    np.fill_diagonal(dist, np.inf)
    ids = basin_ids()
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.4), sharex=True)
    for letter, (ax, (cell, title)) in enumerate(zip(axes, CELLS.items())):
        data = series(cell)
        lstm, idw = data["LSTM"].loc[ids], data["LSTM + IDW"].loc[ids]
        gain = (idw.nse - lstm.nse).to_numpy()
        fold = lstm.fold.to_numpy()
        allowed = fold[None, :] != fold[:, None] if cell == "ungauged" else np.ones_like(dist, bool)
        near = np.where(allowed, dist, np.inf).min(1)

        grid = np.linspace(5, FIT_MAX_KM, 120)
        # Narrower than Fig. 2's 60 km: with 531 points rather than 39k pairs, most within 60 km,
        # a 60 km window would flatten the drop over the first 50 km.
        fit = _loess(near, gain, grid, bandwidth=30)
        ax.scatter(near, np.clip(gain, *YLIM), s=5, color=NEUTRAL, alpha=0.5, linewidths=0,
                   zorder=2)
        ax.plot(grid, fit, color=CORRECTED, lw=1.8, zorder=4, label="NSE change (LOESS fitted)")
        ax.axhline(0, color=MUTE, lw=0.7, ls=(0, (2, 2)), zorder=3)
        ax.set_xlim(0, XMAX)
        ax.set_ylim(*YLIM)
        ax.set_ylabel("change in NSE after IDW", color=INK, fontsize=FS)
        ax.set_title(f"{panel_letter(letter)} {title}", color=INK, fontsize=FS + 0.5, loc="left",
                     pad=3)
        style(ax)
        ax.tick_params(labelsize=FS - 1)
        if letter == 0:
            ax.legend(frameon=False, loc="upper right", fontsize=FS - 1, labelcolor=INK,
                      handlelength=2.2)
        print(f"  {cell:<9} fitted change at 10/25/50/100 km "
              + " ".join(f"{np.interp(at, grid, fit):+.3f}" for at in (10, 25, 50, 100)))
    axes[-1].set_xlabel("distance to nearest donor gauge (km)", color=INK, fontsize=FS)

    fig.tight_layout(pad=0.3, h_pad=0.8)
    os.makedirs(f"{HERE}/{OUT_DIR}", exist_ok=True)
    stem = f"{HERE}/{OUT_DIR}/figS3_gain_distance"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, facecolor="white")
    print(f"-> {stem}.png/.pdf")


if __name__ == "__main__":
    main()
