"""Uncorrected NSE of each model at its training basins.

    python figures/make_nse_map.py    # -> gauged_nse_map.png

One map per model: regional LSTM, regional differentiable HBV (425 basins), local HBV. One
colour scale for all three, darker = better. The scale starts at 0 (below it, worse than the
mean flow) and its bins are finer above 0.5, where most basins fall.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)
from evaluate import read_metrics  # noqa: E402
from palette import INK, LABEL, MUTE, PANEL, SEQUENTIAL, TITLE, draw_us_outline  # noqa: E402
from palette import panel_letter, series_pair, style  # noqa: E402
from shared import BASINS_CSV  # noqa: E402

CELL = "gauged"

PANELS = [("lstm", "Regional LSTM"),
          ("hybrid", "Regional differentiable HBV"),
          ("hbv", "Local HBV")]


def main():
    loc = pd.read_csv(BASINS_CSV, dtype={"gauge_id": str}).set_index("gauge_id")
    bounds = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cmap = SEQUENTIAL.resampled(len(bounds))  # one extra colour for the below-0 bin
    norm = mcolors.BoundaryNorm(bounds, cmap.N, extend="min")

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(6.5, 9.2), sharex=True)
    for i, (ax, (model, title)) in enumerate(zip(axes, PANELS)):
        df = read_metrics(CELL, model)
        nse = df[df.variant == list(series_pair(model))[0]].set_index("basin").nse.dropna()
        lat = loc.loc[nse.index, "gauge_lat"].to_numpy()
        lon = loc.loc[nse.index, "gauge_lon"].to_numpy()
        v = nse.to_numpy()

        draw_us_outline(ax)
        order = np.argsort(-v)  # poorest basins on top, so they aren't buried
        sc = ax.scatter(lon[order], lat[order], c=v[order], cmap=cmap, norm=norm, s=13,
                        edgecolor=MUTE, linewidth=0.35, zorder=3)

        ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
        ax.set_ylabel("latitude (°N)", color=INK, fontsize=LABEL)
        ax.set_title(f"{panel_letter(i)} {title} ({len(v)} basins)", color=INK,
                     fontsize=PANEL, loc="left", pad=6)
        style(ax, grid=False)
        print(f"  {model:<7} {len(v)} basins, median NSE {np.median(v):.3f}, "
              f"{int((v < 0).sum())} below 0")
    axes[-1].set_xlabel("longitude (°E)", color=INK, fontsize=LABEL)

    fig.suptitle("NSE of each model at its training basins, test period",
                 fontsize=TITLE, color=INK, x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0.08, 1, 0.985))
    cax = fig.add_axes((0.2, 0.05, 0.6, 0.013))
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=bounds,
                      spacing="uniform")
    cb.set_label("NSE, Nash-Sutcliffe efficiency", color=INK, fontsize=LABEL - 1.5)
    cb.ax.tick_params(colors=MUTE, labelsize=LABEL - 2.5, length=2)
    cb.outline.set_visible(False)

    out = f"{HERE}/{CELL}_nse_map.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
