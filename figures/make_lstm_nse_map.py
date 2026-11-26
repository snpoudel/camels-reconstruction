"""Fig. 4: maps of the LSTM's NSE and its change after the IDW correction, both settings.

    python figures/make_lstm_nse_map.py   # -> figures/lstm/fig04_nse_maps.{png,svg}
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)
from make_lstm_boxplots import series  # noqa: E402
from palette import BLUE, INK, LABEL, MUTE, ORANGE, PANEL, SEQUENTIAL_ALT  # noqa: E402
from palette import figure_variant  # noqa: E402
from palette import draw_us_outline, panel_letter  # noqa: E402
from shared import BASINS_CSV  # noqa: E402

SUFFIX, OUT_DIR = figure_variant()

CELLS = ["ungauged", "gauged"]
ROW_LABEL = {"ungauged": "Ungauged", "gauged": "Partially gauged"}
CORRECTION = "LSTM + IDW"
COL_TITLE = ["LSTM NSE", "ΔNSE after IDW"]
WIDTH, HEIGHT = 5.6, 4.0


def colorbar(fig, sc, rect, label, **kw):
    cb = fig.colorbar(sc, cax=fig.add_axes(rect), orientation="horizontal", **kw)
    cb.set_label(label, color=INK, fontsize=LABEL - 1.5)
    cb.ax.tick_params(colors=MUTE, labelsize=LABEL - 2.5, length=2)
    cb.outline.set_visible(False)


def main():
    loc = pd.read_csv(BASINS_CSV, dtype={"gauge_id": str}).set_index("gauge_id")
    data = {cell: series(cell) for cell in CELLS}
    nse = {cell: data[cell]["LSTM"].nse.dropna() for cell in CELLS}
    gains = {cell: (data[cell][CORRECTION].nse - nse[cell]).dropna() for cell in CELLS}
    change_bounds = [-0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20]
    # Outer bins go darker than DIVERGING's poles, which barely change lightness at the ends.
    change_cmap = mcolors.ListedColormap([
        "#003d63", BLUE, "#5a9bc6", "#a7bdcc", "#d2b9a6", "#e8925a", ORANGE, "#8c3a00"])
    change_norm = mcolors.BoundaryNorm(change_bounds, change_cmap.N, extend="both")
    bounds = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    nse_cmap = SEQUENTIAL_ALT.resampled(len(bounds))  # one extra colour for the below-0 bin
    nse_norm = mcolors.BoundaryNorm(bounds, nse_cmap.N, extend="min")

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, HEIGHT), sharex=True, sharey=True)
    for row, cell in enumerate(CELLS):
        g = gains[cell]
        panels = [(nse[cell], ""), (g, f" ({100 * (g > 0).mean():.0f}% improved)")]
        print(f"  {cell:<9} {CORRECTION}: {int((g > 0).sum())}/{len(g)} improved, "
              f"median change {g.median():+.2f}")

        for col, (v, note) in enumerate(panels):
            ax = axes[row][col]
            lat = loc.loc[v.index, "gauge_lat"].to_numpy()
            lon = loc.loc[v.index, "gauge_lon"].to_numpy()
            v = v.to_numpy()
            draw_us_outline(ax)
            if col == 0:
                order = np.argsort(-v)  # poorest basins on top, so they aren't buried
                nse_sc = ax.scatter(lon[order], lat[order], c=v[order], cmap=nse_cmap,
                                    norm=nse_norm, s=5, edgecolor=MUTE, linewidth=0.2, zorder=3)
                ax.text(-0.04, 0.5, ROW_LABEL[cell], transform=ax.transAxes, rotation=90,
                        ha="right", va="center", color=INK, fontsize=LABEL - 1)
            else:
                order = np.argsort(np.abs(v))
                change_sc = ax.scatter(lon[order], lat[order], c=v[order], cmap=change_cmap,
                                       norm=change_norm, s=5, edgecolor=MUTE, linewidth=0.15,
                                       zorder=3)
            ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
            ax.set_title(f"{panel_letter(2 * row + col)} {COL_TITLE[col]}{note}", color=INK,
                         fontsize=PANEL - 2, loc="left", pad=2)
            ax.set_axis_off()

    # Margins set by hand: tight_layout sizes the slots before the map aspect shrinks them.
    fig.subplots_adjust(left=0.05, right=0.99, top=1 - 0.2 / HEIGHT, bottom=0.75 / HEIGHT,
                        wspace=0.06, hspace=0.12)
    for ax in axes[1]:
        ax.apply_aspect()  # so get_position() gives where the map is drawn, not its slot
    first, second = (ax.get_position() for ax in axes[1])
    bar_y, bar_h = 0.42 / HEIGHT, 0.08 / HEIGHT
    colorbar(fig, nse_sc, (first.x0 + 0.1 * first.width, bar_y, 0.8 * first.width, bar_h),
             "NSE, Nash-Sutcliffe efficiency", ticks=bounds, spacing="uniform")
    colorbar(fig, change_sc, (second.x0 + 0.05 * second.width, bar_y, 0.9 * second.width, bar_h),
             "change in NSE", ticks=change_bounds, spacing="uniform",
             format=mticker.FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\u2212")))

    os.makedirs(f"{HERE}/{OUT_DIR}", exist_ok=True)
    stem = f"{HERE}/{OUT_DIR}/fig04_nse_maps"
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=300, facecolor="white")
    print(f"-> {stem}.png/.svg")


if __name__ == "__main__":
    main()
