"""CDFs of NSE and KGE across basins for the LSTM before and after each correction, both cells.

    python figures/make_lstm_cdf.py   # -> lstm/fig03_cdf.{png,pdf}

Two panels, NSE then KGE. Colour is the method (uncorrected, IDW, MLP) and line type the cell
(dashed ungauged, solid gauged, labelled "discontinued" as in the manuscript); each legend lists
the ungauged three, then the discontinued three, with their medians.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_lstm_boxplots import SERIES, series  # noqa: E402
from palette import INK, LABEL, MUTE, PANEL, panel_letter, style  # noqa: E402
from palette import figure_variant  # noqa: E402

SUFFIX, OUT_DIR = figure_variant()


PANELS = [("nse", "Nash–Sutcliffe efficiency (NSE)"), ("kge", "Kling–Gupta efficiency (KGE)")]
CELLS = {"ungauged": (0, (4, 2)), "gauged": "-"}
CELL_LABEL = {"ungauged": "ungauged basins", "gauged": "discontinued gauges"}


def main():
    data = {cell: series(cell) for cell in CELLS}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3), sharey=True)
    for letter, (ax, (metric, xlabel)) in enumerate(zip(axes, PANELS)):
        handles = []
        for cell, ls in CELLS.items():
            # A blank-handled entry as each group's header.
            handles.append(Line2D([], [], color="none", label=f"{CELL_LABEL[cell]} (median)"))
            for name, color in SERIES.items():
                v = np.sort(data[cell][name][metric].dropna().to_numpy())
                handles += ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, lw=1.8,
                                   ls=ls, zorder=3, label=f"{name}  {np.median(v):.2f}")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel, color=INK, fontsize=LABEL)
        if letter == 0:
            ax.set_ylabel("fraction of basins", color=INK, fontsize=LABEL)
        legend = ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=LABEL - 2,
                           labelcolor=INK, handlelength=2.4)
        for text, handle in zip(legend.get_texts(), handles):
            if handle.get_color() == "none":
                text.set_color(MUTE)
        ax.set_title(panel_letter(letter), color=INK, fontsize=PANEL, loc="left", pad=4)
        style(ax)

    fig.tight_layout(pad=0.4, w_pad=2.0)
    os.makedirs(f"{HERE}/{OUT_DIR}", exist_ok=True)
    stem = f"{HERE}/{OUT_DIR}/fig03_cdf"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, facecolor="white")
    print(f"-> {stem}.png/.pdf")


if __name__ == "__main__":
    main()
