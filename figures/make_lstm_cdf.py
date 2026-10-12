"""Fig. 3: CDF of NSE across basins for the LSTM before and after each correction, both cells.

    python figures/make_lstm_cdf.py   # -> lstm/fig03_cdf.{png,svg}

Colour is the method (uncorrected, IDW, MLP) and line type the cell (dashed ungauged, solid
gauged, labelled "discontinued" as in the manuscript); the legend lists the ungauged three, then
the discontinued three, with their medians. KGE and the biases are in Table 1 and Fig. S1.
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
from palette import INK, LABEL, MUTE, style  # noqa: E402
from palette import figure_variant  # noqa: E402

SUFFIX, OUT_DIR = figure_variant()


CELLS = {"ungauged": (0, (4, 2)), "gauged": "-"}
CELL_LABEL = {"ungauged": "ungauged basins", "gauged": "discontinued gauges"}


def main():
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    handles = []
    for cell, ls in CELLS.items():
        data = series(cell)
        # A blank-handled entry as each group's header.
        handles.append(Line2D([], [], color="none", label=f"{CELL_LABEL[cell]} (median)"))
        for name, color in SERIES.items():
            v = np.sort(data[name]["nse"].dropna().to_numpy())
            handles += ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, lw=1.6, ls=ls,
                               zorder=3, label=f"{name}  {np.median(v):.2f}")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Nash–Sutcliffe efficiency (NSE)", color=INK, fontsize=LABEL - 1)
    ax.set_ylabel("fraction of basins", color=INK, fontsize=LABEL - 1)
    legend = ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=LABEL - 3,
                       labelcolor=INK, handlelength=2.2)
    for text, handle in zip(legend.get_texts(), handles):
        if handle.get_color() == "none":
            text.set_color(MUTE)
    style(ax)
    ax.tick_params(labelsize=LABEL - 2)

    fig.tight_layout(pad=0.4)
    os.makedirs(f"{HERE}/{OUT_DIR}", exist_ok=True)
    stem = f"{HERE}/{OUT_DIR}/fig03_cdf"
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=300, facecolor="white")
    print(f"-> {stem}.png/.svg")


if __name__ == "__main__":
    main()
