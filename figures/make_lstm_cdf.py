"""CDFs of NSE and KGE across basins for the LSTM before and after each correction, both cells.

    python figures/make_lstm_cdf.py   # -> lstm/03_cdf.png

Two panels, NSE then KGE. Colour is the method (uncorrected, IDW, MLP) and line type the cell
(dashed ungauged, solid gauged); each legend lists the ungauged three, then the gauged three,
with their medians.
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
from palette import INK, LABEL, MUTE, PANEL, TITLE, panel_letter, style  # noqa: E402

PANELS = [("nse", "NSE, Nash-Sutcliffe efficiency"), ("kge", "KGE, Kling-Gupta efficiency")]
CELLS = {"ungauged": (0, (4, 2)), "gauged": "-"}


def main():
    data = {cell: series(cell) for cell in CELLS}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=True)
    for letter, (ax, (metric, xlabel)) in enumerate(zip(axes, PANELS)):
        handles = []
        for cell, ls in CELLS.items():
            # A blank-handled entry as each group's header.
            handles.append(Line2D([], [], color="none", label=f"{cell} basins (median)"))
            for name, color in SERIES.items():
                v = np.sort(data[cell][name][metric].dropna().to_numpy())
                handles += ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, lw=1.8,
                                   ls=ls, zorder=3, label=f"{name}  {np.median(v):.2f}")
        ax.axhline(0.5, color=MUTE, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlabel, color=INK, fontsize=LABEL)
        if letter == 0:
            ax.set_ylabel("fraction of basins", color=INK, fontsize=LABEL)
        legend = ax.legend(handles=handles, frameon=False, loc="upper left", fontsize=LABEL - 1,
                           labelcolor=INK, handlelength=2.6)
        for text, handle in zip(legend.get_texts(), handles):
            if handle.get_color() == "none":
                text.set_color(MUTE)
        ax.set_title(f"{panel_letter(letter)} {xlabel.split(',')[0]}", color=INK,
                     fontsize=PANEL, loc="left", pad=6)
        style(ax)

    fig.suptitle("LSTM, 531 basins: skill before and after each correction", fontsize=TITLE,
                 color=INK, x=0.012, ha="left", y=0.995)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    os.makedirs(f"{HERE}/lstm", exist_ok=True)
    out = f"{HERE}/lstm/03_cdf.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
