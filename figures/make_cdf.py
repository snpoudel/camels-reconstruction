"""CDFs of NSE and KGE across basins, before and after the correction.

    python figures/make_cdf.py --model lstm                 # -> cdf_lstm.png
    python figures/make_cdf.py --cell gauged --model lstm   # one cell -> gauged_cdf_lstm.png

One row per cell the model has. The nearest-gauge-only reference (one donor, all of its error)
lands left of the uncorrected model: the neighbour signal is real, but the gain comes from
pooling and shrinkage. Settings were selected on KGE, so the NSE column is a cross-metric view.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import read_metrics  # noqa: E402
from palette import BENCHMARK, CELL_TITLE, INK, LABEL, MUTE, PANEL, TITLE  # noqa: E402
from palette import available_cells, model_label, panel_letter, series_cdf, style  # noqa: E402

PANELS = [("nse", "NSE, Nash-Sutcliffe efficiency"), ("kge", "KGE, Kling-Gupta efficiency")]


def main(cells, model, prefix):
    fig, axes = plt.subplots(len(cells), 2, figsize=(8.5, 8.5 if len(cells) == 2 else 4.5),
                             sharex=True, squeeze=False)

    letter = 0
    for row, cell in enumerate(cells):
        df = read_metrics(cell, model)
        for col, (metric, xlabel) in enumerate(PANELS):
            ax = axes[row][col]
            for name, color in series_cdf(model).items():
                v = np.sort(df.loc[df.variant == name, metric].dropna().to_numpy())
                label = f"{model_label(model)} + {name}" if name == BENCHMARK else name
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, lw=2, zorder=3,
                        label=f"{label} ({np.median(v):.2f})")
                ax.plot([np.median(v)], [0.5], "o", ms=7, color=color, markeredgecolor="white",
                        markeredgewidth=1.6, zorder=4)
            ax.axhline(0.5, color=MUTE, lw=0.8, ls=(0, (4, 3)), zorder=1)
            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, 1)
            if row == len(cells) - 1:
                ax.set_xlabel(xlabel, color=INK, fontsize=LABEL)
            if col == 0:
                ax.set_ylabel("CDF", color=INK, fontsize=LABEL)
            ax.legend(frameon=False, loc="upper left", fontsize=LABEL - 1, labelcolor=INK)
            metric_name = xlabel.split(",")[0]
            ax.set_title(f"{panel_letter(letter)} {CELL_TITLE[cell]} — {metric_name}",
                         color=INK, fontsize=PANEL, loc="left", pad=6)
            style(ax)
            letter += 1

    fig.suptitle(f"{model_label(model)}: skill before and after the correction",
                 fontsize=TITLE, color=INK, x=0.012, ha="left", y=0.995)
    fig.tight_layout()
    fig.subplots_adjust(top=0.94 if len(cells) == 2 else 0.88)
    out = f"{HERE}/{prefix}cdf_{model}.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=list(CELL_TITLE), help="force one cell; default: every "
                    "cell this model has")
    ap.add_argument("--model", default="lstm")
    a = ap.parse_args()
    cells = [a.cell] if a.cell else available_cells(a.model)
    if not cells:
        raise SystemExit(f"no scored metrics for model '{a.model}' -- run figures/evaluate.py "
                         f"--model {a.model} first")
    main(cells, a.model, f"{a.cell}_" if a.cell else "")
