"""How the skill of every basin shifts when the correction is applied.

    python figures/make_cdf.py --model lstm     # -> figures/cdf_lstm.png (gauged + ungauged)
    python figures/make_cdf.py --model hbv      # -> figures/cdf_hbv.png (gauged only -- HBV
                                                 #    has no ungauged run)
    python figures/make_cdf.py --cell gauged --model lstm   # force one cell -> gauged_cdf_lstm.png

Empirical cumulative distributions, NSE and KGE across basins, one row per cell the model has
(gauged, ungauged, or both -- see `palette.available_cells`). Each basin contributes one point --
the score of its seed-mean hydrograph -- so a curve lying to the right is winning at every
quantile it is right of, not merely on average.

A third line is the parameter-free reference: the single nearest gauge's error applied in full,
no K, lambda or p chosen. It lands to the LEFT of the uncorrected model, which is the point --
the neighbour signal is real (see the error-structure figure) but believing one gauge completely
destroys more than it repairs. What separates the two is pooling and shrinkage, not the
existence of spatial structure.

The settings on the page were chosen to maximise KGE (`figures/evaluate.py`'s default), so the
NSE column is a deliberate cross-metric view and the caption says so. The two columns disagree by
more than their axes alone: the correction improves KGE's correlation and bias terms but
slightly worsens its variability term, which NSE does not charge for, so the NSE gain runs
larger than the KGE gain.

Colour: the only figure drawing three series at once, so it uses all three Okabe-Ito slots from
palette.py. Identity never rests on colour alone: each series is named in the legend with its
median, and the curves are separated in position.
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
from evaluate import fold_settings, read_info, read_metrics  # noqa: E402
from palette import BENCHMARK, CELL_BLURB, CELL_TITLE, INK, LABEL, MUTE, PANEL  # noqa: E402
from palette import TITLE, available_cells, caption, model_label, panel_letter  # noqa: E402
from palette import series_cdf, settings_note, style  # noqa: E402

PANELS = [("nse", "NSE, Nash-Sutcliffe efficiency"), ("kge", "KGE, Kling-Gupta efficiency")]


def main(cells, model, prefix):
    fig, axes = plt.subplots(len(cells), 2, figsize=(8.5, 8.5 if len(cells) == 2 else 4.5),
                             sharex=True, squeeze=False)

    letter = 0
    blurbs = []
    for row, cell in enumerate(cells):
        df = read_metrics(cell, model)
        info = read_info(cell, model)
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
        blurbs.append(f"{CELL_TITLE[cell]} ({info['n_basins']} basins, {CELL_BLURB[cell]}, "
                      f"{info['n_folds']} folds, {info['n_seeds']} seeds, test "
                      f"{info['test_period']}, "
                      f"{settings_note(fold_settings(df, model_label(model)))})")

    fig.suptitle(f"{model_label(model)}: skill before and after the correction",
                fontsize=TITLE, color=INK, x=0.012, ha="left", y=0.995)
    bottom = caption(fig, f"{'; '.join(blurbs)}, chosen by cross-validation to maximise KGE, "
                          f"never reading the basin being corrected.  "
                          f"\"{BENCHMARK}\" is the parameter-free reference: the single closest "
                          f"gauge's error applied in full, nothing selected.", 110)
    top = 0.94 if len(cells) == 2 else 0.88
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.subplots_adjust(top=top)
    out = f"{HERE}/{prefix}cdf_{model}.png"
    fig.savefig(out, dpi=200, facecolor="white")
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
