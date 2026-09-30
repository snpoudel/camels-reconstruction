"""Cross-validation profiles for K, lambda and p.

    python figures/make_hyperparam.py --model lstm                  # -> hyperparameters_lstm.png
    python figures/make_hyperparam.py --cell ungauged --model lstm  # one cell

One row per cell, one line per fold. Each panel sweeps one constant with the other two held at
that fold's optimum; the marker is the selected value. No basin shown here is one being corrected.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import read_tuning  # noqa: E402
from palette import CELL_TITLE, CORRECTED, INK, LABEL, MUTE, PANEL, TITLE  # noqa: E402
from palette import available_cells, model_label, panel_letter, style  # noqa: E402

CV_POOL = {"ungauged": "the fold's own training basins",
           "gauged": "every basin outside the group being corrected"}

AXES = [("k", "K, donor gauges consulted"),
        (r"lam", r"$\lambda$, fraction of donor error applied"),
        ("p", "p, inverse-distance exponent")]


def main(cells, model, prefix):
    fig, axes = plt.subplots(len(cells), 3, figsize=(8.5, 7.5 if len(cells) == 2 else 4),
                             squeeze=False)

    letter, cv_lines = 0, []
    for row, cell in enumerate(cells):
        df = read_tuning(cell, model)
        labels = sorted(df.label.unique())
        single = len(labels) == 1
        lw, alpha = (1.6, 1.0) if single else (1.0, 0.75)
        ms = 5

        optima = {col: [] for col, _ in AXES}
        for label in labels:
            d = df[df.label == label]
            best = d[d.selected].iloc[0]
            for col_i, (col, _) in enumerate(AXES):
                ax = axes[row][col_i]
                others = [c for c, _ in AXES if c != col]
                sub = d[(d[others[0]] == best[others[0]]) & (d[others[1]] == best[others[1]])]
                sub = sub.sort_values(col)
                # Plotted by grid rank, not value: K and p are sampled unevenly.
                grid = sorted(df[col].unique())
                ax.plot([grid.index(v) for v in sub[col]], sub.score, color=CORRECTED, lw=lw,
                        alpha=alpha, zorder=3)
                ax.plot([grid.index(best[col])], [best.score], "o", ms=ms, color=CORRECTED,
                        markeredgecolor="white", markeredgewidth=1.2, zorder=5)
                optima[col].append(best[col])

        for col_i, (col, xlabel) in enumerate(AXES):
            ax = axes[row][col_i]
            grid = sorted(df[col].unique())
            step = max(1, -(-len(grid) // 6))  # at most ~6 labelled ticks
            major = list(range(0, len(grid), step))
            minor = [i for i in range(len(grid)) if i not in major]
            ax.set_xticks(major)
            ax.set_xticklabels([f"{grid[i]:g}" for i in major])
            if minor:
                ax.set_xticks(minor, minor=True)
            if row == len(cells) - 1:
                ax.set_xlabel(xlabel, color=INK, fontsize=LABEL)
            ax.text(0.03, 0.05, panel_letter(letter), color=INK, fontsize=PANEL,
                    transform=ax.transAxes, ha="left", va="bottom")
            letter += 1
            if col_i == 0 and row == 0:
                ax.plot([], [], color=CORRECTED, lw=lw, alpha=alpha,
                        label="profile" if single else "per-fold profile")
                ax.plot([], [], "o", ms=ms, color=CORRECTED, markeredgecolor="white",
                        markeredgewidth=1.2, label="selected")
                ax.legend(frameon=False, loc="lower right", fontsize=LABEL - 2, labelcolor=INK)
            # Floor at 0.7: the surface spans only a few hundredths of KGE, and zooming to it
            # makes sampling noise look like structure.
            pad = 0.06 * (df.score.max() - df.score.min())
            ax.set_ylim(0.7, df.score.max() + pad)
            style(ax)
            ax.tick_params(which="minor", colors=MUTE, length=2)
            if col_i > 0:
                ax.tick_params(labelleft=False)
        axes[row][0].set_ylabel(f"{CELL_TITLE[cell]}\nMedian cross-validation KGE",
                                color=INK, fontsize=LABEL)

        spans = []
        for col, label in (("k", "K"), ("lam", r"$\lambda$"), ("p", "p")):
            v = optima[col]
            spans.append(f"{label} = {v[0]:g}" if len(set(v)) == 1
                         else f"{label} in [{min(v):g}, {max(v):g}]")
        cv_lines.append(f"{CELL_TITLE[cell]}: {len(labels)} fold{'s' if len(labels) > 1 else ''}, "
                        f"{int(df.n_basins.iloc[0])} cross-validation basins each "
                        f"({CV_POOL[cell]}), selected {', '.join(spans)}")

    fig.suptitle(f"{model_label(model)}: cross-validation for K, $\\lambda$ and p",
                 fontsize=TITLE, color=INK, x=0.012, ha="left", y=0.995)
    fig.tight_layout(rect=(0.015, 0, 1, 1))
    fig.subplots_adjust(top=0.92 if len(cells) == 2 else 0.85)
    out = f"{HERE}/{prefix}hyperparameters_{model}.png"
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"-> {out}   " + "; ".join(cv_lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=list(CELL_TITLE), help="force one cell; default: every "
                    "cell this model has")
    ap.add_argument("--model", default="lstm")
    a = ap.parse_args()
    cells = [a.cell] if a.cell else available_cells(a.model)
    if not cells:
        raise SystemExit(f"no tuning surface for model '{a.model}' -- run figures/evaluate.py "
                         f"--model {a.model} first")
    main(cells, a.model, f"{a.cell}_" if a.cell else "")
