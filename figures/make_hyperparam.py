"""Cross-validation for the correction's three constants: K, lambda and p.

    python figures/make_hyperparam.py --cell ungauged
    python figures/make_hyperparam.py --cell gauged

The correction has three free numbers: K, how many gauges to consult; lambda, how much of their
error to believe; and p, how steeply weight falls off with distance (weight ~ 1/d^p). All three
are chosen by cross-validation among basins other than the ones being corrected, so no basin's
own record reaches the settings applied to it -- and no basin on this figure is one being scored
as a result.

One line per fold. Each is a profile through that fold's own optimum: a panel sweeps one
constant with the other two held at that fold's best, and the marker sits on the cell the
cross-validation scored highest, which is what the pipeline applies. Reading across the lines
is the point: the optima cluster in a narrow region and the curves are flat around them.

Folds are replicates of one procedure, not different things being compared, so they share one
hue rather than getting a categorical palette; no line needs to be told apart from another.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import read_info, read_tuning  # noqa: E402
from palette import CELL_TITLE, CORRECTED, INK, LABEL, PANEL, TITLE  # noqa: E402
from palette import caption, style  # noqa: E402

# What the cross-validation pool is, per cell -- not the same thing as the basins being
# corrected, which appear nowhere on this figure.
CV_POOL = {"ungauged": "the fold's own training basins",
           "gauged": "every basin outside the group being corrected"}

AXES = [("k", "K, donor gauges consulted", "(a)"),
        ("lam", r"$\lambda$, fraction of donor error applied", "(b)"),
        ("p", "p, inverse-distance exponent", "(c)")]


def main(cell):
    df = read_tuning(cell)
    info = read_info(cell)
    labels = sorted(df.label.unique())
    single = len(labels) == 1
    lw, alpha = (2.2, 1.0) if single else (1.5, 0.75)

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 4.4), sharey=True)
    optima = {col: [] for col, _, _ in AXES}
    for label in labels:
        d = df[df.label == label]
        best = d[d.selected].iloc[0]
        for ax, (col, _, _) in zip(axes, AXES):
            others = [c for c, _, _ in AXES if c != col]
            sub = d[(d[others[0]] == best[others[0]]) & (d[others[1]] == best[others[1]])]
            sub = sub.sort_values(col)
            # K and p are sampled unevenly, so plotting by value packs the grid unevenly;
            # plotting by rank gives every step an equal slot and the ticks show real values.
            grid = sorted(df[col].unique())
            ax.plot([grid.index(v) for v in sub[col]], sub.score, color=CORRECTED, lw=lw,
                    alpha=alpha, zorder=3)
            ax.plot([grid.index(best[col])], [best.score], "o", ms=7, color=CORRECTED,
                    markeredgecolor="white", markeredgewidth=1.5, zorder=5)
            optima[col].append(best[col])

    for ax, (col, xlabel, title) in zip(axes, AXES):
        grid = sorted(df[col].unique())
        step = max(1, -(-len(grid) // 6))  # at most ~6 labels in a narrow panel
        ax.set_xticks(range(len(grid)))
        ax.set_xticklabels([f"{v:g}" if i % step == 0 else "" for i, v in enumerate(grid)])
        ax.set_xlabel(xlabel, color=INK, fontsize=LABEL)
        # Panel letter in the bottom-left corner, in axis fractions so it sits inside the
        # plot regardless of the y-limits below.
        ax.text(0.03, 0.05, title, color=INK, fontsize=PANEL, transform=ax.transAxes,
               ha="left", va="bottom")
        if ax is axes[0]:  # one legend for the figure: all three panels draw the same two things
            ax.plot([], [], color=CORRECTED, lw=lw, alpha=alpha,
                    label="profile" if single else "per-fold profile")
            ax.plot([], [], "o", ms=7, color=CORRECTED, markeredgecolor="white",
                    markeredgewidth=1.5, label="selected")
            ax.legend(frameon=False, loc="lower right", fontsize=LABEL - 2, labelcolor=INK)
        # Floor at 0.6 rather than the tight span of the surface: the surface itself only
        # spans a few hundredths of KGE, and zooming all the way to it (as an earlier version
        # did) makes ordinary sampling noise look like a dramatic shape. 0.6 keeps the profiles
        # readable while still showing them clearly above the uncorrected floor.
        pad = 0.06 * (df.score.max() - df.score.min())
        ax.set_ylim(0.6, df.score.max() + pad)
        style(ax)
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)
    axes[0].set_ylabel("Median cross-validation KGE", color=INK, fontsize=LABEL)

    fig.suptitle(f"{CELL_TITLE[cell]}: cross-validation for K, $\\lambda$ and p",
                 fontsize=TITLE, color=INK, x=0.012, ha="left", y=0.98)
    spans = []
    for col, label in (("k", "K"), ("lam", r"$\lambda$"), ("p", "p")):
        v = optima[col]
        spans.append(f"{label} = {v[0]:g}" if len(set(v)) == 1
                     else f"{label} in [{min(v):g}, {max(v):g}]")
    bottom = caption(fig, f"{len(labels)} fold{'s' if len(labels) > 1 else ''}, "
                          f"{int(df.n_basins.iloc[0])} cross-validation basins each "
                          f"({CV_POOL[cell]}).  "
                          f"Each line holds the other two constants at that fold's optimum; "
                          f"markers are the selected values: {', '.join(spans)}.  $\\lambda$ = 0 "
                          f"is the uncorrected model.  The axis is zoomed to the surface's full "
                          f"span of {df.score.max() - df.score.min():.3f} KGE -- a wide region "
                          f"scores nearly the same.  No basin is read here to choose the settings "
                          f"applied to it.", 118)
    fig.tight_layout(rect=(0.02, bottom, 1, 0.94))
    out = f"{HERE}/{cell}_hyperparameters.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}   optima: " + ", ".join(spans))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=list(CELL_TITLE), default="ungauged")
    main(ap.parse_args().cell)
