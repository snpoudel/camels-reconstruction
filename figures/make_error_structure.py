"""Whether model errors are spatially structured -- the premise the correction rests on -- and
whether that holds beyond one model family.

    python figures/make_error_structure.py

Three panels, one row per model, all reading the discontinued-gauge ("gauged") cell: a regional
LSTM, a regional differentiable HBV, and a locally-calibrated HBV run one basin at a time with no
cross-basin information at all. One dot per PAIR of basins in each panel: how their two daily
residual series correlate over the whole evaluation period, against how far apart the two gauges
are. Distance is the x axis, not time -- each pair's years have already been collapsed into a
single correlation before it reaches this plot. The line is a LOESS fit through the cloud, drawn
because tens of thousands of overplotted dots show the trend poorly on their own; the dots stay
so the reader can see what the fit is summarising and how wide the spread around it really is.

Three model families rather than one because the argument the figure makes -- errors at nearby
gauges rise and fall together, so a neighbour's error is informative about a target basin's own
-- would be a property of one architecture if only the LSTM were shown. A regionally-trained
differentiable model and a conceptual model calibrated the traditional way, one basin at a time,
show the identical shape, so the pattern is a property of what these models get wrong, not of how
any one of them was built.

Nothing here knows what the correction is: the figure is built from residuals and gauge
coordinates alone, with no K, lambda or p in it, so it establishes a property of each model's
errors rather than a property of this pipeline -- see `evaluate.error_structure`.

Colour: grey for the pairs, the palette's first series colour for the fit -- the cloud is
evidence, the line is the claim, and the ink follows that.
"""
import os
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import LOESS_KM, read_error_structure, read_info  # noqa: E402
from palette import INK, LABEL, MUTE, NEUTRAL, RAW, TITLE, caption, style  # noqa: E402

CELL = "gauged"

# model key, panel title (with the citation the panel's model draws its claim to soundness
# from), and the plain-language model description that goes in the caption
PANELS = [
    ("lstm", "Regional LSTM (Kratzert et al., 2019b)",
     "an LSTM trained jointly on every basin"),
    ("hybrid", "Regional differentiable HBV (Poudel & Steinschneider, 2026)",
     "a differentiable HBV trained jointly on every basin"),
    ("hbv", "Local HBV (Seibert et al., 2018)",
     "an HBV model calibrated to each basin's own record alone, with no cross-basin "
     "information at all"),
]

# The axis is cut here rather than at each cloud's true floor, which keeps the decay legible at
# the cost of a handful of strongly anticorrelated pairs. The caption reports how many, so the
# trim is stated rather than silent.
Y_FLOOR = -0.4


def main():
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(6.5, 8), sharex=True)

    caption_parts = []
    for ax, (model, title, desc) in zip(axes, PANELS):
        pairs, fit = read_error_structure(model, CELL)
        info = read_info(CELL, model)

        ax.scatter(pairs.dist_km, pairs.r, s=5, color=NEUTRAL, alpha=0.10, linewidths=0,
                   zorder=2, rasterized=True, label="one basin pair")
        ax.plot(fit.dist_km, fit.r_fit, color=RAW, lw=2.4, zorder=4, label="LOESS fit")
        ax.axhline(0, color=MUTE, lw=0.8, ls=(0, (4, 3)), zorder=3)

        ax.set_xlim(0, fit.dist_km.max())
        ax.set_ylim(Y_FLOOR, 1)
        ax.set_ylabel("correlation of\ndaily residuals", color=INK, fontsize=LABEL)
        ax.set_title(title, color=INK, fontsize=TITLE, loc="left", pad=8)
        style(ax)

        caption_parts.append(f"{title.split(' (')[0]} -- {desc}, {info['n_basins']} "
                             f"trained-on basins, {info['n_folds']} folds, {info['n_seeds']} "
                             f"seed(s), test period {info['test_period']}, {len(pairs):,} pairs, "
                             f"{int((pairs.r < Y_FLOOR).sum())} below the axis")

    axes[-1].set_xlabel("distance between gauge pair (km)", color=INK, fontsize=LABEL)
    leg = axes[0].legend(frameon=False, loc="upper right", fontsize=LABEL - 1, labelcolor=INK,
                         handletextpad=0.6, borderpad=0.2)
    leg.legend_handles[0].set_alpha(0.8)  # the cloud's own alpha is unreadable at legend size
    leg.legend_handles[0].set_sizes([26])

    text = ("The residual is the same log quantity the correction transfers, "
            "log(sim + eps) - log(obs + eps), evaluated on days outside each "
            "model's own training window; distance is great-circle, truncated at "
            "1000 km, and the fit is a LOESS with a fixed "
            f"{LOESS_KM} km bandwidth. " + "  ".join(caption_parts) + ".")
    width = 105
    caption(fig, text, width)
    # `caption` anchors its text a fixed 0.014 (figure-fraction) above the bottom edge and
    # returns a margin sized for the ~4.8in-tall figures every other script here uses; on this
    # three-panel figure that fraction is measured against triple the height, which leaves a
    # canyon of white space between the last axis and the caption. Reserve just enough room in
    # inches instead: the caption block's own height plus a small gap above it.
    fig_height = fig.get_size_inches()[1]
    n_lines = len(textwrap.wrap(text, width))
    line_height_in = 8.5 * 1.2 / 72
    bottom = 0.014 + (n_lines * line_height_in + 0.12) / fig_height
    fig.tight_layout(rect=(0, bottom, 1, 1))
    out = f"{HERE}/{CELL}_error_structure.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
