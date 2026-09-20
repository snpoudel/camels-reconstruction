"""Whether the model's errors are spatially structured -- the premise the correction rests on.

    python figures/make_error_structure.py

One dot per PAIR of basins: how their two daily residual series correlate over the whole
evaluation period, against how far apart the two gauges are. Distance is the x axis, not time --
each pair's years have already been collapsed into a single correlation before it reaches this
plot. The line is a LOESS fit through the cloud, drawn because tens of thousands of overplotted
dots show the trend poorly on their own; the dots stay so the reader can see what the fit is
summarising and how wide the spread around it really is.

Drawn from every trained-on ("gauged") basin's residual, not just the smaller ungauged held-out
set: the pattern being tested is a property of the regional model itself, not of any one basin's
training status, so testing it across every trained-on basin gives the densest picture of the
same pattern -- see `evaluate.error_structure`.

Nothing here knows what the correction is: the figure is built from residuals and gauge
coordinates alone, with no K, lambda or p in it, so it establishes a property of the model's
errors rather than a property of this pipeline.

Colour: grey for the pairs, the palette's first series colour for the fit -- the cloud is
evidence, the line is the claim, and the ink follows that.
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
from evaluate import LOESS_KM, read_error_structure, read_info  # noqa: E402
from palette import INK, LABEL, MUTE, NEUTRAL, RAW, TITLE, caption, model_label, style  # noqa: E402

CELL = "gauged"

# The axis is cut here rather than at the cloud's true floor, which keeps the decay legible at
# the cost of a few hundred strongly anticorrelated pairs. The caption reports how many, so the
# trim is stated rather than silent.
Y_FLOOR = -0.4


def main(model):
    pairs, fit = read_error_structure(model, CELL)
    info = read_info(CELL, model)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.scatter(pairs.dist_km, pairs.r, s=5, color=NEUTRAL, alpha=0.10, linewidths=0, zorder=2,
               rasterized=True, label="one basin pair")
    ax.plot(fit.dist_km, fit.r_fit, color=RAW, lw=2.4, zorder=4, label="LOESS fit")
    ax.axhline(0, color=MUTE, lw=0.8, ls=(0, (4, 3)), zorder=3)

    ax.set_xlim(0, fit.dist_km.max())
    ax.set_ylim(Y_FLOOR, 0.9)
    ax.set_xlabel("distance between gauge pair (km)", color=INK, fontsize=LABEL)
    ax.set_ylabel("correlation of daily residuals", color=INK, fontsize=LABEL)
    leg = ax.legend(frameon=False, loc="upper right", fontsize=LABEL - 1, labelcolor=INK,
                    handletextpad=0.6, borderpad=0.2)
    leg.legend_handles[0].set_alpha(0.8)  # the cloud's own alpha is unreadable at legend size
    leg.legend_handles[0].set_sizes([26])
    style(ax)
    ax.set_title(f"Spatial structure of {model_label(model)}'s error", color=INK,
                 fontsize=TITLE, loc="left", pad=8)

    bottom = caption(fig, f"{info['n_basins']} trained-on basins, "
                          f"{info['n_folds']} folds, {info['n_seeds']} seeds, test period "
                          f"{info['test_period']}.  The residual is the same log quantity the "
                          f"correction transfers, log(sim + eps) - log(obs + eps), from the "
                          f"model trained on each basin, evaluated on days outside its own "
                          f"training window.  Every pair of gauges within "
                          f"{int(fit.dist_km.max())} km contributes one dot, its correlation "
                          f"taken over the whole test period -- {len(pairs):,} pairs in all, of "
                          f"which {int((pairs.r < Y_FLOOR).sum())} fall below the axis.  Pairs "
                          f"are far denser at long range than short, so the fit is a LOESS with "
                          f"a fixed {LOESS_KM} km bandwidth rather than a fixed share of the "
                          f"points, which would over-smooth exactly where the decay is "
                          f"steepest.", 100)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    suffix = "" if model == "lstm" else f"_{model}"
    out = f"{HERE}/{CELL}_error_structure{suffix}.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}   {len(pairs)} pairs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm")
    main(ap.parse_args().model)
