"""Whether the model's errors are spatially structured -- the premise the correction rests on.

    python figures/make_error_structure.py

How two basins' daily log residuals correlate, against how far apart their gauges are. Nothing
here knows what the correction is: the figure is built from residuals and gauge coordinates
alone, with no K, lambda or p in it, so it establishes a property of the model's errors rather
than a property of this pipeline.

Drawn twice, on the raw residual and with each series' day-of-year climatology removed. The
second line carries the argument: two nearby basins share a seasonal cycle through their
forcings whether or not the model errs together, so only the deseasonalised agreement is
evidence of day-to-day information worth borrowing.

Colour: ink for both lines, the deseasonalised one dashed. These are two views of one quantity
rather than two series, so they are separated by line style instead of taking a palette slot --
blue and orange mean uncorrected/corrected everywhere, and teal is the CDF figure's
nearest-gauge reference.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import read_error_structure, read_info  # noqa: E402
from palette import INK, LABEL, MUTE, TITLE, caption, style  # noqa: E402

SERIES = [("", "solid", "daily log residual"),
          ("_deseason", (0, (5, 2)), "seasonal cycle removed")]


def main():
    dist = read_error_structure()
    info = read_info("ungauged")

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for suffix, dash, label in SERIES:
        ax.fill_between(dist.dist_mid, dist[f"r_q25{suffix}"], dist[f"r_q75{suffix}"],
                        color=INK, alpha=0.10, lw=0, zorder=2)
        ax.plot(dist.dist_mid, dist[f"r_med{suffix}"], color=INK, lw=2, ls=dash, zorder=3,
                label=f"{label}   {dist[f'r_med{suffix}'].iloc[0]:.2f} at the closest bin")
    ax.axhline(0, color=MUTE, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(0, dist.dist_hi.max())
    ax.set_xlabel("distance between gauges (km)", color=INK, fontsize=LABEL)
    ax.set_ylabel("correlation of daily residuals", color=INK, fontsize=LABEL)
    ax.legend(frameon=False, loc="upper right", fontsize=LABEL - 1, labelcolor=INK)
    style(ax)
    ax.set_title("Ungauged basins: the model's errors are spatially structured",
                 color=INK, fontsize=TITLE, loc="left", pad=8)

    n_pairs = int(dist.n_pairs.sum())
    bin_km = int(dist.dist_hi.iloc[0] - dist.dist_lo.iloc[0])
    bottom = caption(fig, f"{info['n_basins']} basins held out of training entirely, "
                          f"{info['n_folds']} folds, {info['n_seeds']} seeds, test period "
                          f"{info['test_period']}.  The residual is the same log quantity the "
                          f"correction transfers, log(sim + eps) - log(obs + eps), from the "
                          f"model that held each basin out.  {n_pairs} basin pairs within "
                          f"{int(dist.dist_hi.max())} km, binned every {bin_km} km: median with "
                          f"the interquartile range.  Deseasonalising removes each series' own "
                          f"day-of-year climatology, smoothed over 31 days, from both sides "
                          f"before correlating.", 100)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    out = f"{HERE}/ungauged_error_structure.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}   {n_pairs} pairs")


if __name__ == "__main__":
    main()
