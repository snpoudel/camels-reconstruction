"""NSE and KGE CDFs, vanilla vs donor, on the 76 ungauged (PUB) basins.

    python figures/make_cdf_figure.py   # -> figures/cdf_vanilla_donor.{png,svg}

Two panels, same x-axis measure top and bottom: for each basin, the ensemble-mean
hydrograph's score (see make_results_figure.py, which this reuses `load()` from) is
one point; the curve is the empirical CDF across the 76 held-out basins, so a curve
to the right of the other is winning at every quantile it's right of, not just on
average. `spatial` is left out here -- this figure is specifically the vanilla/donor
information comparison the project's `README.md` singles out.

Colour: the same vanilla/donor slots as the results-figure palette (slots 1 and 3),
so identity stays consistent across the paper's figures. Re-validated as a pair on
its own -- dropping `spatial` changes which checks are load-bearing, since CVD
separation and the normal-vision floor are pairwise: normal-vision dE 24.0, CVD
(protan/deutan) dE 23.1/23.2, both far past the pass thresholds (>=15, >=8). Donor's
teal sits under the 3:1 contrast floor against white (2.7:1) -- a WARN, not a fail,
mitigated the same way the rest of this project's figures already do it: identity
never rests on colour alone (every curve is legend- and position-identified) and text
stays in ink, never in the series colour.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_results_figure import load  # noqa: E402  (reuses the one true ensemble-scoring path)

ARCHS = ["vanilla", "donor"]
COLORS = {"vanilla": "#2a78d6", "donor": "#1baf7a"}
INK, MUTE, GRID = "#16191c", "#5d6a73", "#dfe3e6"
MEASURES = [("nse", "NSE"), ("kge", "KGE")]
XLIM = (-1.0, 1.0)  # a handful of basins score below this; disclosed in the footer, not hidden

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
})


def ecdf(v):
    v = np.sort(v[np.isfinite(v)])
    return v, np.arange(1, len(v) + 1) / len(v)


def build(data, width=4.6, height=5.6):
    fig, axes = plt.subplots(2, 1, figsize=(width, height))
    clipped = total = 0

    for ax, (m, mlabel) in zip(axes, MEASURES):
        # Medians often sit close together on the x-axis (they're what's being compared),
        # so their labels alternate above/below the y=0.5 line rather than both crowding
        # one spot -- centering on the same point would garble into unreadable overlap.
        for arch, (ly, lva) in zip(ARCHS, [(0.44, "top"), (0.56, "bottom")]):
            v = data[("ungauged/test", m, arch)]
            x, y = ecdf(v)
            total += len(v)
            clipped += int((x < XLIM[0]).sum())
            ax.step(np.clip(x, *XLIM), y, where="post", color=COLORS[arch], lw=1.6,
                    label=arch, zorder=3)
            med = float(np.median(v))
            ax.plot([med, med], [0, 0.5], color=COLORS[arch], lw=0.7, ls=(0, (2, 2)), zorder=2)
            ax.text(med, ly, f"{med:.3f}", ha="center", va=lva, fontsize=6.6,
                    color=INK, zorder=4)

        ax.axhline(0.5, color=MUTE, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.set_xlim(*XLIM)
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction of basins ≤ x", fontsize=7.4, color=INK)
        ax.set_title(f"{mlabel}  (→ better)", fontsize=8.2, color=INK, pad=5)
        ax.grid(axis="both", color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTE)
            ax.spines[s].set_linewidth(0.7)
        ax.tick_params(labelsize=6.8, colors=MUTE, length=2.5, width=0.7)

    axes[0].legend(loc="lower right", fontsize=7.2, frameon=False, labelcolor=INK,
                    handlelength=1.4, handletextpad=0.6)
    axes[1].set_xlabel("score", fontsize=7.4, color=INK)

    fig.text(0.01, 0.005,
             "76 held-out (ungauged) basins, WY2005–14 · one point per basin, from the\n"
             "ensemble-mean hydrograph of 3 seeds · dashed verticals mark the median\n"
             f"x clipped to [{XLIM[0]:.0f}, {XLIM[1]:.0f}] — {clipped} of {total} points "
             "below the left edge, not shown",
             fontsize=5.6, color=MUTE, ha="left", va="bottom", linespacing=1.5)
    fig.subplots_adjust(left=0.135, right=0.97, top=0.95, bottom=0.145, hspace=0.32)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=HERE)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = load(dump=None)  # metrics_ensemble.csv is already written by make_results_figure.py
    fig = build(data)
    png, svg = f"{a.out}/cdf_vanilla_donor.png", f"{a.out}/cdf_vanilla_donor.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    w, h = fig.get_size_inches()
    print(f"{w:.2f} x {h:.2f} in -> {png} ({w*300:.0f}x{h*300:.0f} px), {svg}")


if __name__ == "__main__":
    main()
