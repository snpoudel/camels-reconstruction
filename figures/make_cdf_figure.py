"""NSE and KGE CDFs, vanilla vs donor, on the ungauged (PUB) basins.

    python figures/make_cdf_figure.py   # -> figures/cdf_vanilla_donor.png

Two panels, same x-axis measure top and bottom: for each basin, the ensemble-mean
hydrograph's score (see make_results_figure.py, which this reuses `load()` from) is
one point; the curve is the empirical CDF across the held-out basins, so a curve
to the right of the other is winning at every quantile it's right of, not just on
average. `spatial` is left out here -- this figure is specifically the vanilla/donor
information comparison the project's `README.md` singles out.

`load()` pools every fold an architecture has been run on for this cell (see that
module's docstring) -- vanilla currently has only fold 0 (n=76), donor has folds 0-2
(n=228). The two curves are therefore not drawn from equal-sized samples; each legend
label carries its own n so this is never silently read as a matched comparison.

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
XLIM = (0.0, 1.0)  # zooms into the region that separates the two curves; a handful of
                    # basins score below 0 and are clipped off-plot (note this in the caption)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
})


def ecdf(v):
    v = np.sort(v[np.isfinite(v)])
    return v, np.arange(1, len(v) + 1) / len(v)


def build(data, n_basins, width=4.6, height=5.6):
    fig, axes = plt.subplots(2, 1, figsize=(width, height))

    for ax, (m, mlabel) in zip(axes, MEASURES):
        # The median (and sample size, since folds pool unevenly -- see module docstring)
        # goes in the legend label itself rather than as text on the plot -- one fewer
        # thing competing with the curves and dashed verticals for space, and no ambiguity
        # about which value, or how many basins, belongs to which arch.
        for arch in ARCHS:
            v = data[("ungauged/test", m, arch)]
            x, y = ecdf(v)
            med = float(np.median(v))
            n = n_basins[("ungauged/test", arch)]
            ax.step(np.clip(x, *XLIM), y, where="post", color=COLORS[arch], lw=1.6,
                    label=f"{arch}, n={n} (median = {med:.3f})", zorder=3)
            ax.plot([med, med], [0, 0.5], color=COLORS[arch], lw=0.7, ls=(0, (2, 2)), zorder=2)

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

        # Upper-left is empty in every panel (low scores start at fraction 0), so the
        # legend sits there instead of near the medians around x=0.7-0.8. Each panel
        # gets its own legend since the median differs by measure.
        ax.legend(loc="upper left", fontsize=7.2, frameon=False, labelcolor=INK,
                  handlelength=1.4, handletextpad=0.6)

    axes[1].set_xlabel("score", fontsize=7.4, color=INK)

    fig.suptitle("Out-of-sample (ungauged) basins", fontsize=9, color=INK, y=0.995)
    fig.subplots_adjust(left=0.135, right=0.97, top=0.91, bottom=0.09, hspace=0.35)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=HERE)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data, n_basins = load(dump=None)  # metrics_ensemble.csv already written by make_results_figure.py
    fig = build(data, n_basins)
    png = f"{a.out}/cdf_vanilla_donor.png"
    fig.savefig(png, dpi=300)
    w, h = fig.get_size_inches()
    print(f"{w:.2f} x {h:.2f} in -> {png} ({w*300:.0f}x{h*300:.0f} px)")


if __name__ == "__main__":
    main()
