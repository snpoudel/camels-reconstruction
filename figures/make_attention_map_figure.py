"""For the held-out basins where `donor`'s attention-driven improvement over `vanilla` was a real
win (not just a smaller loss), a US map of exactly what the model borrowed from and how much.

    python figures/make_attention_map_figure.py   # -> figures/attention_map_examples.png

Basin selection is computed here, not hardcoded: from `runs/metrics_ensemble.csv` (fold 0, the
only fold `vanilla` has been run on), rank held-out basins by `donor - vanilla` NSE, keep only
basins where donor's own score is positive (an improved-but-still-broken basin is not a "win"),
and take the top 3. Attention is the mean of the 3 `donor_fold0_seed{0,1,2}` runs, time-averaged
over the 20 evenly-spaced test-period snapshots `attention.npz` stores (there is no denser record
to show a season or event instead -- see `analysis/donor_diagnostics.py`, which this reuses).

Attention is a proper softmax distribution over the 455 gauged donors (sums to 1.0 per query,
asserted below), so weights are shown as % of that basin's total attention throughout -- a color
or a number here is directly "this donor got N% of what this basin borrowed."

Colour: sequential, one hue (the project's blue, `#2a78d6`) light-to-dark for the continuous %
weight -- not the categorical palette, and so no multi-colour CVD pairing to validate.
"""
import argparse
import glob
import json
import os
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, f"{ROOT}/analysis")
from donor_diagnostics import load as load_diag  # noqa: E402  (fold order + distance matrix)

FOLD = 0
N_BASINS = 3
INK, MUTE = "#16191c", "#5d6a73"
CMAP = LinearSegmentedColormap.from_list("blue_seq", ["#eef3fa", "#2a78d6"])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
})


def pick_basins(runs_dir=f"{ROOT}/runs"):
    df = pd.read_csv(f"{runs_dir}/metrics_ensemble.csv", dtype={"basin": str})
    v = df.query("arch == 'vanilla' and cell == 'ungauged/test'").set_index("basin").nse
    d = df.query("arch == 'donor' and cell == 'ungauged/test' and fold == @FOLD") \
          .set_index("basin").nse
    both = pd.DataFrame({"vanilla": v, "donor": d}).dropna()
    both["delta"] = both.donor - both.vanilla
    picked = both.query("donor > 0").sort_values("delta", ascending=False).head(N_BASINS)
    print(picked)
    return picked


def load_attention(d, runs_dir=f"{ROOT}/runs"):
    n = d["n_gauged"]
    mats = [np.load(f"{runs_dir}/donor_fold{FOLD}_seed{s}/attention.npz")["attn"].astype(np.float64)
            for s in (0, 1, 2)]
    A = np.mean(mats, axis=0)
    return A[:, n:, :].mean(0)  # [n_held_out, n_gauged], time-and-seed-averaged


def build(picked, W, d, att, width=4.6, height=9.6):
    """One column, one row per basin -- each map gets full width instead of a third of it.

    `layout="compressed"` is the fix for the dead space a fixed-aspect (cartopy) subplot grid
    otherwise leaves: it shrinks the figure to the actual rendered content instead of centering
    each map inside an evenly-divided, aspect-mismatched row.
    """
    n = d["n_gauged"]
    basins = d["basins"]
    idx = {b: i for i, b in enumerate(basins)}
    proj = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5)
    fig, axes = plt.subplots(N_BASINS, 1, figsize=(width, height),
                             subplot_kw={"projection": proj}, layout="compressed")

    donor_lon = att.loc[basins[:n], "gauge_lon"].to_numpy()
    donor_lat = att.loc[basins[:n], "gauge_lat"].to_numpy()

    vmax = max(100 * W[idx[b] - n].max() for b in picked.index)
    sca = None
    for ax, b in zip(axes, picked.index):
        qi = idx[b] - n
        w = W[qi]
        assert abs(w.sum() - 1.0) < 1e-3, f"{b}: attention row sums to {w.sum():.4f}, not 1.0"
        pct = 100 * w

        ax.set_extent([-125, -66, 24, 50])
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="#f6f7f7", zorder=0)
        ax.add_feature(cfeature.STATES.with_scale("110m"), edgecolor="#c7cbce", linewidth=0.3, zorder=1)
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=MUTE, linewidth=0.4, zorder=1)
        ax.spines["geo"].set_visible(False)

        order = np.argsort(pct)  # draw the biggest weights last, on top
        sca = ax.scatter(donor_lon[order], donor_lat[order], transform=ccrs.PlateCarree(),
                         c=pct[order], cmap=CMAP, vmin=0, vmax=vmax, s=14,
                         edgecolor="white", linewidth=0.15, zorder=2)
        ax.scatter([att.loc[b, "gauge_lon"]], [att.loc[b, "gauge_lat"]], transform=ccrs.PlateCarree(),
                  marker="*", s=140, color=INK, edgecolor="white", linewidth=0.5, zorder=3)

        ax.text(0.01, 0.04, f"{b}   vanilla {picked.loc[b, 'vanilla']:.2f} → donor "
               f"{picked.loc[b, 'donor']:.2f}", transform=ax.transAxes, fontsize=7.6,
               color=INK, va="bottom", ha="left",
               bbox=dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75))

    cbar = fig.colorbar(sca, ax=list(axes), location="bottom", shrink=0.45, pad=0.02, aspect=32)
    cbar.set_label("% of this basin's total attention", fontsize=7.2, color=INK, labelpad=2)
    cbar.ax.tick_params(labelsize=6.4, colors=MUTE, length=2, width=0.6)
    cbar.outline.set_edgecolor(MUTE)

    fig.suptitle("Where donor's real wins over vanilla came from (★ = held-out basin)",
                fontsize=8.6, color=INK)
    fig.get_layout_engine().set(hspace=0.02, h_pad=0.02)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=HERE)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    picked = pick_basins()
    d = load_diag(FOLD)
    W = load_attention(d)
    att = pd.read_csv(f"{ROOT}/data/camels531.csv", dtype={"gauge_id": str}).set_index("gauge_id")

    fig = build(picked, W, d, att)
    png = f"{a.out}/attention_map_examples.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    w, h = fig.get_size_inches()
    print(f"{w:.2f} x {h:.2f} in -> {png} ({w*300:.0f}x{h*300:.0f} px)")


if __name__ == "__main__":
    main()
