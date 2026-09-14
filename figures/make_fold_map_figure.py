"""US map of all 531 CAMELS basins, coloured by which of the 7 cross-validation folds each one
belongs to (see `assign_folds` in `src/data.py`) -- a study-design figure, not a results figure.

    python figures/make_fold_map_figure.py   # -> figures/fold_map.png

Colour: the project's validated 8-slot categorical palette, 7 of 8 slots in the fixed order (see
`CLAUDE.md`). Only the first 3 slots are validated for every simultaneous pairing; at 7 colours on
one scatter of 531 points, some pairs fall under the colourblind-safety floor. Disclosed, not
hidden -- the same WARN-not-fail precedent `make_cdf_figure.py` already sets for donor's teal.
Mitigated the same way too: identity never rests on colour alone (every fold is legend-named), and
a fold's basins are never adjacent to just one other fold's on the map, so a hard-to-separate pair
of hues rarely sits next to each other in practice.
"""
import argparse
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, MUTE = "#16191c", "#5d6a73"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
})


def load(data_dir=f"{ROOT}/data"):
    att = pd.read_csv(f"{data_dir}/camels531.csv", dtype={"gauge_id": str})
    folds = pd.read_csv(f"{data_dir}/basin_list_with_folds.csv", dtype={"basin": str})
    df = att.merge(folds, left_on="gauge_id", right_on="basin", validate="one_to_one")
    assert len(df) == 531, f"expected 531 basins, got {len(df)}"
    return df


def build(df, width=7.2, height=4.6):
    proj = ccrs.AlbersEqualArea(central_longitude=-96, central_latitude=37.5)
    fig = plt.figure(figsize=(width, height))
    ax = plt.axes(projection=proj)
    ax.set_extent([-125, -66, 24, 50])
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#f4f5f6", zorder=0)
    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor=MUTE, linewidth=0.4, zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=MUTE, linewidth=0.5, zorder=1)
    ax.spines["geo"].set_edgecolor(MUTE)
    ax.spines["geo"].set_linewidth(0.7)

    for fold in sorted(df.fold.unique()):
        sub = df[df.fold == fold]
        ax.scatter(sub.gauge_lon, sub.gauge_lat, transform=ccrs.PlateCarree(),
                  s=10, color=PALETTE[fold], edgecolor="white", linewidth=0.25,
                  label=f"fold {fold}  (n={len(sub)})", zorder=2)

    ax.set_title("531 CAMELS basins by cross-validation fold", fontsize=9.5, color=INK, pad=8)
    leg = ax.legend(loc="lower left", fontsize=6.8, frameon=False, labelcolor=INK,
                    handlelength=1.0, handletextpad=0.5, borderaxespad=0.6,
                    markerscale=1.4, ncol=1)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.03)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=HERE)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load()
    fig = build(df)
    png = f"{a.out}/fold_map.png"
    fig.savefig(png, dpi=300)
    w, h = fig.get_size_inches()
    print(f"{w:.2f} x {h:.2f} in -> {png} ({w*300:.0f}x{h*300:.0f} px)")


if __name__ == "__main__":
    main()
