"""The one palette and figure style every figure here uses.

Colours are the Okabe-Ito colourblind-safe qualitative palette (Okabe & Ito, 2008), used as
published rather than designed here -- it is the standard set for scientific figures and can
simply be cited. Slot order is fixed and never cycled, so a series keeps its colour across
figures:

    slot 0  #0072B2  blue        the uncorrected model ("LSTM")
    slot 1  #D55E00  vermillion  the same model after the neighbour-residual correction
    slot 2  #949494  grey        the parameter-free nearest-gauge reference (CDF figure only)

Blue and vermillion are the two series any figure compares. Grey is the reference line rather
than a third series, and reads that way: a baseline the eye passes over on its way to the two
that matter.

Ink, not colour, carries text:  INK #16191c   MUTE #5d6a73   GRID #dfe3e6

Identity never rests on colour alone in any case: every series is named in its legend, and the
figures separate their series by position, marker size or dash as well.

Everything else here exists so the figures agree with each other: one type scale, one axis
treatment, one caption block, one pair of series names, one pair of cell titles. Import them
rather than re-deciding per figure.
"""
import json
import os
import textwrap

import matplotlib.colors as mcolors
import numpy as np

BLUE, ORANGE, NEUTRAL = "#0072B2", "#D55E00", "#949494"
INK, MUTE, GRID = "#16191c", "#5d6a73", "#dfe3e6"

# The signed-quantity ramp both map figures use. A visibly grey zero, not the near-white GRID:
# that reads as "empty" on a map, and a marker filled with it is nearly invisible against the
# page. Same two hues as the series pair, so a vermillion basin carries the same meaning here
# as an vermillion line does anywhere else.
DIVERGE_ZERO = "#b6bec4"
DIVERGING = mcolors.LinearSegmentedColormap.from_list("diverging", [BLUE, DIVERGE_ZERO, ORANGE])

# Series identity, used across every figure. The names are the `variant` values in
# figures/metrics_*.csv, so a figure never invents its own wording for a series.
RAW, CORRECTED = BLUE, ORANGE
SERIES = {"LSTM": RAW, "LSTM + correction": CORRECTED}

# The parameter-free reference -- one donor, all of its error, nothing selected. Only the CDF
# figure draws it, so `SERIES` above stays a pair and the figures that unpack it keep working.
BENCHMARK = "nearest gauge only"
SERIES_CDF = {"LSTM": RAW, BENCHMARK: NEUTRAL, "LSTM + correction": CORRECTED}

CELL_TITLE = {"gauged": "Gauged basins", "ungauged": "Ungauged basins"}
CELL_BLURB = {"gauged": "each basin reconstructed as if its own gauge had gone dark",
              "ungauged": "basins held out of training entirely"}

# One type scale, in points.
TITLE, PANEL, LABEL, TICK, CAPTION = 11.5, 10.5, 10, 9, 8.5


def style(ax, grid=True):
    """Recessive grid and axes: the data carries the ink, the frame does not.

    `grid=False` for the map figures -- lat/lon gridlines compete with the state outline and
    the data points for the same thin grey ink; the axis ticks alone are enough to read
    coordinates off. Tick marks are unaffected either way.
    """
    # Two calls, not one with `visible=grid`: matplotlib re-enables the grid whenever any
    # line-property kwarg is passed alongside visible=False, regardless of its value.
    if grid:
        ax.grid(True, color=GRID, lw=0.7, zorder=0)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelsize=TICK, length=3)


# CONUS + DC state boundaries, trimmed from a public-domain simplified GeoJSON
# (PublicaMundi/MappingAPI) to the 49 features CAMELS-US basins ever fall inside -- no Alaska,
# Hawaii or Puerto Rico. 59 KB, tracked alongside data/camels531.csv rather than fetched at
# plot time, so a figure never depends on network access.
US_STATES_GEOJSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "us_states_conus.geojson")


def draw_us_outline(ax, color=GRID, lw=0.7, zorder=1):
    """State boundaries, drawn dim and behind the data -- geographic context, not a series.

    Reads GeoJSON with the standard library only (no geopandas/shapely dependency for one
    background layer); handles the Polygon and MultiPolygon geometries the states need.
    """
    states = json.load(open(US_STATES_GEOJSON))
    for feature in states["features"]:
        geom = feature["geometry"]
        rings = geom["coordinates"] if geom["type"] == "Polygon" else \
            [ring for poly in geom["coordinates"] for ring in poly]
        for ring in rings:
            xy = np.asarray(ring)
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, zorder=zorder)


def caption(fig, text, width=120):
    """The caption block, in the same place at the same size on every figure. Returns the
    bottom margin to leave for it, for the caller's `tight_layout(rect=...)`."""
    lines = textwrap.wrap(text, width)
    fig.text(0.012, 0.014, textwrap.fill(text, width), fontsize=CAPTION, color=MUTE, va="bottom")
    return 0.035 + 0.021 * len(lines)


def settings_note(params):
    """How a caption states the selected settings: spelled out when there are few, summarised
    when every fold has its own."""
    if len(params) < 4:
        return "; ".join(rf"fold {f}: K={int(k)}, $\lambda$={lam:g}, p={p:g}"
                         for f, k, lam, p in params)
    return r"K, $\lambda$ and p selected per fold"
