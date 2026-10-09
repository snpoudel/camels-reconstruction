"""The shared palette and figure style. Import from here rather than re-deciding per figure.

Series colours are the Okabe-Ito colourblind-safe palette (Okabe & Ito, 2008), in fixed slots:

    slot 0  #0072B2  blue        the uncorrected model
    slot 1  #D55E00  vermillion  the same model after the correction
    slot 2  #949494  grey        the nearest-gauge-only reference (CDF figure only)
    slot 3  #009E73  green       the same model after the learned MLP correction

Text uses ink, not series colour: INK #16191c, MUTE #5d6a73, GRID #dfe3e6.
"""
import json
import os

import matplotlib.colors as mcolors
import numpy as np

BLUE, ORANGE, NEUTRAL = "#0072B2", "#D55E00", "#949494"
INK, MUTE, GRID = "#16191c", "#5d6a73", "#dfe3e6"

# Signed change on a map. The zero is a visible grey, not GRID: a near-white marker reads as
# "missing" rather than "unchanged".
DIVERGE_ZERO = "#b6bec4"
DIVERGING = mcolors.LinearSegmentedColormap.from_list("diverging", [BLUE, DIVERGE_ZERO, ORANGE])
# DIVERGING saturates at this percentile of |change|, so the typical basin reaches full colour
# instead of the ramp being spent on a few extreme near-zero-flow basins.
CAP_PERCENTILE = 85

# Unsigned skill on a map, darker = better. Stepped at equal OKLCH lightness in the series
# blue's hue; stock "Blues" bunches its dark end, which made NSE 0.7-0.9 look identical.
SEQUENTIAL = mcolors.LinearSegmentedColormap.from_list("sequential", [
    "#d4ebff", "#acd9ff", "#81c7ff", "#4eb3ff", "#239ff0", "#038bd8",
    "#0278bb", "#00659f", "#005383", "#004169", "#0a314b"])
# The same steps in purple (OKLCH hue 305), for skill shown beside a DIVERGING panel, where blue
# already means "got worse".
SEQUENTIAL_ALT = mcolors.LinearSegmentedColormap.from_list("sequential_alt", [
    "#ede2fb", "#dccaf4", "#ccb2ec", "#bb99e3", "#aa82d8", "#996bca",
    "#8757b7", "#7446a1", "#603787", "#4c2b6b", "#38204f"])

RAW, CORRECTED = BLUE, ORANGE
# Checked against RAW and CORRECTED under simulated protan/deutan/tritan vision: every pair stays
# >= 8.6 apart (OKLab delta E x 100), the closest being RAW vs LEARNED under tritan.
LEARNED = "#009E73"

# Display names for runs/<model>/; anything missing falls back to the capitalised key.
MODEL_LABEL = {"lstm": "LSTM", "hbv": "HBV"}

BENCHMARK = "nearest gauge only"

CELL_TITLE = {"gauged": "Gauged basins", "ungauged": "Ungauged basins"}

TITLE, PANEL, LABEL, TICK = 11.5, 10.5, 10, 9

# CONUS + DC state outlines, trimmed from a public-domain GeoJSON (PublicaMundi/MappingAPI) and
# tracked in data/ so no figure needs network access.
US_STATES_GEOJSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "us_states_conus.geojson")


def figure_variant():
    """(method suffix, output folder) for an LSTM figure run: ("", "lstm") for the reported
    method, ("_noscale", "lstm_noscale") with --variant noscale -- the comparison whose residuals
    are not divided by each basin's flow scale. Read from the command line, so every module a
    figure script imports agrees on it."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["noscale"])
    v = ap.parse_known_args()[0].variant
    return (f"_{v}", f"lstm_{v}") if v else ("", "lstm")


def model_label(model):
    return MODEL_LABEL.get(model, model.capitalize())


def series_pair(model="lstm"):
    """{name: colour} for the raw and corrected series, in that order. The names are the
    `variant` values in figures/metrics_*.csv."""
    name = model_label(model)
    return {name: RAW, f"{name} + correction": CORRECTED}


def series_cdf(model="lstm"):
    """`series_pair` plus the nearest-gauge-only reference."""
    name = model_label(model)
    return {name: RAW, BENCHMARK: NEUTRAL, f"{name} + correction": CORRECTED}


def available_cells(model, method="idw"):
    """The cells this model has metrics for, gauged first. HBV has no ungauged run."""
    here = os.path.dirname(os.path.abspath(__file__))
    return [c for c in ("gauged", "ungauged")
            if os.path.exists(f"{here}/metrics_{c}_{model}_{method}.csv")]


def panel_letter(i):
    return f"({'abcdefghij'[i]})"


def style(ax, grid=True):
    """Recessive axes; `grid=False` for maps, where gridlines compete with the state outlines."""
    # Two calls, not `ax.grid(grid, ...)`: matplotlib turns the grid back on whenever a
    # line-property kwarg is passed, even alongside visible=False.
    if grid:
        ax.grid(True, color=GRID, lw=0.6, alpha=0.6, zorder=0)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelsize=TICK, length=3)


def draw_us_outline(ax, color=GRID, lw=0.7, zorder=1):
    """State boundaries behind the data, read with the standard library (no geopandas)."""
    states = json.load(open(US_STATES_GEOJSON))
    for feature in states["features"]:
        geom = feature["geometry"]
        rings = geom["coordinates"] if geom["type"] == "Polygon" else \
            [ring for poly in geom["coordinates"] for ring in poly]
        for ring in rings:
            xy = np.asarray(ring)
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, zorder=zorder)
