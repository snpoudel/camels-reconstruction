"""Shared colours and figure style for every figure script.

Series colours are Okabe-Ito (Okabe & Ito, 2008): blue for the uncorrected model, vermillion
after the IDW correction, green after the MLP correction, grey for the nearest-gauge reference.
"""
import json
import os

import matplotlib.colors as mcolors
import numpy as np

BLUE, ORANGE, NEUTRAL = "#0072B2", "#D55E00", "#949494"
INK, MUTE, GRID = "#16191c", "#5d6a73", "#dfe3e6"

# Zero is a visible grey, since a near-white marker reads as "missing" rather than "unchanged".
DIVERGE_ZERO = "#b6bec4"
DIVERGING = mcolors.LinearSegmentedColormap.from_list("diverging", [BLUE, DIVERGE_ZERO, ORANGE])
# Saturate at this percentile of |change| so a few extreme basins do not use up the ramp.
CAP_PERCENTILE = 85

# Equal OKLCH lightness steps in the series blue's hue, darker = better.
SEQUENTIAL = mcolors.LinearSegmentedColormap.from_list("sequential", [
    "#d4ebff", "#acd9ff", "#81c7ff", "#4eb3ff", "#239ff0", "#038bd8",
    "#0278bb", "#00659f", "#005383", "#004169", "#0a314b"])
# The same steps in purple, for skill beside a DIVERGING panel where blue means "got worse".
SEQUENTIAL_ALT = mcolors.LinearSegmentedColormap.from_list("sequential_alt", [
    "#ede2fb", "#dccaf4", "#ccb2ec", "#bb99e3", "#aa82d8", "#996bca",
    "#8757b7", "#7446a1", "#603787", "#4c2b6b", "#38204f"])

RAW, CORRECTED = BLUE, ORANGE
LEARNED = "#009E73"

MODEL_LABEL = {"lstm": "LSTM", "hbv": "HBV"}

BENCHMARK = "nearest gauge only"

CELL_TITLE = {"gauged": "Gauged basins", "ungauged": "Ungauged basins"}

TITLE, PANEL, LABEL, TICK = 11.5, 10.5, 10, 9

# CONUS state outlines from a public-domain GeoJSON (PublicaMundi/MappingAPI).
US_STATES_GEOJSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "us_states_conus.geojson")


def figure_variant():
    """(method suffix, output folder), ("_noscale", "lstm_noscale") under --variant noscale."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["noscale"])
    v = ap.parse_known_args()[0].variant
    return (f"_{v}", f"lstm_{v}") if v else ("", "lstm")


def model_label(model):
    return MODEL_LABEL.get(model, model.capitalize())


def series_pair(model="lstm"):
    """{variant name in metrics_*.csv: colour} for the raw and corrected series."""
    name = model_label(model)
    return {name: RAW, f"{name} + correction": CORRECTED}


def series_cdf(model="lstm"):
    """`series_pair` plus the nearest-gauge-only reference."""
    name = model_label(model)
    return {name: RAW, BENCHMARK: NEUTRAL, f"{name} + correction": CORRECTED}


def available_cells(model, method="idw"):
    """The cells this model has metrics for, gauged first."""
    here = os.path.dirname(os.path.abspath(__file__))
    return [c for c in ("gauged", "ungauged")
            if os.path.exists(f"{here}/metrics_{c}_{model}_{method}.csv")]


def panel_letter(i):
    return f"({'abcdefghij'[i]})"


def style(ax, grid=True):
    """Recessive axes; `grid=False` for maps, where gridlines compete with the state outlines."""
    # Not ax.grid(grid, ...): matplotlib re-enables the grid whenever line kwargs are passed.
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
    """State boundaries drawn behind the data."""
    states = json.load(open(US_STATES_GEOJSON))
    for feature in states["features"]:
        geom = feature["geometry"]
        rings = geom["coordinates"] if geom["type"] == "Polygon" else \
            [ring for poly in geom["coordinates"] for ring in poly]
        for ring in rings:
            xy = np.asarray(ring)
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, zorder=zorder)
