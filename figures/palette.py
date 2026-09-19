"""The one palette and figure style every figure here uses, and the check that it is safe.

Slot order is fixed and never cycled, so a series keeps its colour across figures:

    slot 0  #2a78d6  blue    the uncorrected model ("LSTM")
    slot 1  #eb6834  orange  the same model after the neighbour-residual correction
    slot 2  #1baf7a  teal    reserved -- a third series, if a figure ever needs one

Ink, not colour, carries text:  INK #16191c   MUTE #5d6a73   GRID #dfe3e6

Everything else here exists so the figures agree with each other: one type scale, one axis
treatment, one caption block, one pair of series names, one pair of cell titles. Import them
rather than re-deciding per figure.

Run this file to re-validate:  python figures/palette.py
It reports OKLab dE (x100) between every simultaneous pair under normal vision and under
simulated protanopia and deuteranopia (Vienot et al. 1999). Normal-vision dE >= 15 and CVD
dE >= 8 pass; 6-8 is legal only with a second, non-colour encoding; below that, re-step.
"""
import textwrap

import numpy as np

BLUE, ORANGE, TEAL = "#2a78d6", "#eb6834", "#1baf7a"
COLORS = [BLUE, ORANGE, TEAL]
INK, MUTE, GRID = "#16191c", "#5d6a73", "#dfe3e6"

# Series identity, used across every figure. The names are the `variant` values in
# figures/metrics_*.csv, so a figure never invents its own wording for a series.
RAW, CORRECTED = BLUE, ORANGE
SERIES = {"LSTM": RAW, "LSTM + correction": CORRECTED}

CELL_TITLE = {"gauged": "Gauged basins", "ungauged": "Ungauged basins"}
CELL_BLURB = {"gauged": "each basin reconstructed as if its own gauge had gone dark",
              "ungauged": "basins held out of training entirely"}

# One type scale, in points.
TITLE, PANEL, LABEL, TICK, CAPTION = 11.5, 10.5, 10, 9, 8.5


def style(ax):
    """Recessive grid and axes: the data carries the ink, the frame does not."""
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTE, labelsize=TICK, length=3)


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


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _hex_to_linear(h):
    rgb = np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return _srgb_to_linear(rgb)


def _oklab(lin):
    m = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                  [0.2119034982, 0.6806995451, 0.1073969566],
                  [0.0883024619, 0.2817188376, 0.6299787005]])
    lms = np.cbrt(m @ lin)
    n = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                  [1.9779984951, -2.4285922050, 0.4505937099],
                  [0.0259040371, 0.7827717662, -0.8086757660]])
    return n @ lms


# Vienot, Brettel & Mollon (1999) dichromat simulation, in linear sRGB.
_TO_LMS = np.array([[17.8824, 43.5161, 4.11935],
                    [3.45565, 27.1554, 3.86714],
                    [0.0299566, 0.184309, 1.46709]])
_PROTAN = np.array([[0, 2.02344, -2.52581], [0, 1, 0], [0, 0, 1]])
_DEUTAN = np.array([[1, 0, 0], [0.494207, 0, 1.24827], [0, 0, 1]])


def _simulate(lin, kind):
    lms = _TO_LMS @ lin
    lms = (_PROTAN if kind == "protan" else _DEUTAN) @ lms
    return np.linalg.solve(_TO_LMS, lms)


def delta_e(h1, h2, vision="normal"):
    a, b = _hex_to_linear(h1), _hex_to_linear(h2)
    if vision != "normal":
        a, b = _simulate(a, vision), _simulate(b, vision)
    return float(np.linalg.norm(_oklab(a) - _oklab(b)) * 100)


def report(colors=(BLUE, ORANGE), names=("LSTM", "corrected")):
    print(f"pairs among {len(colors)} simultaneous series")
    worst = {}
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            row = {v: delta_e(colors[i], colors[j], v) for v in ("normal", "protan", "deutan")}
            print(f"  {names[i]:>10} vs {names[j]:<10} "
                  + "  ".join(f"{v} dE {d:5.1f}" for v, d in row.items()))
            for v, d in row.items():
                worst[v] = min(worst.get(v, 1e9), d)
    ok_norm, ok_cvd = worst["normal"] >= 15, min(worst["protan"], worst["deutan"]) >= 8
    print(f"  normal-vision floor {worst['normal']:.1f} (need >=15): {'PASS' if ok_norm else 'FAIL'}")
    print(f"  CVD floor           {min(worst['protan'], worst['deutan']):.1f} (need >=8): "
          f"{'PASS' if ok_cvd else 'FAIL'}")
    return ok_norm and ok_cvd


if __name__ == "__main__":
    two = report()
    print()
    three = report(COLORS, ("LSTM", "corrected", "reserved"))
    print(f"\noverall: {'PASS' if two and three else 'REVIEW'}")
