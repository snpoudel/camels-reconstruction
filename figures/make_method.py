"""The neighbour-residual correction: five steps, each with its equation and a picture.

    python figures/make_method.py    # -> figures/method.png, figures/method.svg

Every step box carries the arithmetic on top and a sketch of the same neighbourhood underneath,
so the notation can be read straight off the picture: step 1 names the basins (q_1, q_2, q_j at
the gauges, q_i at the ungauged target), step 2 turns distance into weight on those same points,
step 3 puts today's error at each gauge, step 4 pools them onto the target, step 5 shows the
shift that comes out. The neighbourhood is the same one throughout -- that is the point, the
eye tracks a single target and its donors the whole way across. Step 1 is the one panel that
draws every gauge alike: nothing has been selected yet, and singling the donors out there
would give away a choice step 2 has not made.

The band above the five steps says where the three constants come from: the seven folds drawn
as seven cells -- one ungauged (held out), six gauged (the donor pool) -- and the one line of
text that K, p and lambda are cross-validated on those gauged folds. The cell colours are the
figure's own: blue is a gauged basin in every step panel, orange is the ungauged target.

Nothing here is measured. It is a method diagram: the sketches are schematic (step 5's
hydrograph is an analytic curve, not a run), and the only numbers on the page -- the search
grid for K, p and lambda -- are imported from `src/parameter_tuning.py` so they cannot drift
from the code. For the same method on real data see `figures/ungauged_hyperparameters.png`.

No footnotes: the distance floor, the epsilon that keeps the logs finite, and the
renormalisation over donors that reported are all in manuscript.md and belong in the caption.
Carrying them on the figure made it busier than the five steps it exists to show.

`check_fit` measures every string against the box it sits in and prints a WARN if one spills,
because the failure mode of this figure is silent: an equation grows, overruns its border, and
nothing complains. Keep the run clean.

Colour: `figures/palette.py` -- slot-0 blue for the uncorrected LSTM and for gauged basins,
slot-1 orange for the ungauged target and the corrected prediction, matching what
`make_cdf.py` and `make_boxplot.py` use for "LSTM" vs "LSTM + correction".
Two hues only; ink and grey carry everything else.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

# Glyphs as outlines, so the .svg renders identically in Illustrator, Inkscape or a LaTeX
# pipeline that has never heard of DejaVu Sans.
matplotlib.rcParams["svg.fonttype"] = "path"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from palette import CORRECTED, GRID, INK, MUTE, RAW  # noqa: E402
from parameter_tuning import K_GRID, LAM_GRID, P_GRID  # noqa: E402

FIG_W, FIG_H = 11.0, 5.05
MARGIN, GAP = 0.32, 0.34
BOX_H, Y0 = 3.40, 0.28
BOX_W = (FIG_W - 2 * MARGIN - 4 * GAP) / 5

# The cross-validation band above the steps: seven fold cells, then one line of text. Sized so
# the five step boxes keep their own geometry -- the room comes from dropping the footnotes.
CV_Y, CV_H = 3.82, 0.66
CELL_W, CELL_H, CELL_GAP = 0.26, 0.24, 0.045
SPLIT_GAP = 0.16          # separates the one ungauged cell from the six gauged ones

# One neighbourhood, drawn five times, in unit coordinates inside each sketch panel. Three
# constraints on these positions: every donor is nearer the target than every other gauge, so
# "the K nearest" is visible rather than asserted; no two donors are collinear with the target,
# so the three weight lines in step 2 read as three; and the donor distances differ enough that
# those lines come out visibly unequal.
TARGET = (0.44, 0.46)
DONORS = ((0.22, 0.70), (0.56, 0.62), (0.78, 0.40))
OTHERS = ((0.06, 0.14), (0.94, 0.84), (0.32, 0.06))
Q_LAB = (r"$q_1$", r"$q_2$", r"$q_j$")
W_LAB = (r"$w_{i1}$", r"$w_{i2}$", r"$w_{ij}$")
R_LAB = (r"$r_1$", r"$r_2$", r"$r_j$")
# Offset, alignment and (for step 2) where along the line the weight label sits -- fixed per
# donor so no label jumps between panels or lands on top of its neighbour.
OFF = (((0.00, 0.10), "center", "bottom", 0.68),
       ((0.07, 0.05), "left", "bottom", 0.40),
       ((0.08, 0.00), "left", "center", 0.70))
LAB_FS = 7.8

STEPS = (
    dict(title="LSTM prediction", accent=RAW, eq_size=10.5,
         eq=r"$\mathrm{LSTM}\rightarrow q^{sim}_i,\; q^{sim}_j$",
         note="one model predicts every\nbasin, gauged or ungauged"),
    dict(title="Spatial weights", accent=None, eq_size=12,
         eq=r"$w_{ij}=\dfrac{d_{ij}^{-p}}{\sum_{j'} d_{ij'}^{-p}}$",
         note="the K nearest gauges,\nweighted by distance"),
    dict(title="Donor errors today", accent=None, eq_size=11.5,
         eq=r"$r_j=\log\dfrac{q^{sim}_j+\epsilon}{q^{obs}_j+\epsilon}$",
         note="how wrong the model is\nat each gauge, in logs"),
    dict(title="Neighbour error", accent=None, eq_size=12.5,
         eq=r"$R_i=\sum_j w_{ij}\,r_j$",
         note="their weighted mean:\nthe error around basin i"),
    dict(title="Correct the target", accent=CORRECTED, eq_size=10.5,
         eq=(r"$q^{corr}_i=(q^{sim}_i+\epsilon)$", r"$\times\; e^{-\lambda R_i}-\epsilon$"),
         note="shift the prediction by\na fraction $\\lambda$ of it"),
)

_fit = []          # (artist, x_lo, x_hi) -- checked once at the end, see check_fit


def track(artist, x_lo, x_hi):
    _fit.append((artist, x_lo, x_hi))
    return artist


def check_fit(fig, ax):
    """Print a WARN for any string wider than the box it was drawn in."""
    fig.canvas.draw()
    inv, rend, bad = ax.transData.inverted(), fig.canvas.get_renderer(), 0
    for artist, x_lo, x_hi in _fit:
        bb = artist.get_window_extent(renderer=rend).transformed(inv)
        if bb.x0 < x_lo - 1e-3 or bb.x1 > x_hi + 1e-3:
            over = max(x_lo - bb.x0, bb.x1 - x_hi)
            print(f"  WARN overflows by {over:.3f} in: {artist.get_text()[:48]!r}")
            bad += 1
    return bad


def weights():
    """Step-2 weights for the sketch, from the drawn geometry -- p = 1, normalised."""
    inv = np.array([1.0 / np.hypot(u - TARGET[0], v - TARGET[1]) for u, v in DONORS])
    return inv / inv.sum()


def pt(rect, u, v):
    x0, y0, w, h = rect
    return x0 + u * w, y0 + v * h


def basins(ax, rect, mark_donors=True):
    """The same map in every panel: three donors, three farther gauges, one ungauged target.

    `mark_donors=False` draws every gauge alike -- step 1, where the LSTM has simply predicted
    everywhere and no donor has been picked yet. The K nearest are chosen in step 2, so that is
    the first panel allowed to single them out.
    """
    faint = OTHERS if mark_donors else ()
    strong = DONORS if mark_donors else OTHERS + DONORS
    for u, v in faint:
        ax.scatter(*pt(rect, u, v), s=15, color=RAW, alpha=0.28, linewidths=0, zorder=4)
    for u, v in strong:
        ax.scatter(*pt(rect, u, v), s=38, color=RAW, edgecolor="white", linewidth=0.7, zorder=5)
    tx, ty = pt(rect, *TARGET)
    ax.scatter([tx], [ty], s=140, marker="*", color=CORRECTED, edgecolor="white",
               linewidth=0.7, zorder=6)
    return tx, ty


def label_donors(ax, rect, labels):
    for (u, v), lab, (off, ha, va, _) in zip(DONORS, labels, OFF):
        ax.text(*pt(rect, u + off[0], v + off[1]), lab, fontsize=LAB_FS, color=INK,
                ha=ha, va=va, zorder=7)


def hydrograph(ax, rect):
    """Schematic, not data: one analytic storm, before and after a downward correction."""
    x0, y0, w, h = rect
    px0, px1 = x0 + 0.10 * w, x0 + 0.96 * w
    py0, py1 = y0 + 0.16 * h, y0 + 0.72 * h
    t = np.linspace(0, 1, 240)
    q = (0.18 * np.exp(-2.6 * t) + np.exp(-((t - 0.40) / 0.085) ** 2)
         + 0.42 * np.exp(-((t - 0.72) / 0.14) ** 2) + 0.10)
    q /= q.max()
    x = px0 + t * (px1 - px0)
    raw, cor = py0 + q * (py1 - py0), py0 + 0.66 * q * (py1 - py0)
    ax.fill_between(x, cor, raw, color=CORRECTED, alpha=0.14, linewidth=0, zorder=4)
    ax.plot(x, raw, color=RAW, lw=1.3, zorder=5)
    ax.plot(x, cor, color=CORRECTED, lw=1.3, zorder=6)
    ax.plot([px0, px1], [py0, py0], color=MUTE, lw=0.6, alpha=0.6, zorder=3)
    # Mini legend, top-left, clear of the peak: labelling the curves in place needs leader
    # lines at this size, and leader lines at this size are clutter.
    for n, (lab, col) in enumerate(((r"$q^{sim}_i$", RAW), (r"$q^{corr}_i$", CORRECTED))):
        ly = y0 + (0.94 - 0.13 * n) * h
        ax.plot([x0 + 0.08 * w, x0 + 0.20 * w], [ly, ly], color=col, lw=1.3, zorder=7)
        ax.text(x0 + 0.24 * w, ly, lab, fontsize=LAB_FS, color=col, ha="left", va="center",
                zorder=7)
    ax.text(px1, py0 - 0.05 * h, "time", fontsize=6.6, color=MUTE, ha="right", va="top",
            zorder=7)


def sketch(ax, rect, i):
    x0, y0, w, h = rect
    # Above the step box's own white fill (zorder 2), or it is painted over and vanishes.
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                linewidth=0, facecolor=GRID, alpha=0.75, zorder=2.5))
    if i == 4:
        hydrograph(ax, rect)
        return

    tx, ty = basins(ax, rect, mark_donors=i != 0)
    if i == 0:
        label_donors(ax, rect, Q_LAB)
        ax.text(*pt(rect, TARGET[0] - 0.08, TARGET[1] - 0.02), r"$q_i$", fontsize=LAB_FS,
                color=CORRECTED, ha="right", va="center", zorder=7)
    elif i == 1:
        for (u, v), lab, wj, (_, _, _, at) in zip(DONORS, W_LAB, weights(), OFF):
            px, py = pt(rect, u, v)
            ax.plot([tx, px], [ty, py], color=MUTE, lw=0.6 + 4.5 * wj, alpha=0.7,
                    solid_capstyle="round", zorder=3)
            ax.text(*pt(rect, TARGET[0] + at * (u - TARGET[0]), TARGET[1] + at * (v - TARGET[1])),
                    lab, fontsize=LAB_FS, color=INK, ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.92))
    elif i == 2:
        label_donors(ax, rect, R_LAB)
    elif i == 3:
        for (u, v), wj in zip(DONORS, weights()):
            ax.add_patch(FancyArrowPatch(pt(rect, u, v), (tx, ty), arrowstyle="-|>",
                                         mutation_scale=7, linewidth=0.6 + 3.5 * wj,
                                         color=MUTE, alpha=0.8, shrinkA=3, shrinkB=8, zorder=3))
        ax.text(*pt(rect, TARGET[0], TARGET[1] - 0.16), r"$R_i$", fontsize=LAB_FS + 0.6,
                color=CORRECTED, ha="center", va="top", zorder=7)


def add_box(ax, x, y, step, n):
    ax.add_patch(FancyBboxPatch((x, y), BOX_W, BOX_H, boxstyle="round,pad=0,rounding_size=0.07",
                                linewidth=1.6 if step["accent"] else 1.0,
                                edgecolor=step["accent"] or MUTE, facecolor="white", zorder=2))
    cx, lo, hi = x + BOX_W / 2, x + 0.07, x + BOX_W - 0.07
    track(ax.text(cx, y + BOX_H - 0.22, step["title"], ha="center", va="top", fontsize=10.0,
                  color=INK, fontweight="bold", zorder=4), lo, hi)
    eq = step["eq"] if isinstance(step["eq"], tuple) else (step["eq"],)
    for k, line in enumerate(eq):
        track(ax.text(cx, y + BOX_H - (0.78 if len(eq) == 1 else 0.66 + 0.30 * k), line,
                      ha="center", va="center", fontsize=step["eq_size"], color=INK, zorder=4),
              lo, hi)
    track(ax.text(cx, y + BOX_H - 1.14, step["note"], ha="center", va="top", fontsize=8.0,
                  color=MUTE, linespacing=1.5, zorder=4), lo, hi)
    sketch(ax, (x + 0.09, y + 0.12, BOX_W - 0.18, BOX_H - 1.74), n - 1)
    ax.add_patch(Circle((x + 0.02, y + BOX_H - 0.02), 0.145, facecolor=step["accent"] or INK,
                        edgecolor="white", linewidth=1.1, zorder=6))
    ax.text(x + 0.02, y + BOX_H - 0.02, str(n), ha="center", va="center", fontsize=8.5,
            color="white", fontweight="bold", zorder=7)


def cv_band(ax):
    """Where K, p and lambda come from: seven fold cells, an arrow, and one line of text.

    The ungauged fold is drawn first and set apart by a wider gap, so the six that follow read
    as the pool the search is allowed to use. Colours are the figure's own -- orange is the
    ungauged target everywhere else on the page, blue is a gauged basin -- so the band needs no
    legend of its own.
    """
    x0, w = MARGIN, FIG_W - 2 * MARGIN
    ax.add_patch(FancyBboxPatch((x0, CV_Y), w, CV_H,
                                boxstyle="round,pad=0,rounding_size=0.08", linewidth=1.0,
                                edgecolor=MUTE, facecolor=GRID, alpha=0.45, zorder=2))
    cy = CV_Y + CV_H - 0.10 - CELL_H / 2      # cell row, sitting above its labels
    cell_y = cy - CELL_H / 2

    ungauged_x = x0 + 0.16
    gauged_x = ungauged_x + CELL_W + SPLIT_GAP
    for n in range(7):
        held = n == 0
        cx = ungauged_x if held else gauged_x + (n - 1) * (CELL_W + CELL_GAP)
        colour = CORRECTED if held else RAW
        ax.add_patch(FancyBboxPatch((cx, cell_y), CELL_W, CELL_H,
                                    boxstyle="round,pad=0,rounding_size=0.03", linewidth=1.1,
                                    edgecolor=colour, facecolor=colour, alpha=0.30, zorder=4))
    gauged_end = gauged_x + 6 * CELL_W + 5 * CELL_GAP

    # Two lines each: "ungauged (target)" on one line is three times its cell's width, and
    # widening the split to fit it would spend exactly the room the sentence needs.
    lab_y = cell_y - 0.05
    track(ax.text(ungauged_x + CELL_W / 2, lab_y, "ungauged\n(target)", fontsize=7.4,
                  color=CORRECTED, ha="center", va="top", linespacing=1.25, zorder=5),
          x0, gauged_x)
    track(ax.text((gauged_x + gauged_end) / 2, lab_y, "gauged\n(donors)", fontsize=7.4,
                  color=RAW, ha="center", va="top", linespacing=1.25, zorder=5),
          gauged_x, gauged_end)

    arr_x0, arr_x1 = gauged_end + 0.15, gauged_end + 0.58
    ax.add_patch(FancyArrowPatch((arr_x0, cy), (arr_x1, cy), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.2, color=MUTE, zorder=5))

    text_x = arr_x1 + 0.16
    track(ax.text(text_x, cy,
                  f"Select $K$ (donor gauges, {K_GRID[0]}–{K_GRID[-1]}), "
                  f"$p$ (IDW exponent, {P_GRID[0]:g}–{P_GRID[-1]:g}) and "
                  f"$\\lambda$ (error shrinkage, {LAM_GRID[0]:g}–{LAM_GRID[-1]:g}) "
                  f"by cross-validation on gauged folds",
                  fontsize=8.4, color=INK, ha="left", va="center", zorder=5),
          text_x, x0 + w - 0.12)


def main(out=None):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    track(ax.text(MARGIN, FIG_H - 0.11,
                  "Correcting a target ungauged basin using donor basins' errors",
                  fontsize=13, color=INK, fontweight="bold", ha="left", va="top"),
          MARGIN, FIG_W - MARGIN)

    cv_band(ax)

    xs = [MARGIN + i * (BOX_W + GAP) for i in range(5)]
    for i, step in enumerate(STEPS):
        add_box(ax, xs[i], Y0, step, i + 1)
        if i:
            ax.add_patch(FancyArrowPatch((xs[i - 1] + BOX_W, Y0 + BOX_H * 0.75),
                                         (xs[i], Y0 + BOX_H * 0.75), arrowstyle="-|>",
                                         mutation_scale=11, linewidth=1.2, color=MUTE, zorder=5))

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    bad = check_fit(fig, ax)
    stem = out or f"{HERE}/method"
    fig.savefig(f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(f"{stem}.svg", facecolor="white")
    print(f"-> {stem}.png, {stem}.svg" + (f"   ({bad} overflowing)" if bad else "   (fit ok)"))


if __name__ == "__main__":
    main()
