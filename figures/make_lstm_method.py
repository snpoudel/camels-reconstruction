"""Fig. 1: the spatial residual correction in four numbered steps, IDW and the MLP sharing step 3.

    python figures/make_lstm_method.py   # -> lstm/fig01_method.{png,svg}

Steps 1, 2 and 4 are common to both corrections. Step 3 turns the donors' scaled errors into one
daily correction delta, in the target's own units, either as an inverse-distance-weighted mean
scaled by lambda (IDW) or with an MLP, so step 4 is a single formula. A one-line legend below
defines the symbols the boxes do not. Kept to two series colours, blue for the model and donors
and orange for the target and its correction. Every sketch shows the same neighbourhood, so the
eye follows one target and its donors across the steps.

`check_fit` prints a WARN for any string that overflows its box, since nothing else would catch
it. Keep the run clean.

Descriptive superscripts (sim, obs, corr) are upright, as Copernicus style requires. Sized for a
17.8 cm wide print, where the smallest text (9 here) comes out at about 6.5 pt.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from palette import CORRECTED, GRID, INK, MUTE, RAW  # noqa: E402
from palette import figure_variant  # noqa: E402

SUFFIX, OUT_DIR = figure_variant()
SCALED = SUFFIX != "_noscale"

SIM, OBS, CORR = r"^{\mathrm{sim}}", r"^{\mathrm{obs}}", r"^{\mathrm{corr}}"

M, G, PAD = 0.16, 0.24, 0.10
W1, W2, W3, W4 = 1.75, 1.75, 3.40, 1.75
H, LEG_H = 3.15, 0.44
FIG_W = 2 * M + 3 * G + W1 + W2 + W3 + W4
Y0 = PAD + LEG_H
TOP = Y0 + H
FIG_H = TOP + 0.18                # room for the badges, which sit on the box corners
X1 = M
X2 = X1 + W1 + G
X3 = X2 + W2 + G
X4 = X3 + W3 + G

TITLE_FS, EQ_FS, SUB_FS, LEG_FS, LAB_FS = 10.0, 11.0, 9.5, 9.5, 9.0
SKETCH_TOP = TOP - 1.18

# One neighbourhood in unit coordinates, drawn in every sketch. Every donor is nearer the target
# than every other gauge, no two donors are collinear with it, and their distances differ enough
# that step 3's weight lines come out visibly unequal.
TARGET = (0.44, 0.46)
DONORS = ((0.22, 0.70), (0.56, 0.62), (0.78, 0.40))
OTHERS = ((0.06, 0.14), (0.94, 0.84), (0.32, 0.06))
# Per-donor label offset and alignment, fixed so labels stay put across sketches.
OFF = (((0.00, 0.10), "center", "bottom"),
       ((0.07, 0.05), "left", "bottom"),
       ((0.08, 0.00), "left", "center"))

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


def pt(rect, u, v):
    x0, y0, w, h = rect
    return x0 + u * w, y0 + v * h


def panel(ax, rect):
    # Above the step box's own white fill (zorder 2), or it is painted over and vanishes.
    ax.add_patch(FancyBboxPatch(rect[:2], *rect[2:], boxstyle="round,pad=0,rounding_size=0.06",
                                linewidth=0, facecolor=GRID, alpha=0.75, zorder=2.5))


def basins(ax, rect):
    """Three donors, three farther gauges and the target; returns the target's position."""
    for u, v in OTHERS:
        ax.scatter(*pt(rect, u, v), s=15, color=RAW, alpha=0.28, linewidths=0, zorder=4)
    for u, v in DONORS:
        ax.scatter(*pt(rect, u, v), s=38, color=RAW, edgecolor="white", linewidth=0.7, zorder=5)
    tx, ty = pt(rect, *TARGET)
    ax.scatter([tx], [ty], s=140, marker="*", color=CORRECTED, edgecolor="white",
               linewidth=0.7, zorder=6)
    return tx, ty


def donor_weights():
    """The sketch's weights from the drawn geometry -- p = 1, normalised."""
    inv = np.array([1.0 / np.hypot(u - TARGET[0], v - TARGET[1]) for u, v in DONORS])
    return inv / inv.sum()


def box(ax, x, w, title, badge):
    ax.add_patch(FancyBboxPatch((x, Y0), w, H, boxstyle="round,pad=0,rounding_size=0.07",
                                linewidth=1.0, edgecolor=MUTE, facecolor="white", zorder=2))
    track(ax.text(x + w / 2, TOP - 0.16, title, ha="center", va="top", fontsize=TITLE_FS,
                      color=INK, fontweight="bold", zorder=4), x + 0.05, x + w - 0.05)
    ax.add_patch(Circle((x + 0.02, TOP - 0.02), 0.14, facecolor=INK, edgecolor="white",
                        linewidth=1.1, zorder=6))
    ax.text(x + 0.02, TOP - 0.02, badge, ha="center", va="center", fontsize=9, color="white",
            fontweight="bold", zorder=7)


def text(ax, x, y, s, lo, hi, size=EQ_FS, color=INK, **kw):
    track(ax.text(x, y, s, ha=kw.pop("ha", "center"), va=kw.pop("va", "center"),
                      fontsize=size, color=color, zorder=4, linespacing=1.4, **kw), lo, hi)


def sketch_rect(x, w, top):
    """A square sketch area centred in the space between the box bottom and `top`."""
    side = min(w - 0.2, top - Y0 - 0.14)
    return (x + (w - side) / 2, Y0 + 0.1 + (top - Y0 - 0.14 - side) / 2, side, side)


def neighbourhood(ax, rect, labels, target_label=None):
    """Steps 1 and 2: the neighbourhood with each donor labelled, and optionally the target."""
    panel(ax, rect)
    basins(ax, rect)
    for (u, v), lab, (off, ha, va) in zip(DONORS, labels, OFF):
        ax.text(*pt(rect, u + off[0], v + off[1]), lab, fontsize=LAB_FS, color=INK,
                ha=ha, va=va, zorder=7)
    if target_label:
        ax.text(*pt(rect, TARGET[0] - 0.08, TARGET[1] - 0.02), target_label, fontsize=LAB_FS,
                color=CORRECTED, ha="right", va="center", zorder=7)


def weights(ax, rect):
    """Step 3's neighbourhood: one line per donor, thicker the larger its weight. Only donor j's
    line is labelled; labels on all three hid the shortest line at this size."""
    panel(ax, rect)
    tx, ty = basins(ax, rect)
    for (u, v), wj in zip(DONORS, donor_weights()):
        ax.plot([tx, pt(rect, u, v)[0]], [ty, pt(rect, u, v)[1]], color=MUTE,
                lw=0.6 + 4.5 * wj, alpha=0.7, solid_capstyle="round", zorder=3)
    u, v = DONORS[2]
    ax.text(*pt(rect, (TARGET[0] + u) / 2, (TARGET[1] + v) / 2 - 0.07), r"$w_{ij}$",
            fontsize=LAB_FS, color=INK, ha="center", va="top", zorder=7)


def legend(ax):
    """One centred line, only the symbols the boxes do not define themselves. r, w, delta and
    q_corr are defined by an equation or a box title; p and lambda are left to the caption."""
    items = [(dict(marker="*", s=110, color=CORRECTED), "target gauge $i$"),
             (dict(marker="o", s=30, color=RAW), "donor gauge $j$ ($K$ nearest)"),
             (dict(marker="o", s=30, color=RAW, alpha=0.28), "other gauge"),
             (f"$q{SIM},\\ q{OBS}$", "predicted, observed flow")]
    if SCALED:
        items.append((r"$s$", f"standard deviation of $q{SIM}$"))
    rend, inv = ax.figure.canvas.get_renderer(), ax.transData.inverted()

    def width(s, size):
        t = ax.text(0, 0, s, fontsize=size)
        w = t.get_window_extent(rend).transformed(inv).width
        t.remove()
        return w

    gap, marker_w = 0.26, 0.20
    widths = [(marker_w if isinstance(k, dict) else width(k, LEG_FS + 1) + 0.07)
              + width(lab, LEG_FS) for k, lab in items]
    x = (FIG_W - sum(widths) - gap * (len(items) - 1)) / 2
    y = PAD + LEG_H / 2 - 0.06
    for (key, label), w in zip(items, widths):
        if isinstance(key, dict):
            ax.scatter([x + 0.07], [y + 0.06], edgecolor="white", linewidth=0.6, zorder=5, **key)
            lx = x + marker_w
        else:
            ax.text(x, y, key, fontsize=LEG_FS + 1, color=INK, va="baseline", zorder=4)
            lx = x + width(key, LEG_FS + 1) + 0.07
        ax.text(lx, y, label, fontsize=LEG_FS, color=MUTE, va="baseline", zorder=4)
        x += w + gap
    if x - gap > FIG_W - M:
        print(f"  WARN legend runs {x - gap - FIG_W + M:.2f} past the right margin")


def hydrograph(ax, rect):
    """Step 4: a schematic storm hydrograph, before and after a downward correction."""
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
    # A mini legend rather than in-place labels, which would need leader lines at this size.
    for n, (lab, col) in enumerate(((f"$q{SIM}_i$", RAW), (f"$q{CORR}_i$", CORRECTED))):
        ly = y0 + (0.94 - 0.13 * n) * h
        ax.plot([x0 + 0.08 * w, x0 + 0.20 * w], [ly, ly], color=col, lw=1.3, zorder=7)
        ax.text(x0 + 0.24 * w, ly, lab, fontsize=LAB_FS, color=col, ha="left", va="center",
                zorder=7)
    ax.text(px1, py0 - 0.05 * h, "time", fontsize=LAB_FS - 1.2, color=MUTE, ha="right",
            va="top", zorder=7)


def network(ax, rect):
    """Inputs named by symbol, two hidden layers, one output: the day's correction."""
    x0, y0, w, h = rect
    panel(ax, rect)
    inputs = (r"$r_1,\dots,r_K$", r"$d_{i1},\dots,d_{iK}$", "day of year",
              f"$q{SIM}_i/s_i$" if SCALED else f"$q{SIM}_i$")
    layers = [[(0.40, v) for v in (0.86, 0.62, 0.38, 0.14)],
              [(0.56, v) for v in (0.80, 0.60, 0.40, 0.20)],
              [(0.71, v) for v in (0.80, 0.60, 0.40, 0.20)],
              [(0.86, 0.50)]]
    for a, b in zip(layers, layers[1:]):
        for ua, va in a:
            for ub, vb in b:
                ax.plot(*zip(pt(rect, ua, va), pt(rect, ub, vb)), color=MUTE, lw=0.4,
                        alpha=0.45, zorder=3)
    for (u, v), lab in zip(layers[0], inputs):
        ax.scatter(*pt(rect, u, v), s=22, color=INK, edgecolor="white", linewidth=0.6,
                   zorder=5)
        text(ax, *pt(rect, u - 0.03, v), lab, x0, x0 + w, size=LAB_FS, ha="right")
    for layer in layers[1:3]:
        for u, v in layer:
            ax.scatter(*pt(rect, u, v), s=18, color="white", edgecolor=MUTE, linewidth=0.7,
                       zorder=5)
    ax.scatter(*pt(rect, 0.86, 0.50), s=55, color=CORRECTED, edgecolor="white",
               linewidth=0.6, zorder=6)
    text(ax, *pt(rect, 0.95, 0.50), r"$\delta_i$", x0, x0 + w, size=LAB_FS + 1.5)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    legend(ax)

    box(ax, X1, W1, "LSTM prediction", "1")
    text(ax, X1 + W1 / 2, TOP - 0.72, rf"$\mathrm{{LSTM}}\rightarrow q{SIM}_i,\, q{SIM}_j$",
         X1 + 0.05, X1 + W1 - 0.05, size=10.0)
    neighbourhood(ax, sketch_rect(X1, W1, SKETCH_TOP),
                  [f"$q{SIM}_{n}$" for n in ("1", "2", "j")], f"$q{SIM}_i$")

    box(ax, X2, W2, "Donor errors", "2")
    text(ax, X2 + W2 / 2, TOP - 0.72, rf"$r_j=\dfrac{{q{SIM}_j-q{OBS}_j}}{{s_j}}$" if SCALED
         else f"$r_j=q{SIM}_j-q{OBS}_j$", X2 + 0.05, X2 + W2 - 0.05)
    neighbourhood(ax, sketch_rect(X2, W2, SKETCH_TOP), (r"$r_1$", r"$r_2$", r"$r_j$"))

    # Step 3, one box: IDW above, the MLP below, both ending in the same delta_i.
    box(ax, X3, W3, "Daily correction", "3")
    mid = Y0 + 0.44 * H
    eq_x, eq_hi = X3 + 0.88, X3 + 1.72
    text(ax, X3 + 0.14, TOP - 0.50, "IDW", X3 + 0.05, eq_hi, size=SUB_FS, ha="left",
         fontweight="bold")
    text(ax, eq_x, TOP - 0.92, r"$w_{ij}=\dfrac{d_{ij}^{-p}}{\sum_{j'} d_{ij'}^{-p}}$",
         X3 + 0.05, eq_hi)
    text(ax, eq_x, mid + 0.20, r"$\delta_i=\lambda\sum_j w_{ij}\,r_j$", X3 + 0.05, eq_hi)
    side = TOP - 0.42 - mid - 0.06
    weights(ax, ((eq_hi + X3 + W3 - 0.1 - side) / 2, mid + 0.06, side, side))
    text(ax, X3 + 0.14, mid - 0.12, "MLP", X3 + 0.05, eq_hi, size=SUB_FS, ha="left",
         fontweight="bold")
    network(ax, (X3 + 0.1, Y0 + 0.1, W3 - 0.2, mid - 0.30 - Y0 - 0.1))

    box(ax, X4, W4, "Corrected flow", "4")
    text(ax, X4 + W4 / 2, TOP - 0.60, f"$q{CORR}_i=\\max(0,$", X4 + 0.05, X4 + W4 - 0.05,
         size=10.0)
    text(ax, X4 + W4 / 2, TOP - 0.92, f"$q{SIM}_i-s_i\\,\\delta_i)$" if SCALED
         else f"$q{SIM}_i-\\delta_i)$", X4 + 0.05, X4 + W4 - 0.05, size=10.0)
    hydrograph(ax, sketch_rect(X4, W4, SKETCH_TOP))

    bad = check_fit(fig, ax)
    os.makedirs(f"{HERE}/{OUT_DIR}", exist_ok=True)
    stem = f"{HERE}/{OUT_DIR}/fig01_method"
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=300, facecolor="white")
    print(f"-> {stem}.png/.svg" + (f"   ({bad} overflowing)" if bad else "   (fit ok)"))


if __name__ == "__main__":
    main()
