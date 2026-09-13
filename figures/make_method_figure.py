"""Method figure for the cross-basin attention LSTM.

    python figures/make_method_figure.py    # -> figures/architecture.{png,svg}

Journal style: the canvas carries shapes, symbols and tensor shapes only; every sentence
of explanation lives in the caption block drawn below the diagram, which is meant to be
cropped off and pasted into the manuscript as the figure caption.

Three conventions do the explaining:
  * rows are basins, everywhere -- the input tensor and h are drawn the same way;
  * blue rows are the 455 gauged basins (the donor pool), amber rows the 76 ungauged;
  * teal is what the `donor` variant switches on, matching its box in the results figure.

The diagram is the *inference* pass -- all 531 basins are queries, which is why h is
[531 x 256]. Training differs only in that the 76 ungauged basins are not encoded at all
(`layout()` in src/data.py); the caption says so.

Everything hangs off one spine at x = 50, with h fanning out to a left bus (queries) and
a right bus (donors) at matching heights. Arrows are drawn in a second pass from measured
box edges, and `report_overlaps` checks every registered pair -- a clean run prints
"no overlaps".
"""
import argparse
import itertools
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))   # so the script runs from anywhere

EDGE = "#3f4a52"      # box border
LINE = "#5d6a73"      # connectors
INK = "#16191c"
MUTE = "#5d6a73"      # secondary text
ACCENT = "#17855f"    # what `donor` switches on; matches its box in the results figure
GAUGED = "#bfd8f0"    # the 455 basins of the donor pool
UNGAUGED = "#f6d29b"  # the 76 held-out basins
PANEL = "#f6f7f8"
PANEL_EDGE = "#d3d9dd"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",  # keep text as text in the SVG
})

CAPTION = (
    "Figure 1.  Cross-basin attention LSTM, drawn as the inference pass. One forward pass "
    "predicts a single day $t$ for every basin at once, so the batch dimension is the basin "
    "dimension. A single LSTM, with weights shared across basins, reads the same 270-day window "
    "of 6 daily forcings and 30 static catchment attributes for each basin and returns its last "
    "hidden state; stacked, these give h. Rows are basins throughout: blue rows are the 455 "
    "gauged basins of the fold, which form the fixed donor pool, and amber rows the 76 held-out "
    "ungauged basins. At inference every basin is a query, so queries supply Q while donors "
    "supply K and V. The self-diagonal of A is masked in training and in evaluation, so a gauged "
    "query cannot read its own row; an ungauged query is not in the pool and therefore keeps all "
    "455 donors. Here tau is a learned softmax temperature. The attended context c "
    "re-enters through a LayerNorm residual, and a two-layer MLP head (256 to 256 to 1, with ReLU "
    "and dropout) emits flow in mm day-1. The "
    "three variants differ only inside the shaded block: vanilla removes it and predicts from "
    "h_q alone; spatial keeps it with V = h_d; donor adds the teal term q_d, an encoding of each "
    "donor's own recent discharge. Its flow over days t-6..t enters as a standardised log-flow "
    "anomaly, which is dimensionless and so transfers between basins of very different size, "
    "alongside one 1/0 flag per day recording whether that gauge reported -- 14 values per "
    "donor, passed through a small MLP to 256 dimensions and added to h_d. A donor with no "
    "observation anywhere in the window is dropped from the "
    "softmax altogether; that masking is applied identically in spatial, so the donor pool is "
    "the same in both. During training only "
    "the 455 gauged basins are encoded and queried, each masked out of its own row, and the 76 "
    "ungauged basins are absent altogether. At no stage does a query basin's own discharge, or "
    "any statistic of it, enter the forward pass."
)


def box(ax, x, y, s, fs=6.4, fc="white", ec=EDGE, lw=0.7, color=INK, pad=0.34):
    """Text with a box shrink-wrapped around it. pad is in font-size units."""
    return ax.text(
        x, y, s, ha="center", va="center", fontsize=fs, color=color, zorder=4,
        linespacing=1.45,
        bbox=dict(boxstyle="round,pad=%.2f" % pad, facecolor=fc, edgecolor=ec, linewidth=lw),
    )


def rect(ax, t):
    """(x0, y0, x1, y1) of a text's box, in data coordinates."""
    patch = t.get_bbox_patch()
    bb = patch.get_window_extent() if patch else t.get_window_extent()
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
    return x0, y0, x1, y1


def colour_box(ax, xc, y, parts, fs=6.4, pad=1.5):
    """One line of text whose segments carry different colours, in a shrink-wrapped box.

    matplotlib has no rich text, so the segments are laid out left to right by measuring
    each one after a draw, then the box is drawn around their union.
    """
    fig = ax.figure
    fig.canvas.draw()
    ts = [ax.text(0, y, s, ha="left", va="center", fontsize=fs, color=c, zorder=5)
          for s, c in parts]
    fig.canvas.draw()
    inv = ax.transData.inverted()
    w = []
    for t in ts:
        bb = t.get_window_extent()
        w.append(inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0])
    h = (inv.transform((0, ts[0].get_window_extent().y1))[1]
         - inv.transform((0, ts[0].get_window_extent().y0))[1])
    x = xc - sum(w) / 2
    for t, wi in zip(ts, w):
        t.set_x(x)
        x += wi
    r = (xc - sum(w) / 2 - pad, y - h / 2 - pad, xc + sum(w) / 2 + pad, y + h / 2 + pad)
    ax.add_patch(FancyBboxPatch((r[0], r[1]), r[2] - r[0], r[3] - r[1],
                                boxstyle="round,pad=0,rounding_size=0.8", facecolor="white",
                                edgecolor=EDGE, lw=0.7, zorder=4))
    return r


def arrow(ax, p0, p1, color=LINE, lw=0.75):
    ax.annotate("", xy=p1, xytext=p0, zorder=3, arrowprops=dict(
        arrowstyle="-|>,head_width=0.14,head_length=0.30", color=color,
        linewidth=lw, shrinkA=0, shrinkB=0))


def elbow(ax, pts, color=LINE, lw=0.75):
    """Orthogonal polyline; only the final leg carries the head."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=3,
                solid_capstyle="round")
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw)


def line(ax, pts, color=LINE, lw=0.75):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, zorder=3,
            solid_capstyle="round")


def junction(ax, x, y):
    ax.plot([x], [y], marker="o", ms=2.1, color=LINE, zorder=4)


def basin_block(ax, x0, x1, ybot, ytop, frac=0.75, fs=6.0, cols=0, depth=0, dz=0.8):
    """A tensor whose rows are basins: gauged band on top, ungauged band below.

    `cols` draws faint column rules, which is how the input tensor shows that each row
    is itself a 270-day sequence, and `depth` stacks sheets behind it for the third axis.
    h has neither -- that is the point: the LSTM consumed both, leaving one flat sheet of
    the same rows. The band labels carry a bbox in the band colour so the rules stop
    behind the text instead of striking through it.
    """
    for i in range(depth, 0, -1):
        o = dz * i
        ax.add_patch(plt.Rectangle((x0 + o, ybot + o), x1 - x0, ytop - ybot,
                                   facecolor="#e9eef2", edgecolor=EDGE, lw=0.55,
                                   zorder=1.5))
    ysplit = ybot + (ytop - ybot) * (1 - frac)
    for y0, y1, fc in [(ysplit, ytop, GAUGED), (ybot, ysplit, UNGAUGED)]:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=fc,
                                   edgecolor="none", zorder=2))
    for i in range(1, cols + 1):
        xr = x0 + (x1 - x0) * i / (cols + 1)
        ax.plot([xr, xr], [ybot, ytop], color="white", lw=0.45, alpha=0.85, zorder=3)
    ax.add_patch(plt.Rectangle((x0, ybot), x1 - x0, ytop - ybot, facecolor="none",
                               edgecolor=EDGE, lw=0.8, zorder=4))
    ax.plot([x0, x1], [ysplit, ysplit], color=EDGE, lw=0.6, zorder=4)
    xc = (x0 + x1) / 2
    for y, s, f, fc in [((ysplit + ytop) / 2, "455 gauged", fs, GAUGED),
                        ((ybot + ysplit) / 2, "76 ungauged", fs - 0.4, UNGAUGED)]:
        ax.text(xc, y, s, ha="center", va="center", fontsize=f, color=INK, zorder=5,
                bbox=dict(boxstyle="square,pad=0.18", facecolor=fc, edgecolor="none"))
    return ysplit


def report_overlaps(named_rects, tol=0.25):
    """Print any pair of boxes that intersect -- the figure's own acceptance test."""
    bad = []
    for (na, a), (nb, b) in itertools.combinations(named_rects, 2):
        if (a[0] < b[2] - tol and b[0] < a[2] - tol
                and a[1] < b[3] - tol and b[1] < a[3] - tol):
            bad.append("%s <-> %s" % (na, nb))
    print("overlap check:", "no overlaps" if not bad else "COLLISIONS: " + "; ".join(bad))
    return bad


def diagram(fig, pos):
    ax = fig.add_axes(pos)
    ax.set_xlim(0, 100)
    ax.set_ylim(12.0, 101.0)
    ax.axis("off")
    R = {}

    # ---- input: one tensor, rows are basins ------------------------------------
    t_in = ax.text(50, 99.6, "Input   $X$    [531 basins × 270 days × 36 features]",
                   ha="center", va="center", fontsize=6.6, color=INK)
    basin_block(ax, 35, 65, 88, 96, frac=0.72, fs=5.7, depth=2)
    ax.text(67.4, 96.4, "36 features", ha="left", va="center", fontsize=5.2, color=MUTE)
    for x, lab in [(36.0, "$t\\!-\\!269$"), (64.0, "$t$")]:
        ax.text(x, 86.9, lab, ha="center", va="center", fontsize=5.4, color=MUTE)
    ax.text(33.3, 92.2, "531 basins", rotation=90, ha="center", va="center",
            fontsize=5.6, color=MUTE)

    n_lstm = box(ax, 50, 83.5, "shared LSTM  ·  256 hidden  ·  last hidden state")

    # ---- h: the same rows, now encoded ------------------------------------------
    t_h = ax.text(52.0, 79.2, "$h$    [531 × 256]", ha="left", va="center",
                  fontsize=6.6, color=INK)
    basin_block(ax, 35, 65, 67, 77)
    ax.text(50, 65.4, "256 hidden units", ha="center", va="center",
            fontsize=5.4, color=MUTE)
    ax.text(33.3, 70.0, "531 basins", rotation=90, ha="center", va="center",
            fontsize=5.4, color=MUTE)
    t_hq = ax.text(19, 76.9, "$h_q$   all 531 rows", ha="center", va="center",
                   fontsize=5.9, color=MUTE)
    t_hd = ax.text(81, 76.9, "$h_d$   455 gauged rows", ha="center", va="center",
                   fontsize=5.9, color=MUTE)

    # ---- the attention block -----------------------------------------------------
    ax.add_patch(FancyBboxPatch((8, 42), 85, 22, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=PANEL, edgecolor=PANEL_EDGE, lw=0.8, zorder=1))
    t_panel = ax.text(50, 61.4, "cross-basin attention   ($d = 64$)", ha="center",
                      va="center", fontsize=6.2, color=MUTE)

    p_q = box(ax, 21, 55.0, "$Q = W_q h_q$    [531 × 64]")
    p_k = box(ax, 78, 55.0, "$K = W_k h_d$    [455 × 64]")
    p_a = box(ax, 50, 55.0, "$A = \\mathrm{softmax}(QK^{\\top}\\!/\\sqrt{d}\\,\\tau)$\n"
                            "[531 × 455]")
    r_v = colour_box(ax, 78, 45.5, [("$V = h_d$ ", INK), ("$+\\ q_d$", ACCENT),
                                    ("    [455 × 256]", INK)])

    p_c = box(ax, 50, 45.5, "$c = A\\,V$    [531 × 256]")
    p_f = box(ax, 82, 37.5, "MLP   14 → 32 → 256", fs=6.0, ec=ACCENT, color=ACCENT)
    t_qd = ax.text(79.5, 40.8, "$q_d$   [455 × 256]", ha="right", va="center",
                   fontsize=6.0, color=ACCENT)
    p_zd = box(ax, 82, 28.5, "donor discharge,  days $t\\!-\\!6…t$\n"
                             "+ one 1/0 flag per day     [455 × 14]\n"
                             "flag 0 = that gauge did not report",
               fs=5.4, ec=ACCENT, color=ACCENT)
    t_donly = ax.text(82, 22.3, "donor variant only", ha="center", va="center",
                      fontsize=5.7, color=ACCENT)

    # ---- head ---------------------------------------------------------------------
    p_ctx = box(ax, 50, 31.0, "$ctx = h_q + \\mathrm{LayerNorm}(c)$    [531 × 256]")
    p_head = box(ax, 50, 23.0, "prediction head :   MLP   256 → 256 → 1")
    p_out = box(ax, 50, 15.0, "$\\hat{Q}_t$    [531]    flow on day $t$,  mm day$^{-1}$")

    fig.canvas.draw()
    R.update({k: rect(ax, t) for k, t in [
        ("input_lab", t_in), ("lstm", n_lstm), ("h_lab", t_h),
        ("hq", t_hq), ("hd", t_hd), ("panel_lab", t_panel), ("Q", p_q), ("K", p_k),
        ("A", p_a), ("c", p_c), ("f", p_f), ("zd", p_zd), ("donly", t_donly),
        ("qd", t_qd), ("ctx", p_ctx),
        ("head", p_head), ("out", p_out)]})
    R["V"] = r_v

    # ---- spine --------------------------------------------------------------------
    arrow(ax, (50, 88.0), (50, R["lstm"][3]))
    arrow(ax, (50, R["lstm"][1]), (50, 77.2))
    arrow(ax, (50, R["A"][1]), (50, R["c"][3]))
    arrow(ax, (50, R["c"][1]), (50, R["ctx"][3]))
    arrow(ax, (50, R["ctx"][1]), (50, R["head"][3]))
    arrow(ax, (50, R["head"][1]), (50, R["out"][3]))

    # ---- query bus (left) and donor bus (right), at matching heights ---------------
    ycy = (R["ctx"][1] + R["ctx"][3]) / 2
    line(ax, [(35, 75), (4, 75), (4, ycy)])
    arrow(ax, (4, 55.0), (R["Q"][0], 55.0))
    arrow(ax, (4, ycy), (R["ctx"][0], ycy))
    junction(ax, 4, 55.0)

    line(ax, [(65, 75), (96, 75), (96, 45.5)])
    arrow(ax, (96, 55.0), (R["K"][2], 55.0))
    arrow(ax, (96, 45.5), (R["V"][2], 45.5))
    junction(ax, 96, 55.0)

    arrow(ax, (R["Q"][2], 55.0), (R["A"][0], 55.0))
    arrow(ax, (R["K"][0], 55.0), (R["A"][2], 55.0))
    arrow(ax, (R["V"][0], 45.5), (R["c"][2], 45.5))
    arrow(ax, (82, R["zd"][3]), (82, R["f"][1]), color=ACCENT)
    arrow(ax, (82, R["f"][3]), (82, R["V"][1]), color=ACCENT)

    fig.canvas.draw()
    return list(R.items())


def build(width=6.0, diagram_in=4.45, wrap=137, fs=5.4):
    # The caption band is sized from the wrapped text, so editing the caption can never
    # silently clip it.
    text = textwrap.fill(CAPTION, wrap)
    cap_in = len(text.split("\n")) * fs * 1.5 / 72 + 0.17
    height = diagram_in + cap_in
    fig = plt.figure(figsize=(width, height))
    checked = diagram(fig, [0, cap_in / height, 1, 1 - cap_in / height])

    cax = fig.add_axes([0, 0, 1, cap_in / height])
    cax.axis("off")
    cax.text(0.025, 0.95, text, ha="left", va="top",
             fontsize=fs, color=INK, linespacing=1.5, transform=cax.transAxes)

    report_overlaps(checked)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=HERE, help="output directory")
    p.add_argument("--name", default="architecture")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig = build()
    png, svg = "%s/%s.png" % (a.out, a.name), "%s/%s.svg" % (a.out, a.name)
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    w, h = fig.get_size_inches()
    print("%.2f x %.2f in -> %s (%dx%d px), %s" % (w, h, png, w * 300, h * 300, svg))


if __name__ == "__main__":
    main()
