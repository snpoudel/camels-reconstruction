"""Where the correction acts: four bias measures, before and after.

    python figures/make_boxplot.py --cell ungauged
    python figures/make_boxplot.py --cell gauged

Overall skill lives in the CDF figure (NSE and KGE); this one is about water balance -- overall
percent bias, then bias split by how high the river actually was that day: top 10%,
middle 30-90%, bottom 30%.

The flow bands are cut on the OBSERVED flow (Yilmaz et al. 2008), so "high-flow bias" is the
model's error on the days the river really was high, not on the days it predicted high. For the
bias panels zero is the target and the dashed line marks it, so a box straddling zero more
tightly is better regardless of which side it sits on.

Each box is drawn over its own basins as a faint jittered scatter, so it summarises what is
actually visible underneath rather than standing in for it. The axis is scaled to the drawn
whiskers, not the full data range -- percent bias on near-zero-flow streams runs to five figures
and would flatten every box -- and the bracketed count under a box is how many of its basins
fall outside the shown range.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import fold_settings, read_info, read_metrics  # noqa: E402
from palette import CELL_BLURB, CELL_TITLE, INK, LABEL, MUTE, PANEL, SERIES, TITLE  # noqa: E402
from palette import caption, settings_note, style  # noqa: E402

PANELS = [
    ("pbias", "overall bias (%)", True),
    ("bias_high", "bias, top 10% of flows (%)", True),
    ("bias_mid", "bias, middle 30-90% of flows (%)", True),
    ("bias_low", "bias, bottom 30% of flows (%)", True),
]


def main(cell):
    df = read_metrics(cell)
    info = read_info(cell)
    series = {name: df[df.variant == name] for name in SERIES}

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.8))
    rng = np.random.default_rng(0)  # fixed, so the jitter is reproducible

    for ax, (col, title, centred) in zip(axes.ravel(), PANELS):
        vals = [s[col].dropna().to_numpy() for s in series.values()]
        bp = ax.boxplot(vals, positions=[0, 1], widths=0.52, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK, lw=1.6),
                        whiskerprops=dict(color=MUTE, lw=1.0),
                        capprops=dict(color=MUTE, lw=1.0))
        for patch, color in zip(bp["boxes"], SERIES.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.70)
            patch.set_edgecolor("white")  # 2px surface gap between adjacent fills
            patch.set_linewidth(1.6)
            patch.set_zorder(3)
        for element in ("medians", "whiskers", "caps"):
            for line in bp[element]:
                line.set_zorder(3)
        if centred:
            ax.axhline(0, color=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=1)
        for x, (v, color) in enumerate(zip(vals, SERIES.values())):
            ax.scatter(x + rng.uniform(-0.16, 0.16, len(v)), v, s=5, color=color, alpha=0.30,
                       linewidths=0, zorder=2)
        # Medians printed: several panels are close, and the reader should not have to estimate
        # the difference off the ink.
        for x, v in enumerate(vals):
            ax.annotate(f"{np.median(v):.2f}", (x, np.median(v)), ha="center", va="center",
                        fontsize=LABEL - 1.5, color=INK, zorder=4,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))

        ax.set_title(title, color=INK, fontsize=PANEL, loc="left", pad=6)
        style(ax)
        # Scale to the whiskers, not the data range.
        wy = np.concatenate([w.get_ydata() for w in bp["whiskers"]])
        lo, hi = (min(wy.min(), 0.0), max(wy.max(), 0.0)) if centred else (wy.min(), wy.max())
        pad = (hi - lo) * 0.16 + 1e-9
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{name}\n({n})" if n else name
                            for name, v in zip(SERIES, vals)
                            for n in [int((v < lo - pad).sum() + (v > hi + pad).sum())]],
                           fontsize=LABEL - 1, color=INK)

    fig.suptitle(f"{CELL_TITLE[cell]}: bias before and after the correction",
                 fontsize=TITLE, color=INK, x=0.012, ha="left", y=0.995)
    bottom = caption(fig, f"{info['n_basins']} basins, {CELL_BLURB[cell]}, {info['n_folds']} "
                          f"folds, {info['n_seeds']} seeds, test period {info['test_period']}.  "
                          f"Boxes: quartiles and median; whiskers 1.5 IQR, outliers omitted; dots "
                          f"are individual basins, jittered.  (n) under a box is how many of its "
                          f"basins fall beyond the shown axis.  {settings_note(fold_settings(df))}, "
                          f"chosen by cross-validation that never reads the basin being corrected.",
                     145)
    fig.tight_layout(rect=(0, bottom, 1, 0.975))
    out = f"{HERE}/{cell}_metrics_boxplots.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}")
    for col, title, _ in PANELS:
        a, b = (series[n][col].median() for n in SERIES)
        print(f"  {title:<34} {a:+8.3f} -> {b:+8.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=list(CELL_TITLE), default="ungauged")
    main(ap.parse_args().cell)
