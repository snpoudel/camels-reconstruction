"""Fig. S1: the LSTM before and after each correction in both cells: NSE, KGE and percent bias.

    python figures/make_lstm_boxplots.py   # -> lstm/figS1_boxplots.{png,svg}

Two groups of three boxes per panel, ungauged then discontinued gauges: the uncorrected LSTM,
after the IDW correction, and after the learned MLP correction, over the same basins. The bias bands are cut on the observed flow (Yilmaz et al.,
2008). Each y axis is scaled to the whiskers because percent bias on near-zero-flow streams runs
to five figures; the red count under a box is how many of its basins fall outside the shown range.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import read_metrics  # noqa: E402
from palette import CORRECTED, INK, LABEL, LEARNED, MUTE, PANEL, RAW  # noqa: E402
from palette import figure_variant  # noqa: E402
from palette import panel_letter, style  # noqa: E402

SUFFIX, OUT_DIR = figure_variant()

PANELS = [
    ("nse", "NSE", False),
    ("kge", "KGE", False),
    ("pbias", "bias (%)", True),
    ("bias_high", "high-flow bias (%)", True),
    ("bias_mid", "medium-flow bias (%)", True),
    ("bias_low", "low-flow bias (%)", True),
]
SERIES = {"LSTM": RAW, "LSTM + IDW": CORRECTED, "LSTM + MLP": LEARNED}
CELLS = {"ungauged": "Ungauged", "gauged": "Discontinued"}
STEP, GROUP_GAP, WIDTH = 0.56, 1.0, 0.50
OFF_COUNT_COLOR = "#c1272d"


def series(cell):
    """{series name: per-basin metrics}, all three on the same basins."""
    idw, mlp = read_metrics(cell, "lstm", f"idw{SUFFIX}"), read_metrics(cell, "lstm", f"mlp{SUFFIX}")
    pick = lambda df, v: df[df.variant == v].set_index("basin")  # noqa: E731
    out = {"LSTM": pick(idw, "LSTM"), "LSTM + IDW": pick(idw, "LSTM + correction"),
           "LSTM + MLP": pick(mlp, "LSTM + correction")}
    assert len({frozenset(d.index) for d in out.values()}) == 1, "scored on different basins"
    return out


def main():
    data = {cell: series(cell) for cell in CELLS}
    boxes = [(cell, name) for cell in CELLS for name in SERIES]
    pos = [g * (2 * STEP + GROUP_GAP) + k * STEP for g in range(len(CELLS)) for k in range(len(SERIES))]
    colors = list(SERIES.values()) * len(CELLS)
    rng = np.random.default_rng(0)  # reproducible jitter
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 7.2))
    for letter, (ax, (col, title, centred)) in enumerate(zip(axes.flat, PANELS)):
        vals = [data[cell][name][col].dropna().to_numpy() for cell, name in boxes]
        bp = ax.boxplot(vals, positions=pos, widths=WIDTH, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.6), whiskerprops=dict(color=MUTE, lw=1.0),
                        capprops=dict(color=MUTE, lw=1.0))
        for patch, color in zip(bp["boxes"], colors):
            patch.set(facecolor=color, alpha=0.70, edgecolor="white", linewidth=1.2, zorder=3)
        for element in ("medians", "whiskers", "caps"):
            for line in bp[element]:
                line.set_zorder(3)
        if centred:
            ax.axhline(0, color=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=1)
        for x, v, color in zip(pos, vals, colors):
            ax.scatter(x + rng.uniform(-0.13, 0.13, len(v)), v, s=2, color=color, alpha=0.30,
                       linewidths=0, zorder=2)

        ax.set_title(f"{panel_letter(letter)} {title}", color=INK, fontsize=PANEL - 1.5,
                     loc="left", pad=4)
        style(ax)
        ax.grid(False, axis="x")
        wy = np.concatenate([w.get_ydata() for w in bp["whiskers"]])
        lo, hi = (min(wy.min(), 0.0), max(wy.max(), 0.0)) if centred else (wy.min(), wy.max())
        pad = (hi - lo) * 0.16 + 1e-9
        ax.set_ylim(lo - pad, hi + pad)
        for i, (x, v) in enumerate(zip(pos, vals)):
            n_off = int((v < lo - pad).sum() + (v > hi + pad).sum())
            if n_off:
                ax.annotate(f"({n_off})", (x, bp["caps"][2 * i].get_ydata()[0]), xytext=(0, -3),
                            textcoords="offset points", ha="center", va="top",
                            fontsize=LABEL - 4, color=OFF_COUNT_COLOR, zorder=4)
        ax.set_xticks([np.mean(pos[g * len(SERIES):(g + 1) * len(SERIES)]) for g in range(len(CELLS))])
        ax.set_xticklabels(CELLS.values(), fontsize=LABEL - 1.5, color=INK)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=LABEL - 2.5)

    fig.legend(handles=[Patch(facecolor=c, alpha=0.70, label=n) for n, c in SERIES.items()],
               loc="upper center", ncol=len(SERIES), frameon=False, fontsize=LABEL - 1.5,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, h_pad=1.0, rect=(0, 0, 1, 0.965))
    os.makedirs(f"{HERE}/{OUT_DIR}", exist_ok=True)
    stem = f"{HERE}/{OUT_DIR}/figS1_boxplots"
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=300, facecolor="white")
    print(f"-> {stem}.png/.svg")
    for col, title, centred in PANELS:
        for cell in CELLS:
            print(f"  {cell:<9} {title:<22} " + "  ".join(
                f"{name} {data[cell][name][col].median():{'.1f' if centred else '.2f'}}"
                for name in SERIES))


if __name__ == "__main__":
    main()
