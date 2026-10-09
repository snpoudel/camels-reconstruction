"""The LSTM before and after each correction, one cell per figure: NSE, KGE and percent bias.

    python figures/make_lstm_boxplots.py   # -> lstm/figS1_ungauged_boxplots.{png,pdf},
                                           #    lstm/figS2_discontinued_boxplots.{png,pdf}

Three boxes per panel: the uncorrected LSTM, after the IDW correction, and after the learned MLP
correction, over the same basins. The bias bands are cut on the observed flow (Yilmaz et al.,
2008). Each y axis is scaled to the whiskers because percent bias on near-zero-flow streams runs
to five figures; the red count under a box is how many of its basins fall outside the shown range.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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
OUT = {"ungauged": "figS1_ungauged_boxplots", "gauged": "figS2_discontinued_boxplots"}
TICK_LABEL = ["LSTM", "+ IDW", "+ MLP"]
OFF_COUNT_COLOR = "#c1272d"


def series(cell):
    """{series name: per-basin metrics}, all three on the same basins."""
    idw, mlp = read_metrics(cell, "lstm", f"idw{SUFFIX}"), read_metrics(cell, "lstm", f"mlp{SUFFIX}")
    pick = lambda df, v: df[df.variant == v].set_index("basin")  # noqa: E731
    out = {"LSTM": pick(idw, "LSTM"), "LSTM + IDW": pick(idw, "LSTM + correction"),
           "LSTM + MLP": pick(mlp, "LSTM + correction")}
    assert len({frozenset(d.index) for d in out.values()}) == 1, "scored on different basins"
    return out


def figure(cell):
    data = series(cell)
    rng = np.random.default_rng(0)  # reproducible jitter
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.6))
    for letter, (ax, (col, title, centred)) in enumerate(zip(axes.flat, PANELS)):
        vals = [data[name][col].dropna().to_numpy() for name in SERIES]
        bp = ax.boxplot(vals, positions=range(len(SERIES)), widths=0.52, patch_artist=True,
                        showfliers=False, medianprops=dict(color=INK, lw=1.6),
                        whiskerprops=dict(color=MUTE, lw=1.0), capprops=dict(color=MUTE, lw=1.0))
        for patch, color in zip(bp["boxes"], SERIES.values()):
            patch.set(facecolor=color, alpha=0.70, edgecolor="white", linewidth=1.6, zorder=3)
        for element in ("medians", "whiskers", "caps"):
            for line in bp[element]:
                line.set_zorder(3)
        if centred:
            ax.axhline(0, color=MUTE, lw=0.9, ls=(0, (4, 3)), zorder=1)
        for x, (v, color) in enumerate(zip(vals, SERIES.values())):
            ax.scatter(x + rng.uniform(-0.16, 0.16, len(v)), v, s=2, color=color, alpha=0.30,
                       linewidths=0, zorder=2)

        ax.set_title(f"{panel_letter(letter)} {title}", color=INK, fontsize=PANEL - 1.5,
                     loc="left", pad=4)
        style(ax)
        wy = np.concatenate([w.get_ydata() for w in bp["whiskers"]])
        lo, hi = (min(wy.min(), 0.0), max(wy.max(), 0.0)) if centred else (wy.min(), wy.max())
        pad = (hi - lo) * 0.16 + 1e-9
        ax.set_ylim(lo - pad, hi + pad)
        fmt = "{:.1f}" if centred else "{:.2f}"
        for x, v in enumerate(vals):
            ax.annotate(fmt.format(np.median(v)), (x, np.median(v)), ha="center", va="center",
                        fontsize=LABEL - 3, color=INK, zorder=4,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
            n_off = int((v < lo - pad).sum() + (v > hi + pad).sum())
            if n_off:
                ax.annotate(f"({n_off})", (x, bp["caps"][2 * x].get_ydata()[0]), xytext=(0, -3),
                            textcoords="offset points", ha="center", va="top",
                            fontsize=LABEL - 3.5, color=OFF_COUNT_COLOR, zorder=4)
        ax.set_xticks(range(len(SERIES)))
        ax.set_xticklabels(TICK_LABEL, fontsize=LABEL - 2, color=INK)
        ax.tick_params(axis="y", labelsize=LABEL - 2.5)

    fig.tight_layout(pad=0.4, h_pad=1.0)
    stem = f"{HERE}/{OUT_DIR}/{OUT[cell]}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=300, facecolor="white")
    print(f"-> {stem}.png/.pdf")
    for col, title, centred in PANELS:
        print(f"  {cell:<9} {title:<34} " + "  ".join(
            f"{name} {data[name][col].median():{'.1f' if centred else '.2f'}}" for name in SERIES))


if __name__ == "__main__":
    os.makedirs(f"{HERE}/{OUT_DIR}", exist_ok=True)
    for cell in OUT:
        figure(cell)
