"""Six-panel summary of the screen: three measures x two kinds of held-out data.

    python figures/make_results_figure.py   # -> figures/results_boxplots.{png,svg}
                                        #    runs/metrics_ensemble.csv

Scoring follows the convention used in the multi-basin LSTM literature: for each day the
three random starts are averaged into one predicted hydrograph, and the metrics are
computed once from that ensemble mean. Each basin therefore contributes exactly one
value per measure, and a box is the spread across basins.

Note this reads higher than scoring each start separately and averaging the scores -- the
gap is the seed-to-seed disagreement that averaging cancels (on unseen basins: +0.029 NSE
for vanilla, +0.040 spatial, +0.042 donor). Use these numbers to compare against published
ensemble results; use the per-start numbers in EXPERIMENTS.md to compare architectures,
since the ensemble gain rewards whichever variant trains least stably.

Rows are the two things being generalised over -- unseen YEARS for basins the model
trained on, and unseen BASINS. `spatial_all` is absent: it was the worst variant on every
measure and is no longer being pursued.

Colour: slots 1, 2 and 3 of the reference categorical palette. `donor` takes aqua rather
than yellow -- against orange, yellow separates by only 13.7 in OKLab dE, under the hard
normal-vision floor of 15. Aqua lifts the worst pair to 27.6 normal / 9.6 simulated
deuteranope (target 8). Identity never rests on colour: every box is named on the x axis.
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repo root, so the script runs from anywhere


ARCHS = ["vanilla", "spatial", "donor"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
CELLS = [("gauged/test", "unseen years\n455 trained basins"),
         ("ungauged/test", "unseen basins\n76 held-out basins")]
MEASURES = [("nse", "NSE  (↑ better)"),
            ("kge", "KGE  (↑ better)"),
            ("rmse", "RMSE mm day$^{-1}$  (↓ better)")]
SIGNED = ("nse", "kge")
INK, MUTE, GRID = "#16191c", "#5d6a73", "#dfe3e6"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
})


def score(sim, obs):
    """Per-basin NSE, KGE and RMSE -- the same formulas as src/evaluate.metrics."""
    sim = np.clip(sim, 0, None)
    out = []
    for s, o in zip(sim, obs):
        m = np.isfinite(o)
        s, o = s[m], o[m]
        if len(o) < 2 or o.std() == 0:
            out.append(dict(nse=np.nan, kge=np.nan, rmse=np.nan))
            continue
        nse = 1 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
        rmse = np.sqrt(((s - o) ** 2).mean())
        alpha, beta = s.std() / o.std(), s.mean() / o.mean()
        r = np.corrcoef(s, o)[0, 1] if s.std() > 0 else np.nan
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        out.append(dict(nse=nse, kge=kge, rmse=rmse))
    return out


def load(runs_dir=f"{ROOT}/runs", cache_dir=f"{ROOT}/data/cache",
         dump=f"{ROOT}/runs/metrics_ensemble.csv"):
    """(cell, measure, arch) -> one value per basin, scored from the ensemble mean."""
    Q = np.load(f"{cache_dir}/Q.npy")
    all_basins = list(json.load(open(f"{cache_dir}/meta.json"))["basins"])

    members = {}
    for path in sorted(glob.glob(f"{runs_dir}/*/predictions.npz")):
        cfg = json.load(open(path.rsplit("/", 1)[0] + "/config.json"))
        if cfg["arch"] not in ARCHS or cfg.get("max_steps") or cfg.get("pred_stride", 1) != 1:
            continue
        members.setdefault(cfg["arch"], []).append(np.load(path, allow_pickle=True))

    data, rows = {}, []
    for arch in ARCHS:
        ps = members[arch]
        basins, n_gauged = list(ps[0]["basins"]), int(ps[0]["n_gauged"])
        obs_all = Q[[all_basins.index(b) for b in basins]][:, ps[0]["test_t"]]
        # average the raw hydrographs first; the single clip happens inside score()
        ens = np.mean([p["test"] for p in ps], axis=0)
        per_basin = score(ens, obs_all)
        print(f"{arch:9s} ensemble of {len(ps)} starts")
        for cell, sl in (("gauged/test", slice(0, n_gauged)),
                         ("ungauged/test", slice(n_gauged, None))):
            for m, _ in MEASURES:
                data[(cell, m, arch)] = np.array([d[m] for d in per_basin[sl]])
        for b, d in zip(basins, per_basin):
            cell = "gauged/test" if basins.index(b) < n_gauged else "ungauged/test"
            rows.append(dict(basin=b, arch=arch, cell=cell, n_starts=len(ps), **d))

    if dump:
        with open(dump, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["basin", "arch", "cell", "n_starts",
                                              "nse", "kge", "rmse"])
            w.writeheader()
            w.writerows(rows)
        print(f"per-basin ensemble scores -> {dump}")
    return data


def whisker_span(series):
    lo, hi = [], []
    for v in series:
        v = v[np.isfinite(v)]
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        inl = v[(v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)]
        lo.append(inl.min())
        hi.append(inl.max())
    return min(lo), max(hi)


def build(data, width=6.0, height=4.3):
    fig, axes = plt.subplots(2, 3, figsize=(width, height), sharex=True)
    hidden = total = 0

    for j, (m, mlabel) in enumerate(MEASURES):
        allser = [data[(c, m, a)] for c, _ in CELLS for a in ARCHS]
        lo, hi = whisker_span(allser)
        if m in SIGNED:                      # reserve a strip for the "below zero" counts
            lo = min(lo, 0.0)
            ylim = (lo - 0.17 * (hi - lo), hi + 0.04 * (hi - lo))
        else:
            pad = 0.04 * (hi - lo)
            ylim = (lo - pad, hi + pad)
        span = ylim[1] - ylim[0]

        for i, (cell, clabel) in enumerate(CELLS):
            ax = axes[i, j]
            series = [data[(cell, m, a)][np.isfinite(data[(cell, m, a)])] for a in ARCHS]
            hidden += sum(int(((v < ylim[0]) | (v > ylim[1])).sum()) for v in series)
            total += sum(len(v) for v in series)

            bp = ax.boxplot(series, widths=0.58, showfliers=False, patch_artist=True,
                            medianprops=dict(color=INK, lw=1.1),
                            whiskerprops=dict(color=MUTE, lw=0.7),
                            capprops=dict(color=MUTE, lw=0.7))
            for patch, c in zip(bp["boxes"], COLORS):
                patch.set(facecolor=c, alpha=0.32, edgecolor=c, linewidth=0.9)

            ax.set_ylim(*ylim)
            ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(MUTE)
                ax.spines[s].set_linewidth(0.7)
            ax.tick_params(labelsize=6.9, colors=MUTE, length=2.5, width=0.7)

            if m in SIGNED:
                ax.axhline(0, color=MUTE, lw=0.6, ls=(0, (3, 2)), zorder=1)
                for k, v in enumerate(series):
                    med = float(np.median(v))
                    ax.text(k + 1, med - 0.022 * span, f"{med:.3f}", ha="center",
                            va="top", fontsize=6.1, color=INK, zorder=4)
                for k, (v, c) in enumerate(zip(series, COLORS)):
                    ax.text(k + 1, ylim[0] + 0.035 * span, str(int((v < 0).sum())),
                            ha="center", va="bottom", fontsize=6.3, color=c,
                            fontweight="bold", zorder=4)

            if i == 0:
                ax.set_title(mlabel, fontsize=7.4, color=INK, pad=4)
            if j == 0:
                ax.set_ylabel(clabel, fontsize=7.2, color=INK, labelpad=3)

    for ax in axes[1]:
        ax.set_xticks(range(1, len(ARCHS) + 1))
        ax.set_xticklabels(ARCHS, rotation=22, ha="right", fontsize=7.2, color=INK)

    fig.text(0.008, 0.012,
             "the 3 random starts are averaged day by day into one hydrograph, then scored "
             f"once per basin · outliers not drawn ({hidden} of {total})\n"
             "boxes: quartiles and median · whiskers 1.5×IQR · coloured figure below the "
             "dashed zero line: basins scoring below 0",
             fontsize=5.6, color=MUTE, ha="left", va="bottom", linespacing=1.5)
    fig.subplots_adjust(left=0.108, right=0.985, top=0.93, bottom=0.215,
                        wspace=0.26, hspace=0.10)
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=HERE)
    p.add_argument("--name", default="results_boxplots")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = load()
    for cell, _ in CELLS:
        line = "  ".join(f"{arch} {np.nanmedian(data[(cell,'nse',arch)]):.3f}"
                         for arch in ARCHS)
        print(f"median NSE, {cell:14s} {line}")
    fig = build(data)
    png, svg = f"{a.out}/{a.name}.png", f"{a.out}/{a.name}.svg"
    fig.savefig(png, dpi=300)
    fig.savefig(svg)
    w, h = fig.get_size_inches()
    print(f"{w:.2f} x {h:.2f} in -> {png} ({int(w*300)}x{int(h*300)} px), {svg}")


if __name__ == "__main__":
    main()
