"""Two three-panel boxplot summaries: OOS-time and OOS-space, each NSE/KGE/RMSE x three
architectures.

    python figures/make_results_figure.py
    # -> figures/results_boxplot_oos_time.png, figures/results_boxplot_oos_space.png
    #    runs/metrics_ensemble.csv

Scoring follows the convention used in the multi-basin LSTM literature: for each day the
random starts of one run are averaged into one predicted hydrograph, and the metrics are
computed once from that ensemble mean. Each basin therefore contributes exactly one
value per measure, and a box is the spread across basins.

Note this reads higher than scoring each start separately and averaging the scores -- the
gap is the seed-to-seed disagreement that averaging cancels (on unseen basins: +0.029 NSE
for vanilla, +0.040 spatial, +0.042 donor). Use these numbers to compare against published
ensemble results; use the per-start numbers in EXPERIMENTS.md to compare architectures,
since the ensemble gain rewards whichever variant trains least stably.

OOS-time (`gauged/test`, unseen years for trained basins) always uses fold 0 only, even
for architectures that have since been run on more folds -- this row isn't the one being
extended. OOS-space (`ungauged/test`, unseen basins) pools every fold an architecture has
been run on: each fold's held-out basins are a disjoint slice of the same 531 (see
`assign_folds` in `src/data.py`), so folds concatenate into a bigger, non-overlapping
basin sample rather than adding noise. `vanilla`/`spatial` currently have only fold 0;
`donor` has folds 0-2. That means the boxes are NOT drawn from equal-sized samples --
each box's basin count (n) is written on its x-axis label so this is never silently read
as a matched comparison. `spatial_all` is absent from both figures: it was the worst
variant on every measure and is no longer being pursued.

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
CELLS = [("gauged/test", "OOS-time\nunseen years, trained basins"),
         ("ungauged/test", "OOS-space\nunseen basins")]
# Whether a cell pools every fold an architecture has been run on, or sticks to fold 0.
POOL_FOLDS = {"gauged/test": False, "ungauged/test": True}
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
    """(cell, measure, arch) -> one value per basin, scored from each fold's ensemble mean.

    Folds are never mixed within a single ensemble-mean hydrograph (each fold's basins are
    ordered differently) -- the seeds are averaged and scored fold by fold, and only the
    resulting per-basin scores are pooled across folds, per `POOL_FOLDS`.
    """
    Q = np.load(f"{cache_dir}/Q.npy")
    all_basins = list(json.load(open(f"{cache_dir}/meta.json"))["basins"])

    members = {}
    for path in sorted(glob.glob(f"{runs_dir}/*/predictions.npz")):
        cfg = json.load(open(path.rsplit("/", 1)[0] + "/config.json"))
        if cfg["arch"] not in ARCHS or cfg.get("max_steps") or cfg.get("pred_stride", 1) != 1:
            continue
        members.setdefault((cfg["arch"], cfg["fold"]), []).append(np.load(path, allow_pickle=True))

    per_basin_cell = {(cell, arch): [] for cell, _ in CELLS for arch in ARCHS}
    rows = []
    for arch in ARCHS:
        folds = sorted(f for (a, f) in members if a == arch)
        for fold in folds:
            ps = members[(arch, fold)]
            basins, n_gauged = list(ps[0]["basins"]), int(ps[0]["n_gauged"])
            obs_all = Q[[all_basins.index(b) for b in basins]][:, ps[0]["test_t"]]
            # average the raw hydrographs first; the single clip happens inside score()
            ens = np.mean([p["test"] for p in ps], axis=0)
            per_basin = score(ens, obs_all)
            print(f"{arch:9s} fold {fold}  ensemble of {len(ps)} starts")
            for cell, sl in (("gauged/test", slice(0, n_gauged)),
                             ("ungauged/test", slice(n_gauged, None))):
                if fold == 0 or POOL_FOLDS[cell]:
                    per_basin_cell[(cell, arch)].extend(per_basin[sl])
            for b, d in zip(basins, per_basin):
                cell = "gauged/test" if basins.index(b) < n_gauged else "ungauged/test"
                if fold == 0 or POOL_FOLDS[cell]:
                    rows.append(dict(basin=b, arch=arch, fold=fold, cell=cell,
                                     n_starts=len(ps), **d))

    data, n_basins = {}, {}
    for (cell, arch), ds in per_basin_cell.items():
        n_basins[(cell, arch)] = len(ds)
        for m, _ in MEASURES:
            data[(cell, m, arch)] = np.array([d[m] for d in ds])

    if dump:
        with open(dump, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["basin", "arch", "fold", "cell", "n_starts",
                                              "nse", "kge", "rmse"])
            w.writeheader()
            w.writerows(rows)
        print(f"per-basin ensemble scores -> {dump}")
    return data, n_basins


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


def build_panel(data, n_basins, cell, clabel, width=6.0, height=2.6):
    fig, axes = plt.subplots(1, 3, figsize=(width, height))
    hidden = total = 0

    for j, (m, mlabel) in enumerate(MEASURES):
        ax = axes[j]
        series = [data[(cell, m, a)][np.isfinite(data[(cell, m, a)])] for a in ARCHS]
        lo, hi = whisker_span(series)
        if m in SIGNED:
            lo = min(lo, 0.0)
            ylim = (lo - 0.17 * (hi - lo), hi + 0.04 * (hi - lo))
        else:
            pad = 0.04 * (hi - lo)
            ylim = (lo - pad, hi + pad)
        span = ylim[1] - ylim[0]
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

        ax.set_title(mlabel, fontsize=7.6, color=INK, pad=4)
        # Sample sizes differ by architecture once folds are pooled unevenly (see module
        # docstring), so n is always on the label -- never left to be assumed equal.
        ax.set_xticks(range(1, len(ARCHS) + 1))
        ax.set_xticklabels([f"{a}\n(n={n_basins[(cell, a)]})" for a in ARCHS],
                           fontsize=6.8, color=INK)

    fig.suptitle(clabel.replace("\n", "  —  "), fontsize=8.4, color=INK, y=1.0)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.14, wspace=0.28)
    return fig, hidden, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=HERE)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data, n_basins = load()
    for cell, _ in CELLS:
        line = "  ".join(f"{arch} {np.nanmedian(data[(cell,'nse',arch)]):.3f} "
                         f"(n={n_basins[(cell, arch)]})" for arch in ARCHS)
        print(f"median NSE, {cell:14s} {line}")

    names = {"gauged/test": "results_boxplot_oos_time", "ungauged/test": "results_boxplot_oos_space"}
    for cell, clabel in CELLS:
        fig, hidden, total = build_panel(data, n_basins, cell, clabel)
        name = names[cell]
        png = f"{a.out}/{name}.png"
        fig.savefig(png, dpi=300)
        w, h = fig.get_size_inches()
        print(f"{w:.2f} x {h:.2f} in -> {png} ({int(w*300)}x{int(h*300)} px) "
             f"(outliers hidden: {hidden}/{total})")


if __name__ == "__main__":
    main()
