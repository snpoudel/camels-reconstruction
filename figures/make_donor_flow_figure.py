"""Did `donor_flow` beat `donor`? Three panels on the basins no model was trained on.

    python figures/make_donor_flow_figure.py    # -> figures/donor_flow_check.png

`donor_flow` (IDEAS.md idea 1) replaced the attended value `h_d + MLP(Z_d)` with the donors'
raw flow anomaly fed straight to the head. This figure is the verdict on that one change.

Scoring is the project's single convention, reused verbatim: `score()` is imported from
`make_results_figure.py` rather than reimplemented, so the seeds' day-by-day predictions are
averaged into one hydrograph per basin and NSE/KGE computed once from that mean.

**Fold 0 only, for every variant drawn.** `donor` has been run on folds 0-6 and `donor_flow`
only on fold 0; `make_results_figure.py` pools folds on this cell, which would compare
different basin samples. Here all three boxes are the same 76 held-out basins, which is also
what makes the third panel possible -- a per-basin difference only means something if the
basins are matched. The basin ordering is asserted identical across variants before pairing.

The third panel is the one that decides it: the boxes in panels 1-2 hide whether a small
median shift is a broad improvement or a couple of rescued basins, and here it is the latter.

Colour: slots 1, 2 and 3 of the reference categorical palette, the same three validated for
simultaneous display in `make_results_figure.py` -- blue #2a78d6 `vanilla`, aqua #1baf7a
`donor`, orange #eb6834 `donor_flow`. `donor` keeps the aqua it has in every other figure.
Identity never rests on colour: every box is named on the x axis.
"""
import argparse
import importlib.util
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repo root, so the script runs from anywhere

_spec = importlib.util.spec_from_file_location("mrf", f"{HERE}/make_results_figure.py")
_mrf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mrf)
score = _mrf.score                    # identical formulas to src/evaluate.metrics

ARCHS = ["vanilla", "donor", "donor_flow"]
COLORS = {"vanilla": "#2a78d6", "donor": "#1baf7a", "donor_flow": "#eb6834"}
INK, MUTE, GRID = "#16191c", "#5d6a73", "#dfe3e6"
FOLD = 0

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
})


def load(runs_dir=f"{ROOT}/runs", cache_dir=f"{ROOT}/data/cache"):
    """arch -> per-basin scores on the held-out basins, from each variant's ensemble mean."""
    Q = np.load(f"{cache_dir}/Q.npy")
    all_basins = list(json.load(open(f"{cache_dir}/meta.json"))["basins"])

    members = {}
    for d in sorted(os.listdir(runs_dir)):
        path = f"{runs_dir}/{d}/predictions.npz"
        if not os.path.exists(path):
            continue
        cfg = json.load(open(f"{runs_dir}/{d}/config.json"))
        if cfg["arch"] not in ARCHS or cfg["fold"] != FOLD:
            continue
        if cfg.get("max_steps") or cfg.get("pred_stride", 1) != 1:
            continue
        members.setdefault(cfg["arch"], []).append(np.load(path, allow_pickle=True))

    scores, basin_ids = {}, None
    for arch in ARCHS:
        ps = members[arch]
        basins, n_gauged = list(ps[0]["basins"]), int(ps[0]["n_gauged"])
        obs = Q[[all_basins.index(b) for b in basins]][:, ps[0]["test_t"]]
        # average the raw hydrographs first; the single clip happens inside score()
        per_basin = score(np.mean([p["test"] for p in ps], axis=0), obs)
        scores[arch] = per_basin[n_gauged:]
        ids = basins[n_gauged:]
        if basin_ids is None:
            basin_ids = ids
        assert ids == basin_ids, f"{arch}: basin ordering differs -- pairing would be unsafe"
        print(f"{arch:11s} fold {FOLD}  ensemble of {len(ps)} starts  n={len(scores[arch])}")
    return scores, basin_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=f"{ROOT}/figures/donor_flow_check.png")
    a = p.parse_args()

    scores, _ = load()
    nse = {k: np.array([d["nse"] for d in v]) for k, v in scores.items()}
    kge = {k: np.array([d["kge"] for d in v]) for k, v in scores.items()}

    print(f"\nmedian NSE, ungauged/test, fold {FOLD}, ensemble mean:")
    for arch in ARCHS:
        print(f"  {arch:11s} {np.nanmedian(nse[arch]):.3f}   "
              f"KGE {np.nanmedian(kge[arch]):.3f}")

    d = nse["donor_flow"] - nse["donor"]
    ok = np.isfinite(d)
    print(f"\npaired donor_flow - donor (n={ok.sum()}): median {np.median(d[ok]):+.4f}, "
          f"mean {np.mean(d[ok]):+.4f}, improved {(d[ok] > 0).sum()}/{ok.sum()}")

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))
    for ax, (vals, lab) in zip(axes[:2], [(nse, "NSE  (↑ better)"), (kge, "KGE  (↑ better)")]):
        bp = ax.boxplot([vals[a_][np.isfinite(vals[a_])] for a_ in ARCHS],
                        patch_artist=True, widths=0.55, showfliers=False,
                        medianprops=dict(color=INK, lw=1.6))
        for patch, a_ in zip(bp["boxes"], ARCHS):
            patch.set(facecolor=COLORS[a_], alpha=0.75, edgecolor=INK, lw=1.0)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(color=MUTE, lw=1.0)
        ax.set_xticklabels([f"{a_}\nn={np.isfinite(vals[a_]).sum()}" for a_ in ARCHS], fontsize=9)
        ax.set_title(lab, fontsize=10, color=INK)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    ax = axes[2]
    ax.axvline(0, color=MUTE, lw=1.2, ls="--")
    ax.hist(d[ok], bins=22, color=COLORS["donor_flow"], alpha=0.8, edgecolor=INK, lw=0.6)
    ax.set_title("paired Δ NSE per basin\ndonor_flow − donor", fontsize=10, color=INK)
    ax.set_xlabel("Δ NSE", fontsize=9)
    ax.text(0.03, 0.95,
            f"median {np.median(d[ok]):+.3f}\n{(d[ok] > 0).sum()}/{ok.sum()} improved",
            transform=ax.transAxes, va="top", fontsize=9, color=INK)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.suptitle(f"donor_flow vs donor — unseen basins, fold {FOLD}, ensemble mean of 3 starts",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(a.out, dpi=170)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
