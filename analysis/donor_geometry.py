"""Which donors the correction actually uses, and how far away they can be before it stops working.

    python analysis/donor_geometry.py                    # this branch's runs (vanilla)
    python analysis/donor_geometry.py --arch donor       # parent branch's 7 folds, all 531 basins

Two questions, one script. Neither needs a GPU -- both read hydrographs already in `runs/`.

**Where are the donors?** For each held-out basin, the distance to its nearest donor, to the
K-th (farthest one it uses), and the inverse-distance-weighted mean of the K. Also how
concentrated the weights are: the share on the nearest donor, and the effective donor count
`1 / sum(w^2)`. This is the answer to "is it just copying the one gauge upstream?" -- a number,
not an assertion.

**How far is too far?** Binning the gain by each basin's observed donor distance confounds
distance with basin difficulty: the basins with distant gauges are also the arid and mountainous
ones the LSTM does worst on, so a decaying gain could be either effect. The clean test deletes
every donor closer than a floor and re-runs, sweeping the floor from 0 to 3200 km. Each basin is
then its own control, and the only thing that changed is how far its donors are.

K, lambda and p are held at whatever the pipeline selected for that fold -- re-tuning at each
floor would answer a different question, since the deployed method cannot re-tune itself when a
region turns out to be sparsely gauged. Basins left with fewer than K donors above the floor
receive no correction at all (weights zero, so R = 0) and are counted separately rather than
being silently corrected from a garbage neighbourhood.

Writes `runs/donor_geometry.csv` (one row per held-out basin) and `runs/donor_range.csv` (one
row per basin and floor); `figures/make_donor_figure.py` draws both.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.correct import (EPS, apply_shrinkage, choose, idw_weights,  # noqa: E402
                         load_all, neighbour_residual, nse_per_basin)

# Donor-distance floors, in km. Doubling past 200 km because the interesting structure is all
# below it; 3200 km is the width of the CONUS gauge network, i.e. as far away as donors can get.
FLOORS = (0, 25, 50, 75, 100, 150, 200, 400, 800, 1600, 3200)


def _corrected_nse(f, d, k, lam, p, eps=EPS):
    """NSE per held-out basin using donor distances `d`, and which basins had K donors at all.

    `d` may contain inf where a donor has been censored. A basin with fewer than K finite
    donors is not corrected: its weights are zeroed, so R = 0 and the hydrograph is the raw
    LSTM's. Leaving those basins in with whatever distant donors remain would answer a
    different question.
    """
    ok = np.isfinite(d).sum(1) >= k
    W = idw_weights(np.where(np.isinf(d), 1e12, d), k, p)
    W[~ok] = 0.0
    R = neighbour_residual(f["pred"][f["pool"]], f["obs"][f["pool"]], W, eps)
    sim = apply_shrinkage(f["pred"][f["held"]], R, lam, eps)
    return nse_per_basin(sim, f["obs"][f["held"]]), ok, W


def geometry(folds, picked):
    """One row per held-out basin: donor distances, weight concentration, and the gain."""
    rows = []
    for f in folds:
        k, lam, p = picked[f["fold"]]
        d = f["d_held"]
        cor, _, W = _corrected_nse(f, d, k, lam, p)
        raw = nse_per_basin(f["raw"] if "raw" in f else f["pred"][f["held"]],
                            f["obs"][f["held"]])
        order = np.argsort(d, axis=1)
        d_sort = np.take_along_axis(d, order, 1)[:, :k]
        w_sort = np.take_along_axis(W, order, 1)[:, :k]
        rows.append(pd.DataFrame(dict(
            fold=f["fold"], basin=f["basins"], k=k, lam=lam, p=p,
            d_near=d_sort[:, 0], d_kth=d_sort[:, -1], d_eff=(w_sort * d_sort).sum(1),
            w_near=w_sort[:, 0], n_eff=1.0 / np.maximum((w_sort ** 2).sum(1), 1e-12),
            raw=raw, cor=cor, gain=cor - raw)))
    return pd.concat(rows, ignore_index=True)


def range_curve(folds, picked, floors=FLOORS):
    """One row per (basin, floor): the gain once every donor nearer than `floor` is deleted."""
    rows = []
    for f in folds:
        k, lam, p = picked[f["fold"]]
        d0 = f["d_held"]
        raw = nse_per_basin(f["pred"][f["held"]], f["obs"][f["held"]])
        for floor in floors:
            d = np.where(d0 < floor, np.inf, d0)
            cor, ok, _ = _corrected_nse(f, d, k, lam, p)
            d_sort = np.sort(np.where(np.isinf(d), np.nan, d), axis=1)[:, :k]
            with np.errstate(invalid="ignore"):
                med_km = np.nanmedian(d_sort, axis=1)
            rows.append(pd.DataFrame(dict(
                fold=f["fold"], basin=f["basins"], floor=floor, corrected=ok,
                med_donor_km=med_km, raw=raw, cor=cor, gain=cor - raw)))
    return pd.concat(rows, ignore_index=True)


def break_even(curve):
    """Floor in km at which the pooled median gain crosses zero, linearly in log-distance.

    Returns None if the curve never crosses -- which is the honest answer when it doesn't.
    """
    g = curve.groupby("floor").gain.median()
    x, y = g.index.to_numpy(float), g.to_numpy()
    for i in range(len(y) - 1):
        if y[i] > 0 >= y[i + 1]:
            lo = np.log10(max(x[i], 1.0))
            return float(10 ** (lo + (np.log10(x[i + 1]) - lo) * y[i] / (y[i] - y[i + 1])))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--arch", default="vanilla",
                    help="run prefix in runs/; 'donor' reads the parent branch's 7 folds")
    ap.add_argument("--mode", choices=["inner", "gauged", "fixed"], default="inner")
    ap.add_argument("--out-geometry", default=None,
                    help="default: runs/donor_geometry_<arch>.csv")
    ap.add_argument("--out-range", default=None,
                    help="default: runs/donor_range_<arch>.csv")
    a = ap.parse_args()
    # Arch-suffixed by default: without it, running --arch donor after --arch vanilla
    # silently overwrites vanilla's CSVs, and a figure drawn afterwards mixes the two
    # archs' outputs without any error. Happened once; this is the fix.
    a.out_geometry = a.out_geometry or f"runs/donor_geometry_{a.arch}.csv"
    a.out_range = a.out_range or f"runs/donor_range_{a.arch}.csv"

    folds = load_all(a.runs, arch=a.arch)
    if not folds:
        raise SystemExit(f"no {a.arch}_fold*_seed*/ runs under {a.runs}/")
    picked, note = choose(folds, mode=a.mode)
    geo, curve = geometry(folds, picked), range_curve(folds, picked)
    n, nf = len(geo), geo.fold.nunique()
    print(f"{a.arch}: {n} held-out basins over {nf} fold(s).  (K, lambda, p) {note}")

    print("\n=== donor distance, km ===")
    print(f"{'':>24} {'p5':>7} {'p25':>7} {'median':>7} {'p75':>7} {'p95':>7}")
    for col, lab in (("d_near", "nearest donor"), ("d_eff", "weighted-mean donor"),
                     ("d_kth", "K-th (farthest used)")):
        q = geo[col].quantile([.05, .25, .5, .75, .95]).to_numpy()
        print(f"{lab:>24} " + " ".join(f"{v:7.1f}" for v in q))
    print(f"\nweight on the nearest donor: median {geo.w_near.median():.2f}"
          f"   effective donors: median {geo.n_eff.median():.1f} of K={int(geo.k.median())}"
          f"\nbasins whose nearest gauge is >100 km away: {int((geo.d_near > 100).sum())} of {n}")

    print("\n=== gain by observed distance to nearest donor (confounded with basin difficulty) ===")
    edges = [0, 10, 25, 50, 100, 200, np.inf]
    labels = ["<10", "10-25", "25-50", "50-100", "100-200", ">200"]
    print(f"{'km':>10} {'n':>5} {'raw NSE':>9} {'corrected':>10} {'median gain':>12} {'improved':>9}")
    for key, v in geo.groupby(pd.cut(geo.d_near, edges, labels=labels), observed=True):
        print(f"{key:>10} {len(v):5d} {v.raw.median():9.3f} {v.cor.median():10.3f} "
              f"{v.gain.median():+12.4f} {(v.gain > 0).mean():8.0%}")

    print("\n=== gain once every donor nearer than the floor is deleted (each basin its own control) ===")
    print(f"{'floor':>8} {'corrected':>12} {'median donor':>13} {'median gain':>12} {'improved':>9}")
    for floor, v in curve.groupby("floor"):
        w = v[v.corrected]
        print(f"{int(floor):>5} km {len(w):>5}/{len(v):<6} {w.med_donor_km.median():10.0f} km "
              f"{w.gain.median():+12.4f} {(w.gain > 0).mean():8.0%}")
    be = break_even(curve)
    print(f"\nbreak-even: {be:.0f} km" if be else "\nbreak-even: curve never crosses zero")

    geo.to_csv(a.out_geometry, index=False)
    curve.to_csv(a.out_range, index=False)
    print(f"-> {a.out_geometry}\n-> {a.out_range}")


if __name__ == "__main__":
    main()
