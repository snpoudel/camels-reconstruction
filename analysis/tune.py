"""Hyperparameter search for the neighbour correction: K (neighbours) and lambda (shrinkage).

    python analysis/tune.py --out runs/tuning.csv

Writes one row per (fold, K, lambda) with the median NSE over that fold's held-out basins,
which is what `figures/make_hyperparam_figure.py` draws.

Selection, when more than one fold is available, is **leave-one-fold-out**: the values applied
to fold f are those maximising median NSE on the held-out basins of the *other* folds. Fold f
contributes nothing to its own choice. With a single fold there is nothing to leave out, so
the script reports the surface and declines to select -- a value picked on the only fold you
are scoring is not a measurement.

The honest reading of the result is not "the optimum is K=8, lambda=0.5" but "the surface is
flat enough that the choice barely matters", which is why `src/correct.py` ships fixed
constants instead of a fitted pair.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.correct import (EPS, apply_shrinkage, distances, idw_weights,  # noqa: E402
                         load_fold, neighbour_residual)

K_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50]
LAM_GRID = np.round(np.arange(0.0, 1.05, 0.05), 2)


def nse(sim, obs):
    """Vectorised over basins: [n, T] -> [n]."""
    sim = np.clip(sim, 0, None)
    m = np.isfinite(obs)
    o, s, n = np.where(m, obs, 0.0), np.where(m, sim, 0.0), m.sum(1)
    mu = o.sum(1) / np.maximum(n, 1)
    den = ((o - mu[:, None]) ** 2 * m).sum(1)
    return np.where((n > 1) & (den > 0), 1 - ((s - o) ** 2 * m).sum(1) / np.maximum(den, 1e-12), np.nan)


def surface(runs_dir="runs", cache_dir="data/cache"):
    """Median NSE for every (fold, K, lambda). Also returns per-basin NSE for pooling."""
    D, _ = distances(cache_dir)
    Q = np.load(f"{cache_dir}/Q.npy")
    rows, per_basin = [], {}
    for fold in range(7):
        got = load_fold(fold, runs_dir, cache_dir)
        if got is None:
            continue
        pred, t_idx, gauged, n_seeds = got
        obs = Q[:, t_idx]
        pool, held = np.flatnonzero(gauged), np.flatnonzero(~gauged)
        d = D[np.ix_(held, pool)]
        base = nse(pred[held], obs[held])
        for k in K_GRID:
            # R does not depend on lambda, so compute it once per K and sweep lambda over it.
            # This calls exactly the functions src/correct.py uses, so the surface drawn here
            # is the surface the shipped correction produces -- not an approximation of it.
            R = neighbour_residual(pred[pool], obs[pool], idw_weights(d, k), EPS)
            for lam in LAM_GRID:
                v = nse(apply_shrinkage(pred[held], R, lam, EPS), obs[held])
                per_basin[(fold, k, float(lam))] = v
                rows.append(dict(fold=fold, k=k, lam=float(lam), median_nse=np.nanmedian(v),
                                 base_nse=np.nanmedian(base), n_basins=len(held), n_seeds=n_seeds))
    return pd.DataFrame(rows), per_basin


def select(df, per_basin):
    """Leave-one-fold-out choice of (K, lambda) per fold. Needs at least two folds."""
    folds = sorted(df.fold.unique())
    if len(folds) < 2:
        return None
    out = []
    for f in folds:
        others = [o for o in folds if o != f]
        best, arg = -np.inf, None
        for k in K_GRID:
            for lam in LAM_GRID:
                v = np.nanmedian(np.concatenate([per_basin[(o, k, float(lam))] for o in others]))
                if v > best:
                    best, arg = v, (k, float(lam))
        applied = np.nanmedian(per_basin[(f, *arg)])
        out.append(dict(fold=f, k=arg[0], lam=arg[1], median_nse_applied=applied))
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="runs/tuning.csv")
    a = p.parse_args()

    df, per_basin = surface(a.runs)
    df.to_csv(a.out, index=False)
    folds = sorted(df.fold.unique())
    print(f"folds present: {folds}")

    pooled = df.groupby(["k", "lam"]).median_nse.mean().reset_index()
    best = pooled.loc[pooled.median_nse.idxmax()]
    base = df.base_nse.mean()
    print(f"uncorrected: {base:.4f}")
    print(f"best on the surface: K={int(best.k)}, lambda={best.lam:.2f} -> {best.median_nse:.4f}")
    shipped = pooled.query("k == 8 and lam == 0.5").median_nse.iloc[0]
    print(f"shipped default:     K=8, lambda=0.50 -> {shipped:.4f}  "
          f"({shipped - best.median_nse:+.4f} vs the best cell)")
    within = (pooled.median_nse > best.median_nse - 0.01).mean()
    print(f"{within:.0%} of the {len(pooled)}-cell grid is within 0.01 NSE of the best cell")

    sel = select(df, per_basin)
    if sel is None:
        print("\nonly one fold available -- reporting the surface, not selecting. "
              "Re-run once more folds exist for leave-one-fold-out selection.")
    else:
        print("\nleave-one-fold-out selection:")
        print(sel.to_string(index=False))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
