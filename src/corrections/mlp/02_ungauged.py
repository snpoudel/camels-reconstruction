"""Correct the ungauged-basin predictions with the learned MLP correction, one fold at a time.

    python src/corrections/mlp/02_ungauged.py
    python src/corrections/mlp/02_ungauged.py --fold 0 --k 10 --seeds 1   # quick check

The same cell and the same donors as src/corrections/idw/02_ungauged.py: a fold's held-out
basins are corrected from its 455 training basins, using that fold's own LSTM. The network is
trained inside those 455 only, and no held-out basin's observations are read until it is
scored. See model.py for how K is chosen.

    reads   runs/lstm/ungauged/_mean/fold{f}.npz
    writes  runs/lstm/mlp/ungauged/{predictions.npz, params.csv, tuning.csv}
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from model import EPOCHS, K_GRID, SEEDS, run_fold  # noqa: E402
from shared import (CACHE_DIR, distance_matrix, fold_labels, kge_per_basin,  # noqa: E402
                    load_mean, nearest_gauge, observed, save_corrected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, nargs="+", help="default: every fold present")
    ap.add_argument("--k", type=int, nargs="+", default=K_GRID)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    a = ap.parse_args()
    mean_dir = "runs/lstm/ungauged/_mean"

    labels = fold_labels()
    dist = distance_matrix()
    folds = a.fold or [f for f in sorted(set(labels))
                       if os.path.exists(f"{mean_dir}/fold{f}.npz")]
    dates = np.load(f"{CACHE_DIR}/dates.npy")

    raw = corrected = nearest = None
    params, tuning = [], []
    for f in folds:
        sim, test_t, n_seeds = load_mean(f"{mean_dir}/fold{f}.npz")
        obs = observed(test_t)
        doy = pd.to_datetime(dates[test_t]).dayofyear.to_numpy() - 1
        if raw is None:
            raw = np.full(sim.shape, np.nan, np.float32)
            corrected, nearest = np.full(sim.shape, np.nan), np.full(sim.shape, np.nan)
        held_mean, held_runs, k, val_group, val_score, rows = run_fold(
            sim, obs, labels, dist, f, "ungauged", doy, a.k, a.seeds)
        tuning += rows

        # The held-out basins' observations are read from here on, and only to score.
        held, pool = np.flatnonzero(labels == f), np.flatnonzero(labels != f)
        raw[held], corrected[held] = sim[held], held_mean
        nearest[held] = nearest_gauge(sim[held], sim[pool], obs[pool], dist[np.ix_(held, pool)])
        before = np.nanmedian(kge_per_basin(raw[held], obs[held]))
        after = np.nanmedian(kge_per_basin(corrected[held], obs[held]))
        seeds = [np.nanmedian(kge_per_basin(c, obs[held])) for c in held_runs]
        params.append(dict(fold=f, k=k, n_basins=len(held), n_donors=len(pool), n_seeds=n_seeds,
                           n_mlp_seeds=a.seeds, epochs=EPOCHS, val_group=val_group,
                           select_on="kge", val_score=val_score,
                           score_before=before, score_after=after))
        print(f"  fold {f} ({len(held)} basins)  K={k}   KGE {before:.2f} -> {after:.2f}; "
              f"seeds {min(seeds):.2f}-{max(seeds):.2f}", flush=True)

    save_corrected("runs/lstm/mlp/ungauged", raw, corrected, nearest, test_t, params, tuning)


if __name__ == "__main__":
    main()
