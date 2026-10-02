"""Correct the gauged-basin predictions with the learned MLP correction: every basin reconstructed
as if its own gauge went dark.

    python src/corrections/mlp/01_gauged.py
    python src/corrections/mlp/01_gauged.py --group 0 --k 10 --seeds 1   # quick check

The same cell and the same donors as src/corrections/idw/01_gauged.py. The LSTM was trained on
all 531 basins, so every basin is out of sample in time only. When group g is corrected, the
network trains on the other six groups only -- group g is neither a training target nor a
training donor -- and each basin of group g is then corrected from all 530 others, its own
record masked out, the network a real gone-dark gauge still has around it.

    reads   runs/lstm/gauged/_mean/predictions.npz
    writes  runs/lstm/mlp/gauged/{predictions.npz, params.csv, tuning.csv}
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
    ap.add_argument("--group", type=int, nargs="+", help="default: every group")
    ap.add_argument("--k", type=int, nargs="+", default=K_GRID)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    a = ap.parse_args()

    sim, test_t, n_seeds = load_mean("runs/lstm/gauged/_mean/predictions.npz")
    obs = observed(test_t)
    doy = pd.to_datetime(np.load(f"{CACHE_DIR}/dates.npy")[test_t]).dayofyear.to_numpy() - 1
    labels = fold_labels()
    dist = distance_matrix()

    raw = np.full(sim.shape, np.nan, np.float32)
    corrected, nearest = np.full(sim.shape, np.nan), np.full(sim.shape, np.nan)
    params, tuning = [], []
    for g in a.group or sorted(set(labels)):
        held_mean, held_runs, k, val_group, val_score, rows = run_fold(
            sim, obs, labels, dist, g, "gauged", doy, a.k, a.seeds)
        tuning += rows

        target = np.flatnonzero(labels == g)
        d = dist[target].copy()
        d[np.arange(len(target)), target] = np.inf  # a basin can never be its own donor
        raw[target], corrected[target] = sim[target], held_mean
        nearest[target] = nearest_gauge(sim[target], sim, obs, d)
        before = np.nanmedian(kge_per_basin(raw[target], obs[target]))
        after = np.nanmedian(kge_per_basin(corrected[target], obs[target]))
        seeds = [np.nanmedian(kge_per_basin(c, obs[target])) for c in held_runs]
        params.append(dict(fold=g, k=k, n_basins=len(target), n_donors=len(sim) - 1,
                           n_seeds=n_seeds, n_mlp_seeds=a.seeds, epochs=EPOCHS,
                           val_group=val_group, select_on="kge", val_score=val_score,
                           score_before=before, score_after=after))
        print(f"  group {g} ({len(target)} basins)  K={k}   KGE {before:.2f} -> {after:.2f}; "
              f"seeds {min(seeds):.2f}-{max(seeds):.2f}", flush=True)

    save_corrected("runs/lstm/mlp/gauged", raw, corrected, nearest, test_t, params, tuning)


if __name__ == "__main__":
    main()
