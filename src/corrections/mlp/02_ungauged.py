"""Correct the ungauged-basin predictions with the learned MLP correction, one fold at a time.

    python src/corrections/mlp/02_ungauged.py --fold 0
    python src/corrections/mlp/02_ungauged.py --fold 0 --k 10 --seeds 1   # timing run

The same cell and the same donors as src/corrections/idw/02_ungauged.py: a fold's held-out
basins are corrected from its 455 training basins, using that fold's own LSTM. The MLP is
trained inside those 455 only, each basin's donors drawn from the other groups -- IDW's
leave-one-group-out search -- and the pool group (fold + 1) % 7 is kept back to decide when
training stops and which K is used. No held-out basin's observations are read until every
model for the fold is trained.

K is chosen like IDW's settings: the median KGE of the seed-averaged validation hydrographs,
ties to the smaller K. The held-out correction is the average of the seeds' hydrographs.

    reads   runs/lstm/ungauged/_mean/fold{f}.npz
    writes  runs/lstm/mlp/ungauged/{predictions.npz, params.csv, tuning.csv}
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from model import context, correct, splits, train  # noqa: E402
from shared import (CACHE_DIR, distance_matrix, fold_labels, kge_per_basin,  # noqa: E402
                    load_mean, nearest_gauge, observed, save_corrected)

MODEL = "lstm"
K_GRID = (1, 3, 5, 10, 15, 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, nargs="+", help="default: every fold present")
    ap.add_argument("--k", type=int, nargs="+", default=K_GRID)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    mean_dir = f"runs/{MODEL}/ungauged/_mean"

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
        ctx = context(sim, obs, labels, f, doy)
        sim_t = torch.as_tensor(sim)
        if raw is None:
            raw = np.full(sim.shape, np.nan, np.float32)
            corrected, nearest = np.full(sim.shape, np.nan), np.full(sim.shape, np.nan)

        val_score, held_runs = {}, {}
        for k in a.k:
            split, val_group = splits(labels, dist, f, k)
            val_runs, held_runs[k] = [], []
            for s in range(a.seeds):
                start = time.time()
                model, epoch, score = train(s, sim_t, obs, ctx, split["train"], split["val"])
                val_runs.append(correct(model, sim_t, ctx, split["val"]))
                held_runs[k].append(correct(model, sim_t, ctx, split["held"]))
                tuning.append(dict(label=f, k=k, seed=s, best_epoch=epoch, score=score))
                print(f"  fold {f} K={k:>2} seed {s}: best epoch {epoch:>3}, validation KGE "
                      f"{score:.4f}  ({(time.time() - start) / 60:.1f} min)", flush=True)
            val_score[k] = np.nanmedian(kge_per_basin(np.mean(val_runs, 0),
                                                      obs[split["val"][0].numpy()]))
            tuning.append(dict(label=f, k=k, seed="mean", best_epoch=np.nan, score=val_score[k]))
            print(f"  fold {f} K={k:>2}: {a.seeds}-seed mean, validation KGE {val_score[k]:.4f}")
        best = max(val_score.values())
        k = min(c for c, v in val_score.items() if v == best)
        for row in tuning:
            row.setdefault("selected", False)
            if row["label"] == f and row["k"] == k:
                row["selected"] = True

        # The held-out basins' observations are read from here on, and only to score.
        held, pool = np.flatnonzero(labels == f), np.flatnonzero(labels != f)
        raw[held] = sim[held]
        corrected[held] = np.mean(held_runs[k], 0)
        nearest[held] = nearest_gauge(sim[held], sim[pool], obs[pool], dist[np.ix_(held, pool)])
        before = np.nanmedian(kge_per_basin(raw[held], obs[held]))
        after = np.nanmedian(kge_per_basin(corrected[held], obs[held]))
        per_seed = [np.nanmedian(kge_per_basin(c, obs[held])) for c in held_runs[k]]
        params.append(dict(fold=f, k=k, n_basins=len(held), n_donors=len(pool), n_seeds=n_seeds,
                           n_mlp_seeds=a.seeds, val_group=val_group, select_on="kge",
                           val_score=val_score[k], score_before=before, score_after=after))
        print(f"  fold {f} ({len(held)} basins)  K={k}   KGE {before:.4f} -> {after:.4f} "
              f"({after - before:+.4f}); per seed "
              + ", ".join(f"{v:.4f}" for v in per_seed), flush=True)

    save_corrected(f"runs/{MODEL}/mlp/ungauged", raw, corrected, nearest, test_t, params, tuning)


if __name__ == "__main__":
    main()
