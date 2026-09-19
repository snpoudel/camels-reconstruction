"""Correct the ungauged-basin predictions: one fold at a time, donors from that fold's gauges.

    python src/02_lstm_ungauged_correction.py
    python src/02_lstm_ungauged_correction.py --model hybrid

Each fold held a seventh of the basins out of training, so those basins are ungauged in space as
well as in time -- the prediction-in-ungauged-basins case. They are corrected using that fold's
other 455 basins, whose gauges the model did see, as donors.

K, lambda and p are cross-validated inside those 455 training basins only, leave-one-group-out:
the training basins keep their original fold labels, so each of the six groups is scored using
donors from the other five -- a thinned gauge network, like a real ungauged basin faces. No
held-out basin is read during selection, of this fold or of any other.

    reads   runs/<model>/ungauged/_mean/fold{f}.npz
    writes  runs/<model>/ungauged_corrected/{predictions.npz, params.csv, tuning.csv}

--select-on picks what the search maximises, kge (default) or nse; a non-default objective
writes to ungauged_corrected_<objective>/ so both can sit side by side.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parameter_tuning import SCORERS, make_job, pick, run_jobs, surface_rows  # noqa: E402

from residual_postprocessing import (basin_ids, correct, distance_matrix,  # noqa: E402
                                     fold_labels, load_mean, observed,
                                     save_corrected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", help="which runs/<model>/ tree to correct")
    ap.add_argument("--workers", type=int, default=None, help="processes for the grid search")
    ap.add_argument("--select-on", choices=list(SCORERS), default="kge",
                    help="which score the grid search maximises (default kge)")
    a = ap.parse_args()
    mean_dir = f"runs/{a.model}/ungauged/_mean"

    labels = fold_labels()
    dist = distance_matrix()
    folds = [f for f in sorted(set(labels)) if os.path.exists(f"{mean_dir}/fold{f}.npz")]
    if not folds:
        raise SystemExit(f"no fold*.npz under {mean_dir}/ -- run src/lstm/02_lstm_ungauged.py")

    # Each fold has its own model, so a fold's donors and its held-out basins must both be
    # predicted by that same model -- hence one matrix per fold rather than one global one.
    pred, n_seeds, test_t = {}, {}, None
    for f in folds:
        pred[f], t, n_seeds[f] = load_mean(f"{mean_dir}/fold{f}.npz")
        assert test_t is None or np.array_equal(t, test_t), "folds cover different test periods"
        test_t = t
    obs = observed(test_t)
    print(f"folds present: {folds}, {len(test_t)} test days")

    pool = {f: np.flatnonzero(labels != f) for f in folds}
    surfaces = run_jobs([make_job(f, pred[f], obs, dist, pool[f], groups=labels,
                                  objective=a.select_on) for f in folds], workers=a.workers)
    score = SCORERS[a.select_on]
    picked = {f: pick(surfaces[f]) for f in folds}

    raw = np.full((len(basin_ids()), len(test_t)), np.nan, np.float32)
    corrected = np.full(raw.shape, np.nan)  # float64: what the correction's arithmetic returns
    params = []
    for f in folds:
        k, lam, p = picked[f]
        held = np.flatnonzero(labels == f)
        raw[held] = pred[f][held]
        corrected[held] = correct(pred[f][held], pred[f][pool[f]], obs[pool[f]],
                                  dist[np.ix_(held, pool[f])], k, lam, p)
        before = np.nanmedian(score(raw[held], obs[held]))
        after = np.nanmedian(score(corrected[held], obs[held]))
        params.append(dict(fold=f, k=k, lam=lam, p=p, n_basins=len(held), n_donors=len(pool[f]),
                           n_seeds=n_seeds[f], select_on=a.select_on,
                           score_before=before, score_after=after))
        print(f"  fold {f} ({n_seeds[f]} seeds, {len(held)} basins)  K={k:>2} lambda={lam:.2f} "
              f"p={p:.2g}   {a.select_on.upper()} {before:.4f} -> {after:.4f} "
              f"({after - before:+.4f})")

    done = np.flatnonzero(np.isfinite(raw).any(1))
    before = np.nanmedian(score(raw[done], obs[done]))
    after = np.nanmedian(score(corrected[done], obs[done]))
    print()
    print(f"{len(done)} held-out basins pooled: {a.select_on.upper()} {before:.4f} -> {after:.4f} "
          f"({after - before:+.4f})")

    suffix = "" if a.select_on == "kge" else f"_{a.select_on}"
    save_corrected(f"runs/{a.model}/ungauged_corrected{suffix}", raw, corrected, test_t, params,
                   surface_rows(surfaces, picked))


if __name__ == "__main__":
    main()
