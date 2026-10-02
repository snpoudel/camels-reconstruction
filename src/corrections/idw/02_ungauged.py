"""Correct the ungauged-basin predictions: one fold at a time, donors from that fold's gauges.

    python src/corrections/idw/02_ungauged.py
    python src/corrections/idw/02_ungauged.py --model hybrid

Each fold held a seventh of the basins out of training, so those basins are ungauged in space as
well as in time -- the prediction-in-ungauged-basins case. They are corrected using that fold's
other 455 basins, whose gauges the model did see, as donors.

K, lambda and p are cross-validated inside those 455 training basins only, self-masked
leave-one-out: each is scored using the other 454 as donors, the same donor network a held-out
basin has. No held-out basin is read during selection, as a target or as a donor.

    reads   runs/<model>/ungauged/_mean/fold{f}.npz
    writes  runs/<model>/idw/ungauged/{predictions.npz, params.csv, tuning.csv}

--select-on picks what the search maximises: kge (default), nse, or nse_star -- the
basin-normalised loss the LSTM itself was trained on. A non-default objective
writes to idw/ungauged_<objective>/ so both can sit side by side.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from correction import correct  # noqa: E402
from shared import (basin_ids, distance_matrix, fold_labels, load_mean,  # noqa: E402
                    observed, save_corrected, train_sigma)
from tuning import SCORERS, make_job, pick, run_jobs, surface_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", help="which runs/<model>/ tree to correct")
    ap.add_argument("--workers", type=int, default=None, help="processes for the grid search")
    ap.add_argument("--select-on", choices=list(SCORERS), default="kge",
                    help="which score the grid search maximises (default kge)")
    ap.add_argument("--folds-csv", default=None,
                    help="override the fold assignment (e.g. a model whose held-out basins "
                         "aren't the LSTM's own 7-way split)")
    a = ap.parse_args()
    mean_dir = f"runs/{a.model}/ungauged/_mean"

    labels = fold_labels(**({"folds_csv": a.folds_csv} if a.folds_csv else {}))
    dist = distance_matrix()
    folds = [f for f in sorted(set(labels)) if os.path.exists(f"{mean_dir}/fold{f}.npz")]
    if not folds:
        raise SystemExit(f"no fold*.npz under {mean_dir}/ -- "
                         f"run src/models/lstm/02_lstm_ungauged.py")

    # Each fold has its own model, so a fold's donors and its held-out basins must both be
    # predicted by that same model -- hence one matrix per fold rather than one global one.
    pred, n_seeds, test_t, sigma = {}, {}, None, None
    for f in folds:
        pred[f], t, n_seeds[f] = load_mean(f"{mean_dir}/fold{f}.npz")
        assert test_t is None or np.array_equal(t, test_t), "folds cover different test periods"
        test_t = t
        if a.select_on == "nse_star":
            sigma = train_sigma(f"{mean_dir}/fold{f}.npz")
    obs = observed(test_t)
    print(f"folds present: {folds}, {len(test_t)} test days")

    pool = {f: np.flatnonzero(labels != f) for f in folds}
    surfaces = run_jobs([make_job(f, pred[f], obs, sigma, dist, pool[f], objective=a.select_on)
                         for f in folds], workers=a.workers)
    picked = {f: pick(surfaces[f], a.select_on) for f in folds}

    obj = SCORERS[a.select_on]

    def report(sim, o, rows):
        """The objective's own number for a set of basins, in the direction it is read: a loss
        is reported positive and falling, not as the negated score the search maximises."""
        v = obj.aggregate(obj.score(sim, o, None if sigma is None else sigma[rows]))
        return -v if obj.is_loss else v

    raw = np.full((len(basin_ids()), len(test_t)), np.nan, np.float32)
    corrected = np.full(raw.shape, np.nan)  # float64: what the correction's arithmetic returns
    nearest = np.full(raw.shape, np.nan)
    params = []
    for f in folds:
        k, lam, p = picked[f]
        held = np.flatnonzero(labels == f)
        raw[held] = pred[f][held]
        d = dist[np.ix_(held, pool[f])]
        corrected[held] = correct(pred[f][held], pred[f][pool[f]], obs[pool[f]], d, k, lam, p)
        # The parameter-free reference: one donor, all of its error. p is irrelevant at K=1,
        # since the single surviving weight normalises to 1 whatever the exponent.
        nearest[held] = correct(pred[f][held], pred[f][pool[f]], obs[pool[f]], d, 1, 1.0, 1.0)
        before = report(raw[held], obs[held], held)
        after = report(corrected[held], obs[held], held)
        params.append(dict(fold=f, k=k, lam=lam, p=p, n_basins=len(held), n_donors=len(pool[f]),
                           n_seeds=n_seeds[f], select_on=a.select_on,
                           score_before=before, score_after=after))
        print(f"  fold {f} ({n_seeds[f]} seeds, {len(held)} basins)  K={k:>2} lambda={lam:.2f} "
              f"p={p:.2g}   {obj.label} {before:.4f} -> {after:.4f} "
              f"({after - before:+.4f})")

    done = np.flatnonzero(np.isfinite(raw).any(1))
    before, after = report(raw[done], obs[done], done), report(corrected[done], obs[done], done)
    print()
    print(f"{len(done)} held-out basins pooled: {obj.label} {before:.4f} -> {after:.4f} "
          f"({after - before:+.4f})")

    suffix = "" if a.select_on == "kge" else f"_{a.select_on}"
    save_corrected(f"runs/{a.model}/idw/ungauged{suffix}", raw, corrected, nearest, test_t,
                   params, surface_rows(surfaces, picked, a.select_on))


if __name__ == "__main__":
    main()
