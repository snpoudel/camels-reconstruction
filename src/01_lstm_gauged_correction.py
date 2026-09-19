"""Correct the gauged-basin predictions: every basin reconstructed as if its own gauge went dark.

    python src/01_lstm_gauged_correction.py
    python src/01_lstm_gauged_correction.py --model hybrid

This model was trained on all 531 basins, so every basin's test-period prediction is
out-of-sample in time only -- the discontinued-gauge case. Each basin is corrected using all 530
others as donors, its own record masked out by an infinite distance to itself, which is the
network a real gone-dark gauge still has around it.

The scenario is one gauge dark at a time, not a whole region going dark together: a basin's
neighbours -- including the ones in its own fold group -- are live gauges and legitimate donors.
What is never read is the basin's own record. Both halves of that were audited: destroying a
basin's discharge leaves its own corrected hydrograph bit-identical while moving its neighbours'
(it is still a donor to them), and destroying a whole group's discharge leaves that group's
selected (K, lambda, p) unchanged.

K, lambda and p are cross-validated with the target gauge absent from the process entirely. The
basins keep their fold labels, and when group g is being corrected the search runs on the other
six groups only -- group g is neither a cross-validation target nor a cross-validation donor, so
no basin's own record can reach the settings applied to it. Within those six groups the search
is self-masked leave-one-out, matching the dense donor network the correction is then applied
under.

    reads   runs/<model>/gauged/_mean/predictions.npz
    writes  runs/<model>/gauged_corrected/{predictions.npz, params.csv, tuning.csv}

--select-on picks what the search maximises, kge (default) or nse; a non-default objective
writes to gauged_corrected_<objective>/ so both can sit side by side.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parameter_tuning import SCORERS, make_job, pick, run_jobs, surface_rows  # noqa: E402

from residual_postprocessing import (correct, distance_matrix, fold_labels,  # noqa: E402
                                     load_mean, observed, save_corrected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", help="which runs/<model>/ tree to correct")
    ap.add_argument("--workers", type=int, default=None, help="processes for the grid search")
    ap.add_argument("--select-on", choices=list(SCORERS), default="kge",
                    help="which score the grid search maximises (default kge)")
    a = ap.parse_args()
    mean_npz = f"runs/{a.model}/gauged/_mean/predictions.npz"
    if not os.path.exists(mean_npz):
        raise SystemExit(f"{mean_npz} not found -- run src/lstm/01_lstm_gauged.py")

    pred, test_t, n_seeds = load_mean(mean_npz)
    obs = observed(test_t)
    labels = fold_labels()
    dist = distance_matrix()
    groups = sorted(set(labels))
    print(f"{len(pred)} basins, {n_seeds} seeds, {len(test_t)} test days")

    # Group g is dropped from its own search: it appears neither as a CV target nor as a donor.
    surfaces = run_jobs([make_job(g, pred, obs, dist, np.flatnonzero(labels != g),
                                  objective=a.select_on) for g in groups], workers=a.workers)
    score = SCORERS[a.select_on]
    picked = {g: pick(surfaces[g]) for g in groups}

    raw = pred.copy()
    corrected = np.full(pred.shape, np.nan)  # float64: what the correction's arithmetic returns
    params = []
    for g in groups:
        k, lam, p = picked[g]
        target = np.flatnonzero(labels == g)
        d = dist[target].copy()
        d[np.arange(len(target)), target] = np.inf  # a basin can never be its own donor
        corrected[target] = correct(pred[target], pred, obs, d, k, lam, p)
        before = np.nanmedian(score(raw[target], obs[target]))
        after = np.nanmedian(score(corrected[target], obs[target]))
        params.append(dict(fold=g, k=k, lam=lam, p=p, n_basins=len(target),
                           n_donors=len(pred) - 1, n_seeds=n_seeds,
                           select_on=a.select_on, score_before=before, score_after=after))
        print(f"  group {g} ({len(target)} basins)  K={k:>2} lambda={lam:.2f} p={p:.2g}   "
              f"{a.select_on.upper()} {before:.4f} -> {after:.4f} ({after - before:+.4f})")

    before = np.nanmedian(score(raw, obs))
    after = np.nanmedian(score(corrected, obs))
    print()
    print(f"{len(pred)} basins pooled: {a.select_on.upper()} {before:.4f} -> {after:.4f} "
          f"({after - before:+.4f})")

    suffix = "" if a.select_on == "kge" else f"_{a.select_on}"
    save_corrected(f"runs/{a.model}/gauged_corrected{suffix}", raw, corrected, test_t, params,
                   surface_rows(surfaces, picked))


if __name__ == "__main__":
    main()
