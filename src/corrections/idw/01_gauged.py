"""Correct the gauged-basin predictions: every basin reconstructed as if its own gauge went dark.

    python src/corrections/idw/01_gauged.py
    python src/corrections/idw/01_gauged.py --model hybrid

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
    writes  runs/<model>/idw/gauged/{predictions.npz, params.csv, tuning.csv}

--select-on picks what the search maximises: nse_star (default) -- the basin-normalised loss
the LSTM itself was trained on -- kge, or nse. A non-default objective
writes to idw/gauged_<objective>/ so both can sit side by side.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from correction import correct  # noqa: E402
from shared import (distance_matrix, fold_labels, load_mean, observed,  # noqa: E402
                    save_corrected, train_sigma)
from tuning import SCORERS, make_job, pick, run_jobs, surface_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", help="which runs/<model>/ tree to correct")
    ap.add_argument("--workers", type=int, default=None, help="processes for the grid search")
    ap.add_argument("--no-scale", action="store_true",
                    help="the unscaled comparison: residuals in plain mm/day -> runs/<model>/idw_noscale/")
    ap.add_argument("--select-on", choices=list(SCORERS), default="nse_star",
                    help="which score the grid search maximises (default nse_star)")
    a = ap.parse_args()
    mean_npz = f"runs/{a.model}/gauged/_mean/predictions.npz"
    if not os.path.exists(mean_npz):
        raise SystemExit(f"{mean_npz} not found -- run src/models/lstm/01_lstm_gauged.py")

    pred, test_t, n_seeds = load_mean(mean_npz)
    sigma = train_sigma(mean_npz) if a.select_on == "nse_star" else None
    obs = observed(test_t)
    labels = fold_labels()
    dist = distance_matrix()
    groups = sorted(set(labels))
    print(f"{len(pred)} basins, {n_seeds} seeds, {len(test_t)} test days")

    # Group g is dropped from its own search: it appears neither as a CV target nor as a donor.
    surfaces = run_jobs([make_job(g, pred, obs, sigma, dist, np.flatnonzero(labels != g),
                                  objective=a.select_on, scaled=not a.no_scale)
                         for g in groups], workers=a.workers)
    picked = {g: pick(surfaces[g], a.select_on) for g in groups}

    obj = SCORERS[a.select_on]

    def report(sim, o, rows):
        """The objective's own number for a set of basins, in the direction it is read: a loss
        is reported positive and falling, not as the negated score the search maximises."""
        v = obj.aggregate(obj.score(sim, o, None if sigma is None else sigma[rows]))
        return -v if obj.is_loss else v

    raw = pred.copy()
    corrected = np.full(pred.shape, np.nan)  # float64: what the correction's arithmetic returns
    nearest = np.full(pred.shape, np.nan)
    params = []
    for g in groups:
        k, lam, p = picked[g]
        target = np.flatnonzero(labels == g)
        d = dist[target].copy()
        d[np.arange(len(target)), target] = np.inf  # a basin can never be its own donor
        corrected[target] = correct(pred[target], pred, obs, d, k, lam, p, not a.no_scale)
        # The parameter-free reference: one donor, all of its error. p is irrelevant at K=1,
        # since the single surviving weight normalises to 1 whatever the exponent.
        nearest[target] = correct(pred[target], pred, obs, d, 1, 1.0, 1.0, not a.no_scale)
        before = report(raw[target], obs[target], target)
        after = report(corrected[target], obs[target], target)
        params.append(dict(fold=g, k=k, lam=lam, p=p, n_basins=len(target),
                           n_donors=len(pred) - 1, n_seeds=n_seeds,
                           select_on=a.select_on, score_before=before, score_after=after))
        print(f"  group {g} ({len(target)} basins)  K={k:>2} lambda={lam:.2f} p={p:.2g}   "
              f"{obj.label} {before:.4f} -> {after:.4f} ({after - before:+.4f})")

    everything = np.arange(len(pred))
    before = report(raw, obs, everything)
    after = report(corrected, obs, everything)
    print()
    print(f"{len(pred)} basins pooled: {obj.label} {before:.4f} -> {after:.4f} "
          f"({after - before:+.4f})")

    suffix = "" if a.select_on == "nse_star" else f"_{a.select_on}"
    method = "idw_noscale" if a.no_scale else "idw"
    save_corrected(f"runs/{a.model}/{method}/gauged{suffix}", raw, corrected, nearest, test_t,
                   params, surface_rows(surfaces, picked, a.select_on))


if __name__ == "__main__":
    main()
