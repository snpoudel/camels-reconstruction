"""IDW correction, gauged cell: each basin is corrected as if its own gauge stopped reporting.

    python src/corrections/idw/01_gauged.py [--model lstm] [--select-on nse_star|kge|nse] [--no-scale]

Each basin is corrected from all other basins as donors, never itself. The settings for group g
are selected on the other groups only, so no basin's own record reaches the settings applied to it.

    reads   runs/<model>/gauged/_mean/predictions.npz
    writes  runs/<model>/idw/gauged/{predictions.npz, params.csv, tuning.csv}
            (idw_noscale/ with --no-scale, gauged_<objective>/ for a non-default --select-on)
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

    # Group g is left out of its own search, as a target and as a donor.
    surfaces = run_jobs([make_job(g, pred, obs, sigma, dist, np.flatnonzero(labels != g),
                                  objective=a.select_on, scaled=not a.no_scale)
                         for g in groups], workers=a.workers)
    picked = {g: pick(surfaces[g], a.select_on) for g in groups}

    obj = SCORERS[a.select_on]

    def report(sim, o, rows):
        """The objective's aggregate over `rows`, with a loss reported as a positive number."""
        v = obj.aggregate(obj.score(sim, o, None if sigma is None else sigma[rows]))
        return -v if obj.is_loss else v

    raw = pred.copy()
    corrected = np.full(pred.shape, np.nan)
    nearest = np.full(pred.shape, np.nan)
    params = []
    for g in groups:
        k, lam, p = picked[g]
        target = np.flatnonzero(labels == g)
        d = dist[target].copy()
        d[np.arange(len(target)), target] = np.inf  # a basin can never be its own donor
        corrected[target] = correct(pred[target], pred, obs, d, k, lam, p, not a.no_scale)
        # Nearest-gauge reference; p has no effect at K=1.
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
