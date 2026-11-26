"""Grid search for IDW's K, lambda and p by cross-validation.

Per fold (ungauged) or group (gauged), every grid point is scored on a pool of basins that
excludes the ones being corrected, each pool basin corrected from the other pool basins
(self-masked leave-one-out). One process per fold/group.
"""
import concurrent.futures as cf
import os
from collections import namedtuple

import numpy as np

from correction import apply_shrinkage, idw_weights, neighbour_residual
from shared import kge_per_basin, nse_per_basin

# NSE* offset from Kratzert et al. (2019), mm/day.
SIGMA_EPS = 0.1


def _kge(sim, obs, sigma):
    return kge_per_basin(sim, obs)


def _nse(sim, obs, sigma):
    return nse_per_basin(sim, obs)


def _nse_star(sim, obs, sigma):
    """[n] negated NSE* loss per basin, so that higher is better like the other objectives."""
    m = np.isfinite(obs)
    err = ((np.clip(sim, 0, None) - obs) / (sigma[:, None] + SIGMA_EPS)) ** 2
    n = m.sum(1)
    return np.where(n > 0, -np.where(m, err, 0.0).sum(1) / np.maximum(n, 1), np.nan)


Objective = namedtuple("Objective", "score aggregate is_loss label")
"""How a surface is turned into one number, and which direction is better."""

SCORERS = {
    "kge": Objective(_kge, np.nanmedian, False, "KGE"),
    "nse": Objective(_nse, np.nanmedian, False, "NSE"),
    "nse_star": Objective(_nse_star, np.nanmean, True, "NSE* loss"),
}

K_GRID = tuple(range(1, 40, 2))
LAM_GRID = tuple(np.round(np.arange(0.0, 1.05, 0.05), 2))
P_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
Job = namedtuple("Job", "label pred obs sigma dist objective scaled")
"""One fold's or group's search. `pred`/`obs` are the CV pool's rows, `dist` the pool's own
distance matrix with an infinite diagonal, `sigma` their training-period flow stds, `objective`
the key in `SCORERS` the search maximises, `scaled` whether residuals are divided by s."""


def _score_grid(pred_t, obs_t, sigma_t, pred_d, obs_d, dist, score, scaled):
    """{(k, lam, p): [n_target] score} for one target/donor split."""
    out = {}
    for k in K_GRID:
        for p in P_GRID:
            R = neighbour_residual(pred_d, obs_d, idw_weights(dist, k, p), scaled)
            for lam in LAM_GRID:
                out[(int(k), float(lam), float(p))] = score(
                    apply_shrinkage(pred_t, R, lam, scaled), obs_t, sigma_t)
    return out


def _surface(job):
    """{(k, lam, p): [n_pool] score}, each pool basin corrected from the others."""
    return job.label, _score_grid(job.pred, job.obs, job.sigma, job.pred, job.obs, job.dist,
                                  SCORERS[job.objective].score, job.scaled)


def _init_worker():
    """One BLAS thread per worker, so the worker processes do not oversubscribe the CPU."""
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def make_job(label, pred, obs, sigma, dist, rows, objective="nse_star", scaled=True):
    """One label's search, self-masked leave-one-out over `rows`, the basins it may read."""
    d = dist[np.ix_(rows, rows)].copy()
    # Without this each basin is its own zero-distance donor and the search picks lambda = 0.
    np.fill_diagonal(d, np.inf)
    return Job(label, pred[rows], obs[rows], None if sigma is None else sigma[rows], d, objective,
               scaled)


def run_jobs(jobs, workers=None):
    """{label: surface}, one process per job."""
    n_workers = min(len(jobs), workers or max(1, (os.cpu_count() or 1) - 4))
    if n_workers <= 1:
        return dict(_surface(j) for j in jobs)
    with cf.ProcessPoolExecutor(n_workers, initializer=_init_worker) as ex:
        return dict(ex.map(_surface, jobs))


def pick(surf, objective="nse_star"):
    """The best-scoring (K, lambda, p); exact ties go to the smallest."""
    obj = SCORERS[objective]
    scored = {c: obj.aggregate(v) for c, v in surf.items()}
    best = max(scored.values())
    return min(c for c, v in scored.items() if v == best)


def surface_rows(surfaces, picked, objective="nse_star"):
    """The searched surfaces as flat records, one row per label x k x lam x p."""
    rows = []
    for label, surf in surfaces.items():
        for (k, lam, p), vals in surf.items():
            rows.append(dict(label=label, k=k, lam=lam, p=p,
                             score=SCORERS[objective].aggregate(vals),
                             n_basins=len(vals), selected=(k, lam, p) == picked[label]))
    return rows
