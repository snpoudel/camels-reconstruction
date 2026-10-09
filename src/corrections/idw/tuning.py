"""Choosing K, lambda and p by cross-validation, without ever reading the target's own gauge.

The correction has three free numbers: K, how many gauges to consult; lambda, how much of their
error to believe; and p, how steeply weight falls off with distance. All three are selected per
fold (ungauged) or per group (gauged) on a surface built **only from basins other than the ones
being corrected** -- the target's observations are not in the surface, so the reported result
cannot have been tuned on itself.

The surface is self-masked leave-one-out in both cells: every basin the search may read is
corrected from every other one, itself excluded. That reproduces the donor network the
correction is then applied under. A fold's held-out basins draw on the 455 pool basins, and a
pool basin under self-masking draws on the other 454, so the two see the same donor distances
rank for rank. Leave-one-group-out would remove a second group on top of the held-out fold,
leaving the search about 10% sparser than the network it chooses settings for.

One process per fold/group, so a cell is a few minutes.
"""
import concurrent.futures as cf
import os
from collections import namedtuple

import numpy as np

from correction import apply_shrinkage, idw_weights, neighbour_residual
from shared import kge_per_basin, nse_per_basin

# Kratzert et al. (2019)'s NSE*, the loss the LSTM itself was trained to minimise: squared
# error divided by each basin's training-period flow std. The 0.1 offset is theirs, in mm/day,
# and stops near-constant basins dominating.
SIGMA_EPS = 0.1


def _kge(sim, obs, sigma):
    return kge_per_basin(sim, obs)


def _nse(sim, obs, sigma):
    return nse_per_basin(sim, obs)


def _nse_star(sim, obs, sigma):
    """[n] negated NSE*, per basin: mean squared error over that basin's reported days,
    normalised by its training-period flow std. Negated so that higher is better, like the
    other two, and the search stays a maximisation.

    Averaged within a basin and then across basins, rather than pooled over all basin-days as
    the training loss is. The test record is 99.99% complete, so the two are numerically
    indistinguishable here, and per-basin values are what the surface and the figures store.
    """
    m = np.isfinite(obs)
    err = ((np.clip(sim, 0, None) - obs) / (sigma[:, None] + SIGMA_EPS)) ** 2
    n = m.sum(1)
    return np.where(n > 0, -np.where(m, err, 0.0).sum(1) / np.maximum(n, 1), np.nan)


Objective = namedtuple("Objective", "score aggregate is_loss label")
"""How a surface is turned into one number, and which direction is better."""

# What the search maximises. The three disagree about this correction by more than a
# relabelling -- see manuscript.md 2.6 and experiments_log.md, where each was measured rather than
# assumed.
SCORERS = {
    "kge": Objective(_kge, np.nanmedian, False, "KGE"),
    "nse": Objective(_nse, np.nanmedian, False, "NSE"),
    "nse_star": Objective(_nse_star, np.nanmean, True, "NSE* loss"),
}

# Odd K from 1 to 39. The surface is flat enough near its optimum that stepping by 2 cannot
# jump over it. Several folds select K near this ceiling under NSE and NSE* -- checked whether
# that is the same overfitting p shows below, and it is not: cross-validation score and
# held-out score move together and both flatten out (fold 0 under NSE*: held-out loss
# 0.3348 -> 0.3324 from K=3 to K=39, converging, never reversing). The pin just means the
# surface is very flat past K~20, not that more donors would help or hurt if the grid reached
# further.
K_GRID = tuple(range(1, 40, 2))
LAM_GRID = tuple(np.round(np.arange(0.0, 1.05, 0.05), 2))
# The IDW exponent: 1 is plain inverse distance, 2 the classic Shepard weighting, and 0.5
# flattens toward a simple average over the K donors. Steps of 0.5 across that range.
#
# Extending this past 3 looks at first like the same truncated-search problem K had -- several
# folds pin at the boundary -- but it is not: at high p, weight concentrates on the single
# nearest donor, so the cross-validation surface keeps climbing indefinitely (checked to p=5,
# never flattening) while the held-out score for the pinned fold gets WORSE at every step
# (fold 6: p=3 -> 0.7248, p=4 -> 0.7200, p=5 -> 0.7184). That is overfitting to the
# cross-validation pool, not information the grid was cutting off, so the ceiling stays at 3.
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
    """{(k, lam, p): [n_pool] score}, every pool basin scored once, self masked by the infinite
    diagonal."""
    return job.label, _score_grid(job.pred, job.obs, job.sigma, job.pred, job.obs, job.dist,
                                  SCORERS[job.objective].score, job.scaled)


def _init_worker():
    """Cap BLAS to one thread per worker: process parallelism and BLAS's own intra-op threads
    must not both be on, or N workers oversubscribe the machine by N x 24 threads."""
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def make_job(label, pred, obs, sigma, dist, rows, objective="nse_star", scaled=True):
    """One label's search, self-masked leave-one-out over `rows` -- the basins it may read."""
    d = dist[np.ix_(rows, rows)].copy()
    # Without this a basin's nearest neighbour is itself at zero distance, the correction
    # collapses to the identity, and the search would return lambda = 0.
    np.fill_diagonal(d, np.inf)
    # sigma is only read by nse_star; the other objectives are handed None and ignore it.
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
    """The (K, lambda, p) the cross-validation scores best. Exact ties break on the smallest
    (K, lambda, p), which Python's tuple ordering already gives, so the choice is deterministic.

    There used to be a tolerance here that preferred the simplest setting within 0.001 of the
    best, on the theory that the surface is flat enough near its optimum for the argmax to be
    noise. Measured on held-out basins, it cost rather than bought: KGE 0.7117 -> 0.7093, and
    on the NSE* objective 0.7410 -> 0.7322 NSE. It also needed a different constant per
    objective, since the three do not share a scale. See experiments_log.md.
    """
    obj = SCORERS[objective]
    scored = {c: obj.aggregate(v) for c, v in surf.items()}
    best = max(scored.values())
    return min(c for c, v in scored.items() if v == best)


def surface_rows(surfaces, picked, objective="nse_star"):
    """The searched surface as flat records -- one row per label x k x lam x p, for the CSV the
    hyperparameter figure draws from."""
    rows = []
    for label, surf in surfaces.items():
        for (k, lam, p), vals in surf.items():
            rows.append(dict(label=label, k=k, lam=lam, p=p,
                             score=SCORERS[objective].aggregate(vals),
                             n_basins=len(vals), selected=(k, lam, p) == picked[label]))
    return rows
