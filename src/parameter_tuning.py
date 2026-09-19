"""Choosing K, lambda and p by cross-validation, without ever reading the target's own gauge.

The correction has three free numbers: K, how many gauges to consult; lambda, how much of their
error to believe; and p, how steeply weight falls off with distance. All three are selected per
fold (ungauged) or per group (gauged) on a surface built **only from basins other than the ones
being corrected** -- the target's observations are not in the surface, so the reported result
cannot have been tuned on itself.

Two surfaces, because the two cells face different gauge networks (see `make_job`):

    leave-one-group-out   each group corrected using donors from the *other* groups only --
                          the thinned network a held-out (ungauged) group faces.
    self-masked           every basin corrected from every other -- what a single discontinued
                          gauge faces, its neighbourhood still fully gauged.

One process per fold/group; ~80 s per surface, so a cell is a couple of minutes.
"""
import concurrent.futures as cf
import os
from collections import namedtuple

import numpy as np

from residual_postprocessing import (EPS, apply_shrinkage, idw_weights, kge_per_basin,
                                     neighbour_residual)

# Odd K from 1 to 39. The surface is flat enough near its optimum that stepping by 2 cannot
# jump over it, and folds do select across the whole range once seed-averaging has de-noised
# each donor's residual.
K_GRID = tuple(range(1, 41, 2))
LAM_GRID = tuple(np.round(np.arange(0.0, 1.05, 0.05), 2))
# The IDW exponent: 1 is plain inverse distance, 2 the classic Shepard weighting. Below 1
# flattens toward a simple average over the K donors and scores worse; above 3 sharpens toward
# the nearest donor with no further gain.
P_GRID = (1.0, 1.5, 2.0, 3.0)
# The surface is flat near its optimum: cells within 0.001 of the maximum recover K roughly
# halved for a median cost of 0.0000, so the raw argmax is mostly noise in the fourth decimal.
# p is different -- at fixed K and lambda it swings the score by 0.01-0.04 -- so it is rarely
# simplified away. See METHOD.md.
TOLERANCE = 0.001

Job = namedtuple("Job", "label pred obs dist groups")
"""One fold's or group's search. `pred`/`obs` are the CV pool's rows, `dist` the pool's own
distance matrix with an infinite diagonal, `groups` the pool's fold labels (thinned) or None."""


def _score_grid(pred_t, obs_t, pred_d, obs_d, dist, eps):
    """{(k, lam, p): [n_target] KGE} for one target/donor split."""
    out = {}
    for k in K_GRID:
        for p in P_GRID:
            R = neighbour_residual(pred_d, obs_d, idw_weights(dist, k, p), eps)
            for lam in LAM_GRID:
                out[(int(k), float(lam), float(p))] = kge_per_basin(
                    apply_shrinkage(pred_t, R, lam, eps), obs_t)
    return out


def _surface(job, eps=EPS):
    """{(k, lam, p): [n_pool] KGE}, every pool basin scored exactly once."""
    n = len(job.pred)
    if job.groups is None:
        splits = [(np.arange(n), np.arange(n))]          # self masked by the infinite diagonal
    else:
        splits = [(np.flatnonzero(job.groups == g), np.flatnonzero(job.groups != g))
                  for g in np.unique(job.groups)]
    out = {}
    for tgt, don in splits:
        part = _score_grid(job.pred[tgt], job.obs[tgt], job.pred[don], job.obs[don],
                           job.dist[np.ix_(tgt, don)], eps)
        for key, vals in part.items():
            out.setdefault(key, np.full(n, np.nan))[tgt] = vals
    return job.label, out


def _init_worker():
    """Cap BLAS to one thread per worker: process parallelism and BLAS's own intra-op threads
    must not both be on, or N workers oversubscribe the machine by N x 24 threads."""
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def make_job(label, pred, obs, dist, rows, groups=None):
    """One label's search, over `rows` -- the basins it is allowed to learn from.

    `groups`, when given, are those basins' fold labels and the search is leave-one-group-out
    (thinned network, the ungauged case). When None it is self-masked leave-one-out over all of
    them (dense network, the gauged case).
    """
    d = dist[np.ix_(rows, rows)].copy()
    # Without this a basin's nearest neighbour is itself at zero distance, the correction
    # collapses to the identity, and the search would return lambda = 0.
    np.fill_diagonal(d, np.inf)
    return Job(label, pred[rows], obs[rows], d, None if groups is None else groups[rows])


def run_jobs(jobs, workers=None):
    """{label: surface}, one process per job."""
    n_workers = min(len(jobs), workers or max(1, (os.cpu_count() or 1) - 4))
    if n_workers <= 1:
        return dict(_surface(j) for j in jobs)
    with cf.ProcessPoolExecutor(n_workers, initializer=_init_worker) as ex:
        return dict(ex.map(_surface, jobs))


def pick(surf, tolerance=TOLERANCE):
    """The (K, lambda, p) maximising median KGE -- or, if another cell scores within
    `tolerance`, the simplest cell that does: smallest K, then lambda, then p.

    Smaller is simpler by the same logic for K and lambda: K is how much of the donor network
    the correction depends on, lambda how far it shrinks toward the uncorrected model
    (lambda = 0 *is* the uncorrected model). A smaller p spreads weight across the K donors
    rather than concentrating it on the very nearest one, so it is the lower-variance choice.
    """
    scored = {c: np.nanmedian(v) for c, v in surf.items()}
    best = max(scored.values())
    return min(c for c, v in scored.items() if v >= best - tolerance)


def surface_rows(surfaces, picked):
    """The searched surface as flat records -- one row per label x k x lam x p, for the CSV the
    hyperparameter figure draws from."""
    rows = []
    for label, surf in surfaces.items():
        for (k, lam, p), vals in surf.items():
            rows.append(dict(label=label, k=k, lam=lam, p=p, score=np.nanmedian(vals),
                             n_basins=len(vals), selected=(k, lam, p) == picked[label]))
    return rows
