"""The inverse-distance-weighted neighbour-residual correction.

A basin with no usable gauge has no discharge of its own, but its nearest gauges do, and their
errors *today* are informative about its error today. For target basin i on day t, over the K
nearest donor basins j that reported that day:

    w_ij = (1/d_ij^p) / sum_j (1/d_ij^p)                   d = great-circle km
    R_i  = sum_j w_ij * (q_sim_j - q_obs_j) / s_j          s = SD of predicted flow + 0.1 mm/day
    q_i  = max(0, q_sim_i - lam * s_i * R_i)

In words: ask the K nearest gauges how wrong the model was today, each in units of its own
flow's spread, weighting closer gauges more steeply the higher p is set; if they agree it ran
low, assume this basin did too by the same share of *its* spread, and believe a fraction lam of
it. The scaling stops a wet donor's errors, large in mm/day, from swamping a dry target, and
unlike a log ratio it does not let a small low-flow error cut a target's flood peak.

Nothing here is trained and nothing is fitted to the basins being corrected. The weights come
from coordinates alone, s from the model's own predictions, and the target's own discharge is
never read. K, lam and p are selected in `tuning.py`.

`scaled=False` drops s (every s = 1): the unscaled comparison the pipelines run with --no-scale.
"""
import numpy as np

from shared import flow_scale


def idw_weights(dist_km, k, p):
    """[n_target, n_donor] inverse-distance^p weights over the k nearest, rows summing to 1."""
    keep = np.argsort(dist_km, axis=1)[:, :k]
    w = np.maximum(np.take_along_axis(dist_km, keep, 1), 1.0) ** -p
    W = np.zeros_like(dist_km)
    np.put_along_axis(W, keep, w / w.sum(1, keepdims=True), 1)
    return W


def neighbour_residual(sim_donor, obs_donor, W, scaled=True):
    """R: [n_target, T] weighted mean scaled residual of the neighbouring gauges.

    Donors that did not report on a given day are dropped and the surviving weights
    renormalised -- a silent gauge must not read as "this donor had zero error today". A row
    with no reporting donor at all returns 0, i.e. no correction.
    """
    res = (np.clip(sim_donor, 0, None) - obs_donor) / flow_scale(sim_donor, scaled)[:, None]
    reported = np.isfinite(res).astype(np.float64)
    num, den = W @ np.nan_to_num(res), W @ reported
    return np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)


def apply_shrinkage(sim_target, R, lam, scaled=True):
    """Push each prediction by a fraction `lam` of its neighbourhood's error, in its own units."""
    return np.clip(np.clip(sim_target, 0, None) - lam * flow_scale(sim_target, scaled)[:, None] * R,
                   0, None)


def correct(sim_target, sim_donor, obs_donor, dist_km, k, lam, p, scaled=True):
    """The whole correction. `dist_km` is [n_target, n_donor]."""
    W = idw_weights(dist_km, k, p)
    return apply_shrinkage(sim_target, neighbour_residual(sim_donor, obs_donor, W, scaled), lam,
                           scaled)
