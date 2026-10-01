"""The inverse-distance-weighted neighbour-residual correction.

A basin with no usable gauge has no discharge of its own, but its nearest gauges do, and their
errors *today* are informative about its error today. For target basin i on day t, over the K
nearest donor basins j that reported that day:

    w_ij = (1/d_ij^p) / sum_j (1/d_ij^p)                   d = great-circle km
    R_i  = sum_j w_ij * [ log(q_sim_j + eps) - log(q_obs_j + eps) ]
    q_i  = (q_sim_i + eps) * exp(-lam * R_i) - eps

In words: ask the K nearest gauges how wrong the model was today, weighting closer gauges more
steeply the higher p is set; if they agree it ran low, assume this basin did too, and believe a
fraction lam of it.

Nothing here is trained and nothing is fitted to the basins being corrected. The weights come
from coordinates alone, and the target basin's own discharge is never read. K, lam and p are
selected in `tuning.py`. See manuscript.md for the method and the evidence.
"""
import numpy as np

from shared import EPS


def idw_weights(dist_km, k, p):
    """[n_target, n_donor] inverse-distance^p weights over the k nearest, rows summing to 1."""
    keep = np.argsort(dist_km, axis=1)[:, :k]
    w = np.maximum(np.take_along_axis(dist_km, keep, 1), 1.0) ** -p
    W = np.zeros_like(dist_km)
    np.put_along_axis(W, keep, w / w.sum(1, keepdims=True), 1)
    return W


def neighbour_residual(sim_donor, obs_donor, W, eps=EPS):
    """R: [n_target, T] weighted mean log-residual of the neighbouring gauges.

    Donors that did not report on a given day are dropped and the surviving weights
    renormalised -- a silent gauge must not read as "this donor had zero error today". A row
    with no reporting donor at all returns 0, i.e. no correction.
    """
    res = np.log(np.clip(sim_donor, 0, None) + eps) - np.log(obs_donor + eps)
    reported = np.isfinite(res).astype(np.float64)
    num, den = W @ np.nan_to_num(res), W @ reported
    return np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)


def apply_shrinkage(sim_target, R, lam, eps=EPS):
    """Push each prediction by a fraction `lam` of its neighbourhood's error."""
    return np.clip(np.exp(np.log(np.clip(sim_target, 0, None) + eps) - lam * R) - eps, 0, None)


def correct(sim_target, sim_donor, obs_donor, dist_km, k, lam, p, eps=EPS):
    """The whole correction. `dist_km` is [n_target, n_donor]."""
    W = idw_weights(dist_km, k, p)
    return apply_shrinkage(sim_target, neighbour_residual(sim_donor, obs_donor, W, eps), lam, eps)
