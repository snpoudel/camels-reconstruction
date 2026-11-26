"""Inverse-distance-weighted neighbour-residual correction.

For target basin i on day t, over the K nearest donor basins j that reported that day:

    w_ij = (1/d_ij^p) / sum_j (1/d_ij^p)                   d = great-circle km
    R_i  = sum_j w_ij * (q_sim_j - q_obs_j) / s_j          s = SD of predicted flow + 0.1 mm/day
    q_i  = max(0, q_sim_i - lam * s_i * R_i)

The target's own discharge is never read. K, lam and p are selected in `tuning.py`.
`scaled=False` (--no-scale) sets every s to 1.
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
    """[n_target, T] weighted mean scaled donor residual over the donors that reported, else 0."""
    res = (np.clip(sim_donor, 0, None) - obs_donor) / flow_scale(sim_donor, scaled)[:, None]
    reported = np.isfinite(res).astype(np.float64)
    num, den = W @ np.nan_to_num(res), W @ reported
    return np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)


def apply_shrinkage(sim_target, R, lam, scaled=True):
    """Subtract a fraction `lam` of the neighbour residual, in each target's own scale."""
    return np.clip(np.clip(sim_target, 0, None) - lam * flow_scale(sim_target, scaled)[:, None] * R,
                   0, None)


def correct(sim_target, sim_donor, obs_donor, dist_km, k, lam, p, scaled=True):
    """The full correction; `dist_km` is [n_target, n_donor]."""
    W = idw_weights(dist_km, k, p)
    return apply_shrinkage(sim_target, neighbour_residual(sim_donor, obs_donor, W, scaled), lam,
                           scaled)
