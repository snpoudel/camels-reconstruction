"""The neighbour-residual correction -- what this project is about.

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
selected in `parameter_tuning.py`. See METHOD.md for the derivation and the evidence.
"""
import json

import numpy as np
import pandas as pd

# mm/day. Keeps log() finite on the 2.8% of days that report exactly zero flow, and is small
# enough to be hydrologically negligible. A far smaller floor (1e-5) makes those days produce
# log-residuals near -11 and costs ~0.005 NSE.
EPS = 0.01


# --------------------------------------------------------------------------- the correction

def great_circle_km(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    h = (np.sin((la[:, None] - la[None, :]) / 2) ** 2
         + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


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


def nse_per_basin(sim, obs):
    """[n] Nash-Sutcliffe efficiency, vectorised over basins."""
    sim = np.clip(sim, 0, None)
    m = np.isfinite(obs)
    o, s, n = np.where(m, obs, 0.0), np.where(m, sim, 0.0), m.sum(1)
    mu = o.sum(1) / np.maximum(n, 1)
    den = ((o - mu[:, None]) ** 2 * m).sum(1)
    return np.where((n > 1) & (den > 0),
                    1 - ((s - o) ** 2 * m).sum(1) / np.maximum(den, 1e-12), np.nan)


def kge_per_basin(sim, obs):
    """[n] Kling-Gupta efficiency, vectorised over basins -- selection calls this thousands of
    times, so it cannot be a Python loop.

    r, alpha and beta are each a ratio of sums over the reported days, so the 1/n that would
    normally turn a sum into a mean or a variance cancels in every one of the three ratios
    except beta -- only beta needs an actual mean.
    """
    sim = np.clip(sim, 0, None)
    m = np.isfinite(obs)
    o, s, n = np.where(m, obs, 0.0), np.where(m, sim, 0.0), m.sum(1)
    mu_o, mu_s = o.sum(1) / np.maximum(n, 1), s.sum(1) / np.maximum(n, 1)
    do, ds = (o - mu_o[:, None]) * m, (s - mu_s[:, None]) * m
    var_o, var_s, cov = (do ** 2).sum(1), (ds ** 2).sum(1), (do * ds).sum(1)
    ok = (n > 1) & (var_o > 0) & (var_s > 0) & (mu_o != 0)
    r = np.where(ok, cov / np.maximum(np.sqrt(var_o * var_s), 1e-12), np.nan)
    alpha = np.where(ok, np.sqrt(var_s / np.maximum(var_o, 1e-12)), np.nan)
    beta = np.where(ok, mu_s / np.where(mu_o != 0, mu_o, np.nan), np.nan)
    return np.where(ok, 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2), np.nan)


# --------------------------------------------------------------------------- inputs on disk

BASINS_CSV = "data/camels531.csv"
FOLDS_CSV = "data/basin_list_with_folds.csv"
CACHE_DIR = "data/cache"

# Everything below is keyed to this one order -- sorted gauge IDs, which is also the order the
# cache and every predictions.npz is written in.


def basin_ids(list_csv=BASINS_CSV):
    return sorted(pd.read_csv(list_csv, dtype={"gauge_id": str})["gauge_id"])


def fold_labels(folds_csv=FOLDS_CSV, list_csv=BASINS_CSV):
    """[n] fold label per basin. Tracked in git: the assignment defines the experiment."""
    df = pd.read_csv(folds_csv, dtype={"basin": str})
    assert list(df.basin) == basin_ids(list_csv), f"{folds_csv} is not in sorted gauge-ID order"
    return df.fold.to_numpy()


def distance_matrix(list_csv=BASINS_CSV):
    """[n, n] great-circle km between every pair of gauges. Coordinates only -- no discharge."""
    loc = pd.read_csv(list_csv, dtype={"gauge_id": str}).set_index("gauge_id")
    ids = basin_ids(list_csv)
    return great_circle_km(loc.loc[ids, "gauge_lat"].to_numpy(),
                           loc.loc[ids, "gauge_lon"].to_numpy())


def observed(test_t, cache_dir=CACHE_DIR):
    """[n, len(test_t)] observed flow, mm/day, NaN where the gauge did not report."""
    Q = np.load(f"{cache_dir}/Q.npy")
    assert len(Q) == len(basin_ids()), "cache and basin list disagree on the basin count"
    return Q[:, test_t]


def load_mean(npz_path):
    """The seed-mean hydrographs one cell's correction runs on: (pred, test_t, n_seeds)."""
    p = np.load(npz_path, allow_pickle=True)
    return p["pred"], p["test_t"], int(p["n_seeds"])


def train_sigma(npz_path):
    """[n] training-period flow std, the denominator of the model's own training loss.

    Published by whatever produced the predictions, not recomputed here: a model trained on a
    different window normalises by a different sigma, and only it knows which.
    """
    p = np.load(npz_path, allow_pickle=True)
    if "sigma" not in p:
        raise SystemExit(f"{npz_path} predates the sigma it needs for NSE* selection -- "
                         f"rebuild it with src/lstm/0*.py --mean-only (seconds, no retraining)")
    return p["sigma"]


def test_period_label(test_t, cache_dir=CACHE_DIR):
    dates = np.load(f"{cache_dir}/dates.npy")[test_t]
    return f"{str(dates[0])[:4]}-{str(dates[-1])[:4]}"


def cache_basins(cache_dir=CACHE_DIR):
    return list(json.load(open(f"{cache_dir}/meta.json"))["basins"])


def save_corrected(out_dir, raw, corrected, test_t, params, tuning):
    """One cell's finished correction: hydrographs before and after, the chosen settings, and
    the surface they were chosen on."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(f"{out_dir}/predictions.npz", raw=raw, corrected=corrected,
                        test_t=test_t, basins=np.array(basin_ids()))
    pd.DataFrame(params).to_csv(f"{out_dir}/params.csv", index=False)
    pd.DataFrame(tuning).to_csv(f"{out_dir}/tuning.csv", index=False)
    print(f"-> {out_dir}/predictions.npz, params.csv, tuning.csv")
