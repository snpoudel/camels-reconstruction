"""What every correction method shares: the basins, folds, distances and observations on disk,
the per-basin scores, and the one format a finished correction is saved in.

A correction method (src/corrections/<method>/) reads a model's seed-mean predictions with
`load_mean`, corrects them, and writes them with `save_corrected`, so figures/evaluate.py scores
every method the same way.
"""
import numpy as np
import pandas as pd

# mm/day. Keeps log() finite on the 2.8% of days that report exactly zero flow, and is small
# enough to be hydrologically negligible. A far smaller floor (1e-5) makes those days produce
# log-residuals near -11 and costs ~0.005 NSE.
EPS = 0.01


def great_circle_km(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    h = (np.sin((la[:, None] - la[None, :]) / 2) ** 2
         + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def nearest_gauge(sim_target, sim_donor, obs_donor, dist_km, eps=EPS):
    """The parameter-free reference every method is reported against: the single nearest
    donor's log error, applied in full (IDW at K=1, lambda=1). No correction on a day that donor
    did not report."""
    j = np.argmin(dist_km, axis=1)
    res = np.log(np.clip(sim_donor[j], 0, None) + eps) - np.log(obs_donor[j] + eps)
    # float64, as IDW's arithmetic is, so this matches its K=1, lambda=1 output bit for bit.
    R = np.nan_to_num(res).astype(np.float64)
    return np.clip(np.exp(np.log(np.clip(sim_target, 0, None) + eps) - R) - eps, 0, None)


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
                         f"rebuild it with src/models/lstm/0*.py --mean-only "
                         f"(seconds, no retraining)")
    return p["sigma"]


def test_period_label(test_t, cache_dir=CACHE_DIR):
    dates = np.load(f"{cache_dir}/dates.npy")[test_t]
    return f"{str(dates[0])[:4]}-{str(dates[-1])[:4]}"


def save_corrected(out_dir, raw, corrected, nearest, test_t, params, tuning):
    """One cell's finished correction: hydrographs before and after, the parameter-free
    nearest-gauge reference, the chosen settings, and the surface they were chosen on."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(f"{out_dir}/predictions.npz", raw=raw, corrected=corrected,
                        nearest=nearest, test_t=test_t, basins=np.array(basin_ids()))
    pd.DataFrame(params).to_csv(f"{out_dir}/params.csv", index=False)
    pd.DataFrame(tuning).to_csv(f"{out_dir}/tuning.csv", index=False)
    print(f"-> {out_dir}/predictions.npz, params.csv, tuning.csv")
