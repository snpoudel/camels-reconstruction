"""Neighbour-residual correction — the post-processing step this branch is about.

    python -m src.correct                      # score vanilla, raw vs corrected
    python -m src.correct --out runs/corrected.csv

An ungauged basin has no discharge of its own, but its nearest gauges do, and their errors
*today* are informative about its error today. For ungauged basin i on day t, over the K
nearest gauged basins j that reported that day:

    w_ij = (1/d_ij) / sum_j (1/d_ij)                       d = great-circle km
    R_i  = sum_j w_ij * [ log(q_sim_j + eps) - log(q_obs_j + eps) ]
    q_i  = (q_sim_i + eps) * exp(-LAMBDA * R_i) - eps

In words: ask the K nearest gauges how wrong the model was today; if they agree it ran low,
assume this basin did too, and believe a fraction LAMBDA of it.

Nothing here is trained and nothing is fitted to the basins being corrected. The weights come
from coordinates alone; the target basin's own discharge is never read. See METHOD.md for the
evidence, the alternatives that were rejected, and the leak audit.
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd

# Defaults sit in the middle of a broad plateau rather than at a fitted peak: every
# combination of K in [3, 20] and LAMBDA in [0.3, 0.7] scores within ~0.01 NSE of the best.
# `analysis/tune.py` reproduces that surface and selects leave-one-fold-out.
K = 8
LAMBDA = 0.5
# mm/day. Keeps log() finite on the 2.8% of days that report exactly zero flow, and is small
# enough to be hydrologically negligible. A far smaller floor (1e-5) makes those days produce
# log-residuals near -11 and costs ~0.005 NSE.
EPS = 0.01


def great_circle_km(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    h = (np.sin((la[:, None] - la[None, :]) / 2) ** 2
         + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def idw_weights(dist_km, k=K):
    """[n_ungauged, n_gauged] inverse-distance weights over the k nearest, rows summing to 1."""
    keep = np.argsort(dist_km, axis=1)[:, :k]
    w = 1.0 / np.maximum(np.take_along_axis(dist_km, keep, 1), 1.0)
    W = np.zeros_like(dist_km)
    np.put_along_axis(W, keep, w / w.sum(1, keepdims=True), 1)
    return W


def neighbour_residual(sim_gauged, obs_gauged, W, eps=EPS):
    """R: [n_ungauged, T] weighted mean log-residual of the neighbouring gauges.

    Donors that did not report on a given day are dropped and the surviving weights
    renormalised -- a silent gauge must not read as "this donor had zero error today". A row
    with no reporting donor at all returns 0, i.e. no correction.
    """
    res = np.log(np.clip(sim_gauged, 0, None) + eps) - np.log(obs_gauged + eps)
    reported = np.isfinite(res).astype(np.float64)
    num, den = W @ np.nan_to_num(res), W @ reported
    return np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)


def apply_shrinkage(sim_ungauged, R, lam=LAMBDA, eps=EPS):
    """Push each prediction by a fraction `lam` of its neighbourhood's error."""
    return np.clip(np.exp(np.log(np.clip(sim_ungauged, 0, None) + eps) - lam * R) - eps, 0, None)


def correct(sim_ungauged, sim_gauged, obs_gauged, dist_km, k=K, lam=LAMBDA, eps=EPS):
    """The whole correction. `dist_km` is [n_ungauged, n_gauged]."""
    W = idw_weights(dist_km, k)
    return apply_shrinkage(sim_ungauged, neighbour_residual(sim_gauged, obs_gauged, W, eps),
                           lam, eps)


def load_fold(fold, runs_dir="runs", cache_dir="data/cache"):
    """Seed-mean test hydrograph for one fold, in cache-basin order.

    Averaging the seeds' hydrographs and scoring once is the convention the literature
    reports (Kratzert's 0.693 is a 10-member ensemble mean); averaging per-seed metrics
    instead reads lower and is not comparable.
    """
    basins = list(json.load(open(f"{cache_dir}/meta.json"))["basins"])
    idx = {b: i for i, b in enumerate(basins)}
    mats, gauged, t_idx = [], None, None
    for path in sorted(glob.glob(f"{runs_dir}/vanilla_fold{fold}_seed*/predictions.npz")):
        cfg = json.load(open(path.rsplit("/", 1)[0] + "/config.json"))
        if cfg.get("max_steps"):
            continue
        p = np.load(path, allow_pickle=True)
        rows = np.array([idx[b] for b in list(p["basins"])])
        m = np.full((len(basins), p["test"].shape[1]), np.nan, np.float32)
        m[rows] = p["test"]
        mats.append(m)
        if gauged is None:
            gauged = np.zeros(len(basins), bool)
            gauged[rows[: int(p["n_gauged"])]] = True
            t_idx = p["test_t"]
    if not mats:
        return None
    return np.mean(mats, axis=0), t_idx, gauged, len(mats)


def distances(cache_dir="data/cache", loc_csv="data/camels531.csv"):
    basins = list(json.load(open(f"{cache_dir}/meta.json"))["basins"])
    loc = pd.read_csv(loc_csv, dtype={"gauge_id": str}).set_index("gauge_id")
    return great_circle_km(loc.loc[basins, "gauge_lat"].to_numpy(),
                           loc.loc[basins, "gauge_lon"].to_numpy()), basins


def apply_all(k=K, lam=LAMBDA, eps=EPS, runs_dir="runs", cache_dir="data/cache"):
    """Raw and corrected hydrographs for every held-out basin of every available fold.

    Yields one dict per fold so callers can score, plot, or pool as they need.
    """
    D, basins = distances(cache_dir)
    Q = np.load(f"{cache_dir}/Q.npy")
    for fold in range(7):
        got = load_fold(fold, runs_dir, cache_dir)
        if got is None:
            continue
        pred, t_idx, gauged, n_seeds = got
        obs = Q[:, t_idx]
        pool, held = np.flatnonzero(gauged), np.flatnonzero(~gauged)
        yield dict(fold=fold, n_seeds=n_seeds, t_idx=t_idx,
                   basins=[basins[i] for i in held],
                   obs=obs[held], raw=pred[held],
                   corrected=correct(pred[held], pred[pool], obs[pool],
                                     D[np.ix_(held, pool)], k, lam, eps))


def main():
    from .evaluate import metrics
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--lam", type=float, default=LAMBDA)
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="runs/corrected_metrics.csv")
    a = p.parse_args()

    rows = []
    for f in apply_all(a.k, a.lam, runs_dir=a.runs):
        raw, cor = metrics(f["raw"], f["obs"]), metrics(f["corrected"], f["obs"])
        raw["variant"], cor["variant"] = "vanilla", "vanilla + correction"
        for frame in (raw, cor):
            frame["basin"], frame["fold"], frame["n_seeds"] = f["basins"], f["fold"], f["n_seeds"]
        rows += [raw, cor]
        print(f"  fold{f['fold']} ({f['n_seeds']} seeds, {len(f['basins'])} held out): "
              f"{raw.nse.median():.4f} -> {cor.nse.median():.4f} "
              f"({cor.nse.median() - raw.nse.median():+.4f})")
    df = pd.concat(rows, ignore_index=True)
    piv = df.pivot_table(index="basin", columns="variant", values="nse")
    # pivot_table drops a basin whose NSE is undefined (a constant observed series, say).
    # Say so rather than quietly reporting a median over fewer basins than were held out.
    n_held = df.basin.nunique()
    if len(piv) != n_held:
        print(f"  note: {n_held - len(piv)} of {n_held} basins have no defined NSE and are "
              f"excluded from the medians below")
    gain = piv["vanilla + correction"] - piv["vanilla"]
    print(f"\nK={a.k}, lambda={a.lam}, eps={EPS} -- nothing fitted to data")
    print(f"{len(piv)} held-out basins: {piv['vanilla'].median():.4f} -> "
          f"{piv['vanilla + correction'].median():.4f} "
          f"({piv['vanilla + correction'].median() - piv['vanilla'].median():+.4f})")
    print(f"improved in {(gain > 0).mean():.0%} of basins; hurt by >0.05 in {int((gain < -0.05).sum())}")
    df.to_csv(a.out, index=False)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
