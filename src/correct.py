"""Neighbour-residual correction — the post-processing step this branch is about.

    python -m src.correct                      # score vanilla, raw vs corrected
    python -m src.correct --out runs/corrected.csv

An ungauged basin has no discharge of its own, but its nearest gauges do, and their errors
*today* are informative about its error today. For ungauged basin i on day t, over the K
nearest gauged basins j that reported that day:

    w_ij = (1/d_ij^P) / sum_j (1/d_ij^P)                   d = great-circle km
    R_i  = sum_j w_ij * [ log(q_sim_j + eps) - log(q_obs_j + eps) ]
    q_i  = (q_sim_i + eps) * exp(-LAMBDA * R_i) - eps

In words: ask the K nearest gauges how wrong the model was today, weighting closer gauges
more steeply the higher P is set; if they agree it ran low, assume this basin did too, and
believe a fraction LAMBDA of it.

Nothing here is trained and nothing is fitted to the basins being corrected. The weights come
from coordinates alone; the target basin's own discharge is never read. See METHOD.md for the
evidence, the alternatives that were rejected, and the leak audit.
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd

from .data import assign_folds

# Used when selection is switched off, and as the fallback when there are too few folds to
# select honestly (see `choose`). All three sit inside the flat region of the surface.
K = 8
LAMBDA = 0.5
P = 1.0

# The search grid. Every K from 1 to 20 -- dense enough not to jump over the optimum -- and no
# higher: 30/50 scored worse on every fold and widening the grid past a basin's real donor
# density buys nothing.
K_GRID = tuple(range(1, 21))
LAM_GRID = tuple(np.round(np.arange(0.0, 1.05, 0.05), 2))
# The IDW exponent: 1 is plain inverse distance, 2 the classic Shepard weighting. Searched
# only over [1, 3] -- below 1 flattens toward a simple average over the K donors and scored
# worse; above 3 sharpens toward the nearest donor with no further gain.
P_GRID = (1.0, 1.5, 2.0, 3.0)
# mm/day. Keeps log() finite on the 2.8% of days that report exactly zero flow, and is small
# enough to be hydrologically negligible. A far smaller floor (1e-5) makes those days produce
# log-residuals near -11 and costs ~0.005 NSE.
EPS = 0.01


def great_circle_km(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    h = (np.sin((la[:, None] - la[None, :]) / 2) ** 2
         + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def idw_weights(dist_km, k=K, p=P):
    """[n_ungauged, n_gauged] inverse-distance^p weights over the k nearest, rows summing to 1."""
    keep = np.argsort(dist_km, axis=1)[:, :k]
    w = np.maximum(np.take_along_axis(dist_km, keep, 1), 1.0) ** -p
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


def correct(sim_ungauged, sim_gauged, obs_gauged, dist_km, k=K, lam=LAMBDA, p=P, eps=EPS):
    """The whole correction. `dist_km` is [n_ungauged, n_gauged]."""
    W = idw_weights(dist_km, k, p)
    return apply_shrinkage(sim_ungauged, neighbour_residual(sim_gauged, obs_gauged, W, eps),
                           lam, eps)


def nse_per_basin(sim, obs):
    """[n] NSE, vectorised over basins -- selection calls this a few hundred times."""
    sim = np.clip(sim, 0, None)
    m = np.isfinite(obs)
    o, s, n = np.where(m, obs, 0.0), np.where(m, sim, 0.0), m.sum(1)
    mu = o.sum(1) / np.maximum(n, 1)
    den = ((o - mu[:, None]) ** 2 * m).sum(1)
    return np.where((n > 1) & (den > 0),
                    1 - ((s - o) ** 2 * m).sum(1) / np.maximum(den, 1e-12), np.nan)


def load_fold(fold, runs_dir="runs", cache_dir="data/cache", arch="vanilla"):
    """Seed-mean test hydrograph for one fold, in cache-basin order.

    Averaging the seeds' hydrographs and scoring once is the convention the literature
    reports (Kratzert's 0.693 is a 10-member ensemble mean); averaging per-seed metrics
    instead reads lower and is not comparable.
    """
    basins = list(json.load(open(f"{cache_dir}/meta.json"))["basins"])
    idx = {b: i for i, b in enumerate(basins)}
    mats, gauged, t_idx, period = [], None, None, None
    for path in sorted(glob.glob(f"{runs_dir}/{arch}_fold{fold}_seed*/predictions.npz")):
        cfg = json.load(open(path.rsplit("/", 1)[0] + "/config.json"))
        if cfg.get("max_steps"):
            continue
        # Runs from different train/test windows have different test axes; averaging them
        # would either crash or, worse, line up the wrong dates. Refuse explicitly.
        this = cfg.get("period", "complete")
        if period is None:
            period = this
        elif this != period:
            raise SystemExit(
                f"{runs_dir}/ mixes periods in fold {fold}: {period!r} and {this!r}. "
                f"Move one set aside -- they are not comparable and must not be pooled.")
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


def load_all(runs_dir="runs", cache_dir="data/cache", arch="vanilla"):
    """Everything the correction and its tuning need, one entry per available fold."""
    D, basins = distances(cache_dir)
    Q = np.load(f"{cache_dir}/Q.npy")
    out = []
    for fold in range(7):
        got = load_fold(fold, runs_dir, cache_dir, arch)
        if got is None:
            continue
        pred, t_idx, gauged, n_seeds = got
        obs = Q[:, t_idx]
        pool, held = np.flatnonzero(gauged), np.flatnonzero(~gauged)
        out.append(dict(fold=fold, n_seeds=n_seeds, t_idx=t_idx, pool=pool, held=held,
                        pred=pred, obs=obs, d_held=D[np.ix_(held, pool)], D=D,
                        # each training basin keeps its own fold label, giving the six groups
                        # the inner leave-one-group-out search uses
                        pool_groups=assign_folds(len(basins), 7)[pool],
                        basins=[basins[i] for i in held]))
    return out


def surface(f, k_grid=K_GRID, lam_grid=LAM_GRID, p_grid=P_GRID, eps=EPS, on="held"):
    """{(k, lam, p): per-basin NSE} for one fold, on one of three sets of basins.

    `on="held"`   the fold's held-out basins -- the real objective, and the surface the figure
                  draws. **Never used to select anything**: it is the test set.
    `on="inner"`  leave-one-group-out *inside the training basins*. The 455 training basins
                  carry their original fold labels, giving six groups; each group is corrected
                  using donors drawn only from the other five, so no basin is ever its own
                  donor and each group faces a thinned gauge network like a real ungauged one.
                  This is the selection surface -- it touches no test basin of any fold.
    `on="gauged"` every training basin corrected from every other, self masked. Simpler than
                  `inner` but lets a basin borrow from its immediate neighbours, which a truly
                  ungauged basin in a held-out *group* could not.
    """
    if on == "held":
        target, donor, d = f["held"], f["pool"], f["d_held"]
        out = {}
        for k in k_grid:
            for p in p_grid:
                R = neighbour_residual(f["pred"][donor], f["obs"][donor],
                                       idw_weights(d, k, p), eps)
                for lam in lam_grid:
                    out[(int(k), float(lam), float(p))] = nse_per_basin(
                        apply_shrinkage(f["pred"][target], R, lam, eps), f["obs"][target])
        return out

    D = f["D"]
    pool = f["pool"]
    if on == "gauged":
        d = D[np.ix_(pool, pool)].copy()
        # Without this a basin's nearest neighbour is itself at zero distance, the correction
        # collapses to the identity, and the search would return lambda = 0.
        np.fill_diagonal(d, np.inf)
        groups = [(np.arange(len(pool)), np.arange(len(pool)))]
    else:
        d = D[np.ix_(pool, pool)].copy()
        np.fill_diagonal(d, np.inf)
        labels = f["pool_groups"]
        groups = []
        for gl in np.unique(labels):
            tgt = np.flatnonzero(labels == gl)
            groups.append((tgt, np.flatnonzero(labels != gl)))

    out = {(int(k), float(lam), float(p)): np.full(len(pool), np.nan)
           for k in k_grid for lam in lam_grid for p in p_grid}
    for tgt, don in groups:
        dd = d[np.ix_(tgt, don)]
        for k in k_grid:
            for p in p_grid:
                R = neighbour_residual(f["pred"][pool[don]], f["obs"][pool[don]],
                                       idw_weights(dd, k, p), eps)
                for lam in lam_grid:
                    v = nse_per_basin(apply_shrinkage(f["pred"][pool[tgt]], R, lam, eps),
                                      f["obs"][pool[tgt]])
                    out[(int(k), float(lam), float(p))][tgt] = v
    return out


def choose(folds, k_grid=K_GRID, lam_grid=LAM_GRID, p_grid=P_GRID, eps=EPS, mode="inner"):
    """(K, lambda, p) per fold, and a note saying how they were arrived at.

    **No mode ever reads a held-out basin of any fold.** The held-out basins are the test set;
    letting them into selection -- even another fold's -- would mean the pooled 531-basin
    result had been tuned on itself, since every basin is some fold's test set.

    `mode="inner"` (default) searches the leave-one-group-out surface inside fold f's own 455
    training basins (`surface(..., on="inner")`). This is selection done entirely within the
    training folds.

    `mode="gauged"` searches the same training basins without the group structure.

    `mode="fixed"` skips the search: K = 8, lambda = 0.5, p = 1.0.

    All three are clean. They differ in how well they *predict* the held-out optimum, which is
    a measurement, not an assumption -- `analysis/tune.py` prints it.
    """
    if mode == "fixed":
        return {f["fold"]: (K, LAMBDA, P) for f in folds}, "fixed constants, no search"
    picked = {}
    for f in folds:
        surf = surface(f, k_grid, lam_grid, p_grid, eps, on=mode)
        picked[f["fold"]] = max(surf, key=lambda c: np.nanmedian(surf[c]))
    where = ("leave-one-group-out within each fold's training basins"
             if mode == "inner" else "each fold's training basins, self masked")
    return picked, f"chosen on {where}; no held-out basin was read"


def apply_all(runs_dir="runs", cache_dir="data/cache", mode="inner", eps=EPS):
    """Raw and corrected hydrographs for every held-out basin of every available fold."""
    folds = load_all(runs_dir, cache_dir)
    if not folds:
        return [], "no runs found"
    picked, note = choose(folds, eps=eps, mode=mode)
    for f in folds:
        k, lam, p = picked[f["fold"]]
        f["k"], f["lam"], f["p"] = k, lam, p
        f["raw"] = f["pred"][f["held"]]
        f["corrected"] = correct(f["pred"][f["held"]], f["pred"][f["pool"]],
                                 f["obs"][f["pool"]], f["d_held"], k, lam, p, eps)
    return folds, note


def main():
    from .evaluate import metrics
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["inner", "gauged", "fixed"], default="inner",
                   help="how (K, lambda, p) are chosen; see correct.choose")
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="runs/corrected_metrics.csv")
    a = p.parse_args()

    folds, note = apply_all(runs_dir=a.runs, mode=a.mode)
    if not folds:
        raise SystemExit(note)
    rows = []
    for f in folds:
        raw, cor = metrics(f["raw"], f["obs"][f["held"]]), metrics(f["corrected"], f["obs"][f["held"]])
        raw["variant"], cor["variant"] = "vanilla", "vanilla + correction"
        for frame in (raw, cor):
            frame["basin"], frame["fold"], frame["n_seeds"] = f["basins"], f["fold"], f["n_seeds"]
            frame["k"], frame["lam"], frame["p"] = f["k"], f["lam"], f["p"]
        rows += [raw, cor]
        print(f"  fold{f['fold']} ({f['n_seeds']} seeds, {len(f['basins'])} held out)  "
              f"K={f['k']:>2} lambda={f['lam']:.2f} p={f['p']:.2g}   "
              f"{raw.nse.median():.4f} -> {cor.nse.median():.4f} "
              f"({cor.nse.median() - raw.nse.median():+.4f})")

    df = pd.concat(rows, ignore_index=True)
    piv = df.pivot_table(index="basin", columns="variant", values="nse")
    n_held = df.basin.nunique()
    if len(piv) != n_held:
        print(f"  note: {n_held - len(piv)} of {n_held} basins have no defined NSE and are "
              f"excluded from the medians below")
    gain = piv["vanilla + correction"] - piv["vanilla"]
    print(f"\n(K, lambda, p) {note};  eps={EPS}")
    print(f"{len(piv)} held-out basins: {piv['vanilla'].median():.4f} -> "
          f"{piv['vanilla + correction'].median():.4f} "
          f"({piv['vanilla + correction'].median() - piv['vanilla'].median():+.4f})")
    print(f"improved in {(gain > 0).mean():.0%} of basins; hurt by >0.05 in {int((gain < -0.05).sum())}")
    df.to_csv(a.out, index=False)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
