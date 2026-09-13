"""What is in the donor pool, and is the model's attention reaching it?

    python analysis/donor_diagnostics.py               # every section, fold 0
    python analysis/donor_diagnostics.py --fold 3      # a different held-out fold

No GPU, no torch, no training. Every number is either a property of the data or a
property of an attention matrix already dumped by a finished run.

The load-bearing number is RETRIEVAL R2: given a set of donor weights -- a learned
attention row or a hand rule -- how well does the weighted donor anomaly reproduce the
query basin's own flow anomaly? It is the ceiling on what the donor pathway can be
carrying, and it is directly comparable between a trained model and a one-line rule.
Run it on every new donor architecture: if retrieval R2 has not moved, the architecture
change did not change where the model is looking.

Leak discipline mirrors src/data.py. Anomalies are standardised on the training period.
Rules are parameter-free or fitted on gauged basins only. The held-out basins' observed
flow appears nowhere except as the thing being scored.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import TEST_PERIOD, TRAIN_PERIOD, Cfg  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIST = 7  # the model's donor_hist; the window these diagnostics describe


def load(fold, cache_dir=f"{ROOT}/data/cache"):
    """Cache plus fold split, in the model's [gauged..., ungauged...] ordering."""
    meta = json.load(open(f"{cache_dir}/meta.json"))
    basins, feat_cols = meta["basins"], meta["feat_cols"]
    Q = np.load(f"{cache_dir}/Q.npy")
    dates = np.load(f"{cache_dir}/dates.npy")
    p_mean = np.load(f"{cache_dir}/X.npy", mmap_mode="r")[:, 0, feat_cols.index("p_mean")]

    folds = pd.read_csv(f"{ROOT}/data/basin_list_with_folds.csv",
                        dtype={"basin": str}).set_index("basin").fold
    gauged = [b for b in basins if folds[b] != fold]
    order = gauged + [b for b in basins if folds[b] == fold]
    row = {b: i for i, b in enumerate(basins)}
    sel = [row[b] for b in order]

    span = lambda lo, hi: np.flatnonzero((dates >= np.datetime64(lo)) & (dates <= np.datetime64(hi)))
    train_t, test_t = span(*TRAIN_PERIOD), span(*TEST_PERIOD)

    # Same anomaly the model is handed: standardised log flow, basin-relative log floor,
    # mean and sd from the training period. See src.data.prepare.
    logq = np.log(Q + (Cfg.q_eps_frac * np.asarray(p_mean))[:, None])
    mu = np.nanmean(logq[:, train_t], 1, keepdims=True)
    sd = np.nanstd(logq[:, train_t], 1, keepdims=True)
    Z = ((logq - mu) / np.maximum(sd, 1e-6))[sel]

    att = pd.read_csv(f"{ROOT}/data/camels531.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    rad = np.radians(np.array([[att.gauge_lat[b], att.gauge_lon[b]] for b in order]))
    dla = rad[None, :, 0] - rad[:, None, 0]
    dlo = rad[None, :, 1] - rad[:, None, 1]
    D = 2 * 6371 * np.arcsin(np.sqrt(
        np.sin(dla / 2) ** 2
        + np.cos(rad[:, None, 0]) * np.cos(rad[None, :, 0]) * np.sin(dlo / 2) ** 2))
    np.fill_diagonal(D, np.inf)  # a basin is never its own donor, here or in the model

    return dict(basins=order, n_gauged=len(gauged), Z=Z, D=D,
                train_t=train_t, test_t=test_t, Q=Q[sel])


def smooth(Z, k=HIST):
    """Trailing-window mean anomaly, NaN-aware: the quantity donor_features hands the model."""
    box = np.ones(k)
    s = np.apply_along_axis(lambda r: np.convolve(r, box, "same"), 1, np.nan_to_num(Z))
    n = np.apply_along_axis(lambda r: np.convolve(r, box, "same"), 1, np.isfinite(Z).astype(float))
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def retrieval_r2(W, d, t_idx, k=HIST):
    """Median per-basin R2 of the weighted donor anomaly against the query's own anomaly.

    W is [n_query, n_gauged]; rows are normalised here, so any positive weighting works.
    Days on which most of a row's weight sits on a donor with no observation are dropped,
    which is what the model's donor_valid mask does to the same rows.

    Also returns the attention-weighted mean donor distance -- the two numbers together say
    whether a change moved the model's gaze, and whether moving it helped.
    """
    n = d["n_gauged"]
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-30)
    S = smooth(d["Z"], k)
    donor, target = S[:n][:, t_idx], S[-W.shape[0]:][:, t_idx]
    live = np.isfinite(donor).astype(float)
    den = W @ live
    pred = (W @ np.nan_to_num(donor)) / np.maximum(den, 1e-9)
    out = []
    for i in range(W.shape[0]):
        m = np.isfinite(target[i]) & (den[i] > 0.2)
        y, f = target[i][m], pred[i][m]
        out.append(1 - ((f - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return np.array(out), (W * d["D"][-W.shape[0]:, :n]).sum(1).mean()


def kernel(d, scale):
    """exp(-distance / scale) over the gauged donors, for every ungauged basin."""
    return np.exp(-d["D"][d["n_gauged"]:, :d["n_gauged"]] / scale)


def section_decay(d):
    """How far donor information actually travels, measured among gauged basins only."""
    print("\n=== donor-information decay (gauged pairs, training period) ===")
    n = d["n_gauged"]
    S = np.nan_to_num(smooth(d["Z"])[:n][:, d["train_t"]])
    S = (S - S.mean(1, keepdims=True)) / np.maximum(S.std(1, keepdims=True), 1e-9)
    C = S @ S.T / S.shape[1]
    np.fill_diagonal(C, np.nan)
    D = d["D"][:n, :n]
    print(f"  {'separation':>16}  {'mean r':>8}  {'frac r>0.5':>11}  {'pairs':>8}")
    for lo, hi in ((0, 25), (25, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 20000)):
        m = (D >= lo) & (D < hi) & np.isfinite(C)
        print(f"  {f'{lo}-{hi} km':>16}  {np.nanmean(C[m]):>+8.3f}  "
              f"{np.nanmean(C[m] > 0.5):>11.3f}  {m.sum():>8}")
    best = np.nanargmax(np.where(np.isfinite(C), C, -9), 1)
    print(f"  best donor by correlation: median {np.median(D[np.arange(n), best]):.0f} km away; "
          f"it is the nearest donor for {np.mean(best == D.argmin(1)):.0%} of basins")


def section_retrieval(d, runs_dir=f"{ROOT}/runs", fold=0):
    """Hand rules against whatever the trained models actually learned."""
    print("\n=== retrieval R2: hand rules vs learned attention ===")
    n, te = d["n_gauged"], d["test_t"]
    nq = len(d["basins"]) - n
    print(f"  {'donor weighting':>34}  {'R2':>7}  {'eff. distance':>14}")
    Du = d["D"][n:, :n]
    for k in (1, 4, 16, 64, len(d["basins"]) - nq):
        idx = np.argsort(Du, 1)[:, :k]
        W = np.zeros((nq, n))
        np.put_along_axis(W, idx, 1.0 / k, 1)
        r2, dist = retrieval_r2(W, d, te)
        print(f"  {f'{k} nearest, uniform':>34}  {np.median(r2):>+7.3f}  {dist:>11.0f} km")
    for scale in (10, 20, 25, 50, 100, 200, 400):
        r2, dist = retrieval_r2(kernel(d, scale), d, te)
        print(f"  {f'exp(-d/{scale} km)':>34}  {np.median(r2):>+7.3f}  {dist:>11.0f} km")
    for path in sorted(__import__("glob").glob(f"{runs_dir}/*_fold{fold}_seed*/attention.npz")):
        arch = os.path.basename(os.path.dirname(path)).rsplit("_fold", 1)[0]
        z = np.load(path)
        A = z["attn"].astype(np.float64)
        if int(z["n_gauged"]) != n or A.shape[2] != n:
            print(f"  {'(skipped ' + arch + ': donor pool is not the gauged set)':>34}")
            continue
        W = A[:, n:, :].mean(0)  # time-mean attention profile
        r2, dist = retrieval_r2(W, d, te)
        tag = os.path.basename(os.path.dirname(path)).replace(f"_fold{fold}", "")
        print(f"  {f'LEARNED {tag}':>34}  {np.median(r2):>+7.3f}  {dist:>11.0f} km")


def section_gate(d):
    """Is donor reliability predictable from something you know at an unmonitored site?

    If it is, a gate on the donor pathway has something to condition on. If it is not,
    no gate can work and the far-from-any-gauge basins cannot be protected.
    """
    print("\n=== is donor reliability knowable in advance? ===")
    n = d["n_gauged"]
    r2, _ = retrieval_r2(kernel(d, 25.0), d, d["test_t"])
    nn = d["D"][n:, :n].min(1)
    ok = np.isfinite(r2)
    from scipy.stats import spearmanr
    s = spearmanr(nn[ok], r2[ok])
    print(f"  retrieval R2 vs distance to nearest gauge: rho={s.statistic:+.3f}  p={s.pvalue:.1e}"
          f"  (n={ok.sum()})")
    for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 10000)):
        m = (nn >= lo) & (nn < hi) & ok
        if m.sum():
            print(f"    nearest gauge {f'{lo}-{hi} km':>12}  n={m.sum():>3}  "
                  f"median retrieval R2 {np.median(r2[m]):+.3f}")


def section_network(d):
    """Does river-network structure add anything beyond straight-line distance?

    CAMELS ships no network topology, so this uses what the gauge ID already encodes: the
    first two digits are the USGS part number (one of 14 major drainage divisions, close to
    a 2-digit HUC region) and the remaining six run in downstream order within it, so two
    stations on one stream have numerically close IDs. A crude proxy -- but if real network
    proximity mattered much, the proxy should show something once distance is held fixed.
    """
    print("\n=== river-network structure beyond distance (USGS part / downstream order) ===")
    n = d["n_gauged"]
    S = np.nan_to_num(smooth(d["Z"])[:n][:, d["train_t"]])
    S = (S - S.mean(1, keepdims=True)) / np.maximum(S.std(1, keepdims=True), 1e-9)
    C = S @ S.T / S.shape[1]
    np.fill_diagonal(C, np.nan)
    D = d["D"][:n, :n]
    ids = d["basins"][:n]
    same = np.array([b[:2] for b in ids])[:, None] == np.array([b[:2] for b in ids])[None, :]
    num = np.array([int(b[2:]) for b in ids], float)
    gap = np.abs(num[:, None] - num[None, :])
    print(f"  {'separation':>16}  {'same part':>10}  {'other part':>11}"
          f"  {'near in order':>14}  {'far in order':>13}")
    for lo, hi in ((0, 25), (25, 50), (50, 100), (100, 200)):
        m = (D >= lo) & (D < hi) & np.isfinite(C)
        near = m & same & (gap < 2000)
        far = m & same & (gap >= 2000)
        f = lambda x: f"{np.nanmean(C[x]):+.3f}" if x.sum() >= 5 else "     -"
        print(f"  {f'{lo}-{hi} km':>16}  {f(m & same):>10}  {f(m & ~same):>11}"
              f"  {f(near):>14}  {f(far):>13}")


def section_history(d):
    """Does donor history buy anything over the donor's flow today? (donor_hist sensitivity)"""
    print("\n=== donor_hist sensitivity ===")
    print("  shared linear weights fitted on gauged basins over the training period, then")
    print("  applied unchanged to the held-out basins over the test period.")
    n = d["n_gauged"]
    W = kernel(d, 25.0)
    Wg = np.exp(-d["D"][:n, :n] / 25.0)  # gauged basins query the pool too, self excluded by inf
    ok = np.isfinite(d["Z"][:n]).astype(float)
    z0 = np.nan_to_num(d["Z"][:n])
    pool = np.vstack([(Wg @ z0) / np.maximum(Wg @ ok, 1e-9),
                      (W @ z0) / np.maximum(W @ ok, 1e-9)])
    print(f"  {'lags used':>34}  {'held-out R2':>11}")
    for lags, label in (([0], "0 (donor's flow today)"),
                        (list(range(3)), "0-2"),
                        (list(range(HIST)), f"0-{HIST - 1}  (current donor_hist)"),
                        (list(range(15)), "0-14"),
                        (list(range(30)), "0-29"),
                        ([0, 1, 2, 3, 4, 5, 6, 9, 13, 20, 29, 44, 59, 89], "multi-scale to 90 d")):
        mx = max(lags) + 1
        tr = d["train_t"][d["train_t"] >= mx]
        te = d["test_t"][d["test_t"] >= mx]
        A = lambda idx: np.stack([pool[:, idx - l] for l in lags], -1)
        y = d["Z"][:n][:, tr]
        m = np.isfinite(y)
        Xf = np.concatenate([A(tr)[:n][m], np.ones((m.sum(), 1))], 1)
        w = np.linalg.lstsq(Xf.T @ Xf + 1e-3 * np.eye(Xf.shape[1]), Xf.T @ y[m], rcond=None)[0]
        Xe = A(te)[n:]
        pr = np.concatenate([Xe, np.ones(Xe.shape[:2] + (1,))], -1) @ w
        yt = d["Z"][n:][:, te]
        out = []
        for i in range(yt.shape[0]):
            mm = np.isfinite(yt[i])
            yy, ff = yt[i][mm], pr[i][mm]
            out.append(1 - ((ff - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
        print(f"  {label:>34}  {np.median(out):>+11.4f}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--runs", default=f"{ROOT}/runs")
    a = p.parse_args()
    d = load(a.fold)
    print(f"fold {a.fold}: {d['n_gauged']} gauged donors, "
          f"{len(d['basins']) - d['n_gauged']} held-out queries")
    section_decay(d)
    section_retrieval(d, a.runs, a.fold)
    section_gate(d)
    section_network(d)
    section_history(d)


if __name__ == "__main__":
    main()
