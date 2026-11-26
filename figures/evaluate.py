"""Score the corrected predictions into the CSVs every figure reads.

    python figures/evaluate.py                  # every corrected cell; --model lstm --method idw
    python figures/evaluate.py --cell ungauged
    python figures/evaluate.py --method mlp

Reads runs/<model>/<method>/<cell>/ and writes, into figures/: metrics_<cell>_<model>_<method>.csv
(per-basin scores), tuning_<cell>_<model>_<method>.csv (the cross-validation surface),
run_info.json, and error_structure[_fit]_gauged_<model>.csv (residual correlation vs distance).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from palette import BENCHMARK, model_label  # noqa: E402
from shared import (CACHE_DIR, FOLDS_CSV, basin_ids, distance_matrix,  # noqa: E402
                    fold_labels, kge_per_basin, load_mean, observed, test_period_label)

CELLS = ("gauged", "ungauged")


def variants(model):
    """{key in predictions.npz: display name}."""
    name = model_label(model)
    return {"raw": name, "nearest": BENCHMARK, "corrected": f"{name} + correction"}


def _suffix(model, method):
    return f"_{model}_{method}"


def folds_csv_for(model):
    """Fold assignment for a model's ungauged cell: its own override file if present."""
    override = f"{os.path.dirname(FOLDS_CSV)}/basin_list_with_folds_{model}.csv"
    return override if model != "lstm" and os.path.exists(override) else FOLDS_CSV


# Error structure: max pair distance (km), climatology window (days), LOESS bandwidth (km).
DIST_MAX_KM, CLIM_WINDOW, LOESS_KM = 1000, 31, 60


def read_metrics(cell, model="lstm", method="idw"):
    return pd.read_csv(f"{HERE}/metrics_{cell}{_suffix(model, method)}.csv",
                       dtype={"basin": str})


def read_tuning(cell, model="lstm", method="idw"):
    return pd.read_csv(f"{HERE}/tuning_{cell}{_suffix(model, method)}.csv")


def _es_suffix(model, cell):
    parts = ([] if cell == "ungauged" else [cell]) + [model]
    return "_" + "_".join(parts)


def read_error_structure(model="lstm", cell="gauged"):
    """(per-pair correlations, the LOESS fit through them)."""
    suffix = _es_suffix(model, cell)
    return (pd.read_csv(f"{HERE}/error_structure{suffix}.csv"),
            pd.read_csv(f"{HERE}/error_structure_fit{suffix}.csv"))


def fold_settings(df, raw_label="LSTM"):
    """[(fold, K, lambda, p)] as selected, in fold order."""
    first = df[df.variant == raw_label].groupby("fold")[["k", "lam", "p"]].first()
    return [(int(f), int(r.k), float(r.lam), float(r.p)) for f, r in first.iterrows()]


def flow_band_bias(sim, obs, lo, hi):
    """Percent bias over the [lo, hi] quantile band of observed flow (Yilmaz et al., 2008)."""
    m = np.isfinite(obs)
    if m.sum() < 10:
        return np.nan
    o, s = obs[m], sim[m]
    band = (o >= np.quantile(o, lo)) & (o <= np.quantile(o, hi))
    if band.sum() < 5 or o[band].sum() == 0:
        return np.nan
    return 100.0 * (s[band].sum() - o[band].sum()) / o[band].sum()


def metrics_row(sim, obs):
    """NSE, KGE and its components, RMSE, and bias by flow band for one basin."""
    sim = np.clip(sim, 0, None)
    m = np.isfinite(obs)
    s, o = sim[m], obs[m]
    if len(o) < 2 or o.std() == 0:
        return {k: np.nan for k in ("nse", "kge", "r", "alpha", "beta", "rmse", "pbias",
                                    "bias_high", "bias_mid", "bias_low")}
    nse = 1 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    alpha, beta = s.std() / o.std(), s.mean() / o.mean()
    # r and KGE are undefined for a constant prediction, but NSE and RMSE are still scored.
    r = np.corrcoef(s, o)[0, 1] if s.std() > 0 else np.nan
    return dict(nse=nse, kge=1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2),
                r=r, alpha=alpha, beta=beta,
                rmse=np.sqrt(((s - o) ** 2).mean()),
                pbias=100.0 * (s.sum() - o.sum()) / o.sum(),
                bias_high=flow_band_bias(sim, obs, 0.90, 1.00),
                bias_mid=flow_band_bias(sim, obs, 0.30, 0.90),
                bias_low=flow_band_bias(sim, obs, 0.00, 0.30))


def seed_spread(cell, model):
    """Median KGE of each individual seed run, as a check on the seed mean."""
    root = f"runs/{model}/{cell}"
    runs = sorted(d for d in os.listdir(root) if not d.startswith("_"))
    labels, order, out = fold_labels(), {b: i for i, b in enumerate(basin_ids())}, []
    for d in runs:
        p = np.load(f"{root}/{d}/predictions.npz", allow_pickle=True)
        rows = np.array([order[b] for b in list(p["basins"])])
        scored = kge_per_basin(p["test"], observed(p["test_t"])[rows])
        if cell == "ungauged":
            fold = int(d.split("fold")[1].split("_")[0])
            scored = scored[labels[rows] == fold]
        out.append((d, np.nanmedian(scored)))
    return out


def evaluate(cell, model, method):
    src = f"runs/{model}/{method}/{cell}"
    p = np.load(f"{src}/predictions.npz", allow_pickle=True)
    params = pd.read_csv(f"{src}/params.csv").set_index("fold")
    obs = observed(p["test_t"])
    ids = np.array(basin_ids())
    labels = fold_labels(folds_csv=folds_csv_for(model)) if cell == "ungauged" else fold_labels()
    scored = np.flatnonzero(np.isfinite(p["raw"]).any(1))
    raw_label, corr_label = model_label(model), f"{model_label(model)} + correction"

    frames = []
    for key, variant in variants(model).items():
        frame = pd.DataFrame([metrics_row(p[key][i], obs[i]) for i in scored])
        frame["basin"], frame["fold"], frame["variant"] = ids[scored], labels[scored], variant
        settings = [c for c in ("k", "lam", "p", "n_seeds") if c in params]
        frames.append(frame.join(params[settings], on="fold"))
    df = pd.concat(frames, ignore_index=True)
    suffix = _suffix(model, method)
    df.to_csv(f"{HERE}/metrics_{cell}{suffix}.csv", index=False)
    pd.read_csv(f"{src}/tuning.csv").to_csv(f"{HERE}/tuning_{cell}{suffix}.csv", index=False)

    med = df.groupby("variant").kge.median()
    print(f"\n{cell} ({model}, {method}): {len(scored)} basins, "
          f"{int(params.n_seeds.max())} seeds, test {test_period_label(p['test_t'])}")
    print(f"  median KGE  {med[raw_label]:.4f} -> {med[corr_label]:.4f} "
          f"({med[corr_label] - med[raw_label]:+.4f})"
          f"   nearest gauge only {med[BENCHMARK]:.4f}")
    print("  per-seed median KGE: "
          + ", ".join(f"{d} {v:.3f}" for d, v in seed_spread(cell, model)))
    return dict(n_basins=len(scored), n_seeds=int(params.n_seeds.max()),
                n_folds=int(params.index.nunique()), test_period=test_period_label(p["test_t"]))


def _deseason(x, doy):
    """`x` minus its circularly smoothed day-of-year climatology, removing the seasonal cycle."""
    m = np.isfinite(x)
    onehot = np.zeros((x.shape[1], 366))
    onehot[np.arange(x.shape[1]), doy] = 1.0
    total, count = np.where(m, x, 0.0) @ onehot, m.astype(float) @ onehot
    clim = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    pad = CLIM_WINDOW // 2
    wrapped = np.concatenate([clim[:, -pad:], clim, clim[:, :pad]], 1)
    smooth = np.apply_along_axis(np.convolve, 1, wrapped, np.ones(CLIM_WINDOW) / CLIM_WINDOW,
                                 "valid")
    return x - smooth[:, doy]


def _corr_rows(x):
    """[n, n] correlation between every pair of rows, missing days filled with the row mean."""
    m = np.isfinite(x)
    a = np.where(m, x - (np.where(m, x, 0.0).sum(1) / np.maximum(m.sum(1), 1))[:, None], 0.0)
    norm = np.sqrt((a ** 2).sum(1))
    return (a @ a.T) / np.outer(norm, norm)


def _loess(x, y, grid, bandwidth=LOESS_KM, iterations=2):
    """Robust LOESS (Cleveland, 1979) with a fixed bandwidth in km, evaluated on `grid`."""
    def fit(weights):
        out = np.empty(len(grid))
        for i, x0 in enumerate(grid):
            w = np.clip(1 - (np.abs(x - x0) / bandwidth) ** 3, 0, None) ** 3 * weights
            sw = w.sum()
            if sw <= 0:
                out[i] = np.nan
                continue
            mx, my = (w * x).sum() / sw, (w * y).sum() / sw
            sxx = (w * (x - mx) ** 2).sum()
            slope = (w * (x - mx) * (y - my)).sum() / sxx if sxx > 0 else 0.0
            out[i] = my + slope * (x0 - mx)
        return out

    robust, curve = np.ones(len(x)), None
    for _ in range(iterations + 1):
        curve = fit(robust)
        resid = y - np.interp(x, grid, curve)
        s = np.median(np.abs(resid))
        robust = np.clip(1 - (resid / (6 * s)) ** 2, 0, None) ** 2 if s > 0 else robust
    return curve


def error_structure(model, cell="gauged"):
    """Correlation of uncorrected daily residuals between basin pairs against gauge distance."""
    pred, test_t, _ = load_mean(f"runs/{model}/{cell}/_mean/predictions.npz")
    scored = np.flatnonzero(np.isfinite(pred).any(1))
    obs = observed(test_t)[scored]
    res = np.clip(pred[scored], 0, None) - obs
    doy = pd.to_datetime(np.load(f"{CACHE_DIR}/dates.npy")[test_t]).dayofyear.to_numpy() - 1
    dist = distance_matrix()[np.ix_(scored, scored)]

    iu = np.triu_indices(len(dist), 1)
    d, c, cd = dist[iu], _corr_rows(res)[iu], _corr_rows(_deseason(res, doy))[iu]
    keep = d < DIST_MAX_KM
    d, c, cd = d[keep], c[keep], cd[keep]
    suffix = _es_suffix(model, cell)
    pd.DataFrame({"dist_km": d, "r": c, "r_deseason": cd}).to_csv(
        f"{HERE}/error_structure{suffix}.csv", index=False, float_format="%.5g")

    # Starts at 10 km: the local linear fit curls upward near zero distance.
    grid = np.linspace(10, DIST_MAX_KM, 160)
    fit, fit_d = _loess(d, c, grid), _loess(d, cd, grid)
    pd.DataFrame({"dist_km": grid, "r_fit": fit, "r_fit_deseason": fit_d}).to_csv(
        f"{HERE}/error_structure_fit{suffix}.csv", index=False, float_format="%.5g")

    print(f"\nerror structure ({cell}, {len(scored)} basins): {len(d)} basin pairs within "
          f"{DIST_MAX_KM} km")
    for at in (25, 100, 500, 900):
        print(f"  fitted correlation at {at:4d} km {np.interp(at, grid, fit):.3f}, "
              f"{np.interp(at, grid, fit_d):.3f} deseasonalised")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=CELLS, help="default: whichever have been corrected")
    ap.add_argument("--model", default="lstm")
    ap.add_argument("--method", default="idw", help="which runs/<model>/<method>/ to score")
    a = ap.parse_args()

    corrected = f"runs/{a.model}/{a.method}/{{}}/predictions.npz"
    cells = [a.cell] if a.cell else [c for c in CELLS if os.path.exists(corrected.format(c))]
    if not cells:
        raise SystemExit(f"nothing corrected under runs/{a.model}/{a.method}/ yet")

    info_path = f"{HERE}/run_info.json"
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    key = a.model if a.method == "idw" else f"{a.model}_{a.method}"
    info.setdefault(key, {})
    for cell in cells:
        info[key][cell] = evaluate(cell, a.model, a.method)
    json.dump(info, open(info_path, "w"), indent=1)
    if "gauged" in cells:
        error_structure(a.model)
    suffix = _suffix(a.model, a.method)
    print(f"\n-> {HERE}/metrics_*{suffix}.csv, tuning_*{suffix}.csv, "
          f"error_structure_gauged_{a.model}.csv, run_info.json")


if __name__ == "__main__":
    main()
