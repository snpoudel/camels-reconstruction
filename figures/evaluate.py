"""Score the corrected hydrographs once, so every figure just reads a CSV.

    python figures/evaluate.py                  # whichever cells have been corrected
    python figures/evaluate.py --cell ungauged

Reads the finished corrections out of runs/ and writes, into this folder:

    metrics_<cell>.csv   one row per basin x variant: NSE, KGE and its components, RMSE, and
                         bias split by how high the river actually was
    tuning_<cell>.csv    the cross-validation surface the settings were chosen on
    run_info.json        test window, seed and basin counts, for the figure captions
    error_structure.csv  whether the model's errors are spatially structured at all, by gauge
                         separation -- the premise the correction rests on

Nothing downstream of here touches runs/, so a figure can never disagree with the numbers.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from residual_postprocessing import (CACHE_DIR, EPS, basin_ids, distance_matrix,  # noqa: E402
                                     fold_labels, kge_per_basin, observed, test_period_label)

CELLS = ("gauged", "ungauged")
# "nearest" is the parameter-free reference the correction scripts save alongside their own
# output: the single nearest gauge's error, applied in full. Only the CDF figure draws it.
VARIANTS = {"raw": "LSTM", "nearest": "nearest gauge only", "corrected": "LSTM + correction"}

# The error-structure analysis: pairs out to 1000 km, a 31-day circular window on the
# day-of-year climatology that deseasonalising removes, and the LOESS bandwidth the trend
# through the pair cloud is fitted with.
DIST_MAX_KM, CLIM_WINDOW, LOESS_KM = 1000, 31, 60


def read_metrics(cell):
    """The per-basin scores every figure draws from."""
    return pd.read_csv(f"{HERE}/metrics_{cell}.csv", dtype={"basin": str})


def read_tuning(cell):
    """The cross-validation surface the settings were chosen on."""
    return pd.read_csv(f"{HERE}/tuning_{cell}.csv")


def read_info(cell=None):
    info = json.load(open(f"{HERE}/run_info.json"))
    return info[cell] if cell else info


def read_error_structure():
    """The premise figure's two tables: (one row per basin pair, the fitted trend through
    them)."""
    return (pd.read_csv(f"{HERE}/error_structure.csv"),
            pd.read_csv(f"{HERE}/error_structure_fit.csv"))


def fold_settings(df):
    """[(fold, K, lambda, p)] as selected, one per fold, in fold order."""
    first = df[df.variant == "LSTM"].groupby("fold")[["k", "lam", "p"]].first()
    return [(int(f), int(r.k), float(r.lam), float(r.p)) for f, r in first.iterrows()]


def flow_band_bias(sim, obs, lo, hi):
    """Percent bias over the observed-flow band between the `lo` and `hi` quantiles.

    Yilmaz et al. (2008) style: the band is chosen on the OBSERVED flow, so "high-flow bias"
    means the model's error on the days the river really was high, not on the days it predicted
    high.
    """
    m = np.isfinite(obs)
    if m.sum() < 10:
        return np.nan
    o, s = obs[m], sim[m]
    band = (o >= np.quantile(o, lo)) & (o <= np.quantile(o, hi))
    if band.sum() < 5 or o[band].sum() == 0:
        return np.nan
    return 100.0 * (s[band].sum() - o[band].sum()) / o[band].sum()


def metrics_row(sim, obs):
    """Per-basin NSE, KGE and its components, RMSE, and bias by flow band."""
    sim = np.clip(sim, 0, None)
    m = np.isfinite(obs)
    s, o = sim[m], obs[m]
    if len(o) < 2 or o.std() == 0:
        return {k: np.nan for k in ("nse", "kge", "r", "alpha", "beta", "rmse", "pbias",
                                    "bias_high", "bias_mid", "bias_low")}
    nse = 1 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    alpha, beta = s.std() / o.std(), s.mean() / o.mean()
    # A constant prediction has no correlation, so r and KGE are undefined -- but NSE and RMSE
    # are perfectly well defined and must still be scored.
    r = np.corrcoef(s, o)[0, 1] if s.std() > 0 else np.nan
    return dict(nse=nse, kge=1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2),
                r=r, alpha=alpha, beta=beta,
                rmse=np.sqrt(((s - o) ** 2).mean()),
                pbias=100.0 * (s.sum() - o.sum()) / o.sum(),
                bias_high=flow_band_bias(sim, obs, 0.90, 1.00),
                bias_mid=flow_band_bias(sim, obs, 0.30, 0.90),
                bias_low=flow_band_bias(sim, obs, 0.00, 0.30))


def seed_spread(cell, model):
    """Median KGE of each individual seed, as a sanity check on the ensemble mean."""
    root = f"runs/{model}/{cell}"
    runs = sorted(d for d in os.listdir(root) if not d.startswith("_"))
    labels, order, out = fold_labels(), {b: i for i, b in enumerate(basin_ids())}, []
    for d in runs:
        p = np.load(f"{root}/{d}/predictions.npz", allow_pickle=True)
        rows = np.array([order[b] for b in list(p["basins"])])
        scored = kge_per_basin(p["test"], observed(p["test_t"])[rows])
        if cell == "ungauged":  # score only the basins this run held out
            fold = int(d.split("fold")[1].split("_")[0])
            scored = scored[labels[rows] == fold]
        out.append((d, np.nanmedian(scored)))
    return out


def evaluate(cell, model):
    src = f"runs/{model}/{cell}_corrected"
    p = np.load(f"{src}/predictions.npz", allow_pickle=True)
    params = pd.read_csv(f"{src}/params.csv").set_index("fold")
    obs = observed(p["test_t"])
    ids, labels = np.array(basin_ids()), fold_labels()
    scored = np.flatnonzero(np.isfinite(p["raw"]).any(1))

    frames = []
    for key, variant in VARIANTS.items():
        frame = pd.DataFrame([metrics_row(p[key][i], obs[i]) for i in scored])
        frame["basin"], frame["fold"], frame["variant"] = ids[scored], labels[scored], variant
        frames.append(frame.join(params[["k", "lam", "p", "n_seeds"]], on="fold"))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(f"{HERE}/metrics_{cell}.csv", index=False)
    pd.read_csv(f"{src}/tuning.csv").to_csv(f"{HERE}/tuning_{cell}.csv", index=False)

    med = df.groupby("variant").kge.median()
    print(f"\n{cell}: {len(scored)} basins, {int(params.n_seeds.max())} seeds, "
          f"test {test_period_label(p['test_t'])}")
    print(f"  median KGE  {med['LSTM']:.4f} -> {med['LSTM + correction']:.4f} "
          f"({med['LSTM + correction'] - med['LSTM']:+.4f})"
          f"   nearest gauge only {med['nearest gauge only']:.4f}")
    print("  per-seed median KGE: "
          + ", ".join(f"{d} {v:.3f}" for d, v in seed_spread(cell, model)))
    return dict(n_basins=len(scored), n_seeds=int(params.n_seeds.max()),
                n_folds=int(params.index.nunique()), test_period=test_period_label(p["test_t"]))


def _deseason(x, doy):
    """`x` with its own day-of-year climatology removed, smoothed circularly over CLIM_WINDOW.

    Both sides of every correlation below get this, so what survives is day-to-day agreement
    rather than the seasonal cycle two nearby basins would share anyway through their forcings.
    """
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
    """[n, n] correlation between every pair of rows.

    The 0.01% of days a gauge did not report are filled at that basin's own mean, a zero
    deviation: pairwise-complete correlations over 140k pairs would cost minutes to move the
    fourth decimal.
    """
    m = np.isfinite(x)
    a = np.where(m, x - (np.where(m, x, 0.0).sum(1) / np.maximum(m.sum(1), 1))[:, None], 0.0)
    norm = np.sqrt((a ** 2).sum(1))
    return (a @ a.T) / np.outer(norm, norm)


def _loess(x, y, grid, bandwidth=LOESS_KM, iterations=2):
    """Robust locally weighted regression (Cleveland 1979) evaluated on `grid`.

    A fixed bandwidth in kilometres, not the usual nearest-neighbour span: pairs are very
    unevenly spread along the distance axis -- a few hundred under 25 km against thousands
    beyond 800 km -- so a span that adapts to point density would reach hundreds of kilometres
    out at the short-range end and flatten the very drop the figure is about.

    The robustness iterations matter here rather than being a formality. Pair correlations are
    left-skewed: a minority of pairs correlate poorly or negatively, and a plain local mean is
    dragged well below the typical pair by them (0.48 against a median of 0.51 at short range).
    Bisquare reweighting discounts that tail, so the line tracks the bulk of the cloud.
    """
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
        # Residuals are taken against the curve interpolated back to the points: fitting at all
        # 39k of them directly would be quadratic for no visible gain.
        resid = y - np.interp(x, grid, curve)
        s = np.median(np.abs(resid))
        robust = np.clip(1 - (resid / (6 * s)) ** 2, 0, None) ** 2 if s > 0 else robust
    return curve


def error_structure(model):
    """Are the model's errors spatially structured? The premise the correction rests on.

    How two basins' daily residuals correlate, against how far apart their gauges are, asked of
    the log residual the correction transfers and asked twice -- once raw, once with the
    seasonal cycle removed from both sides. Coordinates and residuals only: no K, lambda or p
    enters this, so what it establishes is a property of the model's errors rather than of the
    correction built on them.

    Ungauged only: these are the out-of-sample residuals of the model that held each basin out,
    which is the error the reconstruction problem actually faces.
    """
    p = np.load(f"runs/{model}/ungauged_corrected/predictions.npz", allow_pickle=True)
    obs = observed(p["test_t"])
    res = np.log(np.clip(p["raw"], 0, None) + EPS) - np.log(obs + EPS)
    doy = pd.to_datetime(np.load(f"{CACHE_DIR}/dates.npy")[p["test_t"]]).dayofyear.to_numpy() - 1
    dist = distance_matrix()

    iu = np.triu_indices(len(dist), 1)
    d, c, cd = dist[iu], _corr_rows(res)[iu], _corr_rows(_deseason(res, doy))[iu]
    keep = d < DIST_MAX_KM
    d, c, cd = d[keep], c[keep], cd[keep]
    pd.DataFrame({"dist_km": d, "r": c, "r_deseason": cd}).to_csv(
        f"{HERE}/error_structure.csv", index=False, float_format="%.5g")

    # The curve starts at 10 km, not at the closest pair: a local linear fit against a kernel
    # truncated by the x = 0 boundary extrapolates the steep decay and curls upward, which is an
    # artefact of the fit rather than anything in the data.
    grid = np.linspace(10, DIST_MAX_KM, 160)
    fit, fit_d = _loess(d, c, grid), _loess(d, cd, grid)
    pd.DataFrame({"dist_km": grid, "r_fit": fit, "r_fit_deseason": fit_d}).to_csv(
        f"{HERE}/error_structure_fit.csv", index=False, float_format="%.5g")

    print(f"\nerror structure: {len(d)} basin pairs within {DIST_MAX_KM} km")
    for at in (25, 100, 500, 900):
        print(f"  fitted correlation at {at:4d} km {np.interp(at, grid, fit):.3f}, "
              f"{np.interp(at, grid, fit_d):.3f} deseasonalised")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=CELLS, help="default: whichever have been corrected")
    ap.add_argument("--model", default="lstm")
    a = ap.parse_args()

    corrected = f"runs/{a.model}/{{}}_corrected/predictions.npz"
    cells = [a.cell] if a.cell else [c for c in CELLS if os.path.exists(corrected.format(c))]
    if not cells:
        raise SystemExit(f"nothing corrected under runs/{a.model}/ yet")

    info_path = f"{HERE}/run_info.json"
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    for cell in cells:
        info[cell] = evaluate(cell, a.model)
    info["model"] = a.model
    json.dump(info, open(info_path, "w"), indent=1)
    if "ungauged" in cells:
        error_structure(a.model)
    print(f"\n-> {HERE}/metrics_*.csv, tuning_*.csv, error_structure.csv, run_info.json")


if __name__ == "__main__":
    main()
