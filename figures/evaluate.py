"""Score the corrected hydrographs once, so every figure just reads a CSV.

    python figures/evaluate.py                  # whichever cells have been corrected
    python figures/evaluate.py --cell ungauged

Reads the finished corrections out of runs/ and writes, into this folder:

    metrics_<cell>.csv   one row per basin x variant: NSE, KGE and its components, RMSE, and
                         bias split by how high the river actually was
    tuning_<cell>.csv    the cross-validation surface the settings were chosen on
    run_info.json        test window, seed and basin counts, for the figure captions

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
from residual_postprocessing import (basin_ids, fold_labels, kge_per_basin,  # noqa: E402
                                     observed, test_period_label)

CELLS = ("gauged", "ungauged")
VARIANTS = {"raw": "LSTM", "corrected": "LSTM + correction"}


def read_metrics(cell):
    """The per-basin scores every figure draws from."""
    return pd.read_csv(f"{HERE}/metrics_{cell}.csv", dtype={"basin": str})


def read_tuning(cell):
    """The cross-validation surface the settings were chosen on."""
    return pd.read_csv(f"{HERE}/tuning_{cell}.csv")


def read_info(cell=None):
    info = json.load(open(f"{HERE}/run_info.json"))
    return info[cell] if cell else info


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
          f"({med['LSTM + correction'] - med['LSTM']:+.4f})")
    print("  per-seed median KGE: " + ", ".join(f"{d} {v:.3f}" for d, v in seed_spread(cell, model)))
    return dict(n_basins=len(scored), n_seeds=int(params.n_seeds.max()),
                n_folds=int(params.index.nunique()), test_period=test_period_label(p["test_t"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=CELLS, help="default: whichever have been corrected")
    ap.add_argument("--model", default="lstm")
    a = ap.parse_args()

    cells = [a.cell] if a.cell else [c for c in CELLS
                                     if os.path.exists(f"runs/{a.model}/{c}_corrected/predictions.npz")]
    if not cells:
        raise SystemExit(f"nothing corrected under runs/{a.model}/ yet")

    info_path = f"{HERE}/run_info.json"
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    for cell in cells:
        info[cell] = evaluate(cell, a.model)
    info["model"] = a.model
    json.dump(info, open(info_path, "w"), indent=1)
    print(f"\n-> {HERE}/metrics_*.csv, tuning_*.csv, run_info.json")


if __name__ == "__main__":
    main()
