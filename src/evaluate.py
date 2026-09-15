"""Prediction and metrics.

    python -m src.train ...   # each run writes runs/<name>/predictions.npz
    python -m src.evaluate    # aggregates every run into runs/metrics.csv

Two cells carry the comparison:
    gauged/test     out-of-sample in TIME  (trained basins, unseen years)
    ungauged/test   out-of-sample in SPACE (held-out basins, unseen years) -- the PUB case
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd
import torch

from .config import Cfg

CELLS = ("gauged/test", "ungauged/test")


@torch.no_grad()
def predict(cfg: Cfg, model, data, t_idx, chunk=256):
    """[n_basins, len(t_idx)] flow in mm/day. Every basin is queried, held out or not."""
    model.eval()
    n = len(data.basins)
    out = np.empty((n, len(t_idx)), np.float32)
    for k, t in enumerate(t_idx):
        t = int(t)
        window = data.X[:, t - cfg.seq_len + 1 : t + 1]
        preds = [model(window[i : i + chunk]).float().cpu().numpy() for i in range(0, n, chunk)]
        out[:, k] = np.concatenate(preds)
    return out


def write_predictions(cfg: Cfg, model, data, out_dir, save_train=False):
    """Test-period predictions for every basin.

    The training period is skipped by default: nothing in this branch reads it, and it is
    5479 of 9131 inference days -- about 60% of the post-training cost on a rented GPU,
    multiplied by every run in the grid. `--save-train-predictions` puts it back for
    diagnostics that want it.
    """
    out = dict(test=predict(cfg, model, data, data.test_t), test_t=data.test_t,
               basins=np.array(data.basins), n_gauged=data.n_gauged)
    if save_train:
        out.update(train=predict(cfg, model, data, data.train_t), train_t=data.train_t)
    np.savez_compressed(f"{out_dir}/predictions.npz", **out)


def flow_band_bias(sim, obs, lo, hi):
    """Percent bias over the observed-flow band between the `lo` and `hi` quantiles.

    Yilmaz et al. (2008) style: the band is chosen on the OBSERVED flow, so "high-flow bias"
    means the model's error on the days the river really was high, not on the days it
    predicted high.
    """
    m = np.isfinite(obs)
    if m.sum() < 10:
        return np.nan
    o, s = obs[m], sim[m]
    q_lo, q_hi = np.quantile(o, lo), np.quantile(o, hi)
    band = (o >= q_lo) & (o <= q_hi)
    if band.sum() < 5 or o[band].sum() == 0:
        return np.nan
    return 100.0 * (s[band].sum() - o[band].sum()) / o[band].sum()


def metrics_row(sim, obs):
    """Per-basin NSE, KGE and its components, RMSE, and bias by flow band."""
    sim = np.clip(sim, 0, None)
    m = np.isfinite(obs)
    s, o = sim[m], obs[m]
    if len(o) < 2 or o.std() == 0:
        return {k: np.nan for k in ("nse", "kge", "r", "alpha", "beta", "rmse",
                                    "pbias", "bias_high", "bias_mid", "bias_low")}
    nse = 1 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    alpha, beta = s.std() / o.std(), s.mean() / o.mean()
    # A constant prediction has no correlation, so r and KGE are undefined -- but NSE and
    # RMSE are perfectly well defined and must still be scored.
    r = np.corrcoef(s, o)[0, 1] if s.std() > 0 else np.nan
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return dict(nse=nse, kge=kge, r=r, alpha=alpha, beta=beta,
                rmse=np.sqrt(((s - o) ** 2).mean()),
                pbias=100.0 * (s.sum() - o.sum()) / o.sum(),
                bias_high=flow_band_bias(sim, obs, 0.90, 1.00),
                bias_mid=flow_band_bias(sim, obs, 0.30, 0.90),
                bias_low=flow_band_bias(sim, obs, 0.00, 0.30))


def metrics(sim, obs):
    return pd.DataFrame([metrics_row(s, o) for s, o in zip(sim, obs)])


def load_runs(runs_dir="runs", cache_dir="data/cache"):
    """Score every completed run into one long table: basin x seed x cell."""
    Q = np.load(f"{cache_dir}/Q.npy")
    all_basins = list(json.load(open(f"{cache_dir}/meta.json"))["basins"])
    rows = []
    for path in sorted(glob.glob(f"{runs_dir}/vanilla_fold*_seed*/predictions.npz")):
        cfg = json.load(open(path.rsplit("/", 1)[0] + "/config.json"))
        if cfg.get("max_steps"):
            print(f"skipping smoke run {path}")
            continue
        p = np.load(path, allow_pickle=True)
        basins, n_gauged = list(p["basins"]), int(p["n_gauged"])
        # Observations are keyed by gauge ID, never by position, so a fold reordering cannot
        # silently pair the wrong basin with the wrong prediction.
        obs_all = Q[[all_basins.index(b) for b in basins]]
        frame = metrics(p["test"], obs_all[:, p["test_t"]])
        frame["basin"], frame["seed"], frame["fold"] = basins, cfg["seed"], cfg["fold"]
        frame["cell"] = np.where(np.arange(len(basins)) < n_gauged, "gauged/test", "ungauged/test")
        rows.append(frame)
    if not rows:
        raise SystemExit(f"nothing to score under {runs_dir}/")
    return pd.concat(rows, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="runs/metrics.csv")
    a = p.parse_args()
    df = load_runs(a.runs)
    df.to_csv(a.out, index=False)
    for cell in CELLS:
        cur = df.query("cell == @cell")
        if cur.empty:
            continue
        per_seed = cur.groupby(["fold", "seed"]).nse.median()
        print(f"\n{cell}  ({cur.basin.nunique()} basins, {cur.seed.nunique()} seeds)")
        print(f"  median NSE per seed: {per_seed.min():.3f} - {per_seed.max():.3f}")
    print(f"\nper-basin metrics -> {a.out}")
    print("NOTE: the headline number is the ensemble-mean hydrograph scored once per basin "
          "(see src/correct.py), not these per-seed medians.")


if __name__ == "__main__":
    main()
