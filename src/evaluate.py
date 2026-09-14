"""Prediction, metrics, and the results table.

    python -m src.train ...     # each run writes runs/<name>/predictions.npz
    python -m src.evaluate      # aggregates every run into runs/metrics.csv

`metrics.csv` is one row per basin x architecture x seed x cell, which is the shape the
CDF plots are drawn from. Two cells carry the comparison:

    gauged/test     out-of-sample in TIME  (trained basins, unseen years)
    ungauged/test   out-of-sample in SPACE (held-out basins, unseen years)
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd
import torch

from .config import Cfg
from .data import layout, step_inputs

CELLS = ("gauged/train", "gauged/test", "ungauged/test")
METRICS = ["nse", "kge", "rmse"]


@torch.no_grad()
def predict(cfg: Cfg, model, data, t_idx):
    """Flow in mm/day for every basin over `t_idx`. Returns [n_basins, len(t_idx)]."""
    n_enc, n_query, n_donor = layout(cfg, data, training=False)
    model.eval()
    out = np.empty((n_query, len(t_idx)), np.float32)
    for k, t in enumerate(t_idx):
        x, dz, valid, dist = step_inputs(cfg, data, int(t), n_enc, n_donor, n_query)
        pred, _, _ = model(x, n_query, n_donor, dz, valid, dist)
        out[:, k] = pred.cpu().numpy()
    return out


def write_predictions(cfg: Cfg, model, data, out_dir, stride=1):
    """Both periods for all basins -- the 2x2 cells are slices of this, not extra runs.

    `stride` subsamples evaluation dates; it exists for smoke tests and must stay 1 for real runs.
    """
    train_t, test_t = data.train_t[::stride], data.test_t[::stride]
    np.savez_compressed(
        f"{out_dir}/predictions.npz",
        train=predict(cfg, model, data, train_t),
        test=predict(cfg, model, data, test_t),
        train_t=train_t, test_t=test_t,
        basins=np.array(data.basins), n_gauged=data.n_gauged,
    )


def metrics(sim, obs):
    """Per-basin NSE, KGE with its r/alpha/beta components, and RMSE.

    Negative predictions are clipped here rather than at write time, so every architecture is
    scored through exactly the same transformation.
    """
    sim = np.clip(sim, 0, None)
    rows = []
    for s, o in zip(sim, obs):
        m = np.isfinite(o)
        s, o = s[m], o[m]
        if len(o) < 2 or o.std() == 0:  # nothing is defined against a constant observation
            rows.append({k: np.nan for k in ("nse", "kge", "r", "alpha", "beta", "rmse")})
            continue
        nse = 1 - ((s - o) ** 2).sum() / ((o - o.mean()) ** 2).sum()
        rmse = np.sqrt(((s - o) ** 2).mean())
        alpha, beta = s.std() / o.std(), s.mean() / o.mean()
        # A constant prediction has no correlation, so r and KGE are undefined -- but NSE and
        # RMSE are perfectly well defined and must still be scored. Dropping the basin instead
        # would delete it from the median for whichever architecture failed hardest, which
        # manufactures a gain for the others.
        r = np.corrcoef(s, o)[0, 1] if s.std() > 0 else np.nan
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        rows.append(dict(nse=nse, kge=kge, r=r, alpha=alpha, beta=beta, rmse=rmse))
    return pd.DataFrame(rows)


def load_runs(runs_dir="runs", cache_dir="data/cache"):
    """Score every completed run into one long table: basin x arch x seed x cell."""
    Q = np.load(f"{cache_dir}/Q.npy")
    all_basins = list(json.load(open(f"{cache_dir}/meta.json"))["basins"])
    rows, skipped = [], 0
    for path in sorted(glob.glob(f"{runs_dir}/*/predictions.npz")):
        cfg = json.load(open(f"{path.rsplit('/', 1)[0]}/config.json"))
        if cfg.get("max_steps") or cfg.get("pred_stride", 1) != 1:
            print(f"skipping smoke run {path} (max_steps={cfg.get('max_steps')}, "
                  f"pred_stride={cfg.get('pred_stride')})")
            skipped += 1
            continue
        p = np.load(path, allow_pickle=True)
        basins, n_gauged = list(p["basins"]), int(p["n_gauged"])
        # Observations are keyed by gauge ID, never by position, so a fold reordering cannot
        # silently pair the wrong basin with the wrong prediction.
        obs_all = Q[[all_basins.index(b) for b in basins]]
        for period in ("train", "test"):
            frame = metrics(p[period], obs_all[:, p[f"{period}_t"]])
            frame["basin"], frame["arch"] = basins, cfg["arch"]
            frame["seed"], frame["fold"] = cfg["seed"], cfg["fold"]
            frame["cell"] = np.where(np.arange(len(basins)) < n_gauged,
                                     f"gauged/{period}", f"ungauged/{period}")
            rows.append(frame)
    if not rows:
        raise SystemExit(f"nothing to score under {runs_dir}/"
                         + (f" ({skipped} smoke run(s) skipped)" if skipped else ""))
    return pd.concat(rows, ignore_index=True).query("cell in @CELLS")


def summarise(df):
    """Median per-basin metric for each architecture, with the spread across seeds."""
    # fold is a grouping key so that k-fold runs never silently average across folds
    per_seed = df.groupby(["cell", "arch", "fold", "seed"])[METRICS].median()
    agg = per_seed.groupby(["cell", "arch"]).agg(["mean", "min", "max"])
    for cell in CELLS:
        if cell not in agg.index.get_level_values("cell"):
            continue
        cur = df.query("cell == @cell")
        n, scored = cur.basin.nunique(), int(cur.groupby("arch").nse.count().min())
        note = "" if scored == n else f", {scored} scored for the worst architecture"
        print(f"\n{cell}  ({n} basins{note})\n")
        print(f"  {'arch':14s} {'NSE':>24s} {'KGE':>8s} {'RMSE':>8s}")
        for arch, r in agg.loc[cell].iterrows():
            print(f"  {arch:14s} {r[('nse', 'mean')]:>9.3f} "
                  f"[{r[('nse', 'min')]:.3f}, {r[('nse', 'max')]:.3f}]"
                  f" {r[('kge', 'mean')]:>8.3f} {r[('rmse', 'mean')]:>8.3f}")
    print("\n  NSE shown as mean over seeds [min, max]; KGE and RMSE as mean over seeds.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="runs/metrics.csv")
    a = p.parse_args()
    df = load_runs(a.runs)
    df.to_csv(a.out, index=False)
    summarise(df)
    print(f"\nper-basin metrics -> {a.out}")


if __name__ == "__main__":
    main()
