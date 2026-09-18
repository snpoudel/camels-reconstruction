"""Data cache, fold assignment, and all leak-sensitive normalisation.

Every statistic that could leak information about held-out basins or the test period is
computed here and nowhere else: the feature scaler and the per-basin flow sigma used by the
loss, both fit on gauged basins over the training period only.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .config import DROP_COLS, N_DAYS, N_FEAT, Cfg


def build_cache(raw_dir="data/raw", list_csv="data/camels531.csv", cache_dir="data/cache"):
    """Read the per-basin CSVs once into dense arrays and cache them as .npy."""
    basins = sorted(pd.read_csv(list_csv, dtype={"gauge_id": str})["gauge_id"])
    header = pd.read_csv(f"{raw_dir}/input_{basins[0]}.csv", nrows=0).columns.tolist()
    feat_cols = [c for c in header if c not in ("Year", "Mnth", "Day", "qobs(mm/day)", *DROP_COLS)]
    # Asserted by name, not position: if the target survived the drop, NSE would be ~0.99
    # and the result would look like a triumph.
    assert "qobs(mm/day)" not in feat_cols, "target leaked into the features"
    assert len(feat_cols) == N_FEAT, f"expected {N_FEAT} features, got {len(feat_cols)}"

    X = np.empty((len(basins), N_DAYS, N_FEAT), np.float32)
    Q = np.empty((len(basins), N_DAYS), np.float32)
    dates = None
    for i, b in enumerate(basins):
        df = pd.read_csv(f"{raw_dir}/input_{b}.csv")
        assert len(df) == N_DAYS, f"{b} has {len(df)} rows"
        d = pd.to_datetime(dict(year=df.Year, month=df.Mnth, day=df.Day)).values
        if dates is None:
            dates = d
        else:
            assert np.array_equal(d, dates), f"{b} has a different date axis"
        X[i] = df[feat_cols].to_numpy(np.float32)
        Q[i] = df["qobs(mm/day)"].to_numpy(np.float32)
    assert np.isfinite(X).all(), "NaN in features after dropping DROP_COLS"

    assign_folds(len(basins), Cfg.n_folds, out_csv="data/basin_list_with_folds.csv", basins=basins)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(f"{cache_dir}/X.npy", X)
    np.save(f"{cache_dir}/Q.npy", Q)
    np.save(f"{cache_dir}/dates.npy", dates)
    with open(f"{cache_dir}/meta.json", "w") as f:
        json.dump({"basins": basins, "feat_cols": feat_cols, "drop_cols": list(DROP_COLS)}, f, indent=1)
    print(f"cached {X.shape} features, {np.isnan(Q).mean():.4%} of flow values missing")


def assign_folds(n_basins, n_folds, seed=42, out_csv=None, basins=None):
    """Balanced random fold labels. Deterministic in `seed`, independent of the run seed."""
    perm = np.random.default_rng(seed).permutation(n_basins)
    folds = np.empty(n_basins, int)
    folds[perm] = np.arange(n_basins) % n_folds
    if out_csv:
        pd.DataFrame({"basin": basins, "fold": folds}).to_csv(out_csv, index=False)
    return folds


@dataclass
class Data:
    """Tensors on the compute device, basins reordered as [gauged..., ungauged...].

    Position therefore encodes fold membership by construction -- `basins` is the only mapping
    back to gauge IDs, and every output is keyed through it rather than by index.
    """

    X: torch.Tensor      # [N, T, 36] standardised features
    Q: torch.Tensor      # [N, T] observed flow, mm/day, NaN where missing
    sigma: torch.Tensor  # [N] train-period flow std, denominator of the NSE loss
    basins: list
    n_gauged: int
    train_t: np.ndarray  # target date indices
    test_t: np.ndarray


def prepare(cfg: Cfg, device) -> Data:
    """Load the cache, split by fold, and fit every statistic on gauged x train only."""
    X = np.load(f"{cfg.cache_dir}/X.npy")
    Q = np.load(f"{cfg.cache_dir}/Q.npy")
    dates = np.load(f"{cfg.cache_dir}/dates.npy")
    meta = json.load(open(f"{cfg.cache_dir}/meta.json"))
    basins, feat_cols = meta["basins"], meta["feat_cols"]
    # Re-checked here and not merely in build_cache: a stale cache of the right width would
    # otherwise pass silently on a machine that only ever received data/cache/.
    assert "qobs(mm/day)" not in feat_cols, "target leaked into the cached features"
    assert meta.get("drop_cols") == list(DROP_COLS), "cache built with different DROP_COLS"
    assert X.shape == (len(basins), N_DAYS, N_FEAT), f"unexpected cache shape {X.shape}"

    def bounds(period):
        lo, hi = (np.datetime64(d) for d in period)
        idx = np.flatnonzero((dates >= lo) & (dates <= hi))
        return idx[0], idx[-1] + 1

    tr0, tr1 = bounds(cfg.train_period)
    te0, te1 = bounds(cfg.test_period)
    # A target day needs `seq_len - 1` days of history behind it, and the cache begins on
    # 1980-01-01. The "kratzert" window opens at day 274, so anything past a 275-day sequence
    # cannot be trained from its first day. Move the first target forward to the earliest day
    # that has its full history rather than quietly running short windows -- and say so,
    # loudly, because it is a change to the training period.
    if tr0 < cfg.seq_len - 1:
        lost = cfg.seq_len - 1 - tr0
        tr0 = cfg.seq_len - 1
        print(f"  NOTE: period {cfg.period!r} starts {lost} days before seq_len={cfg.seq_len} "
              f"has enough history; first target moved to {str(dates[tr0])[:10]} "
              f"({lost} of {tr1 - (tr0 - lost)} training days dropped, "
              f"{100 * lost / (tr1 - (tr0 - lost)):.1f}%)", flush=True)
    assert tr1 - tr0 > 365, "training period too short after warm-up trimming"
    assert te0 >= cfg.seq_len - 1, "not enough warm-up before the test period"

    folds = assign_folds(len(basins), cfg.n_folds)
    if cfg.fold is None:
        gauged, ungauged = np.arange(len(basins)), np.array([], dtype=int)
    else:
        gauged = np.flatnonzero(folds != cfg.fold)
        ungauged = np.flatnonzero(folds == cfg.fold)
    order = np.concatenate([gauged, ungauged])
    X, Q = X[order], Q[order]
    basins = [basins[i] for i in order]
    n_gauged = len(gauged)

    sigma = np.nanstd(Q[:, tr0:tr1], axis=1)

    # Feature scaler: gauged basins, training period. Nothing else is in scope.
    mean = X[:n_gauged, tr0:tr1].mean(axis=(0, 1))
    std = np.maximum(X[:n_gauged, tr0:tr1].std(axis=(0, 1)), 1e-3)  # several statics are near-constant
    X = (X - mean) / std
    assert np.isfinite(X).all()

    t = lambda a: torch.as_tensor(a, device=device)
    return Data(X=t(X), Q=t(Q), sigma=t(sigma.astype(np.float32)), basins=basins,
                n_gauged=n_gauged, train_t=np.arange(tr0, tr1), test_t=np.arange(te0, te1))


def sample_batch(data: Data, rng, batch_size, seq_len):
    """One literature-standard batch: independently sampled (basin, date-window) pairs.

    Each row has its own basin *and* its own end date, gathered in one call. Only gauged
    basins are sampled -- held-out basins have no loss term.
    """
    basin_idx = rng.integers(0, data.n_gauged, batch_size)
    date_idx = rng.choice(data.train_t, batch_size)
    day_idx = date_idx[:, None] - seq_len + 1 + np.arange(seq_len)[None, :]

    dev = data.X.device
    b = torch.as_tensor(basin_idx, device=dev)
    days = torch.as_tensor(day_idx, device=dev)
    dates = torch.as_tensor(date_idx, device=dev)
    return data.X[b[:, None], days], data.Q[b, dates], data.sigma[b]
