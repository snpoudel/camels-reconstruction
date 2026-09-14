"""Data cache, fold assignment, and all leak-sensitive normalisation.

Every statistic that could leak information about held-out basins or the test period is
computed in this module and nowhere else: the feature scaler, the per-basin flow sigma used
by the loss, and the donor flow anomalies. All three are fit on gauged basins over the
training period only.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .config import DROP_COLS, N_DAYS, N_FEAT, TEST_PERIOD, TRAIN_PERIOD, Cfg


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
        # Attention pools across basins at a fixed date, so a shifted date axis in one file
        # would corrupt every basin's context invisibly.
        if dates is None:
            dates = d
        else:
            assert np.array_equal(d, dates), f"{b} has a different date axis"
        X[i] = df[feat_cols].to_numpy(np.float32)
        Q[i] = df["qobs(mm/day)"].to_numpy(np.float32)

    assert np.isfinite(X).all(), "NaN in features after dropping DROP_COLS"

    # Written once here rather than from prepare(): it is the versioned definition of the
    # experiment, and 12 concurrent training processes rewriting it can leave it torn.
    assign_folds(len(basins), Cfg.n_folds, out_csv="data/basin_list_with_folds.csv",
                 basins=basins)

    os.makedirs(cache_dir, exist_ok=True)
    np.save(f"{cache_dir}/X.npy", X)
    np.save(f"{cache_dir}/Q.npy", Q)
    np.save(f"{cache_dir}/dates.npy", dates)
    with open(f"{cache_dir}/meta.json", "w") as f:
        json.dump({"basins": basins, "feat_cols": feat_cols,
                   "drop_cols": list(DROP_COLS)}, f, indent=1)
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
    """Tensors resident on the GPU, with basins reordered as [gauged..., ungauged...].

    The reordering is what makes `X[:n_gauged, t0:t1]` a cheap contiguous slice instead of a
    gather over the full 977 MB tensor. Position therefore encodes fold membership *by
    construction* — `basins` is the only mapping back to gauge IDs, and every output is keyed
    through it rather than by index.
    """

    X: torch.Tensor  # [N, T, 36] standardised features
    Q: torch.Tensor  # [N, T] observed flow, mm/day, NaN where missing
    Z: torch.Tensor  # [N, T] standardised log-flow anomaly, 0 where missing
    Zobs: torch.Tensor  # [N, T] 1.0 where flow was observed
    sigma: torch.Tensor  # [N] train-period flow std, denominator of the NSE loss
    D: torch.Tensor  # [N, N] great-circle distance between basins, in 100s of km
    basins: list
    n_gauged: int
    train_t: np.ndarray  # target date indices
    test_t: np.ndarray


def prepare(cfg: Cfg, device, list_csv="data/camels531.csv") -> Data:
    """Load the cache, split by fold, and fit every statistic on gauged x train only."""
    X = np.load(f"{cfg.cache_dir}/X.npy")
    Q = np.load(f"{cfg.cache_dir}/Q.npy")
    dates = np.load(f"{cfg.cache_dir}/dates.npy")
    meta = json.load(open(f"{cfg.cache_dir}/meta.json"))
    basins, feat_cols = meta["basins"], meta["feat_cols"]
    # Only data/cache/ is copied to the GPU box, so these must be re-checked here and not
    # merely in build_cache -- a stale cache of the right width would otherwise pass silently.
    assert "qobs(mm/day)" not in feat_cols, "target leaked into the cached features"
    assert meta.get("drop_cols") == list(DROP_COLS), "cache built with different DROP_COLS"
    assert X.shape == (len(basins), N_DAYS, N_FEAT), f"unexpected cache shape {X.shape}"

    def bounds(period):
        lo, hi = (np.datetime64(d) for d in period)
        idx = np.flatnonzero((dates >= lo) & (dates <= hi))
        return idx[0], idx[-1] + 1

    tr0, tr1 = bounds(TRAIN_PERIOD)
    te0, te1 = bounds(TEST_PERIOD)
    assert tr0 >= cfg.seq_len - 1, "not enough warm-up before the training period"

    folds = assign_folds(len(basins), cfg.n_folds)
    gauged = np.flatnonzero(folds != cfg.fold)
    ungauged = np.flatnonzero(folds == cfg.fold)
    order = np.concatenate([gauged, ungauged])
    X, Q = X[order], Q[order]
    basins = [basins[i] for i in order]
    n_gauged = len(gauged)

    # Donor anomaly: dimensionless "how far above my own normal am I", so it transfers across
    # basins of different size and wetness. Computed from raw features, before scaling.
    p_mean = X[:, 0, feat_cols.index("p_mean")]  # static, constant in time
    logq = np.log(Q + (cfg.q_eps_frac * p_mean)[:, None])
    mu = np.nanmean(logq[:, tr0:tr1], axis=1, keepdims=True)
    sd = np.nanstd(logq[:, tr0:tr1], axis=1, keepdims=True)
    Z = np.nan_to_num((logq - mu) / np.maximum(sd, 1e-6))
    Zobs = np.isfinite(Q).astype(np.float32)

    sigma = np.nanstd(Q[:, tr0:tr1], axis=1)

    # Feature scaler: gauged basins, training period. Nothing else is in scope.
    mean = X[:n_gauged, tr0:tr1].mean(axis=(0, 1))
    std = np.maximum(X[:n_gauged, tr0:tr1].std(axis=(0, 1)), 1e-3)  # several statics are near-constant
    X = (X - mean) / std
    assert np.isfinite(X).all()

    # Great-circle distance between every pair of basins, in the reordered [gauged...,
    # ungauged...] layout, scaled to hundreds of km. `donor_gated` uses this as a physical prior
    # on attention (analysis/donor_diagnostics.py: content-only attention never finds geography
    # on its own). Scaled down from raw km (up to ~4500) so a single learned scalar multiplying
    # it stays in a numerically gentle range -- unscaled, its gradient would dominate the global
    # grad-clip norm and silently starve every other parameter's update.
    coords = pd.read_csv(list_csv, dtype={"gauge_id": str}).set_index("gauge_id")
    assert coords.index.is_unique, "camels531.csv has a duplicate gauge_id"
    assert set(coords.index) == set(basins), "camels531.csv basin set does not match the cache"
    # `basins` (not `order`) is the already-reordered list of gauge-ID strings -- `order` itself
    # is the integer position array used to build it, and coords is indexed by gauge ID.
    rad = np.radians(coords.loc[basins, ["gauge_lat", "gauge_lon"]].to_numpy())
    dlat = rad[None, :, 0] - rad[:, None, 0]
    dlon = rad[None, :, 1] - rad[:, None, 1]
    hav = np.sin(dlat / 2) ** 2 + np.cos(rad[:, None, 0]) * np.cos(rad[None, :, 0]) * np.sin(dlon / 2) ** 2
    D = (2 * 6371 * np.arcsin(np.sqrt(hav)) / 100).astype(np.float32)  # 100s of km

    t = lambda a: torch.as_tensor(a, device=device)
    return Data(
        X=t(X), Q=t(Q), Z=t(Z.astype(np.float32)), Zobs=t(Zobs), sigma=t(sigma.astype(np.float32)),
        D=t(D), basins=basins, n_gauged=n_gauged,
        train_t=np.arange(tr0, tr1), test_t=np.arange(te0, te1),
    )


def donor_features(data: Data, t: int, hist: int, n: int):
    """Donor flow anomaly and observation mask over the `hist` days ending at `t`.

    Returns ([n, 2*hist] features, [n] validity). A donor is masked out entirely only if it
    has no observation at all in the window; individual missing days carry z=0 plus an
    indicator bit so the model can tell "normal flow" from "no data".
    """
    z = data.Z[:n, t - hist + 1 : t + 1]
    obs = data.Zobs[:n, t - hist + 1 : t + 1]
    return torch.cat([z * obs, obs], dim=1), obs.any(dim=1)


def layout(cfg: Cfg, data: Data, training: bool):
    """(n_encode, n_query, n_donor) for one forward pass.

    Queries and donors are both prefixes of the [gauged..., ungauged...] ordering. During
    training only gauged basins are queried, because only they have a loss term.
    """
    n_all, n_gauged = len(data.basins), data.n_gauged
    n_query = n_gauged if training else n_all
    n_donor = 0 if not cfg.spatial else (n_all if cfg.donors_include_ungauged else n_gauged)
    return max(n_query, n_donor), n_query, n_donor


def step_inputs(cfg: Cfg, data: Data, t: int, n_enc: int, n_donor: int, n_query: int = None):
    """Encoder window and donor tensors for target date `t`."""
    assert n_query or not cfg.gated, "donor_gated needs n_query to slice the distance matrix"
    x = data.X[:n_enc, t - cfg.seq_len + 1 : t + 1]
    if not cfg.spatial:
        return x, None, None, None
    # The availability mask is applied for every spatial arch, not just `donor`, so the
    # donor pool is bit-identical across configs and the comparison stays single-variable.
    dz, valid = donor_features(data, t, cfg.donor_hist, n_donor)
    dist = data.D[:n_query, :n_donor] if (cfg.gated and n_query) else None
    return x, (dz if cfg.use_donor else None), valid, dist
