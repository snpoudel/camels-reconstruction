"""The LSTM: data cache, fold split, model, training, prediction.

Everything the two driver scripts in this folder share. Nothing outside `src/models/lstm/`
imports this -- the corrections in `src/corrections/` read finished predictions off disk, so
they work just as well on hydrographs produced by somebody else's model.

Hyperparameters follow Kratzert et al. (2019, WRR 55, 11344) where the paper specifies a value
and NeuralHydrology defaults otherwise. Nothing here is tuned: the model has to stay the one
the literature already uses for the comparison to mean anything.
"""
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Kratzert et al. (2019): train WY1981-1995, test WY1996-2010.
TRAIN_PERIOD = ("1980-10-01", "1995-09-30")
TEST_PERIOD = ("1995-10-01", "2010-09-30")

# Static attributes with missing values (root_depth_50 in 13 basins, geol_porostiy in 3).
DROP_COLS = ("root_depth_50", "geol_porostiy")

LR_SCHEDULE = {0: 1e-3, 10: 5e-4, 20: 1e-4}  # NeuralHydrology default, keyed by epoch

N_FEAT = 36     # 6 daily forcings + 30 static attributes, after DROP_COLS
N_DAYS = 12784  # 1980-01-01 to 2014-12-31, no gaps


@dataclass(frozen=True)
class Cfg:
    """One training run. Only `fold` and `seed` are meant to vary."""

    fold: int | None = 0  # which of n_folds is held out as ungauged; None trains on every basin
    seed: int = 0

    # Kratzert et al. 2019
    hidden: int = 256
    layers: int = 1
    dropout: float = 0.4
    seq_len: int = 365  # 364 preceding days + the target day
    forget_bias: float = 5.0
    epochs: int = 30
    clip_norm: float = 1.0

    # Fixed-size batches of independently sampled (basin, date-window) pairs. Kratzert used
    # 2000, the NeuralHydrology default is 256; 512 sits between them.
    batch_size: int = 512

    n_folds: int = 7
    cache_dir: str = "data/cache"


# --------------------------------------------------------------------------- data

def build_cache(raw_dir="data/raw", list_csv="data/camels531.csv", cache_dir="data/cache",
                folds_csv="data/basin_list_with_folds.csv"):
    """Read the per-basin CSVs once into dense arrays and cache them as .npy."""
    basins = sorted(pd.read_csv(list_csv, dtype={"gauge_id": str})["gauge_id"])
    header = pd.read_csv(f"{raw_dir}/input_{basins[0]}.csv", nrows=0).columns.tolist()
    feat_cols = [c for c in header if c not in ("Year", "Mnth", "Day", "qobs(mm/day)", *DROP_COLS)]
    # Asserted by name, not position: if the target survived the drop, NSE would be ~0.99 and
    # the result would look like a triumph.
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

    assign_folds(len(basins), Cfg.n_folds, out_csv=folds_csv, basins=basins)
    os.makedirs(cache_dir, exist_ok=True)
    np.save(f"{cache_dir}/X.npy", X)
    np.save(f"{cache_dir}/Q.npy", Q)
    np.save(f"{cache_dir}/dates.npy", dates)
    with open(f"{cache_dir}/meta.json", "w") as f:
        json.dump({"basins": basins, "feat_cols": feat_cols, "drop_cols": list(DROP_COLS)}, f,
                  indent=1)
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

    tr0, tr1 = bounds(TRAIN_PERIOD)
    te0, te1 = bounds(TEST_PERIOD)
    # A target day needs `seq_len - 1` days of history behind it, and the cache begins on
    # 1980-01-01 while the training window opens at day 274. Move the first target forward to
    # the earliest day with full history rather than quietly running short windows -- and say
    # so, loudly, because it is a change to the training period.
    if tr0 < cfg.seq_len - 1:
        lost = cfg.seq_len - 1 - tr0
        tr0 = cfg.seq_len - 1
        print(f"  NOTE: training period starts {lost} days before seq_len={cfg.seq_len} has "
              f"enough history; first target moved to {str(dates[tr0])[:10]} "
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
    std = np.maximum(X[:n_gauged, tr0:tr1].std(axis=(0, 1)), 1e-3)  # some statics near-constant
    X = (X - mean) / std
    assert np.isfinite(X).all()

    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    return Data(X=t(X), Q=t(Q), sigma=t(sigma.astype(np.float32)), basins=basins,
                n_gauged=n_gauged, train_t=np.arange(tr0, tr1), test_t=np.arange(te0, te1))


def sample_batch(data: Data, rng, batch_size, seq_len):
    """One batch: independently sampled (basin, date-window) pairs, each row its own basin and
    its own end date. Only gauged basins are sampled -- held-out basins have no loss term."""
    basin_idx = rng.integers(0, data.n_gauged, batch_size)
    date_idx = rng.choice(data.train_t, batch_size)
    day_idx = date_idx[:, None] - seq_len + 1 + np.arange(seq_len)[None, :]

    dev = data.X.device
    b = torch.as_tensor(basin_idx, device=dev)
    days = torch.as_tensor(day_idx, device=dev)
    dates = torch.as_tensor(date_idx, device=dev)
    return data.X[b[:, None], days], data.Q[b, dates], data.sigma[b]


# --------------------------------------------------------------------------- model

class LSTMModel(nn.Module):
    """LSTM encoder over a `seq_len` window, one flow value per window."""

    def __init__(self, cfg: Cfg):
        super().__init__()
        h = cfg.hidden
        self.lstm = nn.LSTM(N_FEAT, h, cfg.layers, batch_first=True)
        # Forget-gate bias init. PyTorch gate order is (i, f, g, o) and both bias vectors are
        # summed, so the whole value goes in one and the other is zeroed.
        for layer in range(cfg.layers):
            getattr(self.lstm, f"bias_ih_l{layer}").data[h:2 * h] = cfg.forget_bias
            getattr(self.lstm, f"bias_hh_l{layer}").data[h:2 * h] = 0.0
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        """x: [batch, seq_len, N_FEAT] -> [batch] predicted flow in mm/day."""
        return self.head(self.dropout(self.lstm(x)[0][:, -1])).squeeze(-1)


# --------------------------------------------------------------------------- training

def nse_loss(pred, obs, sigma, eps=0.1):
    """Basin-normalised squared error (Kratzert et al. 2019 NSE*).

    Dividing by each basin's own training-period flow std stops wet, large basins from
    dominating the gradient. `sigma` is a training-period statistic of the query basin, used
    only to weight the loss -- it never touches the prediction itself.
    """
    m = ~torch.isnan(obs)
    return (((pred[m] - obs[m]) / (sigma[m] + eps)) ** 2).mean()


def steps_per_epoch(cfg: Cfg, data):
    return (data.n_gauged * len(data.train_t)) // cfg.batch_size


def train(cfg: Cfg, data):
    torch.manual_seed(cfg.seed)
    model = LSTMModel(cfg).to(data.X.device)
    opt = torch.optim.Adam(model.parameters(), lr=LR_SCHEDULE[0])
    rng = np.random.default_rng(cfg.seed)
    n_steps = steps_per_epoch(cfg, data)
    start = time.time()

    for epoch in range(cfg.epochs):
        if epoch in LR_SCHEDULE:
            for g in opt.param_groups:
                g["lr"] = LR_SCHEDULE[epoch]
        model.train()
        total = 0.0
        for _ in range(n_steps):
            x, obs, sigma = sample_batch(data, rng, cfg.batch_size, cfg.seq_len)
            loss = nse_loss(model(x), obs, sigma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1:02d}/{cfg.epochs} | loss {total / n_steps:.4f} "
              f"| lr {opt.param_groups[0]['lr']:.0e} | {(time.time() - start) / 60:.1f} min",
              flush=True)
    return model


@torch.no_grad()
def predict(cfg: Cfg, model, data, t_idx, chunk=256):
    """[n_basins, len(t_idx)] flow in mm/day. Every basin is queried, held out or not."""
    model.eval()
    n = len(data.basins)
    out = np.empty((n, len(t_idx)), np.float32)
    for k, t in enumerate(t_idx):
        t = int(t)
        window = data.X[:, t - cfg.seq_len + 1:t + 1]
        preds = [model(window[i:i + chunk]).float().cpu().numpy() for i in range(0, n, chunk)]
        out[:, k] = np.concatenate(preds)
    return out


def device():
    return torch.device("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")


def run_one(cfg: Cfg, out_dir):
    """Train one (fold, seed) and write predictions.npz, model.pt and config.json."""
    data = prepare(cfg, device())
    print(f"{out_dir} | train {TRAIN_PERIOD[0]}..{TRAIN_PERIOD[1]}, "
          f"test {TEST_PERIOD[0]}..{TEST_PERIOD[1]}", flush=True)
    print(f"  {data.n_gauged} gauged, {len(data.basins) - data.n_gauged} ungauged | "
          f"{len(data.train_t)} train dates, {len(data.test_t)} test dates | {data.X.device}",
          flush=True)

    model = train(cfg, data)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{out_dir}/model.pt")
    json.dump({**vars(cfg), "commit": os.popen("git rev-parse --short HEAD").read().strip()},
              open(f"{out_dir}/config.json", "w"), indent=1)
    # Only the test period: nothing downstream reads the training period, and it is 60% of the
    # inference cost. Basins stay in fold order here; `seed_mean` keys them back by gauge ID.
    np.savez_compressed(f"{out_dir}/predictions.npz",
                        test=predict(cfg, model, data, data.test_t), test_t=data.test_t,
                        basins=np.array(data.basins), n_gauged=data.n_gauged)
    print(f"done -> {out_dir}", flush=True)


def seed_mean(run_dirs, out_npz, cache_dir="data/cache"):
    """Average the seeds' test hydrographs, in cache-basin order, and write one .npz.

    Averaging the hydrographs and scoring once is the convention the literature reports
    (Kratzert's 0.693 is a 10-member ensemble mean); averaging per-seed metrics instead reads
    lower and is not comparable. Rows are keyed by gauge ID, never by position, so a fold
    reordering cannot silently pair the wrong basin with the wrong prediction.
    """
    basins = list(json.load(open(f"{cache_dir}/meta.json"))["basins"])
    idx = {b: i for i, b in enumerate(basins)}
    mats, test_t = [], None
    for d in run_dirs:
        p = np.load(f"{d}/predictions.npz", allow_pickle=True)
        m = np.full((len(basins), p["test"].shape[1]), np.nan, np.float32)
        m[[idx[b] for b in list(p["basins"])]] = p["test"]
        mats.append(m)
        test_t = p["test_t"]
    # The training-period flow std this model's loss normalised by, published alongside the
    # predictions so the post-processing can select on the same objective without importing
    # anything from here -- and so a model trained elsewhere can state its own. It must be the
    # sigma `prepare` actually built, warm-up trim included, or it is a different constant than
    # the one the loss divided by.
    seq_len = json.load(open(f"{run_dirs[0]}/config.json"))["seq_len"]
    Q = np.load(f"{cache_dir}/Q.npy")
    dates = np.load(f"{cache_dir}/dates.npy")
    lo, hi = (np.datetime64(d) for d in TRAIN_PERIOD)
    idx = np.flatnonzero((dates >= lo) & (dates <= hi))
    tr0, tr1 = max(idx[0], seq_len - 1), idx[-1] + 1
    sigma = np.nanstd(Q[:, tr0:tr1], axis=1)

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(out_npz, pred=np.mean(mats, axis=0), test_t=test_t,
                        basins=np.array(basins), n_seeds=len(mats), sigma=sigma)
    print(f"-> {out_npz}   {len(mats)} seeds averaged")
