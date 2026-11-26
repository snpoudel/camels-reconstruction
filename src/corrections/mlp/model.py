"""Learned neighbour-residual correction: an MLP predicts each day's correction for a target basin
from its K nearest donors.

    r_j(t)     = (q_sim_j(t) - q_obs_j(t)) / s_j                  s = flow_scale, SD of predicted flow
    delta_i(t) = f( r_1..r_K(t), missing_1..K(t), d_1..K, q_sim_i(t) / s_i, sin/cos doy(t) )
    q_i(t)     = max(0, q_sim_i(t) - s_i * delta_i(t))

IDW is the special case delta = lam * sum_j w_j r_j. The network is trained on NSE* using only the
basins outside the fold (ungauged) or group (gauged), one of whose groups is held back to choose K,
the only setting searched. `scaled=False` (--no-scale) sets every s to 1.
"""
import numpy as np
import torch
import torch.nn as nn

from shared import SCALE_FLOOR, flow_scale, nse_per_basin

K_GRID = (1, 3, 5, 10, 15, 20, 30, 39)
SEEDS = 5
HIDDEN = 64
DROPOUT = 0.2
LR = 1e-3
CLIP_NORM = 1.0
BATCH_BASINS = 16
EPOCHS = 40
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def splits(labels, dist, fold, k, cell):
    """({train, val, held}: (targets, donors nearest first, standardised distances), val_group)."""
    pool = labels != fold
    val_group = (fold + 1) % len(set(labels))
    rows = {"train": np.flatnonzero(pool & (labels != val_group)),
            "val": np.flatnonzero(labels == val_group),
            "held": np.flatnonzero(~pool)}
    out = {}
    for name, targets in rows.items():
        candidates = pool if name != "held" or cell == "ungauged" else np.ones_like(pool)
        allowed = np.broadcast_to(candidates, (len(targets), len(pool))).copy()
        allowed[np.arange(len(targets)), targets] = False   # never its own donor
        d = np.where(allowed, dist[targets], np.inf)
        donors = np.argsort(d, axis=1)[:, :k]
        d = np.take_along_axis(d, donors, 1)
        assert np.isfinite(d).all(), f"fewer than {k} allowed donors"
        out[name] = (targets, donors, d)
    mu, sd = out["train"][2].mean(), out["train"][2].std()
    return {name: (torch.as_tensor(t, device=DEVICE), torch.as_tensor(dn, device=DEVICE),
                   torch.as_tensor((d - mu) / sd, dtype=torch.float32, device=DEVICE))
            for name, (t, dn, d) in out.items()}, val_group


def context(sim, obs, readable, pool, doy, scaled=True):
    """Tensors shared by every K: residuals of `readable` rows, own scaled flow, season, and s."""
    s = flow_scale(sim, scaled)
    res = np.full(sim.shape, np.nan)
    res[readable] = (np.clip(sim[readable], 0, None) - obs[readable]) / s[readable, None]
    on_pool = res[pool & readable]
    res = (res - np.nanmean(on_pool)) / np.nanstd(on_pool)
    own = np.clip(sim, 0, None) / s[:, None]
    own = (own - own[pool].mean()) / own[pool].std()
    angle = 2 * np.pi * doy / 365.25
    season = np.stack([np.sin(angle), np.cos(angle)])
    return tuple(torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
                 for a in (res, own, season, s))


def features(ctx, targets, donors, dist):
    """[B, T, 3K+3] inputs for a batch of targets."""
    res, own, season, _ = ctx
    r = res[donors]
    missing = torch.isnan(r)
    B, K, T = r.shape
    x = torch.cat([torch.nan_to_num(r), missing.float(), dist[:, :, None].expand(B, K, T),
                   own[targets][:, None], season[None].expand(B, 2, T)], 1)
    return x.transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * k + 3, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
                                 nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT),
                                 nn.Linear(HIDDEN, 1))
        # Start from no correction, IDW's lambda = 0.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, sim, s):
        """[B, T] corrected flow, mm/day."""
        delta = self.net(x).squeeze(-1)
        return torch.clamp(torch.clamp(sim, min=0) - s[:, None] * delta, min=0)


def nse_star(sim, obs, sigma):
    """[B] NSE* loss per basin over its reported days."""
    m = ~torch.isnan(obs)
    err = torch.where(m, ((sim - torch.nan_to_num(obs)) / (sigma[:, None] + SCALE_FLOOR)) ** 2, 0.0)
    return err.sum(1) / m.sum(1)


@torch.no_grad()
def correct(model, sim, ctx, split, chunk=64):
    """[n_target, T] corrected flow for one split, as numpy."""
    model.eval()
    targets, donors, dist = split
    s = ctx[3]
    return torch.cat([model(features(ctx, targets[i:i + chunk], donors[i:i + chunk],
                                     dist[i:i + chunk]), sim[targets[i:i + chunk]],
                            s[targets[i:i + chunk]])
                      for i in range(0, len(targets), chunk)]).cpu().numpy()


def train(seed, sim, obs, sigma, ctx, split, epochs=EPOCHS):
    """One seed's model, the learning rate cosine-decayed to 0 over `epochs` passes."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP(split[1].shape[1]).to(DEVICE)
    n = len(split[0])
    n_batches = -(-n // BATCH_BASINS)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * n_batches)
    for _ in range(epochs):
        model.train()
        for b in np.array_split(rng.permutation(n), n_batches):
            targets, donors, dist = (a[torch.as_tensor(b, device=DEVICE)] for a in split)
            out = model(features(ctx, targets, donors, dist), sim[targets], ctx[3][targets])
            loss = nse_star(out, obs[targets], sigma[targets]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()
            schedule.step()
    return model


def run_fold(sim, obs, sigma, labels, dist, fold, cell, doy, k_grid, seeds, epochs=EPOCHS,
             scaled=True, log=print):
    """Train every K x seed for one fold, choose K on validation, return the held-out correction.

    The fold's observations are never read in the ungauged cell, and in the gauged cell only as
    donor inputs to their neighbours after training.
    """
    pool = labels != fold
    ctx = context(sim, obs, pool if cell == "ungauged" else np.ones_like(pool), pool, doy, scaled)
    sim_t = torch.as_tensor(sim, device=DEVICE)
    obs_t = torch.as_tensor(np.where(pool[:, None], obs, np.nan), device=DEVICE)
    sigma_t = torch.as_tensor(np.where(pool, sigma, np.nan), dtype=torch.float32, device=DEVICE)
    val_score, held_runs, tuning = {}, {}, []
    for k in k_grid:
        split, val_group = splits(labels, dist, fold, k, cell)
        val_runs, held_runs[k] = [], []
        for s in range(seeds):
            model = train(s, sim_t, obs_t, sigma_t, ctx, split["train"], epochs)
            val_runs.append(correct(model, sim_t, ctx, split["val"]))
            held_runs[k].append(correct(model, sim_t, ctx, split["held"]))
        v = split["val"][0].cpu().numpy()
        val_mean = np.mean(val_runs, 0)
        val_score[k] = float(np.nanmean(np.nanmean(
            ((val_mean - obs[v]) / (sigma[v, None] + SCALE_FLOOR)) ** 2, 1)))
        tuning.append(dict(label=fold, k=k, score=val_score[k], n_basins=len(v)))
        log(f"  {cell} {fold} K={k:>2}: validation NSE* loss {val_score[k]:.4f}, median NSE "
            f"{np.nanmedian(nse_per_basin(val_mean, obs[v])):.3f} ({seeds} seeds)", flush=True)
    best = min(val_score.values())
    k = min(c for c, v in val_score.items() if v == best)
    for row in tuning:
        row["selected"] = row["k"] == k
    return np.mean(held_runs[k], 0), held_runs[k], k, val_group, val_score[k], tuning
