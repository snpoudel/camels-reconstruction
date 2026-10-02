"""The learned neighbour-residual correction: a small MLP reads the K nearest gauges' errors
today, the season, and the target's own simulated flow, and predicts today's correction.

    delta_i(t) = f( r_1(t)..r_K(t), missing_1..K(t), log d_1..K, sin/cos doy(t), log q_sim_i(t) )
    q_i(t)     = (q_sim_i(t) + eps) * exp(delta_i(t)) - eps

r_j is donor j's log residual, exactly as in IDW. IDW is the special case delta = -lam *
sum_j w_j r_j, with weights fixed by distance alone; here the rule can change with the season
and the flow. Trained on 1 - KGE, the objective IDW's settings are selected on. K is the only
setting searched; everything else below is fixed in advance and never tuned.
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from shared import EPS, kge_per_basin

HIDDEN = 64
LR = 1e-3
CLIP_NORM = 1.0
BATCH_BASINS = 16
MAX_EPOCHS = 300
PATIENCE = 30
# Caps the daily correction at a factor of e^5 either way, so exp() cannot overflow while the
# network is still untrained. IDW's own corrections stay far inside this.
DELTA_MAX = 5.0


def splits(labels, dist, fold, k):
    """{train, val, held}: (targets, donors, log distance) for one fold, donors nearest first.

    Mirrors IDW's leave-one-group-out search: a basin's donors are pool basins outside its own
    group, so a pool basin trains under the thinned network a held-out basin faces, and a
    held-out basin (its group is the fold) draws on the whole pool. The pool group
    (fold + 1) % n_folds is kept back for validation.
    """
    n_folds = len(set(labels))
    pool = labels != fold
    val_group = (fold + 1) % n_folds
    rows = {"train": np.flatnonzero(pool & (labels != val_group)),
            "val": np.flatnonzero(labels == val_group),
            "held": np.flatnonzero(~pool)}
    out = {}
    for name, targets in rows.items():
        allowed = pool[None, :] & (labels[None, :] != labels[targets][:, None])
        d = np.where(allowed, dist[targets], np.inf)
        donors = np.argsort(d, axis=1)[:, :k]
        d = np.take_along_axis(d, donors, 1)
        assert np.isfinite(d).all(), f"fewer than {k} allowed donors"
        out[name] = (torch.as_tensor(targets), torch.as_tensor(donors),
                     torch.as_tensor(np.log(d / 100.0), dtype=torch.float32))
    return out, val_group


def context(sim, obs, labels, fold, doy):
    """The per-fold tensors every K shares: donor log residuals for pool basins only (no
    held-out observation is read), every basin's standardised log simulated flow, the season."""
    pool = labels != fold
    res = np.full(sim.shape, np.nan, np.float32)
    res[pool] = np.log(np.clip(sim[pool], 0, None) + EPS) - np.log(obs[pool] + EPS)
    logq = np.log(np.clip(sim, 0, None) + EPS)
    logq = (logq - logq[pool].mean()) / logq[pool].std()
    angle = 2 * np.pi * doy / 365.25
    season = np.stack([np.sin(angle), np.cos(angle)])
    return tuple(torch.as_tensor(a, dtype=torch.float32) for a in (res, logq, season))


def features(ctx, targets, donors, logd):
    """[B, T, 3K+3] inputs for a batch of targets, gathered on the fly."""
    res, logq, season = ctx
    r = res[donors]                                   # [B, K, T]
    missing = torch.isnan(r)
    B, K, T = r.shape
    x = torch.cat([torch.nan_to_num(r), missing.float(), logd[:, :, None].expand(B, K, T),
                   logq[targets][:, None], season[None].expand(B, 2, T)], 1)
    return x.transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3 * k + 3, HIDDEN), nn.SiLU(),
                                 nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, 1))
        # Zero output layer: training starts from no correction at all, IDW's lambda = 0.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, sim):
        """[B, T] corrected flow, mm/day."""
        delta = DELTA_MAX * torch.tanh(self.net(x).squeeze(-1))
        return torch.clamp((torch.clamp(sim, min=0) + EPS) * torch.exp(delta) - EPS, min=0)


def kge(sim, obs):
    """[B] KGE per basin over its reported days -- shared.kge_per_basin, in torch."""
    m = ~torch.isnan(obs)
    n = m.sum(1)
    o, s = torch.where(m, obs, 0.0), torch.where(m, sim, 0.0)
    do, ds = (o - (o.sum(1) / n)[:, None]) * m, (s - (s.sum(1) / n)[:, None]) * m
    var_o, var_s, cov = (do ** 2).sum(1), (ds ** 2).sum(1), (do * ds).sum(1)
    r = cov / torch.sqrt(var_o * var_s)
    alpha, beta = torch.sqrt(var_s / var_o), s.sum(1) / o.sum(1)
    return 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


@torch.no_grad()
def correct(model, sim, ctx, split, chunk=64):
    """[n_target, T] corrected flow for one split, as numpy."""
    model.eval()
    targets, donors, logd = split
    return torch.cat([model(features(ctx, targets[i:i + chunk], donors[i:i + chunk],
                                     logd[i:i + chunk]), sim[targets[i:i + chunk]])
                      for i in range(0, len(targets), chunk)]).numpy()


def train(seed, sim, obs, ctx, train_split, val_split):
    """One seed: (model at its best validation epoch, that epoch, its validation median KGE)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MLP(train_split[1].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    obs_t = torch.as_tensor(obs)
    n = len(train_split[0])
    best, best_epoch, best_state = -np.inf, 0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for b in np.array_split(rng.permutation(n), -(-n // BATCH_BASINS)):
            targets, donors, logd = (a[torch.as_tensor(b)] for a in train_split)
            loss = (1 - kge(model(features(ctx, targets, donors, logd), sim[targets]),
                            obs_t[targets])).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()
        score = np.nanmedian(kge_per_basin(correct(model, sim, ctx, val_split),
                                           obs[val_split[0].numpy()]))
        if score > best:
            best, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, best_epoch, best
