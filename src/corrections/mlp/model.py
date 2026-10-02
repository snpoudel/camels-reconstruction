"""The learned neighbour-residual correction: a small MLP reads the K nearest gauges' errors
today, the season, and the target's own simulated flow, and predicts today's correction.

    delta_i(t) = f( r_1(t)..r_K(t), missing_1..K(t), log d_1..K, sin/cos doy(t), log q_sim_i(t) )
    q_i(t)     = (q_sim_i(t) + eps) * exp(delta_i(t)) - eps

r_j is donor j's log residual, exactly as in IDW. IDW is the special case delta = -lam *
sum_j w_j r_j, with weights fixed by distance alone; here the rule can change with the season
and the flow. Trained on 1 - KGE, the objective IDW's settings are selected on. K is the only
setting searched; everything else below is fixed in advance and never tuned.

Each fold (ungauged) or group (gauged) is handled like IDW's search: the network trains only on
the pool -- the basins outside the fold -- each pool basin drawing on every other pool basin as
a donor, itself excluded. One pool group is kept back to choose K. The fold's basins are then
corrected from the pool (ungauged), or from every other basin (gauged: one gauge dark at a time).
"""
import numpy as np
import torch
import torch.nn as nn

from shared import EPS, kge_per_basin

K_GRID = (1, 3, 5, 10, 15, 20, 30)
SEEDS = 5
HIDDEN = 64
DROPOUT = 0.2
LR = 1e-3
CLIP_NORM = 1.0
BATCH_BASINS = 16
# A fixed number of passes over the training basins, the learning rate decaying to zero along a
# cosine, rather than stopping at the validation group's best pass: validation KGE jumps by
# ~0.01 from one pass to the next, so its best pass is mostly noise. Chosen once on validation
# basins (20, 40 and 80 compared), then frozen.
EPOCHS = 40
# Caps the daily correction at a factor of e^5 either way, so exp() cannot overflow while the
# network is still untrained. IDW's own corrections stay far inside this.
DELTA_MAX = 5.0
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def splits(labels, dist, fold, k, cell):
    """{train, val, held}: (targets, donors, log distance) on DEVICE, donors nearest first."""
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
        out[name] = (torch.as_tensor(targets, device=DEVICE),
                     torch.as_tensor(donors, device=DEVICE),
                     torch.as_tensor(np.log(d / 100.0), dtype=torch.float32, device=DEVICE))
    return out, val_group


def context(sim, obs, readable, pool, doy):
    """The tensors every K shares: donor log residuals for `readable` rows only, every basin's
    log simulated flow standardised on the pool, and the season."""
    res = np.full(sim.shape, np.nan, np.float32)
    res[readable] = np.log(np.clip(sim[readable], 0, None) + EPS) - np.log(obs[readable] + EPS)
    logq = np.log(np.clip(sim, 0, None) + EPS)
    logq = (logq - logq[pool].mean()) / logq[pool].std()
    angle = 2 * np.pi * doy / 365.25
    season = np.stack([np.sin(angle), np.cos(angle)])
    return tuple(torch.as_tensor(a, dtype=torch.float32, device=DEVICE)
                 for a in (res, logq, season))


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
        self.net = nn.Sequential(nn.Linear(3 * k + 3, HIDDEN), nn.SiLU(), nn.Dropout(DROPOUT),
                                 nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Dropout(DROPOUT),
                                 nn.Linear(HIDDEN, 1))
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
                      for i in range(0, len(targets), chunk)]).cpu().numpy()


def train(seed, sim, obs, ctx, split, epochs=EPOCHS):
    """One seed, `epochs` passes over the training basins, learning rate cosine-decayed to 0."""
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
            targets, donors, logd = (a[torch.as_tensor(b, device=DEVICE)] for a in split)
            loss = (1 - kge(model(features(ctx, targets, donors, logd), sim[targets]),
                            obs[targets])).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()
            schedule.step()
    return model


def run_fold(sim, obs, labels, dist, fold, cell, doy, k_grid, seeds, epochs=EPOCHS, log=print):
    """Train every K x seed for one fold, choose K on the validation group, and return the
    held-out basins' corrected flow (the seeds' average at that K) with the bookkeeping.

    The fold's observations are read only where `cell` allows: never in the ungauged cell, and
    in the gauged cell only as donor inputs for their neighbours once training is done.
    """
    pool = labels != fold
    ctx = context(sim, obs, pool if cell == "ungauged" else np.ones_like(pool), pool, doy)
    sim_t = torch.as_tensor(sim, device=DEVICE)
    obs_t = torch.as_tensor(np.where(pool[:, None], obs, np.nan), device=DEVICE)
    val_score, held_runs, tuning = {}, {}, []
    for k in k_grid:
        split, val_group = splits(labels, dist, fold, k, cell)
        val_runs, held_runs[k] = [], []
        for s in range(seeds):
            model = train(s, sim_t, obs_t, ctx, split["train"], epochs)
            val_runs.append(correct(model, sim_t, ctx, split["val"]))
            held_runs[k].append(correct(model, sim_t, ctx, split["held"]))
        val_obs = obs[split["val"][0].cpu().numpy()]
        val_score[k] = np.nanmedian(kge_per_basin(np.mean(val_runs, 0), val_obs))
        tuning.append(dict(label=fold, k=k, score=val_score[k], n_basins=len(val_obs)))
        log(f"  {cell} {fold} K={k:>2}: validation KGE {val_score[k]:.3f} ({seeds} seeds)")
    best = max(val_score.values())
    k = min(c for c, v in val_score.items() if v == best)
    for row in tuning:
        row["selected"] = row["k"] == k
    return np.mean(held_runs[k], 0), held_runs[k], k, val_group, val_score[k], tuning
