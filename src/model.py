"""LSTM encoder with optional cross-basin (spatial) attention."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import N_FEAT, Cfg


class PUBModel(nn.Module):
    """LSTM encoder plus optional attention over a shared pool of donor basins.

    Basins are ordered so that queries and donors are both prefixes of the encoded batch, so
    a query and a donor are the same basin exactly when their indices match. That keeps the
    self-mask to one comparison and makes it impossible to forget at evaluation time.
    """

    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.lstm = nn.LSTM(N_FEAT, h, cfg.layers, batch_first=True)
        # Forget-gate bias init. PyTorch gate order is (i, f, g, o) and both bias vectors are
        # summed, so the whole value goes in one and the other is zeroed.
        self.lstm.bias_ih_l0.data[h : 2 * h] = cfg.forget_bias
        self.lstm.bias_hh_l0.data[h : 2 * h] = 0.0

        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Sequential(
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(h, 1)
        )

        if cfg.spatial:
            # Low-rank Q/K give the model a similarity subspace separate from the space it
            # predicts flow in. Without them, attention can only compare basins in the exact
            # representation used for prediction -- the collapse in the previous version.
            self.wq = nn.Linear(h, cfg.attn_rank, bias=False)
            self.wk = nn.Linear(h, cfg.attn_rank, bias=False)
            self.log_temp = nn.Parameter(torch.zeros(()))  # learnable softmax temperature
            self.norm = nn.LayerNorm(h)
        if cfg.use_donor:
            self.donor_mlp = nn.Sequential(
                nn.Linear(2 * cfg.donor_hist, 32), nn.ReLU(), nn.Linear(32, h)
            )
        if cfg.gated:
            # Physical distance prior on attention: content-only attention never finds geography
            # on its own (analysis/donor_diagnostics.py -- `spatial`'s attention is spatially
            # indistinguishable from uniform), so it is given here instead of hoped for. Init at
            # 0 (softplus(0) = 0.69 per 100 km, a ~145 km length scale) is NOT inside the useful
            # band the diagnostics found (fold-optimal kernel scales run 20-50 km, R2 ~0.86; 145
            # km scores ~0.78, comparable to a crude 16-nearest-neighbour rule) -- it is a gentler
            # stepping stone training can sharpen from. An earlier, more cautious init (-4.6,
            # chosen to protect the global grad-clip norm) was measured to move by ~0.001 in 6
            # real steps with the gradient flipping sign, i.e. effectively unreachable in 30
            # epochs; measured over 60 real steps from 0, dist_w moves consistently (mean grad
            # negative, 68% of steps) and reaches the 50 km scale in under one epoch. Measured
            # directly at both inits: the distance term is <1% of the pre-clip gradient norm
            # throughout, so grad-clip was never actually at risk (`data.D` is still pre-scaled
            # to 100s of km, which is most of why) -- that risk was the reason for the original
            # cautious init, and it was the wrong thing to be cautious about.
            self.dist_w = nn.Parameter(torch.tensor(0.0))
            # Reliability gate on the donor injection, conditioned only on the query's own state
            # and its distance to the nearest AVAILABLE donor -- never on the attention output
            # itself (entropy, max weight), which would let the gate see and steer the very
            # attention it is meant to be judging. Donor reliability is predictable from exactly
            # this information (section_gate: rho ~ -0.55, holds on all 7 folds), so the gate has
            # something real to condition on. Final-layer bias init at +2 (sigmoid(2) ~ 0.88)
            # so the gate starts mostly OPEN -- donor_gated begins training close to `donor`'s
            # unconditional injection and only has to learn where to close, rather than starting
            # at a neutral 0.5 and having to learn both to open and to close from scratch.
            self.gate_mlp = nn.Sequential(nn.Linear(h + 1, 32), nn.ReLU(), nn.Linear(32, 1))
            nn.init.constant_(self.gate_mlp[-1].bias, 2.0)

    def forward(self, x, n_query, n_donor=0, donor_z=None, donor_valid=None, donor_dist=None,
                want_attn=False):
        """x: [n_enc, seq_len, N_FEAT]. Returns predictions in mm/day for the first n_query.

        Nothing here touches the query basin's own observed flow or its flow statistics --
        that is the leak that would make ungauged prediction trivial.
        """
        h = self.lstm(x)[0][:, -1]  # [n_enc, hidden]
        ctx = h[:n_query]
        attn = gate = None

        if self.cfg.spatial:
            logits = self.wq(ctx) @ self.wk(h[:n_donor]).T
            logits = logits / (math.sqrt(self.cfg.attn_rank) * self.log_temp.exp())
            idx = torch.arange(max(n_query, n_donor), device=x.device)
            mask = idx[:n_query, None] == idx[None, :n_donor]  # never attend to yourself
            if donor_valid is not None:
                mask = mask | ~donor_valid[None, :]
            if self.cfg.gated:
                # Added before masking, so a masked entry's bias can never leak through --
                # masked_fill below overwrites it with -inf regardless of the bias value.
                logits = logits - F.softplus(self.dist_w) * donor_dist
            logits = logits.masked_fill(mask, float("-inf"))
            # A fully masked row softmaxes to NaN; nan_to_num gives it zero attention, so it
            # contributes only the LayerNorm bias. Near-unreachable: it needs every donor
            # missing for the whole window.
            attn = torch.nan_to_num(logits.softmax(-1))
            value = h[:n_donor]
            if donor_z is not None:
                value = value + self.donor_mlp(donor_z)
            injection = self.norm(attn @ value)
            if self.cfg.gated:
                # Distance to the nearest donor this row could actually use -- masked/unavailable
                # donors pushed to "infinitely far" first, so the gate never sees a distance for
                # a donor it isn't allowed to attend to.
                d_nearest = (donor_dist + mask.float() * 1e6).min(dim=-1, keepdim=True).values
                g = torch.sigmoid(self.gate_mlp(torch.cat([ctx, torch.log1p(d_nearest)], dim=-1)))
                injection = g * injection
                if want_attn:
                    gate = g  # so a null result is diagnosable: did the gate ever open?
            ctx = ctx + injection
            if not want_attn:
                attn = None

        return self.head(self.dropout(ctx)).squeeze(-1), attn, gate
