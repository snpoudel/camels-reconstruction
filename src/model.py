"""LSTM encoder with optional cross-basin (spatial) attention."""
import math

import torch
import torch.nn as nn

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
        # `donor_flow` concatenates the attended donor anomaly to the query state, so the
        # head is that much wider. Two layers either way: recovering flow from a log anomaly
        # is ~exp(mu + sigma*z), a product a linear head cannot express.
        head_in = h + cfg.donor_hist if cfg.flow_value else h
        self.head = nn.Sequential(
            nn.Linear(head_in, h), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(h, 1)
        )

        if cfg.spatial:
            # Low-rank Q/K give the model a similarity subspace separate from the space it
            # predicts flow in. Without them, attention can only compare basins in the exact
            # representation used for prediction -- the collapse in the previous version.
            self.wq = nn.Linear(h, cfg.attn_rank, bias=False)
            self.wk = nn.Linear(h, cfg.attn_rank, bias=False)
            self.log_temp = nn.Parameter(torch.zeros(()))  # learnable softmax temperature
            if not cfg.flow_value:
                self.norm = nn.LayerNorm(h)
        if cfg.use_donor:
            self.donor_mlp = nn.Sequential(
                nn.Linear(2 * cfg.donor_hist, 32), nn.ReLU(), nn.Linear(32, h)
            )

    def forward(self, x, n_query, n_donor=0, donor_z=None, donor_valid=None, want_attn=False):
        """x: [n_enc, seq_len, N_FEAT]. Returns predictions in mm/day for the first n_query.

        Nothing here touches the query basin's own observed flow or its flow statistics --
        that is the leak that would make ungauged prediction trivial.
        """
        h = self.lstm(x)[0][:, -1]  # [n_enc, hidden]
        ctx = h[:n_query]
        attn = None

        if self.cfg.spatial:
            logits = self.wq(ctx) @ self.wk(h[:n_donor]).T
            logits = logits / (math.sqrt(self.cfg.attn_rank) * self.log_temp.exp())
            idx = torch.arange(max(n_query, n_donor), device=x.device)
            mask = idx[:n_query, None] == idx[None, :n_donor]  # never attend to yourself
            if donor_valid is not None:
                mask = mask | ~donor_valid[None, :]
            logits = logits.masked_fill(mask, float("-inf"))
            # A fully masked row softmaxes to NaN; nan_to_num gives it zero attention, so it
            # contributes only the LayerNorm bias (or, under flow_value, a zero context --
            # see below). Near-unreachable: it needs every donor missing for the whole window.
            attn = torch.nan_to_num(logits.softmax(-1))
            if self.cfg.flow_value:
                # Load-bearing, not defensive: with no indicator bits, an unmasked donor that
                # did not report contributes z=0 -- "exactly normal" -- instead of dropping
                # out. The `.all` rule in donor_features is only enforced through this mask.
                assert donor_valid is not None, "flow_value requires donor_valid"
                # The value is the donors' own recent anomaly, so c is the attention-weighted
                # regional anomaly in z-space, handed to the head beside h_q rather than
                # summed into it. Three things are deliberate and load-bearing:
                #   * dropout on h_q only, never on c -- at p=0.4 it would zero ~3 of the 7
                #     donor-days, and 0 is a real value here ("exactly normal"), so it would
                #     feed a plausible lie rather than blur c. The head keeps its own dropout.
                #   * no LayerNorm on c -- normalising a 7-vector subtracts the mean across
                #     the seven days, and that mean is the signal.
                #   * no residual add, so the two pathways arrive at the head separately --
                #     head[0] mixes them at once, so this buys a clean input, not a clean
                #     attribution. Crediting either pathway still needs an ablation.
                # A fully masked row contributes c = 0, i.e. "the region is exactly normal".
                # It needs every donor missing for the whole window, which has never occurred
                # (the pool bottoms out at 414 of 455), but there is no learned fallback here
                # the way the LayerNorm bias was one for `donor`.
                ctx = torch.cat([self.dropout(ctx), attn @ donor_z], dim=-1)
            else:
                value = h[:n_donor]
                if donor_z is not None:
                    value = value + self.donor_mlp(donor_z)
                ctx = self.dropout(ctx + self.norm(attn @ value))
            if not want_attn:
                attn = None
        else:
            ctx = self.dropout(ctx)

        return self.head(ctx).squeeze(-1), attn
