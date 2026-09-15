"""The literature-standard LSTM. One layer, statics concatenated at every time step."""
import torch.nn as nn

from .config import N_FEAT, Cfg


class LSTMModel(nn.Module):
    """LSTM encoder over a `seq_len` window, one flow value per window.

    Deliberately plain: this branch's contribution is `src/correct.py`, and the model has to
    stay the one the literature already uses for the comparison to mean anything.
    """

    def __init__(self, cfg: Cfg):
        super().__init__()
        h = cfg.hidden
        self.lstm = nn.LSTM(N_FEAT, h, cfg.layers, batch_first=True)
        # Forget-gate bias init. PyTorch gate order is (i, f, g, o) and both bias vectors are
        # summed, so the whole value goes in one and the other is zeroed. Applied to every
        # layer, not just the first -- `cfg.layers` is configurable and a second layer left at
        # the PyTorch default would be a silent inconsistency.
        for layer in range(cfg.layers):
            getattr(self.lstm, f"bias_ih_l{layer}").data[h : 2 * h] = cfg.forget_bias
            getattr(self.lstm, f"bias_hh_l{layer}").data[h : 2 * h] = 0.0
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        """x: [batch, seq_len, N_FEAT] -> [batch] predicted flow in mm/day."""
        return self.head(self.dropout(self.lstm(x)[0][:, -1])).squeeze(-1)
