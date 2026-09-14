"""Experiment configuration.

Hyperparameters follow Kratzert et al. (2019, HESS 23, 5089) where the paper specifies a
value, and NeuralHydrology code defaults otherwise. Every value applies identically to all
architectures, so none can confound the architecture comparison.
"""
from dataclasses import dataclass

# Water-year periods. WY1990-2004 is 100.000% qobs-complete across all 531 basins, which is
# why it was chosen for training: no target masking, no donor gaps, no basin exclusions.
TRAIN_PERIOD = ("1989-10-01", "2004-09-30")  # WY1990-2004, 5479 days
TEST_PERIOD = ("2004-10-01", "2014-09-30")  # WY2005-2014, 3652 days

# Static attributes with missing values (root_depth_50 in 13 basins, geol_porostiy in 3).
# A single NaN donor would propagate through the shared softmax and NaN every query's logits.
DROP_COLS = ("root_depth_50", "geol_porostiy")

# NeuralHydrology default schedule, keyed by epoch.
LR_SCHEDULE = {0: 1e-3, 10: 5e-4, 20: 1e-4}

ARCHS = ("vanilla", "spatial", "donor")

N_FEAT = 36  # 6 daily forcings + 30 static attributes, after DROP_COLS
N_DAYS = 12784  # 1980-01-01 to 2014-12-31, no gaps


@dataclass(frozen=True)
class Cfg:
    """One training run. `arch` selects the architecture; everything else is fixed."""

    arch: str = "vanilla"
    fold: int = 0  # which of n_folds is held out as ungauged
    seed: int = 0

    # Kratzert et al. 2019
    hidden: int = 256
    layers: int = 1
    dropout: float = 0.4
    seq_len: int = 270  # 269 preceding days + the target day

    # NeuralHydrology defaults
    forget_bias: float = 3.0
    epochs: int = 30
    clip_norm: float = 1.0

    # Spatial attention
    attn_rank: int = 64  # low-rank Q/K subspace, separate from the prediction space
    donor_hist: int = 7  # days of donor flow anomaly
    q_eps_frac: float = 0.01  # log-flow floor = q_eps_frac * p_mean, basin-relative

    n_folds: int = 7
    cache_dir: str = "data/cache"
    out_dir: str = "runs"

    @property
    def spatial(self) -> bool:
        return self.arch in ("spatial", "donor")

    @property
    def use_donor(self) -> bool:
        return self.arch == "donor"

    @property
    def name(self) -> str:
        return f"{self.arch}_fold{self.fold}_seed{self.seed}"
