"""Experiment configuration — one LSTM, no variants.

Hyperparameters follow Kratzert et al. (2019, WRR 55, 11344), the standard PUB reference,
where the paper specifies a value and NeuralHydrology code defaults otherwise. Nothing here
is tuned: the point of this branch is that the *model* is the literature-standard one and the
gain comes from the post-processing step in `src/correct.py`.
"""
from dataclasses import dataclass

# Water-year periods. WY1990-2004 is 100.000% qobs-complete across all 531 basins, which is
# why it is the training window: no target masking, no basin exclusions.
TRAIN_PERIOD = ("1989-10-01", "2004-09-30")  # WY1990-2004, 5479 days
TEST_PERIOD = ("2004-10-01", "2014-09-30")   # WY2005-2014, 3652 days

# Static attributes with missing values (root_depth_50 in 13 basins, geol_porostiy in 3).
DROP_COLS = ("root_depth_50", "geol_porostiy")

LR_SCHEDULE = {0: 1e-3, 10: 5e-4, 20: 1e-4}  # NeuralHydrology default, keyed by epoch

N_FEAT = 36    # 6 daily forcings + 30 static attributes, after DROP_COLS
N_DAYS = 12784  # 1980-01-01 to 2014-12-31, no gaps


@dataclass(frozen=True)
class Cfg:
    """One training run. Only `fold` and `seed` are meant to vary."""

    fold: int = 0   # which of n_folds is held out as ungauged
    seed: int = 0

    # Kratzert et al. 2019
    hidden: int = 256
    layers: int = 1
    dropout: float = 0.4
    seq_len: int = 270      # 269 preceding days + the target day
    forget_bias: float = 5.0
    epochs: int = 30
    clip_norm: float = 1.0

    # Literature-standard training: fixed-size batches of independently sampled
    # (basin, date-window) pairs. Kratzert used 2000; 256 is the NeuralHydrology default and
    # what this project has used for its literature-faithful runs.
    batch_size: int = 256

    n_folds: int = 7
    cache_dir: str = "data/cache"
    out_dir: str = "runs"

    @property
    def name(self) -> str:
        return f"vanilla_fold{self.fold}_seed{self.seed}"
