"""Experiment configuration — one LSTM, no variants.

Hyperparameters follow Kratzert et al. (2019, WRR 55, 11344), the standard PUB reference,
where the paper specifies a value and NeuralHydrology code defaults otherwise. Nothing here
is tuned: the point of this branch is that the *model* is the literature-standard one and the
gain comes from the post-processing step in `src/correct.py`.
"""
from dataclasses import dataclass

# Water-year periods, selected by `Cfg.period`.
#
#   "complete"  WY1990-2004 train / WY2005-2014 test. The training window is 100.000%
#               qobs-complete across all 531 basins -- no target masking, no basin
#               exclusions -- which is why this project picked it originally.
#
#   "kratzert"  WY1981-1995 train / WY1996-2010 test, matching Kratzert et al. (2019, WRR)
#               exactly, so the 0.693 benchmark becomes a much closer comparison. Two costs:
#               the training window is 96.7% qobs-complete (55 basins have gaps, worst 40%),
#               which `nse_loss` masks but which the "complete" window avoids entirely; and
#               it starts at day index 274, leaving only 5 days of spare warm-up for a
#               270-day sequence. The test window is the better of the two: 99.99% complete
#               over 15 years, against 98.7% over 10.
PERIODS = {
    "complete": (("1989-10-01", "2004-09-30"), ("2004-10-01", "2014-09-30")),
    "kratzert": (("1980-10-01", "1995-09-30"), ("1995-10-01", "2010-09-30")),
}

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

    # Which of PERIODS to train and test on. Recorded in every run's config.json, and runs
    # from different periods must never be pooled -- their test axes are different lengths.
    period: str = "complete"

    n_folds: int = 7
    cache_dir: str = "data/cache"
    out_dir: str = "runs"

    @property
    def train_period(self):
        return PERIODS[self.period][0]

    @property
    def test_period(self):
        return PERIODS[self.period][1]

    @property
    def name(self) -> str:
        return f"vanilla_fold{self.fold}_seed{self.seed}"
