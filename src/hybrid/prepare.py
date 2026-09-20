"""Turn the hybrid model's raw per-basin CSVs into the runs/hybrid/ layout the correction
pipeline reads.

    python src/hybrid/prepare.py

Reads hybrid/pred_input_<gauge_id>.csv (date, qobs, qsim_1) and hybrid/test_basins.csv (the
106 basins the hybrid model never trained on -- true out-of-sample). The other 425 are trained
on ("gauged"). Both groups cover all 531 basins in the same file layout, dates 1980-2014; only
1980-01-01 through 1989-12-31 is used here, per the hybrid run's own test window.

Writes:

    runs/hybrid/gauged/_mean/predictions.npz    -- pred for the 425 gauged basins, NaN for the
                                                    106 ungauged ones (01_lstm_gauged_correction.py
                                                    already drops all-NaN rows via nanmedian/
                                                    isfinite, so it needs no change)
    runs/hybrid/ungauged/_mean/fold7.npz        -- pred for all 531; fold 7 is a sentinel the
                                                    LSTM's 0-6 split never uses
    data/basin_list_with_folds_hybrid.csv       -- the real fold assignment with the 106 test
                                                    basins relabelled fold 7, so
                                                    02_lstm_ungauged_correction.py's existing
                                                    labels==f / labels!=f logic recovers exactly
                                                    this gauged/ungauged split while the search
                                                    still leave-one-group-outs the real 0-6
                                                    labels among the 425 donors

Only one realisation exists (qsim_1): n_seeds=1, no sigma, KGE-only selection -- this is a
secondary check on the LSTM result, not a replica of its multi-seed, multi-fold design.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from residual_postprocessing import BASINS_CSV, CACHE_DIR, FOLDS_CSV, basin_ids  # noqa: E402

HYBRID_DIR = "hybrid"
TEST_START, TEST_END = "1980-01-01", "1989-12-31"
UNGAUGED_FOLD = 7


def load_qsim(gauge_id, dates):
    df = pd.read_csv(f"{HYBRID_DIR}/pred_input_{gauge_id}.csv", parse_dates=["date"])
    return df.set_index("date")["qsim_1"].reindex(dates).to_numpy(np.float32)


def main():
    ids = basin_ids(BASINS_CSV)
    cache_dates = pd.to_datetime(np.load(f"{CACHE_DIR}/dates.npy", allow_pickle=True))
    in_window = (cache_dates >= TEST_START) & (cache_dates <= TEST_END)
    test_t, test_dates = np.flatnonzero(in_window), cache_dates[in_window]

    ungauged = set(pd.read_csv(f"{HYBRID_DIR}/test_basins.csv", dtype={"name": str})
                   ["name"].str.zfill(8))
    assert ungauged <= set(ids), f"test_basins.csv has ids outside {BASINS_CSV}"

    pred = np.full((len(ids), len(test_t)), np.nan, np.float32)
    for i, gid in enumerate(ids):
        pred[i] = load_qsim(gid, test_dates)
    assert np.isfinite(pred).all(), "gap between a hybrid CSV and the cache's test window"

    gauged_mask = np.array([gid not in ungauged for gid in ids])
    gauged_pred = np.where(gauged_mask[:, None], pred, np.nan)

    os.makedirs("runs/hybrid/gauged/_mean", exist_ok=True)
    np.savez_compressed("runs/hybrid/gauged/_mean/predictions.npz",
                        pred=gauged_pred, test_t=test_t, n_seeds=1)

    os.makedirs("runs/hybrid/ungauged/_mean", exist_ok=True)
    np.savez_compressed(f"runs/hybrid/ungauged/_mean/fold{UNGAUGED_FOLD}.npz",
                        pred=pred, test_t=test_t, n_seeds=1)

    folds = pd.read_csv(FOLDS_CSV, dtype={"basin": str})
    assert list(folds.basin) == ids, f"{FOLDS_CSV} is not in sorted gauge-ID order"
    folds.loc[folds.basin.isin(ungauged), "fold"] = UNGAUGED_FOLD
    hybrid_folds_csv = "data/basin_list_with_folds_hybrid.csv"
    folds.to_csv(hybrid_folds_csv, index=False)

    print(f"{gauged_mask.sum()} gauged, {len(ungauged)} ungauged basins, "
          f"{len(test_t)} test days ({TEST_START} to {TEST_END})")
    print("-> runs/hybrid/gauged/_mean/predictions.npz")
    print(f"-> runs/hybrid/ungauged/_mean/fold{UNGAUGED_FOLD}.npz")
    print(f"-> {hybrid_folds_csv}")


if __name__ == "__main__":
    main()
