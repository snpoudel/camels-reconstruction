"""Turn the published HBV benchmark simulations into the runs/hbv/ layout the correction
pipeline reads.

    python src/hbv/prepare.py

Reads data/raw/camels_benchmarks/netcdf/netcdf/<gauge_id>_benchmark_models.nc, one NetCDF per
basin (the CAMELS benchmark models data set, Kratzert et al., collected for "Benchmarking a
Catchment-Aware LSTM for Large-Scale Hydrological Modeling", arXiv:1907.08456). Only `HBV_ub` is
read: the ensemble mean of 100 HBV models independently calibrated to that one basin's own record
(Seibert et al., 2018) -- SAC-SMA, VIC, mHM and FUSE live in the same files but aren't wired in.

Every one of the 531 basins has a file, so unlike the hybrid model there is no ungauged split
here: HBV never saw any other basin at calibration time, so every basin's "gauged" prediction is
exactly what HBV actually produced, out-of-sample only in time -- the same discontinued-gauge
scenario the LSTM's gauged cell corrects.

Dates run 1989-10-01 to 1999-09-30, the validation window the data set was released for (each
model calibrated on 1999-2008); mapped onto this project's own date cache so every downstream
script reads it the same way as every other model. Observed discharge is always looked up fresh
from that cache (`residual_postprocessing.observed`), never from the NetCDFs' own `QObs` -- the
two agree to within a small, constant scale factor (~2.4%, the kind of basin-area-normalisation
difference that shows up between independently processed CAMELS derivatives) but every model
compared in this project must be scored against the one shared observation record, or NSE/KGE
between models would stop being comparable.

Writes:

    runs/hbv/gauged/_mean/predictions.npz    -- pred for all 531 basins

Only one realisation exists (the published upper-bound ensemble mean): n_seeds=1, no sigma,
KGE-only selection -- the same secondary-check status as the hybrid model.
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from residual_postprocessing import BASINS_CSV, CACHE_DIR, basin_ids  # noqa: E402

NETCDF_DIR = "data/raw/camels_benchmarks/netcdf/netcdf"


def main():
    ids = basin_ids(BASINS_CSV)
    cache_dates = pd.to_datetime(np.load(f"{CACHE_DIR}/dates.npy", allow_pickle=True))

    ref_dates, test_t, pred = None, None, None
    for i, gid in enumerate(ids):
        ds = xr.open_dataset(f"{NETCDF_DIR}/{gid}_benchmark_models.nc")
        dates = pd.to_datetime(ds["index"].values)
        if ref_dates is None:
            ref_dates = dates
            test_t = np.flatnonzero(cache_dates.isin(ref_dates))
            assert len(test_t) == len(ref_dates), \
                "HBV's date window is not fully inside data/cache/dates.npy"
            assert np.array_equal(cache_dates[test_t].to_numpy(), ref_dates.to_numpy()), \
                "cache and HBV disagree on date order within the shared window"
            pred = np.empty((len(ids), len(test_t)), np.float32)
        elif not dates.equals(ref_dates):
            raise SystemExit(f"{gid}: date axis differs from the first basin's")
        pred[i] = ds["HBV_ub"].to_numpy()
        ds.close()
    assert np.isfinite(pred).all(), "gap in a basin's HBV_ub series"

    os.makedirs("runs/hbv/gauged/_mean", exist_ok=True)
    np.savez_compressed("runs/hbv/gauged/_mean/predictions.npz",
                        pred=pred, test_t=test_t, n_seeds=1)
    print(f"{len(ids)} basins, {len(test_t)} test days "
          f"({str(ref_dates[0])[:10]} to {str(ref_dates[-1])[:10]})")
    print("-> runs/hbv/gauged/_mean/predictions.npz")


if __name__ == "__main__":
    main()
