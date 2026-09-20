"""Independent replication check: is the spatial error structure a property of regional
training, or would it show up even in a model trained on one basin alone?

    python figures/make_error_structure_kratzert.py

Two panels, the same statistic as `make_error_structure.py` (see that file and
`evaluate.error_structure` for the full derivation): one dot per basin pair, correlation of daily
log-residuals against gauge distance, with a LOESS fit. Left panel: Kratzert et al.'s regional
LSTM (10-seed ensemble, trained on all 531 CAMELS-US basins). Right panel: their single-basin
LSTMs (10-seed ensemble per basin, trained on that basin alone, so it has no way to learn
cross-basin structure). Both panels use the same basin subset, so the comparison is
apples-to-apples.

This project doesn't train an LSTM to answer this -- that would mean 531 separate trainings for
a side question the paper isn't about (see CLAUDE.md). Kratzert et al. (2024, "Never train an
LSTM on a single basin", Zenodo 10.5281/zenodo.11247607) already ran exactly this comparison and
published the simulations, so this script reads their `test_results.p` directly. It touches
nothing else: no entry in figures/evaluate.py, no runs/kratzert, no figures/*.csv -- one script,
one figure, using only their pre-computed obs/sim.

Extraction of the two relevant run_dirs subtrees out of their ~5.4 GB archive:

    curl -L -o data/raw/kratzert/never-paper.tar.gz \\
        "https://zenodo.org/records/11247607/files/never-paper.tar.gz?download=1"
    tar -xzf data/raw/kratzert/never-paper.tar.gz -C data/raw/kratzert/extracted --wildcards \\
        'single-basin-vs-regional-model/run_dirs/regional_model/*/test/*/test_results.p' \\
        'single-basin-vs-regional-model/run_dirs/single_basins_multi_forcings/*/test/*/test_results.p'
"""
import glob
import os
import pickle
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from evaluate import DIST_MAX_KM, LOESS_KM, _corr_rows, _loess  # noqa: E402
from palette import INK, LABEL, MUTE, NEUTRAL, RAW, TITLE, caption, style  # noqa: E402
from residual_postprocessing import EPS, great_circle_km  # noqa: E402

KRATZERT_DIR = f"{ROOT}/data/raw/kratzert/extracted/single-basin-vs-regional-model/run_dirs"
REGIONAL_GLOB = f"{KRATZERT_DIR}/regional_model/*/test/*/test_results.p"
SINGLE_GLOB = f"{KRATZERT_DIR}/single_basins_multi_forcings/*/test/*/test_results.p"

Y_FLOOR = -0.4


def _patch_pandas_compat():
    """Kratzert's pickles were written under pandas < 2.0, which had a real Int64Index class;
    2.0 folded it into the plain Index and dropped the module, so unpickling needs it back as an
    alias or `pickle.load` fails on the class reference before any of our code runs."""
    if "pandas.core.indexes.numeric" in sys.modules:
        return
    shim = types.ModuleType("pandas.core.indexes.numeric")
    shim.Int64Index = shim.UInt64Index = shim.Float64Index = pd.Index
    sys.modules["pandas.core.indexes.numeric"] = shim


def _load_series(path):
    """basin_id -> (dates, obs, sim) for every basin this one test_results.p holds."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    out = {}
    for basin, freqs in d.items():
        ds = freqs["1D"]["xr"]
        out[basin] = (ds["date"].values,
                      ds["QObs(mm/d)_obs"].values[:, 0].astype(np.float64),
                      ds["QObs(mm/d)_sim"].values[:, 0].astype(np.float64))
    return out


def load_ensemble(glob_pattern):
    """basin_id -> (dates, obs, ensemble-mean sim), averaged over every seed file that matches.

    The ensemble mean of the hydrographs, not of per-seed scores -- the same convention this
    project's own `src/lstm/common.py:seed_mean` uses, for the same reason (it's what the
    literature reports).
    """
    paths = sorted(glob.glob(glob_pattern))
    if not paths:
        raise SystemExit(f"no files matched {glob_pattern} -- extract test_results.p first "
                         f"(see this script's module docstring)")
    dates, obs, sims = {}, {}, {}
    for path in paths:
        for basin, (d, o, s) in _load_series(path).items():
            if basin not in dates:
                dates[basin], obs[basin] = d, o
            elif not np.array_equal(dates[basin], d):
                raise SystemExit(f"{path}: date axis disagrees with an earlier seed for {basin}")
            sims.setdefault(basin, []).append(s)
    n_seeds = {len(v) for v in sims.values()}
    print(f"  {glob_pattern.split('/')[-5]}: {len(sims)} basins, "
          f"{n_seeds if len(n_seeds) > 1 else n_seeds.pop()} seeds")
    return dates, obs, {b: np.mean(v, axis=0) for b, v in sims.items()}


def stack(dates, obs, sim, basins, ref_dates):
    """[n_basins, n_days] obs and sim, `basins` order, every basin checked against one shared
    date axis -- Kratzert's temporal split is meant to be identical across every run, and a
    silent misalignment here would poison every pairwise correlation that touches it."""
    obs_m, sim_m = np.empty((len(basins), len(ref_dates))), np.empty((len(basins), len(ref_dates)))
    for i, b in enumerate(basins):
        if not np.array_equal(dates[b], ref_dates):
            raise SystemExit(f"{b}: date axis does not match the reference")
        obs_m[i], sim_m[i] = obs[b], sim[b]
    return obs_m, sim_m


def main():
    _patch_pandas_compat()
    print("loading Kratzert et al. ensembles (10 seeds each)...")
    reg_dates, reg_obs, reg_sim = load_ensemble(REGIONAL_GLOB)
    sb_dates, sb_obs, sb_sim = load_ensemble(SINGLE_GLOB)

    camels = pd.read_csv(f"{ROOT}/data/camels531.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    basins = sorted(set(reg_dates) & set(sb_dates) & set(camels.index))
    print(f"{len(basins)} basins common to Kratzert's regional ({len(reg_dates)}), single-basin "
          f"({len(sb_dates)}) and this project's camels531 ({len(camels)})")

    ref_dates = reg_dates[basins[0]]
    reg_obs_m, reg_sim_m = stack(reg_dates, reg_obs, reg_sim, basins, ref_dates)
    sb_obs_m, sb_sim_m = stack(sb_dates, sb_obs, sb_sim, basins, ref_dates)

    dist = great_circle_km(camels.loc[basins, "gauge_lat"].to_numpy(),
                           camels.loc[basins, "gauge_lon"].to_numpy())
    iu = np.triu_indices(len(basins), 1)
    keep = dist[iu] < DIST_MAX_KM
    d = dist[iu][keep]
    grid = np.linspace(10, DIST_MAX_KM, 160)

    panels = {}
    for name, obs_m, sim_m in (("regional", reg_obs_m, reg_sim_m),
                              ("single_basin", sb_obs_m, sb_sim_m)):
        res = np.log(np.clip(sim_m, 0, None) + EPS) - np.log(obs_m + EPS)
        c = _corr_rows(res)[iu][keep]
        panels[name] = (c, _loess(d, c, grid))

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    titles = {"regional": "Regional LSTM (Kratzert et al.)",
             "single_basin": "Single-basin LSTM (Kratzert et al.)"}
    for ax, key in zip(axes, ("regional", "single_basin")):
        c, fit = panels[key]
        ax.scatter(d, c, s=5, color=NEUTRAL, alpha=0.10, linewidths=0, zorder=2,
                   rasterized=True, label="one basin pair")
        ax.plot(grid, fit, color=RAW, lw=2.4, zorder=4, label="LOESS fit")
        ax.axhline(0, color=MUTE, lw=0.8, ls=(0, (4, 3)), zorder=3)
        ax.set_xlim(0, DIST_MAX_KM)
        ax.set_ylim(Y_FLOOR, 0.9)
        ax.set_xlabel("distance between gauge pair (km)", color=INK, fontsize=LABEL)
        ax.set_title(titles[key], color=INK, fontsize=TITLE, loc="left", pad=8)
        style(ax)
    axes[0].set_ylabel("correlation of daily residuals", color=INK, fontsize=LABEL)
    leg = axes[0].legend(frameon=False, loc="upper right", fontsize=LABEL - 1, labelcolor=INK,
                         handletextpad=0.6, borderpad=0.2)
    leg.legend_handles[0].set_alpha(0.8)
    leg.legend_handles[0].set_sizes([26])

    period = f"{str(ref_dates.min())[:4]}-{str(ref_dates.max())[:4]}"
    bottom = caption(fig, f"{len(basins)} CAMELS-US basins for which Kratzert et al. published "
                          f"both a regionally-trained and an independently single-basin-trained "
                          f"LSTM, 10-seed ensemble mean each, test period {period}. Same "
                          f"residual as Figure 1, log(sim + eps) - log(obs + eps), and the same "
                          f"{LOESS_KM} km LOESS bandwidth, {len(d):,} pairs within "
                          f"{DIST_MAX_KM} km. Simulations from Kratzert et al. (2024, Zenodo "
                          f"10.5281/zenodo.11247607); neither model is trained or corrected by "
                          f"this project -- an external check of whether the structure survives "
                          f"when a model has no way to learn cross-basin patterns.", 100)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    out = f"{HERE}/gauged_error_structure_Kratzert.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
