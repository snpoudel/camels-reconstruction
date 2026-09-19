# Reconstructing streamflow with a neighbour-residual correction

A post-processing step that improves daily streamflow predictions by asking the nearest gauges
how wrong the model was today. It trains nothing, costs milliseconds, and works on any model's
output — the LSTM in `src/lstm/` is just the one that produced the predictions here.

> Ask the nearest gauges how wrong the model was today. If they agree it ran low, assume this
> basin did too, and believe most of it.

Over the K nearest donor basins *j* that reported that day:

```
w_j = (1/d_j^p) / sum(1/d^p)                              d = great-circle km
R   = sum_j w_j * [ log(q_sim_j + eps) - log(q_obs_j + eps) ]
q_corrected = (q_sim + eps) * exp(-lambda * R) - eps
```

K, λ and p are cross-validated per fold, never on a basin they will go on to correct. The
target basin's own discharge is never read. `METHOD.md` has the derivation, the leak audit and
the results.

## Two cells

| | what is missing | donors |
|---|---|---|
| **ungauged** | the basin was never in training — out of sample in space and time | that fold's 455 training basins |
| **gauged** | the basin's gauge stops reporting for the test years — out of sample in time | all 530 other basins, self masked |

Ungauged, 531 CAMELS-US basins, 7 folds, 3 seeds: **median KGE 0.693 → 0.709**, median NSE
0.676 → 0.736. The published benchmark for this setting is median NSE 0.693 (Kratzert et al.
2019, WRR 55, 11344 — same 531 gauge IDs, same train/test windows, a 10-member ensemble).
The gauged cell is pending its all-basins model.

## Layout

| path | contents |
|---|---|
| `src/residual_postprocessing.py` | the correction itself, and where its inputs live on disk |
| `src/parameter_tuning.py` | the K × λ × p search, and the tie-break that picks from it |
| `src/01_lstm_gauged_correction.py` | correct the gauged cell → `runs/lstm/gauged_corrected/` |
| `src/02_lstm_ungauged_correction.py` | correct the ungauged cell → `runs/lstm/ungauged_corrected/` |
| `src/lstm/common.py` | the LSTM: cache, folds, model, training, prediction |
| `src/lstm/01_lstm_gauged.py` | train on every basin, one run per seed |
| `src/lstm/02_lstm_ungauged.py` | train one run per (fold, seed) |
| `figures/evaluate.py` | score the corrections into the CSVs the figures read |
| `figures/make_*.py` | one script per figure; `palette.py` holds the shared style |
| `data/` | `camels531.csv` and the fold assignment (tracked); `raw/`, `cache/` (gitignored) |
| `runs/<model>/` | predictions and corrections, per model family (gitignored) |

Everything is run as `python <path>.py` — no packages, no `-m`. The post-processing never
imports from `src/lstm/`: it reads finished hydrographs off disk, so another model's
predictions drop into `runs/<model>/` and the same two scripts correct them.

## Running it

```bash
# once: build the cache from data/raw/
python -c "import sys; sys.path.insert(0,'src/lstm'); import common; common.build_cache()"

# train (GPU; the cluster). Each writes runs/lstm/<cell>/..., then the seed-mean the
# correction reads. --fold/--seed run one at a time so they can be launched concurrently.
python src/lstm/02_lstm_ungauged.py
python src/lstm/01_lstm_gauged.py

# correct (CPU, minutes) -> runs/lstm/<cell>_corrected/
python src/02_lstm_ungauged_correction.py
python src/01_lstm_gauged_correction.py

# score and plot (CPU) -> figures/*.csv, figures/*.png
python figures/evaluate.py
python figures/make_cdf.py --cell ungauged
python figures/make_boxplot.py --cell ungauged
python figures/make_hyperparam.py --cell ungauged
python figures/make_donor_map.py --cell ungauged
python figures/make_method.py
```

Only training needs a GPU. Everything downstream runs on any machine: the correction is a few
minutes of grid search, and the correction itself is milliseconds.
