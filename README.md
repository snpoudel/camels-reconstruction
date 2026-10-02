# Reconstructing streamflow with a neighbour-residual correction

A post-processing step that improves daily streamflow predictions by asking the nearest gauges
how wrong the model was today. It trains nothing, costs milliseconds, and works on any model's
output — the LSTM in `src/models/lstm/` is just the one that produced the predictions here.

> Ask the nearest gauges how wrong the model was today. If they agree it ran low, assume this
> basin did too, and believe most of it.

Over the K nearest donor basins *j* that reported that day:

```
w_j = (1/d_j^p) / sum(1/d^p)                              d = great-circle km
R   = sum_j w_j * [ log(q_sim_j + eps) - log(q_obs_j + eps) ]
q_corrected = (q_sim + eps) * exp(-lambda * R) - eps
```

K, λ and p are cross-validated per fold, never on a basin they will go on to correct. The
target basin's own discharge is never read. `manuscript.md` is the paper: method, evaluation and
results.

## Two cells

| | what is missing | donors |
|---|---|---|
| **ungauged** | the basin was never in training — out of sample in space and time | that fold's 455 training basins |
| **gauged** | the basin's gauge stops reporting for the test years — out of sample in time | all 530 other basins, self masked |

Ungauged, 531 CAMELS-US basins, 7 folds, 5 seeds: **median KGE 0.700 → 0.716**, median NSE
0.686 → 0.743. The published benchmark for this setting is median NSE 0.693 (Kratzert et al.
2019, WRR 55, 11344 — same 531 gauge IDs, same train/test windows, a 10-member ensemble).
Gauged, the same basins with 5 seeds: **median KGE 0.788 → 0.825**, median NSE 0.764 → 0.817.

Both cells also carry a parameter-free reference — the single nearest gauge's error applied in
full, K = 1 and λ = 1, nothing selected. It scores *below* the uncorrected model in both cells
(ungauged KGE 0.680, gauged 0.785), which is what separates "the neighbours know something"
from "the correction works".

K, λ and p are selected by cross-validation on three possible objectives -- KGE (default), NSE,
or NSE* (the basin-normalised loss the LSTM itself was trained on, `--select-on nse_star`),
measured against each other rather than assumed; see manuscript.md 2.6.

## Layout

| path | contents |
|---|---|
| `src/shared.py` | what every correction method shares: basins, folds, distances, observations, scores, save format |
| `src/corrections/idw/correction.py` | the correction itself |
| `src/corrections/idw/tuning.py` | the K x lambda x p search: three selection objectives (KGE, NSE, NSE*) |
| `src/corrections/idw/01_gauged.py` | correct the gauged cell → `runs/<model>/idw/gauged/` |
| `src/corrections/idw/02_ungauged.py` | correct the ungauged cell → `runs/<model>/idw/ungauged/` |
| `src/corrections/mlp/` | a learned alternative: an MLP predicts each day's correction from the K nearest donors' errors, the season and the target's own simulated flow (LSTM, ungauged cell; in progress) |
| `src/models/lstm/common.py` | the LSTM: cache, folds, model, training, prediction |
| `src/models/lstm/01_lstm_gauged.py` | train on every basin, one run per seed |
| `src/models/lstm/02_lstm_ungauged.py` | train one run per (fold, seed) |
| `src/models/{hbv,hybrid}/prepare.py` | drop another model's published predictions into `runs/<model>/` |
| `figures/evaluate.py` | score a correction method into the CSVs the figures read |
| `figures/make_*.py` | one script per figure; `palette.py` holds the shared style |
| `data/` | `camels531.csv`, fold assignment, US state outlines for the map figures (tracked); `raw/`, `cache/` (gitignored) |
| `runs/<model>/` | predictions (`<cell>/`) and their corrections (`<method>/<cell>/`), per model family (gitignored) |

Everything is run as `python <path>.py` — no packages, no `-m`. `src/models/` produces
predictions and `src/corrections/<method>/` fixes them. A correction never imports from
`src/models/`: it reads finished hydrographs off disk, so another model's predictions drop into
`runs/<model>/` and the same scripts correct them.

## Running it

```bash
# once: build the cache from data/raw/
python -c "import sys; sys.path.insert(0,'src/models/lstm'); import common; common.build_cache()"

# train (GPU) -- already done: runs/lstm/<cell>/ holds the final predictions, so this is a
# record of how they were made, not a step to repeat. Each writes runs/lstm/<cell>/..., then
# the seed-mean the correction reads. --fold/--seed run one at a time.
python src/models/lstm/02_lstm_ungauged.py
python src/models/lstm/01_lstm_gauged.py

# correct (CPU, minutes) -> runs/lstm/idw/<cell>/
python src/corrections/idw/02_ungauged.py
python src/corrections/idw/01_gauged.py

# score and plot (CPU) -> figures/*.csv, figures/*.png
python figures/evaluate.py              # --method idw by default
python figures/make_cdf.py              # gauged + ungauged, one figure -> figures/cdf_lstm.png
python figures/make_boxplot.py          # -> figures/metrics_boxplots_lstm.png
python figures/make_hyperparam.py       # -> figures/hyperparameters_lstm.png
python figures/make_change_map.py       # -> figures/change_map_lstm.png
python figures/make_donor_map.py        # ungauged only, by design
python figures/make_error_structure.py  # the premise, before any correction -- needs
                                         # evaluate.py run for --model lstm, hybrid and hbv
python figures/make_method.py
```

The four combined figures take `--model` (default `lstm`) and auto-detect which cells that
model has -- HBV has no ungauged run, so `--model hbv` produces a gauged-only figure instead of
a two-row one. `--cell gauged`/`--cell ungauged` forces a single cell even for a model that has
both, for one-off inspection; that always writes a `gauged_`/`ungauged_`-prefixed file so it
never overwrites the combined one.

Only training needed a GPU, and it is finished. Everything downstream runs on any machine: the correction is a few
minutes of grid search, and the correction itself is milliseconds.
