# camels-reconstruction

A regional LSTM predicts daily streamflow at every CAMELS-US basin. At a basin whose gauge is
missing, its prediction is corrected using the same day's errors at the K nearest gauges, each
scaled by that gauge's standard deviation of predicted flow (s):

```
r_k     = (q_sim_k - q_obs_k) / s_k
delta_i = lambda * sum_k w_ik * r_k          w_ik = d_ik^-p / sum_k' d_ik'^-p
q_corr  = max(0, q_sim_i - s_i * delta_i)
```

K, lambda and p are chosen by cross-validation on NSE\*, never on the basin being corrected.
A small MLP that predicts delta_i from the same donors is the learned alternative.

There are two settings:
- **Ungauged:** the basin was never used in training (7-fold split).
- **Partially gauged:** the basin was used in training, but its gauge stops reporting in the test years.

## Layout

| Path | Contents |
|---|---|
| `src/shared.py` | Basins, folds, distances, observations and the saved-correction format |
| `src/models/lstm/` | The LSTM: data cache, training per seed (`01_lstm_gauged.py`) and per fold and seed (`02_lstm_ungauged.py`) |
| `src/corrections/idw/` | IDW correction (`correction.py`), cross-validated grid search (`tuning.py`), pipelines for each setting |
| `src/corrections/mlp/` | MLP correction (`model.py`) and pipelines for each setting |
| `figures/evaluate.py` | Scores the corrections and writes the CSVs the figures read |
| `figures/make_lstm_*.py` | One script per figure; `palette.py` holds the shared style |

## Data

Included in `data/`:
- `camels531.csv`: gauge ID, location, elevation, slope and area of the 531 CAMELS-US basins.
- `us_states_conus.geojson`: state outlines for the maps.

Not included: `data/raw/input_<gauge_id>.csv`, one file per basin with daily Daymet forcings,
static attributes and observed flow (`qobs(mm/day)`) for 1980–2014. Build these from CAMELS-US
(Newman et al., 2015; Addor et al., 2017). The inputs are listed in Table A1 of the paper.

## Running

Everything runs as `python <path>.py` from the repository root.

```bash
pip install -r requirements.txt

# Build the cache and the fold assignment from data/raw/
python -c "import sys; sys.path.insert(0, 'src/models/lstm'); import common; common.build_cache()"

# Train the LSTM (GPU): 5 seeds, and 7 folds x 5 seeds for the ungauged setting
python src/models/lstm/01_lstm_gauged.py
python src/models/lstm/02_lstm_ungauged.py

# Correct (IDW: minutes on CPU; MLP: about an hour per setting)
python src/corrections/idw/01_gauged.py
python src/corrections/idw/02_ungauged.py
python src/corrections/mlp/01_gauged.py
python src/corrections/mlp/02_ungauged.py

# Score, then draw the figures into figures/lstm/
python figures/evaluate.py --method idw
python figures/evaluate.py --method mlp
python figures/make_lstm_method.py           # Fig. 1
python figures/make_lstm_error_structure.py  # Fig. 2
python figures/make_lstm_cdf.py              # Fig. 3
python figures/make_lstm_nse_map.py          # Fig. 4
python figures/make_lstm_boxplots.py         # Fig. B1
python figures/make_lstm_gain_distance.py    # Fig. B2
```

Table 1 gives the per-variant medians from `figures/metrics_<setting>_lstm_<method>.csv`.

Note: This repo is developed with help of Claude Code (Anthropic, Opus 5.5).
