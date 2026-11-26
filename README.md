# camels-reconstruction

A regional LSTM predicts daily streamflow at 531 CAMELS-US basins. Where a basin's gauge is
missing, its prediction is corrected with the same day's errors at the nearest gauges, either
by inverse-distance weighting (IDW) or by a small neural network (MLP).

Two settings are tested:
- **Ungauged:** the basin was never used in training.
- **Partially gauged:** the basin was used in training, but its gauge stops reporting in the test years.

## Layout

| Path | Contents |
|---|---|
| `src/models/lstm/` | The LSTM and its training |
| `src/corrections/idw/` | IDW correction |
| `src/corrections/mlp/` | MLP correction |
| `src/shared.py` | Helpers shared by both corrections |
| `figures/` | Scoring (`evaluate.py`) and one script per figure (`make_lstm_*.py`) |

## Data

`data/` includes the basin list (`camels531.csv`) and US state outlines. The per-basin input
files (`data/raw/input_<gauge_id>.csv`: daily forcings, catchment attributes and observed flow)
are not included. Build them from CAMELS-US (Newman et al., 2015; Addor et al., 2017).

## Running

```bash
pip install -r requirements.txt
python -c "import sys; sys.path.insert(0, 'src/models/lstm'); import common; common.build_cache()"

python src/models/lstm/01_lstm_gauged.py        # partially gauged (GPU)
python src/models/lstm/02_lstm_ungauged.py      # ungauged (GPU)

python src/corrections/idw/01_gauged.py         # likewise 02_ungauged.py, and mlp/
python figures/evaluate.py --method idw         # likewise --method mlp
python figures/make_lstm_cdf.py                 # likewise each make_lstm_*.py
```

Note: This repo is developed with help of Claude Code (Anthropic, Opus 5.5).
