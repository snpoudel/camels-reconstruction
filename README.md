# Attention-LSTM-PUB

Does cross-basin (spatial) attention improve streamflow prediction in ungauged basins (PUB)
beyond a vanilla LSTM? This repository runs the experiment that answers it.

## The question, split into three

A vanilla LSTM trained on many basins already transfers well to ungauged ones — it is the
current state of the art for PUB. Adding attention across basins could help for two quite
different reasons, which previous work here conflated into a single model. They are separated
into four architectures so each mechanism can be tested on its own:

| Architecture | Attention keys / values | What it isolates |
|---|---|---|
| `vanilla` | — | Baseline |
| `spatial` | 455 gauged basins | **Reference.** Does retrieving from gauged donors help at all? |
| `spatial_all` | 455 gauged + 76 ungauged | Does letting ungauged basins *participate in training* help? |
| `donor` | 455 gauged, values carry observed flow | Does donor *discharge* add beyond donor *state*? |

Three single-variable comparisons: `vanilla`→`spatial`, `spatial`→`spatial_all`,
`spatial`→`donor`.

`spatial_all` is transductive: it needs the ungauged basins present during training, so it
reconstructs flow for basins designated in advance rather than generalising to new ones.

`donor` is a different problem statement, not just a different architecture — it assumes
gauged neighbours reporting in real time, so it has access to information the other three do
not. Read `donor` vs `vanilla` as an information comparison, not an architecture one.

## Setup

**Basins.** 531 CAMELS-US basins, 7 random folds. One fold (76 basins) is held out as
ungauged; the other six (455) are gauged. No validation fold — training runs a fixed number
of epochs, so the held-out fold is never used for model selection.

**Periods.** Train WY1990–2004 (5,479 days), test WY2005–2014 (3,652 days). The training
window was chosen because **qobs is 100.000% complete for every basin on every day** of it —
which removes donor-eligibility floors, target masking, and basin exclusions entirely. All
missingness in the record sits in WY1981–89 and WY2010–14.

**Evaluation** is a 2×2 over the *same* trained models — these are slices, not extra runs:

|  | Train period | Test period |
|---|---|---|
| Gauged basins (455) | fit quality | **gauged ceiling** |
| Ungauged basins (76) | — | **PUB — the headline** |

The gauged-ceiling cell is what makes a small PUB improvement interpretable. The two cells
that carry the comparison are `gauged/test` (out-of-sample in **time**) and `ungauged/test`
(out-of-sample in **space**).

`src/evaluate.py` writes `runs/metrics.csv` — one row per basin × architecture × seed × cell,
with NSE, KGE (and its r/α/β components) and RMSE — which is the shape the CDF plots are drawn
from. Three seeds are paired across architectures, so initialisation and date order match and
part of the seed noise is common to all of them.

**Hyperparameters** follow Kratzert et al. (2019, HESS 23, 5089) where the paper specifies a
value and NeuralHydrology code defaults otherwise — hidden 256, dropout 0.4, sequence length
270, initial forget bias 3, 30 epochs, LR `{0: 1e-3, 10: 5e-4, 20: 1e-4}`, gradient clipping
at norm 1, Adam. The one deliberate deviation is the loss: basin-normalised NSE (NSE*) rather
than MSE, which the same paper recommends for multi-basin training. Every value is identical
across architectures, so none can confound the comparison.

Reference point: Kratzert et al. (2019, WRR 55) report median NSE **0.69** for a k-fold PUB
LSTM on these same 531 basins, against 0.64 for calibrated SAC-SMA and 0.58 for the National
Water Model. `BENCHMARKS.md` records that study in full, including what about it is and
is not comparable to the runs here.

## Running it

```bash
pip install -r requirements.txt
python -c "from src.data import build_cache; build_cache()"   # ~10 s, writes data/cache/

python -m src.train --arch spatial --benchmark 200            # measure ms/step first
CONCURRENCY=4 ./run_screen.sh                                 # 12 runs: 4 archs x 3 seeds
python -m src.evaluate                                        # 2x2 table + paired tests
```

`--epochs`, `--seq-len`, `--max-steps` and `--pred-stride` exist for smoke tests and must be
left at their defaults for real runs.

## Layout

| File | |
|---|---|
| `src/config.py` | Hyperparameters and periods. No logic. |
| `src/data.py` | Cache, folds, and **every leak-sensitive statistic** — feature scaler, loss sigma, donor anomalies, all fit on gauged basins over the training period only. |
| `src/model.py` | LSTM encoder and cross-basin attention, including the self-attention mask. |
| `src/train.py` | NSE* loss, training loop, benchmark mode. |
| `src/evaluate.py` | Per-basin metrics, the results table, and `metrics.csv`. |
| `BENCHMARKS.md` | Published reference points this experiment is read against — what was run, and what is and is not comparable. |
| `src/description.md` | How the architecture works, end to end, with the real tensor shapes and worked inference cases. |
| `figures/` | The paper figures and the scripts that build them — `make_method_figure.py` (architecture) and `make_results_figure.py` (six-panel screen summary). Both run from any directory. |

## Design notes

**Attention uses low-rank learned projections.** `W_q, W_k: 256 → 64` plus a learnable
softmax temperature, adding 33k parameters (+9%). An earlier version scored similarity as
`softmax(context @ context.T)` with no projections, forcing the model to compare basins in the
exact representation it uses to predict flow. With 531 near-uniform logits that collapses to a
global mean vector added to every basin — a regulariser, not cross-basin retrieval.

**A basin never attends to itself.** In `donor`, the value carries the donor's standardised
log-flow anomaly, and inverting that anomaly recovers the basin's own flow at NSE 1.000000 —
the self-value *is* the target. The diagonal is masked in training *and* evaluation, since two
of the four 2×2 cells query gauged basins that are themselves in the donor pool.

**Predictions are emitted in mm/day.** No code path scales the output by the query basin's
own flow statistics; that would inject the single most informative unknown in PUB.

**Basins are reordered as `[gauged..., ungauged...]`** so the encoder window is a contiguous
GPU slice. Position therefore encodes fold membership by construction, and every output is
keyed back through gauge ID rather than index.

## Prior version

`agu2025_poster.png` presents an earlier iteration (median NSE 0.685 vanilla → 0.714
space-time attention). Those numbers came from a setup with no temporal holdout and a single
seed, and are superseded by this design.
