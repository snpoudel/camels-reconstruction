# Better ungauged-basin prediction from a standard LSTM and one correction step

Predicting streamflow where there is no gauge is the standard hard problem in hydrology. The
best published result is a plain LSTM trained across many basins at once — **median NSE 0.693**
over 531 CAMELS-US basins (Kratzert et al. 2019).

This branch keeps that model exactly as it is and adds **one post-processing step** that costs
milliseconds and trains nothing:

> Ask the nearest gauges how wrong the model was today. If they agree it ran low, assume this
> basin did too, and believe half of it.

On fold 0, that moves median NSE on held-out basins from **0.7086 to 0.7457**, KGE from 0.704
to 0.723, and halves the bias on the lowest 30% of flows. Applied to the parent branch's
7-fold predictions across all 531 basins the same step gives 0.7235 → 0.7429, better on 7 of
7 folds.

`METHOD.md` is the full account: how the idea arrived, the arithmetic, the leak audit, and the
list of things that were tried and thrown away.

## The correction

Over the K nearest gauged basins *j* that reported that day:

```
w_j = (1/d_j) / sum(1/d)                                d = great-circle km
R   = sum_j w_j * [ log(q_sim_j + eps) - log(q_obs_j + eps) ]
q_corrected = (q_sim + eps) * exp(-lambda * R) - eps
```

**eps = 0.01 mm/day.** K and lambda are selected per fold by grid search — it costs seconds —
searching **only inside that fold's training basins**, never a held-out basin of any fold
(`METHOD.md` §5; `analysis/leak_audit.py` check [6] verifies it by destroying every held-out
observation and confirming the selection does not move). The choice barely matters: over
K in [3,20] and lambda in [0.3,0.7] the whole box spans 0.012 NSE on the parent branch's
531-basin surface and 0.027 on this branch's single fold, against a gain of ~0.04–0.05.

The target basin's own discharge is never read. Deleting it from the record entirely leaves the
corrected hydrographs bit-identical; deleting a *gauged* neighbour instead moves them, which is
what makes that check meaningful. Run `python analysis/leak_audit.py` to reproduce it.

## Layout

```
src/config.py      hyperparameters -- Kratzert et al. 2019, nothing tuned
src/data.py        cache, folds, and every leak-sensitive statistic
src/model.py       the LSTM: one layer, statics concatenated
src/train.py       literature-standard batched training
src/evaluate.py    prediction, NSE/KGE/RMSE, bias by flow band
src/correct.py     THE CORRECTION -- about six lines of arithmetic
analysis/tune.py       the K x lambda surface, leave-one-fold-out selection
analysis/leak_audit.py the correction cannot see the basin it corrects -- with a positive control
figures/           palette.py plus one script per figure
```

## Running it

```bash
python -c "from src.data import build_cache; build_cache()"   # -> data/cache/
python -m src.train --benchmark 200                            # measure ms/step first
for f in 0 1 2 3 4 5 6; do FOLD=$f CONCURRENCY=4 ./run_screen.sh; done
python -m src.evaluate                                         # -> runs/metrics.csv
python -m src.correct --out runs/corrected_metrics.csv         # selects K, lambda; raw vs corrected
python analysis/tune.py --out runs/tuning.csv                  # the sensitivity surface
python analysis/leak_audit.py                                  # prove there is no leak
python figures/make_method_figure.py                           # the idea
python figures/make_method_detail_figure.py                    # the arithmetic, worked
python figures/make_cdf_figure.py
python figures/make_boxplot_figure.py
python figures/make_hyperparam_figure.py
```

The correction needs no GPU and no training — `python -m src.correct` runs on any machine in
under a second once predictions exist.

## Status

The model code here is the **literature-standard batched** LSTM, at forget-gate bias 5.0
(Kratzert et al. 2019) rather than the parent branch's frozen 3.0 — see `METHOD.md` §8. The predictions currently in
`runs/` are fold 0 only, and were produced by the parent branch's date-batched variant; they
stand in so the figures and the pipeline are exercised end to end. Every number in this README
and in `METHOD.md` is marked with the fold and the model it came from, and will be replaced
once the 7-fold grid runs. `METHOD.md` §8 lists what that leaves unproven.
