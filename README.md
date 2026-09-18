# Better ungauged-basin prediction from a standard LSTM and one correction step

Predicting streamflow where there is no gauge is the standard hard problem in hydrology. The
best published result is a plain LSTM trained across many basins at once — **median NSE 0.693**
over 531 CAMELS-US basins (Kratzert et al. 2019).

This branch keeps that model exactly as it is and adds **one post-processing step** that costs
milliseconds and trains nothing:

> Ask the nearest gauges how wrong the model was today. If they agree it ran low, assume this
> basin did too, and believe most of it.

Trained end to end on an A100 — real 7-fold grid, literature-standard configuration
(`seq_len=365`, `batch_size=512`, `forget_bias=5.0`, Kratzert's train/test windows), 1 seed per
fold. Median NSE on held-out basins: **0.6313 → 0.7095**, positive on **7 of 7 folds**, 81% of
basins improved. That single-seed number already exceeds the published 0.693 benchmark, before
any seed-ensembling (which is what the literature's own 0.693 is — a 10-member ensemble mean).

`METHOD.md` is the full account: how the idea arrived, the arithmetic, the leak audit, the real
per-fold results, how far the correction's reach extends before it stops paying, and what a
wider seed grid would still add.

## The correction

Over the K nearest gauged basins *j* that reported that day:

```
w_j = (1/d_j^p) / sum(1/d^p)                            d = great-circle km
R   = sum_j w_j * [ log(q_sim_j + eps) - log(q_obs_j + eps) ]
q_corrected = (q_sim + eps) * exp(-lambda * R) - eps
```

**eps = 0.01 mm/day.** K, lambda and the distance exponent p are selected per fold by grid
search — it costs seconds — searching **only inside that fold's training basins**, never a
held-out basin of any fold (`METHOD.md` §5; `analysis/leak_audit.py` check [6] verifies it by
destroying every held-out observation and confirming the selection does not move). Selection
measurably beats fixed constants: paired median +0.0062, bootstrap 95% CI [+0.0049, +0.0079],
entirely positive (`METHOD.md` §5). Every fold selects p > 1: the correction leans more on its
nearest few donors than plain inverse distance would.

It needs a gauge nearby, not merely a gauged country: the gain decays with donor distance and
crosses zero at 326 km (`METHOD.md` §8, `figures/recon_ungauged_donor_range.png`).

The target basin's own discharge is never read. Deleting it from the record entirely leaves the
corrected hydrographs bit-identical; deleting a *gauged* neighbour instead moves them, which is
what makes that check meaningful. Run `python analysis/leak_audit.py` to reproduce it.

## Layout

```
src/config.py           hyperparameters -- Kratzert et al. 2019, nothing tuned
src/data.py             cache, folds, and every leak-sensitive statistic
src/model.py            the LSTM: one layer, statics concatenated
src/train.py             literature-standard batched training
src/evaluate.py          prediction, NSE/KGE/RMSE, bias by flow band
src/correct.py           THE CORRECTION -- about six lines of arithmetic
analysis/tune.py        the K x lambda x p surface, cross-validated selection
analysis/leak_audit.py  the correction cannot see the basin it corrects -- with a positive control
analysis/donor_geometry.py  donor distances, and the range at which the correction stops paying
figures/                palette.py plus one script per figure
```

## Running it

```bash
python -c "from src.data import build_cache; build_cache()"   # -> data/cache/
python -m src.train --benchmark 200                            # measure ms/step first
FOLDS="0 1 2 3 4 5 6" SEEDS=0 CONCURRENCY=4 ./run_screen.sh   # stage 1: one seed, all folds
python -m src.evaluate                                         # -> runs/metrics.csv
python -m src.correct --out runs/recon_ungauged_metrics.csv    # selects K, lambda, p; raw vs corrected
python analysis/tune.py --out runs/recon_ungauged_tuning.csv   # the sensitivity surface
python analysis/leak_audit.py                                  # prove there is no leak
python analysis/donor_geometry.py                              # donor distances, and the range limit
python figures/make_method_figure.py                           # method diagram, steps + equations
python figures/make_cdf_figure.py
python figures/make_boxplot_figure.py
python figures/make_hyperparam_figure.py
python figures/make_donor_figure.py                            # -> figures/recon_ungauged_donor_map.png
```

The correction needs no GPU and no training — `python -m src.correct` runs on any machine in
under a second once predictions exist.

## Status

Stage 1 is complete: the literature-standard batched LSTM (`forget_bias=5.0`,
Kratzert et al. 2019, `METHOD.md` §8), 1 seed per fold, all 7 folds, trained end to end on
an A100 (≈8.5 h wall-clock, ≈$31). A third free constant (p, the IDW distance exponent) and the
donor-geometry / break-even analysis were added in a second pass, on a separate checkout, and
reconciled against these same real predictions rather than left as a placeholder-based draft
(`METHOD.md` §5, §7). Every number in this README and `METHOD.md` comes from that real grid.
`METHOD.md` §9 covers what a wider seed grid (`SEEDS="0 1 2"`) would still settle.
