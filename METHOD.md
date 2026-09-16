# The neighbour-residual correction

How the idea arrived, what it does, and the evidence for it — including a real 7-fold,
531-basin grid trained on an A100 with the literature-standard configuration (§7), not a
placeholder. Plain language throughout; the arithmetic is in `src/correct.py`, about six
lines long.

---

## 1. The problem

Predicting streamflow at a basin with no gauge — "prediction in ungauged basins", PUB — is
the standard hard case in hydrology. The best published result is a plain LSTM trained across
many basins at once: **median NSE 0.693** over 531 CAMELS-US basins (Kratzert et al. 2019,
`BENCHMARKS.md`). Being ungauged costs about 0.05 NSE against the same model on basins it
trained on, and that 0.05 is the headroom anything new has to compete for.

## 2. How the idea arrived

This repository started somewhere else. It spent its first phase asking whether **cross-basin
attention** could beat the plain LSTM: let the model learn, for each basin, which other basins
to borrow from. Three variants were built and run (`EXPERIMENTS.md` has the full record).
Letting attention borrow other basins' hidden *states* did nothing. Letting it borrow their
measured *discharge* helped — but that is an information advantage, not an architectural one,
and it collided with concurrent published work.

The useful question fell out of a diagnostic rather than a model. If the attention variants
gained by reading neighbours' discharge, **how much of a basin's error is visible in its
neighbours' errors in the first place?** That is answerable with no training at all, from
predictions already on disk:

> For each held-out basin, take its daily error. Take the distance-weighted mean of its
> gauged neighbours' errors on the same days. Correlate them.
>
> **r = 0.48** (median across basins), so **r² = 0.23**.

About a quarter of the error variance at an ungauged site is already visible in the gauges
around it. Two follow-ups said this was real signal and not an artefact:

- After removing the seasonal cycle from both sides, neighbour agreement is still **r = 0.86**
  — so it is genuine day-to-day information, not the shared seasonality the LSTM already
  knows from its weather inputs.
- The gain decays smoothly with how far the nearest gauge is, and basins with a gauge *under*
  10 km away gain **less** than basins 10–25 km away — the opposite of what "it is really just
  copying an upstream gauge" would produce.

A correlation of 0.48 that costs nothing to exploit is worth more than an architecture. So the
architecture work stopped and this branch began.

## 3. What the method does

Your LSTM has predicted flow everywhere. Pick an ungauged basin and a day.

1. **Ask the neighbours.** Find the K nearest gauged basins. At each you have both the model's
   prediction and the real measurement, so you know how wrong the model was there today.
2. **See whether they agree.** If they all say the model ran low by about a quarter, that is a
   regional signal — the storm was bigger than the weather data said, or the catchments were
   wetter than the model assumed.
3. **Assume the ungauged basin has the same problem — but hedge.** Apply a fraction λ of the
   neighbours' average error to its prediction.

In symbols, over the K nearest gauged basins *j* that reported that day:

```
w_j = (1/d_j) / sum(1/d)                                  d = great-circle km
R   = sum_j w_j * [ log(q_sim_j + eps) - log(q_obs_j + eps) ]
q_corrected = (q_sim + eps) * exp(-lambda * R) - eps
```

**eps = 0.01 mm/day**; K and lambda are selected per fold (§5), with K = 8, lambda = 0.5 as the no-search fallback.

`figures/method.png` shows the idea in three panels; `figures/method_detail.png` works the arithmetic through on one real basin and one real day, stage by stage, with every intermediate number printed.

### Why these choices

**Why log space.** Streamflow errors are multiplicative. "The model ran 30% low" transfers
between a 0.5 mm/day creek and a 50 mm/day river; "the model was off by 2 mm/day" does not.

**Why lambda < 1.** The neighbours are an imperfect proxy, so believing them fully overshoots.
Fitting lambda by least squares in log space gives 0.83 (measured over 276,126 basin-days).
The value that actually maximises NSE is about 0.5, because NSE is computed in flow space and
dominated by high flows, where `exp()` amplifies an over-correction. Fitting the shrinkage
rather than choosing it scores *worse* — the two facts are the same fact.

**Why eps = 0.01 mm/day, a plain constant.** It keeps `log()` finite on the 2.8% of days that
report exactly zero flow. A much smaller floor (1e-5) makes those days produce log-residuals
near −11 and costs about 0.005 NSE. Tying eps to each basin's mean precipitation was tried and
is not worth the extra machinery.

**Why nothing is fitted.** See §5 — the surface is flat, so fixed constants cost nothing and
remove a whole class of "how did you tune that?" objections.

## 4. What it is not allowed to see

The correction never reads the target basin's own discharge. The weights are built from
coordinates only. Two audits, each with a positive control that proves the test can fail:

| test | result |
|---|---|
| Does a basin borrow from itself? | PASS — held-out basins are not in the donor pool at all |
| **Delete every held-out basin's discharge from the entire record** | PASS — corrected hydrographs **bit-identical**, max difference `0.000e+00` |
| Replace held-out discharge with pure noise | PASS — bit-identical |
| **Positive control: delete the most-used gauged donor instead** | PASS — output moves by 9.085 mm/day, so the tests above are not inert |
| **Destroy every held-out observation and re-run selection** | PASS — the chosen (K, λ) does not move, for either search rule |
| Do the weights read discharge? | PASS — distance only |

Run it yourself: **`python analysis/leak_audit.py`** (exits non-zero on any failure, so it can
go in CI). The second test is the decisive one — the pipeline was told those basins have never
been gauged, all 3,652 days deleted, and its output did not change by a single bit. The fourth
is what makes the second mean anything.

**One honest scope limit.** The method reads *concurrent* discharge at neighbouring gauges. The
published 0.693 uses no discharge anywhere at prediction time. So this is a method for gauged
*regions* containing ungauged *sites*, and the fair comparison is against the same LSTM without
the correction — which is exactly the comparison this branch is built to make.

## 5. Choosing K and lambda, without ever touching the test basins

The grid search takes seconds, so there is no reason not to run it. The only question is
*where* it is allowed to look.

**The rule: no held-out basin of any fold is ever read during selection.** This is stricter
than it first appears. Searching another fold's held-out basins would leave fold *f*'s own
test set untouched, and is tempting for that reason — but every basin is some fold's test set,
so the pooled 531-basin number would end up tuned on itself. That route is closed.

What is left is fold *f*'s own 455 **training** basins, and the model's predictions at them.
`src/correct.py:choose` offers three rules, all clean. Measured across all 7 folds, 531
basins, and reported as the *paired* per-basin comparison (not a difference of two medians —
see the note in §7 about why that distinction matters here):

| rule | where it searches | 531-basin median | vs `fixed`, paired |
|---|---|---|---|
| **`inner`** (default) | leave-one-group-out inside the training basins: they keep their original fold labels, so each of the six groups is corrected using donors from the other five — a thinned gauge network, like a real ungauged basin faces | **0.7057** | **+0.0043** (95% CI [+0.0027, +0.0056]) |
| `gauged` | the same basins without the group structure, self masked | 0.7021 | statistically identical to `inner` (paired diff 0.0000, CI [0, 0]) |
| `fixed` | no search, K=8, λ=0.5 | 0.7002 | — |
| *oracle* | *the held-out basins themselves — shown only to price the others; not selectable* | *0.7025* | — |

This settles the question the single-fold placeholder left open. **Selection now beats fixed
constants by a real, bootstrap-confirmed margin** — small (+0.004 median NSE) but the 95%
interval sits entirely above zero, not straddling it as the single-fold comparison did.
`inner` and `gauged` are statistically indistinguishable from each other despite choosing
different (K, λ) per fold in most cases, which is itself informative: it is more evidence for
the flatness argument below than for the group-structure argument the single-fold placeholder
suggested. That framing (`inner` beating `gauged` by exploiting a thinned gauge network) does
not survive contact with real data and is retracted.

`analysis/leak_audit.py` check [6] is what makes "no held-out basin was read" a measurement
rather than a promise: it destroys every held-out observation, re-runs selection, and checks
the chosen (K, lambda) does not move. It doesn't — verified on the real 7-fold grid, not just
the single-fold placeholder.

### How much the choice matters at all

`figures/hyperparameters.png` draws the cross-validation surface itself: one line per fold,
each a profile through that fold's own CV optimum, with a marker on the optimum. Nothing on
that figure is scored on a held-out basin, and nothing is annotated as "selected" — the point
is the shape, not a winner. As more folds run, more lines appear.

What it shows, and what a paper would quote: **the optima cluster in a narrow region and the
curves are flat around them**, so the exact pair matters far less than that it lies in that
region. Across the real 7-fold grid the per-fold CV optima are K in [5, 10], lambda in
[0.65, 0.80] — five values of K spanning a factor of 2, lambda spanning 0.15 — which is the
statement a paper should make, not a single tuned pair.

Measured on the held-out basins afterwards — as a check on the method, never as an input to
it — the flatness carries over:

| surface | box K in [3,20], lambda in [0.3,0.7] | best cell (oracle) |
|---|---|---|
| 531 basins, 7 folds, `donor` base (parent branch) | 0.7312–0.7429, **range 0.012** | 0.7429 |
| 531 basins, 7 folds, `vanilla` base (this branch, 1 seed) | 0.6757–0.7017, **range 0.026** | 0.7025 |

Against a gain of about 0.05–0.07 over uncorrected, a 0.012–0.026 spread across a 5-fold range
in K and a 2-fold range in lambda is the real message: **the correction works across a wide
region, and cross-validating inside it is cheap insurance rather than a load-bearing step.**
Both rows are now real 7-fold, 531-basin surfaces — no placeholder or single-fold caveat
remains here.

The one genuinely bad corner is K = 1 with large lambda: a single over-trusted neighbour.

## 6. What was tried and rejected

Each was run on all 531 basins using the parent branch's 7-fold predictions. This matters:
the simple form is the *chosen* one, not the only one anybody looked at.

| variant | 531-basin median NSE |
|---|---|
| **1/d over K nearest, fixed constants (shipped)** | **0.7429** |
| two-parameter kernel using static attributes, tuned | 0.7436 — two parameters buy +0.0007 |
| standardise (x, y, attributes) into one space, Euclidean | 0.7275 |
| cosine similarity on (x, y, attributes) | 0.7278 |
| attributes only, no geography | 0.7306 |
| weights fit by ridge regression instead of chosen | 0.7357 |
| lambda varying per basin with distance to nearest gauge | 0.7399–0.7434 |
| lambda split by high/low flow day | +0.001 |
| neighbour residual smoothed over 3 / 7 / 15 days | 0.7360 / 0.7297 / 0.7260 |
| eps = 1e-5 instead of 0.01 | 0.7379 |
| global variance inflation of the hydrograph | 0.7235 — exactly zero |
| neighbour flow as extra LSTM *input channels* | 0.675 vs 0.688 for the same model without |

Two are worth dwelling on. **Putting geography and attributes into one standardised space and
using a single distance is the worst option** — it makes position 2 of 9 dimensions and dilutes
the only signal that matters. And **smoothing hurts at every window**, so the signal is
genuinely same-day, not a slow drift.

## 7. What it buys

All 531 basins, real 7-fold grid, literature-standard batched LSTM (`period=kratzert`,
`seq_len=365`, `batch_size=512`, `forget_bias=5.0`), **1 seed per fold** — the first stage of
the VM run, before any seed-ensembling. K and lambda selected per fold on that fold's own
training basins (`inner`, §5).

| fold | LSTM | + correction | gain | K, λ |
|---|---|---|---|---|
| 0 | 0.6359 | 0.7133 | +0.0773 | 10, 0.70 |
| 1 | 0.6638 | 0.7407 | +0.0769 | 8, 0.75 |
| 2 | 0.6092 | 0.6583 | +0.0491 | 5, 0.65 |
| 3 | 0.6669 | 0.7380 | +0.0711 | 8, 0.80 |
| 4 | 0.6274 | 0.6715 | +0.0441 | 8, 0.75 |
| 5 | 0.6258 | 0.7163 | +0.0904 | 5, 0.65 |
| 6 | 0.5907 | 0.6603 | +0.0696 | 8, 0.65 |
| **all 531** | **0.6313** | **0.7057** | **+0.0744** | — |

Positive on **7 of 7 folds**. Median NSE already exceeds the published 0.693 benchmark on a
single seed, before any seed-ensembling — the step that historically gains the most in this
literature (Kratzert's own number is a 10-member ensemble mean).

The pooled-median gain (+0.0744) is not quite the same claim as the paired one, and both are
worth having: pairing each basin's own corrected score against its own raw score gives a
**median per-basin gain of +0.051, bootstrap 95% CI [+0.046, +0.059]** — entirely positive, so
this is not a difference-of-medians artifact (see §5 for why that check matters here). 422 of
531 basins improve (79%), 45 are hurt by more than 0.05 NSE, and mean NSE rises 0.441 → 0.524 —
the correction rescues the badly-predicted tail rather than only polishing basins already fine.

| | LSTM | + correction |
|---|---|---|
| median KGE | 0.670 | **0.682** |
| median RMSE (mm/day) | 1.383 | **1.236** |
| median overall bias | +9.2% | **−2.4%** |
| bias, top 10% of flows | −9.3% | −14.1% (worse) |
| bias, middle 30–90% | +21.1% | **+6.7%** |
| bias, bottom 30% | +73.6% | **+34.8%** |

Read honestly:

- **The correction works on every measure except one.** NSE, KGE, RMSE, overall bias, and bias
  in the middle and lowest flow bands all improve — overall bias actually flips from biased
  high (+9.2%) to nearly balanced (−2.4%), which the single-fold placeholder did not show.
- **The one exception is bias in the top 10% of flows, which gets worse** (−9.3% → −14.1%).
  The correction trades a little high-flow accuracy for a large reduction in day-to-day error
  everywhere else — worth stating plainly rather than leaving in the figure only.
- **Selection now measurably beats fixed constants** (§5): +0.0043 median NSE, 95% CI
  [+0.0027, +0.0056], entirely positive. The single-fold placeholder had this backwards (fixed
  appeared to win, and even that was inside the noise); with real folds the effect is small
  but real, and `analysis/tune.py --mode inner` (the default) is the recommended setting.

See `figures/cdf_nse.png`, `figures/metrics_boxplots.png`, `figures/method.png`,
`figures/method_detail.png`.

**Consistent with the parent branch's independent evidence** (different base model — `donor`,
not `vanilla` — so not the same claim, but the same direction): applied to 7-fold `donor`
predictions, the same correction code gives 0.7235 → 0.7429, better on 7 of 7 folds, 73% of
basins improved. Two different base models, two independent 7-fold grids, the same qualitative
result each time.

## 8. Configuration this run actually used

`forget_bias` is **5.0** here, not the 3.0 the parent branch froze across its architectures —
Kratzert et al. (2019, WRR)'s own value, taken because this branch trains the
literature-standard model. `seq_len` is 365 (the common literature choice) and `batch_size` is
512 (between the framework's 256 and the paper's 2000). §10 covers the train/test window.

The 7-fold grid in §7 was trained end to end with exactly this configuration on an A100 —
there is no longer a version mismatch to track between this section and the results above.

## 9. What is still unproven

1. **Single seed.** §7's numbers are one seed per fold. The correction is deterministic given
   a hydrograph, so it is unaffected by seed noise directly, but the *baseline* LSTM's own
   score carries seed variance that has not yet been averaged out. This literature typically
   reports ensemble means (Kratzert's 0.693 is a 10-member ensemble) precisely because a single
   seed reads lower and noisier — which is consistent with §7's uncorrected 0.6313 sitting
   below the historical 3-seed placeholder's 0.7086. The natural next step, if pursued, is
   `SEEDS="0 1 2"` on the same `run_screen.sh` grid.
2. **Gauge density is a real limit.** Measured on the parent branch's `donor`-based evidence
   (§6/§7): the gain falls from about +0.027 NSE where the nearest gauge is 10–25 km away to
   about +0.002 beyond 100 km. Not yet re-measured on this branch's own predictions.

## 10. Train/test windows

Two are available, selected with `--period` and recorded in every run's `config.json`. Runs
from different windows are refused rather than pooled — their test axes differ in length, and
lining up the wrong dates would be invisible.

| | train | test | train qobs complete | test qobs complete |
|---|---|---|---|---|
| `kratzert` (default; used for §7) | WY1981–1995, 5478 d | WY1996–2010, 5479 d | 96.7% — 55 basins have gaps, worst 40% | **99.99%**, 530/531 basins |
| `complete` | WY1990–2004, 5479 d | WY2005–2014, 3652 d | **100.000%**, all 531 basins | 98.7% |

`kratzert` matches Kratzert et al. (2019, WRR) exactly, which makes the 0.693 benchmark a much
closer comparison — same basins, same windows, same lengths. What still differs is the forcing
product (Daymet here, extended NLDAS there), the number of folds (7 vs 12) and the ensemble
size, so it is a closer comparison, not an identical one.

Its costs are real but small. The training window has ~2.8% missing targets, which `nse_loss`
masks per-sample but which `complete` avoids entirely. And it begins at day index 274 of the
cache, while a 365-day sequence needs 364 days of history, so **the first 90 training days are
unreachable** — `prepare` moves the first target to 1980-12-30 and prints exactly what it
dropped (90 of 5478 days, 1.6%) rather than silently running short windows. Training therefore
covers 5388 days instead of 5478.

Its test window is the better of the two — 15 years at 99.99% complete against 10 years at
98.7% — which matters more for the reported number than the training gaps do.

## 11. What was run, and the natural next step

Stage 1 — one seed, all 7 folds — has been run on an A100 and is what §7 reports:

```bash
python -m src.train --benchmark 200                              # measured first, per CLAUDE.md
FOLDS="0 1 2 3 4 5 6" SEEDS=0 CONCURRENCY=4 ./run_screen.sh       # 7 runs
python -m src.evaluate
python -m src.correct --out runs/corrected_metrics.csv
python analysis/tune.py --out runs/tuning.csv
python analysis/leak_audit.py
python figures/make_cdf_figure.py && python figures/make_boxplot_figure.py
python figures/make_hyperparam_figure.py && python figures/make_method_figure.py
python figures/make_method_detail_figure.py
```

7 runs (training + full-record inference), ~8.5 h wall-clock at CONCURRENCY=4 on one A100
(22.1 ms/step measured single-process; actual pace was ~7 min/epoch once 4 runs shared the
GPU), ≈$31 at $3.67/h. All 7 succeeded, 0 failures.

The one thing §9 leaves open is seed noise in the baseline. Widening to
`SEEDS="0 1 2"` on the same grid — 14 more runs, since fold 0's seed 0 already exists and
`run_screen.sh` skips completed runs — would give the 3-seed ensemble-mean baseline this
literature conventionally reports, and is the natural next step if a fully seed-stable number
is wanted.