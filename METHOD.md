# The neighbour-residual correction

How the idea arrived, what it does, what was tried and thrown away, and what is still
unproven. Plain language throughout; the arithmetic is in `src/correct.py`, which is about
six lines long.

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

**K = 8, lambda = 0.5, eps = 0.01 mm/day.**

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
| Do the weights read discharge? | PASS — distance only |

Run it yourself: **`python analysis/leak_audit.py`** (exits non-zero on any failure, so it can
go in CI). The second test is the decisive one — the pipeline was told those basins have never
been gauged, all 3,652 days deleted, and its output did not change by a single bit. The fourth
is what makes the second mean anything.

**One honest scope limit.** The method reads *concurrent* discharge at neighbouring gauges. The
published 0.693 uses no discharge anywhere at prediction time. So this is a method for gauged
*regions* containing ungauged *sites*, and the fair comparison is against the same LSTM without
the correction — which is exactly the comparison this branch is built to make.

## 5. How much the two constants matter

`figures/hyperparameters.png`. Each panel sweeps one constant while the thin grey lines hold
the other at every value on the grid, so the spread between them *is* the sensitivity to the
constant not on the x-axis.

Two surfaces, and they do not say the same thing:

| surface | box K in [3,20], lambda in [0.3,0.7] | best cell | K=8, lambda=0.5 |
|---|---|---|---|
| 531 basins, 7 folds (parent branch, `donor` base) | 0.7312–0.7429, **range 0.012** | 0.7429 | 0.7429 |
| 76 basins, fold 0 only (this branch, `vanilla` base) | 0.7308–0.7579, **range 0.027** | 0.7579 | 0.7494 |

On the 531-basin surface the whole box is within about 0.01 NSE and the shipped constants
land on the best cell. **On the single fold in this branch the box spans 0.027 and the shipped
constants sit 0.009 below the best cell** — a sixth of the headline gain. One fold of 76 basins
is a much noisier surface than seven folds of 531, and neither the flatness nor the apparent
optimum on it should be over-read.

So the defensible claim today is the weaker one: the correction helps across a wide range of
both constants, and the shipped pair is a reasonable point inside that range rather than a
fitted optimum. Whether the tighter 531-basin flatness carries over to a `vanilla` base is a
question the 7-fold grid answers, not this branch.

The one genuinely bad corner on both surfaces is K = 1 with large lambda: a single
over-trusted neighbour.

`analysis/tune.py` reproduces the surface and, once more than one fold exists, selects
leave-one-fold-out — the values applied to a fold are chosen on the *other* folds, so a fold
never tunes anything applied to itself. With a single fold it reports the surface and declines
to select, because a value picked on the only fold being scored is not a measurement. **That
is the situation now**, which is why the shipped constants are fixed rather than fitted.

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

Fold 0, 76 held-out basins, 3-seed mean hydrograph, nothing tuned:

| | LSTM | + correction |
|---|---|---|
| **median NSE** | **0.7086** | **0.7494** (+0.0407) |
| median KGE | 0.704 | 0.710 |
| median RMSE (mm/day) | 1.207 | 1.138 |
| median overall bias | −2.2% | **−4.4%** (worse) |
| bias, top 10% of flows | −11.9% | −10.8% |
| bias, middle 30–90% | +13.4% | +6.5% |
| bias, bottom 30% | +49.1% | +32.4% |

78% of basins improve; 3 of 76 are hurt by more than 0.05 NSE. Note the one row that moves the
wrong way: **overall water balance gets worse**, −2.2% to −4.4%. The correction trades a little
long-run bias for a large reduction in day-to-day error, which is what NSE, KGE and RMSE all
reward — worth stating plainly rather than leaving in the figure. See `figures/cdf_nse.png` and
`figures/metrics_boxplots.png`. The correction mostly removes bias in the middle and lower
parts of the flow range, where the LSTM runs high.

**Supporting evidence from the parent branch** (different base model, so not this branch's
headline): applied to 7-fold `donor` predictions across all 531 basins, the same correction
gives 0.7235 → 0.7429, better on **7 of 7 folds**, 73% of basins improved.

## 8. One deliberate change from the parent branch

`forget_bias` is **5.0** here, not the 3.0 the parent branch froze across its architectures.
5.0 is what Kratzert et al. (2019, WRR) specify and what the parent branch's own
literature-faithful variant used; this branch is the literature-standard model, so it takes
the literature value. Two consequences worth stating rather than discovering later:

- The fold-0 predictions currently in `runs/` were trained at **bias 3.0** by the parent
  branch's date-batched model. Every number in §7 therefore prices the correction against
  *that* baseline, not against the model this branch trains.
- When the 7-fold grid runs, the baseline and the corrected number will both come from the
  bias-5.0 batched model, and §7 must be regenerated wholesale. Do not mix the two.

## 9. What is still unproven

1. **One fold, and the wrong model.** This branch carries fold 0 only, borrowed from the
   parent branch's date-batched `vanilla` at forget bias 3.0 — not the literature-standard
   batched model at bias 5.0 that `src/` now trains (§8). The headline is a placeholder.
2. **No uncertainty bars.** The correction is deterministic given a hydrograph, but the
   underlying predictions carry seed noise that has not been propagated.
3. **Gauge density is a real limit.** The gain falls from about +0.027 NSE where the nearest
   gauge is 10–25 km away to about +0.002 beyond 100 km.

## 10. To settle it

```bash
for f in 0 1 2 3 4 5 6; do FOLD=$f SEEDS="0 1 2" CONCURRENCY=4 ./run_screen.sh; done
python -m src.evaluate
python -m src.correct --out runs/corrected_metrics.csv
python analysis/tune.py --out runs/tuning.csv
python figures/make_cdf_figure.py && python figures/make_boxplot_figure.py
python figures/make_hyperparam_figure.py && python figures/make_method_figure.py
```

21 runs. That produces the 7-fold `vanilla` baseline this project has never had, and the
corrected number beside it, from the same predictions.
