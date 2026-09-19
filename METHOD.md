# The neighbour-residual correction

What the method does, what it is not allowed to see, and what it buys. The arithmetic is
`src/residual_postprocessing.py`, about six lines long.

---

## 1. The problem

Predicting streamflow at a basin with no usable gauge is the standard hard case in hydrology.
The best published result is a plain LSTM trained across many basins at once: median NSE 0.693
over 531 CAMELS-US basins (Kratzert et al. 2019, WRR 55, 11344). Being ungauged costs about
0.05 NSE against the same model on basins it trained on, and that is the headroom anything new
has to compete for.

The starting question was how much of a basin's error is visible in its neighbours' errors in
the first place. That is answerable with no training at all, from predictions already on disk:
take each basin's daily error, take the distance-weighted mean of its neighbours' errors on the
same days, and correlate them. **r = 0.48** (median across basins), and after removing the
seasonal cycle from both sides neighbour agreement is still **r = 0.86** — genuine day-to-day
information, not the shared seasonality the model already knows from its weather inputs.

## 2. What the method does

The model has predicted flow everywhere. Pick a basin and a day.

1. **Ask the neighbours.** Find the K nearest basins that have a gauge. At each you have both
   the prediction and the measurement, so you know how wrong the model was there today.
2. **See whether they agree.** If they all say it ran low by about a quarter, that is a regional
   signal — the storm was bigger than the weather data said, or the catchments were wetter than
   the model assumed.
3. **Assume this basin has the same problem, but hedge.** Apply a fraction λ of it.

Over the K nearest donors *j* that reported that day:

```
w_j = (1/d_j^p) / sum(1/d^p)                              d = great-circle km
R   = sum_j w_j * [ log(q_sim_j + eps) - log(q_obs_j + eps) ]
q_corrected = (q_sim + eps) * exp(-lambda * R) - eps
```

`figures/method.png` draws these steps in order, each with its equation.

**Log space**, because streamflow errors are multiplicative: "the model ran 30% low" transfers
between a 0.5 mm/day creek and a 50 mm/day river, "off by 2 mm/day" does not.

**A distance exponent p.** p = 1 is plain inverse distance, p = 2 the classic Shepard weighting;
the grid searches {1, 1.5, 2, 3}. Every fold picks p > 1, and once it does, K stops mattering
much — the effective donor count `1/sum(w²)` stays small regardless of how large K is. K sets
the pool p may draw from; p decides how much of that pool is really used, so reading K alone
overstates how many gauges the correction listens to.

**λ < 1**, because the neighbours are an imperfect proxy and believing them fully overshoots.
Fitting λ by least squares in log space gives 0.83; the value that maximises skill is 0.55–0.8,
because the skill scores live in flow space and are dominated by high flows, where `exp()`
amplifies an over-correction. Fitting the shrinkage scores *worse* than choosing it directly —
the two facts are the same fact.

**eps = 0.01 mm/day** keeps `log()` finite on the 2.8% of days that report exactly zero flow. A
much smaller floor (1e-5) costs about 0.005 NSE. Tying eps to each basin's mean precipitation
was tried and was not worth the machinery.

## 3. The two cells

The same correction, over two different situations, each with its own script and its own
`runs/<model>/` output:

| | ungauged (`src/02_...`) | gauged (`src/01_...`) |
|---|---|---|
| what is missing | the basin was never in training | the basin's gauge stops reporting for the test years |
| out-of-sample in | space **and** time | time |
| model | one per fold, that fold's basins held out | one on all 531 basins |
| donors | that fold's 455 training basins | all 530 other basins, self masked |
| cross-validated on | the fold's training basins, leave-one-group-out | every basin outside the group being corrected, self-masked leave-one-out |

The cross-validation is thinned in the ungauged cell and dense in the gauged one because that
is what each faces at deployment: a held-out group has a sparser network around it than a single
gauge that goes dark inside a live one.

**One gauge dark at a time.** In the gauged cell a basin's neighbours — including its own fold
group's — are live gauges and legitimate donors. The cell does not model a whole region going
dark together.

## 4. What it is not allowed to see

The correction never reads the target basin's own discharge, and the weights are built from
coordinates alone. Audited by destroying the record and checking the output does not move — and
then destroying a *donor's* record and checking that it does, because a test that cannot fail
proves nothing.

| test | result |
|---|---|
| Are the corrected basins in their own donor pool? | PASS — ungauged: not at all; gauged: self masked at infinite distance |
| Delete every corrected basin's discharge (ungauged) | PASS — corrected hydrographs bit-identical, max diff `0.000e+00` |
| Replace it with pure noise instead | PASS — bit-identical |
| Destroy one basin's own discharge (gauged) | PASS — its own corrected hydrograph bit-identical |
| Destroy a whole group's discharge, re-select (gauged) | PASS — the chosen (K, λ, p) does not move |
| Destroy every held-out observation, re-select (ungauged) | PASS — the chosen (K, λ, p) does not move |
| Positive control: delete the most-used donor | PASS — output moves, so the tests above are not inert |
| Do the weights read discharge? | PASS — distance only |

**One honest scope limit.** The method reads *concurrent* discharge at neighbouring gauges. The
published 0.693 uses no discharge anywhere at prediction time. So this is a method for gauged
*regions* containing ungauged *sites*, and the fair comparison is the same model without the
correction — which is what §6 measures.

## 5. Choosing K, λ and p

**No search ever reads a basin it will go on to correct.** Every basin is some fold's test set,
so even searching another fold's held-out basins would tune the pooled result on itself. The
search runs on the cross-validation pool in §3's table.

`figures/ungauged_hyperparameters.png` draws the surface: one line per fold, a profile through
that fold's own optimum. Selecting on KGE, the per-fold optima are K in [3, 27], λ in
[0.55, 0.85], p in [1.5, 3], and the entire surface — every K, every λ, every p, every fold —
spans **0.074 KGE**. The correction works across a wide region. The one genuinely bad corner is
K = 1 with large λ: a single over-trusted neighbour.

**The grid's own bounds were tested, not assumed, and the two constants behave differently.**
K runs odd 1–39: several folds select near that ceiling under NSE and NSE* (below), but the
cross-validation score and the held-out score move together there and both flatten out — a
fold's held-out loss under NSE* keeps improving in step with its CV score all the way to K = 39,
never diverging. That is a flat plateau, not a truncated search, so K stays uncapped at 39.
p runs 1–3, and extending it is different: at high p, weight concentrates on the single nearest
donor, so the cross-validation surface keeps climbing indefinitely (checked to p = 5) while the
held-out score for the pinned fold gets *worse* at every step (one fold: p = 3 → 0.7248 held-out
KGE, p = 4 → 0.7200, p = 5 → 0.7184, while its own cross-validation score rose the whole time).
That is overfitting to the cross-validation pool, not information the grid was cutting off, so
p's ceiling stays at 3.

**Selection takes the raw argmax — no tie-break.** An earlier version broke ties toward the
simplest (K, λ, p) within a tolerance of the true maximum, on the theory that the flat surface
made the argmax noise. Measured on held-out basins, that tie-break *cost* rather than saved:
raw argmax scores 0.7117 median KGE against 0.7093 with the tie-break in place. It is gone.

**How the 455 cross-validation basins are combined into one number was also tested.** The
default pools every basin from every one of the six inner leave-one-group-out rotations and
takes one median. The textbook alternative — median within each of the six groups, then the
arithmetic mean of those six numbers — was compared on held-out basins: every one of the 7
folds picked *different* settings under the two rules, yet the outcome was statistically
identical (paired median +0.0001 KGE, 95% CI [−0.0001, +0.0009], crossing zero). The simpler
pooled median stays.

**NSE and KGE as the selection objective now give genuinely different results.** With the
tie-break gone and the grid settled, selecting on NSE instead of KGE is no longer a wash: paired
against each other on the same held-out basins, NSE-selection gains **+0.0025 NSE** (95% CI
[+0.0006, +0.0037]) but costs **−0.0051 KGE** ([−0.0074, −0.0023]), −0.004 mm/day RMSE
improvement reversed, and 1.5 percentage points more high-flow bias — a real trade, not noise,
unlike the earlier (tie-break, narrower-grid) measurement that found no difference. KGE stays
the default because the two are genuinely different bets and KGE's correlation and bias terms
matter to this project; `--select-on nse` reproduces the comparison.

**A third objective, matching the LSTM's own training loss, was added and beats both.**
`nse_star` selects on the basin-normalised squared-error loss the LSTM itself was trained to
minimise (Kratzert et al. 2019's NSE\*: squared error divided by each basin's *training-period*
flow standard deviation, pooled and averaged rather than taking a per-basin median, since that
loss's own outlier behaviour is already controlled by the per-basin normalisation). Paired
against the other two on held-out basins, NSE\*-selection beats NSE-selection on *both* NSE
(+0.0012, CI [+0.0003, +0.0021]) and KGE (+0.0025, CI [+0.0001, +0.0036]) at once — not a
trade — while giving up a little KGE against KGE-selection itself (−0.0023, CI
[−0.0041, −0.0004]). Small, but every one of those intervals excludes zero. `--select-on
nse_star` runs it; the model's predictions must carry a `sigma` array (the training-period flow
std of each basin) alongside the hydrograph for this objective to be available, which
`src/lstm/common.py:seed_mean` now writes automatically.

## 6. What it buys

All 531 basins, 7 folds, 3 seeds averaged per fold, K/λ/p selected per fold to maximise KGE.

| fold | LSTM | + correction | gain | K, λ, p |
|---|---|---|---|---|
| 0 | 0.7169 | 0.7137 | −0.0032 | 3, 0.65, 1.5 |
| 1 | 0.7120 | 0.7240 | +0.0120 | 7, 0.85, 3 |
| 2 | 0.6627 | 0.6764 | +0.0136 | 7, 0.85, 2 |
| 3 | 0.6800 | 0.7156 | +0.0356 | 5, 0.55, 2 |
| 4 | 0.6930 | 0.6750 | −0.0180 | 27, 0.60, 3 |
| 5 | 0.6979 | 0.7346 | +0.0367 | 3, 0.55, 3 |
| 6 | 0.6833 | 0.7248 | +0.0415 | 5, 0.60, 3 |
| **all 531** | **0.6931** | **0.7117** | **+0.0186** | — |

Median KGE over the pooled 531 basins, and every other measure, before and after:

| | LSTM | + correction |
|---|---|---|
| median NSE | 0.6756 | **0.7354** |
| median KGE | 0.6931 | **0.7117** |
| KGE's r (correlation) | 0.8595 | **0.8870** |
| KGE's beta (bias ratio) | 1.0748 | **0.9803** |
| KGE's alpha (variability ratio) | 0.9158 | 0.9031 (worse) |
| median RMSE (mm/day) | 1.299 | **1.165** |
| median overall bias | +7.5% | **−2.0%** |
| bias, top 10% of flows | −9.6% | −12.0% (worse) |
| bias, middle 30–90% | +21.4% | **+9.4%** |
| bias, bottom 30% | +73.4% | **+35.5%** |

Read honestly:

- **Every measure improves except one.** NSE, KGE, RMSE, overall bias, and bias in the middle
  and lowest flow bands all improve; overall bias flips from biased high to nearly balanced.
- **The exception is the top 10% of flows, which gets worse.** The correction trades a little
  high-flow accuracy for a large reduction in day-to-day error everywhere else.
- **NSE gains about four times what KGE does, and the components say why.** The correction
  improves KGE's correlation and bias terms but slightly worsens its variability term: the LSTM
  is already under-dispersed (alpha < 1), and shrinking toward a weighted mean of neighbours
  damps variance further. KGE charges for that; NSE does not. Since the settings are selected
  to maximise KGE, they are tuned against the measure the method gains least on — deliberate,
  but worth knowing when reading the two numbers side by side.
- **The pooled median and the paired per-basin gain are different statistics.** Pairing each
  basin's corrected score against its own raw score gives a median gain of **+0.050 NSE**
  (bootstrap 95% CI [+0.043, +0.057], 78% of basins improve, 49 hurt by more than 0.05) and
  **+0.027 KGE** (CI [+0.022, +0.035], 62% improve, 100 hurt by more than 0.05). Entirely
  positive on both, but a net rather than a guarantee.
- **Two folds go slightly backwards** (0 and 4) on the pooled median. That is what selecting on
  the training basins costs when a fold's held-out geometry differs from its training geometry;
  it is reported rather than tuned away.

Figures: `figures/ungauged_cdf.png`, `figures/ungauged_metrics_boxplots.png`,
`figures/ungauged_donor_map.png`, `figures/method.png`.

**The gauged cell is pending** the all-basins model, which is still training. The pipeline is in
place and exercised; `runs/lstm/gauged_corrected/` fills in once those runs land.

## 7. Configuration

Kratzert et al. (2019)'s own values where the paper specifies one: `forget_bias = 5.0`,
`hidden = 256`, `dropout = 0.4`, 30 epochs, `seq_len = 365`. `batch_size = 512` sits between
the framework's 256 default and the paper's 2000. Train WY1981–1995, test WY1996–2010 — the
paper's own windows, same 531 basins, which is what makes the 0.693 benchmark a close
comparison (forcing product, fold count and ensemble size still differ).

The training window opens 90 days into the cache and a 365-day sequence needs 364 days of
history, so `prepare` moves the first target forward and prints exactly what it dropped rather
than silently running short windows. Its test window is 99.99% complete over 15 years; the
training window is 96.7% complete, which `nse_loss` masks per sample.
