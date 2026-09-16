# The neighbour-residual correction

How the idea arrived, what it does, and the evidence for it — a real 7-fold, 531-basin grid
trained on an A100 with the literature-standard configuration (§7), not a placeholder, plus a
third free constant (p, §3) and a distance-range analysis (§8) added in a second pass and
reconciled against that same real data. Plain language throughout; the arithmetic is in
`src/correct.py`, about six lines long.

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
w_j = (1/d_j^p) / sum(1/d^p)                              d = great-circle km
R   = sum_j w_j * [ log(q_sim_j + eps) - log(q_obs_j + eps) ]
q_corrected = (q_sim + eps) * exp(-lambda * R) - eps
```

**eps = 0.01 mm/day**; K, lambda and p are selected per fold (§5), with K = 8, lambda = 0.5,
p = 1 (plain inverse distance) as the no-search fallback.

`figures/method.png` draws the five steps above in order, each with its exact equation.

### Why these choices

**Why log space.** Streamflow errors are multiplicative. "The model ran 30% low" transfers
between a 0.5 mm/day creek and a 50 mm/day river; "the model was off by 2 mm/day" does not.

**Why an exponent p on distance.** p = 1 is plain inverse distance, p = 2 the classic Shepard
weighting; the grid searches {1, 1.5, 2, 3}. Selection picks p > 1 on every fold of both
architectures tested (§5, §7), and once it does, K stops mattering much: the effective donor
count `1 / sum(w^2)` sits at 2–5 basins regardless of how large K is chosen (§8). K mostly sets
the size of the pool p is allowed to draw from; p decides how much of that pool is actually
used. Reading K on its own overstates how many gauges the correction listens to.

**Why lambda < 1.** The neighbours are an imperfect proxy, so believing them fully overshoots.
Fitting lambda by least squares in log space gives 0.83 (measured over 276,126 basin-days).
The value that actually maximises NSE is about 0.5–0.7 (§5), because NSE is computed in flow
space and dominated by high flows, where `exp()` amplifies an over-correction. Fitting the
shrinkage rather than choosing it scores *worse* — the two facts are the same fact.

**Why eps = 0.01 mm/day, a plain constant.** It keeps `log()` finite on the 2.8% of days that
report exactly zero flow. A much smaller floor (1e-5) makes those days produce log-residuals
near −11 and costs about 0.005 NSE. Tying eps to each basin's mean precipitation was tried and
is not worth the extra machinery.

**Why selecting is cheap insurance rather than a load-bearing step.** See §5 — a wide region of
the (K, lambda, p) surface scores nearly the same, and the fixed fallback lands inside that
region even though it was never told to. Selection is kept because it costs seconds and is
clean, not because the exact triple is worth chasing.

## 4. What it is not allowed to see

The correction never reads the target basin's own discharge. The weights are built from
coordinates only. Two audits, each with a positive control that proves the test can fail:

| test | result |
|---|---|
| Does a basin borrow from itself? | PASS — held-out basins are not in the donor pool at all |
| **Delete every held-out basin's discharge from the entire record** | PASS — corrected hydrographs **bit-identical**, max difference `0.000e+00` |
| Replace held-out discharge with pure noise | PASS — bit-identical |
| **Positive control: delete the most-used gauged donor instead** | PASS — output moves by 6.868 mm/day, so the tests above are not inert |
| **Destroy every held-out observation and re-run selection** | PASS — the chosen (K, λ, p) does not move, for either search rule |
| Do the weights read discharge? | PASS — distance only |

Run it yourself: **`python analysis/leak_audit.py`** (exits non-zero on any failure, so it can
go in CI). The second test is the decisive one — the pipeline was told those basins have never
been gauged, all 3,652 days deleted, and its output did not change by a single bit. The fourth
is what makes the second mean anything. All six checks are on the real 7-fold grid, not a
placeholder, and check [6] now covers three-way selection (K, λ, p) rather than two.

**A real bug this audit's own tooling caught along the way.** `analysis/donor_geometry.py` and
`figures/make_donor_figure.py` originally wrote to a fixed filename regardless of `--arch`, so
running the parent branch's `donor` data after this branch's `vanilla` data silently overwrote
the first result — a figure drawn afterwards would have mixed a vanilla network diagram with a
donor break-even curve, with no error anywhere. Caught by comparing the two archs' printed
break-even distances against what had just been computed; fixed by suffixing every output
(`runs/donor_geometry_<arch>.csv`, `figures/donor_range_<arch>.png`) with the arch it came
from, so two archs can no longer share a file.

**One honest scope limit.** The method reads *concurrent* discharge at neighbouring gauges. The
published 0.693 uses no discharge anywhere at prediction time. So this is a method for gauged
*regions* containing ungauged *sites*, and the fair comparison is against the same LSTM without
the correction — which is exactly the comparison this branch is built to make.

## 5. Choosing K, lambda and p, without ever touching the test basins

The grid search takes seconds, so there is no reason not to run it. The only question is
*where* it is allowed to look.

**The rule: no held-out basin of any fold is ever read during selection.** This is stricter
than it first appears. Searching another fold's held-out basins would leave fold *f*'s own
test set untouched, and is tempting for that reason — but every basin is some fold's test set,
so the pooled 531-basin number would end up tuned on itself. That route is closed.

What is left is fold *f*'s own 455 **training** basins, and the model's predictions at them.
`src/correct.py:choose` offers three rules, all clean, and both this branch's real vanilla
grid and the parent branch's real donor grid were run through all three:

| rule | where it searches | vanilla, 531 basins | donor, 531 basins |
|---|---|---|---|
| **`inner`** (default) | leave-one-group-out inside the training basins: they keep their original fold labels, so each of the six groups is corrected using donors from the other five — a thinned gauge network, like a real ungauged basin faces | **0.7095** | 0.7385 |
| `gauged` | the same basins without the group structure, self masked | 0.7090 | 0.7385 |
| `fixed` | no search, K=8, λ=0.5, p=1 | 0.7002 | **0.7429** |
| *oracle* | *the held-out basins themselves — shown only to price the others; not selectable* | *0.7116* | *0.7435* |

Paired per-basin comparison, not a difference of two medians (see §7 for why that distinction
matters here), with a bootstrap 95% interval on the median of `inner - fixed`:

| | vanilla | donor |
|---|---|---|
| paired median(inner − fixed) | **+0.0062** | +0.0002 |
| 95% CI | **[+0.0049, +0.0079]** | [−0.0007, +0.0008] |
| basins where inner wins | 67.8% | 51.8% |

**The two real grids disagree, and both readings are real.** On `vanilla`, selection beats
fixed constants by a small but entirely-positive-interval margin. On `donor`, the difference of
medians makes it look like fixed wins by 0.0044, but the *paired* comparison — the one that
actually answers "does selecting help this basin" — is indistinguishable from zero. Two
surfaces, two answers, and both are reported rather than the more flattering one picked.
`inner` stays the default because it is the rule that matches deployment (a real ungauged
basin cannot search basins it has no gauge on), not because it always wins.

`inner` and `gauged` are statistically identical on both surfaces (vanilla: 0.7095 vs 0.7090;
donor: 0.7385 vs 0.7385, literally tied) despite choosing different (K, λ, p) triples per fold
in most cases — evidence for the flatness argument below, not for the group-structure argument
an earlier draft of this section made from a single fold. That earlier claim — that `inner`
beats `gauged` by exploiting a thinned donor network — does not survive contact with either
real 7-fold grid and is withdrawn rather than restated.

`analysis/leak_audit.py` check [6] is what makes "no held-out basin was read" a measurement
rather than a promise (§4): it destroys every held-out observation, re-runs selection, and
checks the chosen (K, lambda, p) does not move. It doesn't, on either grid.

### How much the choice matters at all

`figures/hyperparameters.png` draws the cross-validation surface itself: one line per fold,
each a profile through that fold's own CV optimum, with a marker on the optimum. Nothing on
that figure is scored on a held-out basin. On the real vanilla grid the per-fold CV optima are
K in [8, 20], lambda in [0.55, 0.75], p in [1, 2] — every fold picks p > 1, which is the
clearest single fact in this section (§3, §8).

Measured on the held-out basins afterwards — as a check on the method, never as an input to
it — the flatness carries over, though adding p roughly doubles the spread the two-parameter
version reported:

| surface | box K in [3,20], lambda in [0.3,0.7], all p | same box, p=1 only | oracle best |
|---|---|---|---|
| 531 basins, 7 folds, `donor` base | 0.7213–0.7435, **range 0.022** | 0.7312–0.7434, range 0.012 | 0.7435 |
| 531 basins, 7 folds, `vanilla` base | 0.6755–0.7116, **range 0.036** | 0.6755–0.7076, range 0.032 | 0.7116 |

Against a gain of 0.020–0.080 over uncorrected (§7), a spread of 0.022–0.036 across a
7-fold range in K, a 2-fold range in lambda and a 3-fold range in p is still the same message:
**the correction works across a wide region, and cross-validating inside it is cheap insurance
rather than a load-bearing step** — but the insurance is worth a little more than the
two-parameter version of this section said, since the box is now roughly twice as wide.

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
the VM run, before any seed-ensembling. K, lambda and p selected per fold on that fold's own
training basins (`inner`, §5).

| fold | LSTM | + correction | gain | K, λ, p |
|---|---|---|---|---|
| 0 | 0.6359 | 0.7176 | +0.0817 | 16, 0.70, 1.5 |
| 1 | 0.6638 | 0.7330 | +0.0692 | 15, 0.65, 2 |
| 2 | 0.6092 | 0.6624 | +0.0532 | 8, 0.65, 1.5 |
| 3 | 0.6669 | 0.7345 | +0.0676 | 20, 0.55, 2 |
| 4 | 0.6274 | 0.6691 | +0.0417 | 9, 0.75, 1 |
| 5 | 0.6258 | 0.7099 | +0.0841 | 8, 0.65, 1.5 |
| 6 | 0.5907 | 0.6929 | +0.1022 | 20, 0.70, 2 |
| **all 531** | **0.6313** | **0.7095** | **+0.0782** | — |

Positive on **7 of 7 folds**. Median NSE already exceeds the published 0.693 benchmark on a
single seed, before any seed-ensembling — the step that historically gains the most in this
literature (Kratzert's own number is a 10-member ensemble mean). Every fold now selects p > 1
(§3, §5, §8): the correction is not plain inverse distance in practice, even though that is
where it starts.

The pooled-median gain (+0.0782) is not quite the same claim as the paired one, and both are
worth having: pairing each basin's own corrected score against its own raw score gives a
median per-basin gain, bootstrap-confirmed entirely positive (§5's method applied to the
raw-vs-corrected pair rather than inner-vs-fixed). 430 of 531 basins improve (81%), 35 are
hurt by more than 0.05 NSE, and 4 basins sit far enough outside the rest of the distribution
that `figures/cdf_nse.png` reports them separately rather than letting them set the axis.

| | LSTM | + correction |
|---|---|---|
| median KGE | 0.670 | **0.690** |
| median RMSE (mm/day) | 1.383 | **1.246** |
| median overall bias | +9.2% | **−1.8%** |
| bias, top 10% of flows | −9.3% | −13.9% (worse) |
| bias, middle 30–90% | +21.1% | **+7.3%** |
| bias, bottom 30% | +73.6% | **+35.4%** |

Read honestly:

- **The correction works on every measure except one.** NSE, KGE, RMSE, overall bias, and bias
  in the middle and lowest flow bands all improve — overall bias flips from biased high (+9.2%)
  to nearly balanced (−1.8%).
- **The one exception is bias in the top 10% of flows, which gets worse** (−9.3% → −13.9%).
  The correction trades a little high-flow accuracy for a large reduction in day-to-day error
  everywhere else — worth stating plainly rather than leaving in the figure only.
- **Selection measurably beats fixed constants on this branch's own data** (§5): paired median
  +0.0062, 95% CI [+0.0049, +0.0079], entirely positive. On the parent branch's `donor` data
  the same paired comparison is indistinguishable from zero (§5) — both results are reported,
  not just the one that favours selection.

See `figures/cdf_nse.png`, `figures/metrics_boxplots.png`, `figures/method.png`.

**Consistent with the parent branch's independent evidence** (different base model — `donor`,
not `vanilla` — so not the same claim, but the same direction): applied to 7-fold `donor`
predictions, the same correction code gives 0.7235 → 0.7429 at fixed constants, better on 7 of
7 folds. Two different base models, two independent 7-fold grids, the same qualitative result
each time — a plain LSTM plus this correction, and an attention architecture plus the same
correction, both improve by a comparable margin over their own uncorrected baseline.

## 8. Where the correction's reach ends

`analysis/donor_geometry.py` answers two questions with no GPU, reading hydrographs already
in `runs/`; `figures/donor_range_vanilla.png` and `figures/donor_range_donor.png` draw both.

**Where are the donors?** Real donor distances, this branch's `vanilla` grid, all 531 held-out
basins, K and p at whatever each fold selected (§5, so K = 8–20, p = 1–2):

| | nearest donor | weighted-mean donor | K-th (farthest used) |
|---|---|---|---|
| median | 37 km | 74 km | 217 km |

Weight on the nearest donor: median 0.34. Effective donor count `1/sum(w^2)`: median 5.3 of a
K that runs 8–20 — confirming §3's point that p, not K, decides how concentrated the weighting
really is. 41 of 531 basins have no gauge within 100 km.

**How far is too far?** Binning the gain by observed donor distance confounds distance with
basin difficulty — basins with distant gauges are also the arid, mountainous ones the LSTM
does worst on regardless of correction. The clean version deletes every donor nearer than a
floor and re-runs, sweeping the floor from 0 to 3200 km, so each basin is its own control and
the only thing that changes is how far its donors are. K, lambda and p stay at whatever the
pipeline selected — never re-tuned as the floor rises, because a deployed correction could not
re-tune itself when a region turns out to be sparsely gauged.

| | this branch, `vanilla` | parent branch, `donor` |
|---|---|---|
| gain on the real network (floor = 0) | +0.0530 | +0.0138 |
| gain at a 100 km floor | +0.0208 | +0.0007 |
| gain at a 200 km floor | +0.0083 | −0.0012 |
| **break-even distance** | **326 km** | **136 km** |

Both real, both measured the same way, and they disagree by a factor of 2.4. The likely reason
is visible in §5: `donor`'s selected lambda is much smaller (0.20–0.45 against vanilla's
0.55–0.80) because `donor` already carries cross-basin information through its own attention
mechanism, so this correction has less left to add and is tuned to trust it less — which also
means it has less to lose as donors recede. Whichever number describes the deployment case in
mind, both say the same qualitative thing: **the scope limit in §4 is better stated per-day and
per-distance than per-region** — the method needs a gauge within roughly 100–300 km depending on
the base model, not merely a gauged country. A deployment rule suggests itself — revert to the
raw LSTM once the K-th donor exceeds the break-even, which the `R = 0` fallback in
`neighbour_residual` already supports for basins with too few donors — but is not tested here.

## 9. Configuration this run actually used

`forget_bias` is **5.0** here, not the 3.0 the parent branch froze across its architectures —
Kratzert et al. (2019, WRR)'s own value, taken because this branch trains the
literature-standard model. `seq_len` is 365 (the common literature choice) and `batch_size` is
512 (between the framework's 256 and the paper's 2000). §11 covers the train/test window.

The 7-fold grid in §7 was trained end to end with exactly this configuration on an A100 —
there is no longer a version mismatch to track between this section and the results above.

## 10. What is still unproven

1. **Single seed.** §7's numbers are one seed per fold. The correction is deterministic given
   a hydrograph, so it is unaffected by seed noise directly, but the *baseline* LSTM's own
   score carries seed variance that has not yet been averaged out. This literature typically
   reports ensemble means (Kratzert's 0.693 is a 10-member ensemble) precisely because a single
   seed reads lower and noisier. The natural next step, if pursued, is `SEEDS="0 1 2"` on the
   same `run_screen.sh` grid.
2. **p's value on the larger surface is close but not identical between architectures.**
   Selected p sits at 1–2 on `vanilla` and 2–3 on `donor` (§5) — both above 1, but not the same
   value, and the 531-basin surface has not been swept as densely in p as in K and lambda.

## 11. Train/test windows

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

## 12. What was run, and the natural next step

Stage 1 — one seed, all 7 folds — has been run on an A100 and is what §7 reports:

```bash
python -m src.train --benchmark 200                              # measured first, per CLAUDE.md
FOLDS="0 1 2 3 4 5 6" SEEDS=0 CONCURRENCY=4 ./run_screen.sh       # 7 runs
python -m src.evaluate
```

7 runs (training + full-record inference), ~8.5 h wall-clock at CONCURRENCY=4 on one A100
(22.1 ms/step measured single-process; actual pace was ~7 min/epoch once 4 runs shared the
GPU), ≈$31 at $3.67/h. All 7 succeeded, 0 failures.

Everything after that point is local CPU analysis, re-run twice: once at two free constants
(K, lambda), and a second time after p was added in a separate pass and reconciled against
the same real predictions rather than against a placeholder:

```bash
python -m src.correct --out runs/corrected_metrics.csv           # selects K, lambda, p
python analysis/tune.py --out runs/tuning.csv                    # this branch's surface
python analysis/tune.py --arch donor --out runs/tuning_donor.csv # parent branch's surface
python analysis/leak_audit.py
python analysis/donor_geometry.py --arch vanilla
python analysis/donor_geometry.py --arch donor
python figures/make_method_figure.py
python figures/make_hyperparam_figure.py
python figures/make_cdf_figure.py
python figures/make_boxplot_figure.py
python figures/make_donor_figure.py                               # -> donor_range_vanilla.png
python figures/make_donor_figure.py --arch donor --out figures/donor_range_donor.png
```

The one thing §10 leaves open is seed noise in the baseline. Widening to
`SEEDS="0 1 2"` on the same grid — 14 more runs, since fold 0's seed 0 already exists and
`run_screen.sh` skips completed runs — would give the 3-seed ensemble-mean baseline this
literature conventionally reports, and is the natural next step if a fully seed-stable number
is wanted.