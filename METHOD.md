# The neighbour-residual correction

How the idea arrived, what it does, and the evidence for it — a real 7-fold, 531-basin grid,
literature-standard LSTM (§8), one post-processing step on top. Plain language throughout; the
arithmetic is in `src/correct.py`, about six lines long.

---

## 1. The problem

Predicting streamflow at a basin with no gauge — "prediction in ungauged basins", PUB — is
the standard hard case in hydrology. The best published result is a plain LSTM trained across
many basins at once: **median NSE 0.693** over 531 CAMELS-US basins (Kratzert et al. 2019,
`BENCHMARKS.md`). Being ungauged costs about 0.05 NSE against the same model on basins it
trained on, and that 0.05 is the headroom anything new has to compete for.

## 2. How the idea arrived

The starting question was simple: **how much of a basin's error is visible in its neighbours'
errors in the first place?** That is answerable with no training at all, from predictions
already on disk:

> For each held-out basin, take its daily error. Take the distance-weighted mean of its
> gauged neighbours' errors on the same days. Correlate them.
>
> **r = 0.48** (median across basins), so **r² = 0.23**.

About a quarter of the error variance at an ungauged site is already visible in the gauges
around it. Two checks said this was real signal and not an artefact:

- After removing the seasonal cycle from both sides, neighbour agreement is still **r = 0.86**
  — genuine day-to-day information, not the shared seasonality the LSTM already knows from its
  weather inputs.
- The gain decays smoothly with how far the nearest gauge is, and basins with a gauge *under*
  10 km away gain **less** than basins 10–25 km away — the opposite of what "it is really just
  copying an upstream gauge" would produce.

A correlation of 0.48 that costs nothing to exploit is worth pursuing on its own — no new
architecture, no training, just a post-processing step on top of the standard LSTM.

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
p = 1 (plain inverse distance) as the no-search fallback. `figures/method.png` draws these five
steps in order, each with its exact equation.

### Why these choices

**Log space**, because streamflow errors are multiplicative: "the model ran 30% low" transfers
between a 0.5 mm/day creek and a 50 mm/day river, "off by 2 mm/day" does not.

**A distance exponent p.** p = 1 is plain inverse distance, p = 2 the classic Shepard weighting;
the grid searches {1, 1.5, 2, 3}. Every fold picks p > 1 (§5, §7), and once it does, K stops
mattering much — the effective donor count `1/sum(w^2)` sits at 2–5 basins regardless of how
large K is (§7). K sets the size of the pool p is allowed to draw from; p decides how much of
that pool is actually used, so reading K alone overstates how many gauges the correction
listens to.

**λ < 1**, because the neighbours are an imperfect proxy and believing them fully overshoots.
Fitting λ by least squares in log space gives 0.83 (over 276,126 basin-days); the value that
maximises NSE is 0.5–0.7 (§5), because NSE lives in flow space and is dominated by high flows,
where `exp()` amplifies an over-correction. Fitting the shrinkage scores *worse* than choosing
it by NSE directly — the two facts are the same fact.

**eps = 0.01 mm/day**, a plain constant, keeps `log()` finite on the 2.8% of days that report
exactly zero flow. A much smaller floor (1e-5) costs about 0.005 NSE. Tying eps to each basin's
mean precipitation was tried and wasn't worth the extra machinery.

**Selecting (K, λ, p) is cheap insurance, not a load-bearing step** (§5) — a wide region of the
surface scores nearly the same, and the fixed fallback lands inside that region without being
told to.

## 4. What it is not allowed to see

The correction never reads the target basin's own discharge. The weights are built from
coordinates only. Six checks, each with a positive control that proves the test can fail:

| test | result |
|---|---|
| Does a basin borrow from itself? | PASS — held-out basins are not in the donor pool at all |
| **Delete every held-out basin's discharge from the entire record** | PASS — corrected hydrographs **bit-identical**, max difference `0.000e+00` |
| Replace held-out discharge with pure noise | PASS — bit-identical |
| **Positive control: delete the most-used gauged donor instead** | PASS — output moves by 6.868 mm/day, so the tests above are not inert |
| **Destroy every held-out observation and re-run selection** | PASS — the chosen (K, λ, p) does not move, for either search rule |
| Do the weights read discharge? | PASS — distance only |

Run it yourself: **`python analysis/leak_audit.py`** (exits non-zero on any failure, so it can
go in CI). The second test is decisive — the pipeline was told those basins have never been
gauged, all 3,652 days deleted, and its output did not change by a single bit. The fourth is
what makes the second mean anything.

**One honest scope limit.** The method reads *concurrent* discharge at neighbouring gauges. The
published 0.693 uses no discharge anywhere at prediction time. So this is a method for gauged
*regions* containing ungauged *sites*, and the fair comparison is against the same LSTM without
the correction — which is what §6 measures.

## 5. Choosing K, lambda and p, without ever touching the test basins

**No rule ever reads a held-out basin of any fold.** Every basin is some fold's test set, so
even searching another fold's held-out basins would tune the pooled result on itself. What is
left is fold *f*'s own 455 training basins. `src/correct.py:choose` offers three rules:

| rule | where it searches | 531 basins |
|---|---|---|
| **`inner`** (default) | leave-one-group-out inside the training basins — a thinned gauge network, like a real ungauged basin faces | **0.7095** |
| `gauged` | the same basins without the group structure, self masked | 0.7090 |
| `fixed` | no search, K=8, λ=0.5, p=1 | 0.7002 |
| *oracle* | *the held-out basins themselves — priced only, not selectable* | *0.7116* |

Paired per-basin comparison (see §6 for why pairing rather than a difference of medians
matters here), bootstrap 95% CI on the median of `inner - fixed`:

| | vanilla |
|---|---|
| paired median(inner − fixed) | **+0.0062** |
| 95% CI | **[+0.0049, +0.0079]** |
| basins where inner wins | 67.8% |

Selection beats fixed constants by a small but entirely-positive margin. `inner` stays the
default because it matches deployment (a real ungauged basin cannot search basins it has no
gauge on), not because the margin is large — `inner` and `gauged` score statistically
identically (0.7095 vs 0.7090) despite choosing different (K, λ, p) per fold, which is more
evidence for flatness than for a group-structure advantage. `analysis/leak_audit.py` check [6]
is what makes "no held-out basin was read" a measurement: it destroys every held-out
observation, re-runs selection, and confirms the choice does not move.

### How much the choice matters at all

`figures/recon_ungauged_hyperparameters.png` draws the cross-validation surface: one line per
fold, a profile through that fold's own optimum, nothing scored on a held-out basin. The
per-fold CV optima are K in [8, 20], lambda in [0.55, 0.75], p in [1, 2] — every fold picks
p > 1, the clearest single fact in this section (§3, §7).

Measured on the held-out basins afterwards — as a check, never as an input:

| box (K in [3,20], lambda in [0.3,0.7], all p) | oracle best |
|---|---|
| 0.6757–0.7043, range 0.0286 | 0.7053 (K=13, λ=0.80, p=2) |

A 0.0286 spread across a 7-fold range in K, a 2-fold range in λ and a 3-fold range in p is the
same message as §3: the correction works across a wide region, and cross-validating inside it
is cheap insurance rather than a load-bearing step. The one genuinely bad corner is K = 1 with
large lambda — a single over-trusted neighbour.

### Breaking ties toward the simplest triple

The raw argmax over that wide, flat region is not a real preference — it is noise in the
fourth decimal place dressed up as a choice. `src/correct.py:choose` therefore does not take
the argmax directly; it takes `pick(surf, tolerance=TOLERANCE)`, `TOLERANCE = 0.001`: among
every (K, λ, p) scoring within `TOLERANCE` of the true maximum, it keeps the simplest — smallest
K, then λ, then p, in that order (that ordering happens to be exactly what Python's tuple
comparison gives for free, since every surface is keyed `(K, λ, p)`).

K and λ are simpler when smaller for the same reason: K is how much of the donor network the
correction depends on, and λ is how far it shrinks toward the uncorrected model (λ = 0 *is* the
uncorrected model), so smaller is less reliant on external information either way. Measured on
the real grid, letting K and λ range freely within `TOLERANCE` recovers K roughly halved in
every fold (e.g. 35→27, 37→21, 17→9) for a median cost of 0.0000 and a worst case of 0.0009 —
the plateau this buys from is wide.

p is not the same kind of knob, and does **not** get fixed to a constant. Holding K and λ at
their optimum, p alone swings the score by 0.01–0.04 across its range — ten to forty times
`TOLERANCE` — because it sets how concentrated the correction's weight is on the very nearest
donor versus spread across all K of them (effective donor count, `1/Σw²`, roughly halves every
time p doubles). That is a real, not a free, choice: fixing p = 2 in advance and letting K and
λ compensate costs at most 0.0013 when selecting on NSE, but up to **0.0042** when selecting on
KGE (three folds, all of which want p = 3, the steepest the grid offers) — over four times
`TOLERANCE`, and no coincidence: KGE-selection already prefers smaller K for the same reason
(§9 / `EXPERIMENTS.md`, the variability finding) — concentrating weight on fewer, closer donors
fights the same over-damping that hurts KGE's variability term. Smaller p is still preferred
*within* genuinely tied cells (lower variance, more averaged), but the grid keeps searching it,
because for KGE in particular it is doing real work, not redundant work already done by K.

## 6. What it buys

All 531 basins, real 7-fold grid, literature-standard batched LSTM (`period=kratzert`,
`seq_len=365`, `batch_size=512`, `forget_bias=5.0`), **1 seed per fold** — the first stage of
the VM run, before any seed-ensembling (§9). K, lambda and p selected per fold on that fold's
own training basins (`inner`, §5).

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
literature (Kratzert's own number is a 10-member ensemble mean). Every fold selects p > 1
(§3, §5, §7): the correction is not plain inverse distance in practice, even though that is
where it starts.

The pooled-median gain (+0.0782) and the paired per-basin gain are different statistics and
both worth having: pairing each basin's own corrected score against its own raw score gives a
median per-basin gain, bootstrap-confirmed entirely positive. 430 of 531 basins improve (81%),
35 are hurt by more than 0.05 NSE, and 4 basins sit far enough outside the rest of the
distribution that `figures/recon_ungauged_cdf_nse.png` reports them separately rather than
letting them set the axis.

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
  everywhere else.
- **Selection measurably beats fixed constants** (§5): paired median +0.0062, 95% CI
  [+0.0049, +0.0079], entirely positive.

See `figures/recon_ungauged_cdf_nse.png`, `figures/recon_ungauged_metrics_boxplots.png`,
`figures/method.png`.

## 7. Where the correction's reach ends

`analysis/donor_geometry.py` answers two questions with no GPU, reading hydrographs already in
`runs/`; `figures/recon_ungauged_donor_range.png` draws both.

**Where are the donors?** Real donor distances, all 531 held-out basins, K and p at whatever
each fold selected (§5, so K = 8–20, p = 1–2):

| | nearest donor | weighted-mean donor | K-th (farthest used) |
|---|---|---|---|
| median | 37 km | 74 km | 217 km |

Weight on the nearest donor: median 0.34. Effective donor count `1/sum(w^2)`: median 5.3 of a
K that runs 8–20 — confirming §3's point that p, not K, decides how concentrated the weighting
really is. 41 of 531 basins have no gauge within 100 km.

**How far is too far?** Binning the gain by observed donor distance confounds distance with
basin difficulty — basins with distant gauges are also the arid, mountainous ones the LSTM does
worst on regardless of correction. The clean version deletes every donor nearer than a floor
and re-runs, sweeping the floor from 0 to 3200 km, so each basin is its own control. K, lambda
and p stay at whatever the pipeline selected — never re-tuned as the floor rises, because a
deployed correction could not re-tune itself when a region turns out to be sparsely gauged.

| | value |
|---|---|
| gain on the real network (floor = 0) | +0.0530 |
| gain at a 100 km floor | +0.0208 |
| gain at a 200 km floor | +0.0083 |
| **break-even distance** | **326 km** |

**The scope limit in §4 is better stated per-day and per-distance than per-region** — the
method needs a gauge within roughly 300 km, not merely a gauged country. A deployment rule
suggests itself — revert to the raw LSTM once the K-th donor exceeds the break-even, which the
`R = 0` fallback in `neighbour_residual` already supports for basins with too few donors — but
is not tested here.

## 8. Configuration

`forget_bias` is **5.0** — Kratzert et al. (2019, WRR)'s own value, taken because this branch
trains the literature-standard model. `seq_len` is 365 and `batch_size` is 512 (between the
framework's 256 default and the paper's 2000). §10 covers the train/test window.

## 9. What is still unproven, and what running it cost

**Single seed.** §6's numbers are one seed per fold. The correction is deterministic given a
hydrograph, so it is unaffected by seed noise directly, but the *baseline* LSTM's own score
carries seed variance not yet averaged out. This literature typically reports ensemble means
(Kratzert's 0.693 is a 10-member ensemble) precisely because a single seed reads lower and
noisier. The next step, if pursued: `SEEDS="0 1 2"` on the same `run_screen.sh` grid — 14 more
runs, since all 7 seed-0 runs already exist and `run_screen.sh` skips completed ones — giving
the 3-seed ensemble-mean baseline this literature conventionally reports.

Stage 1 (§6's 7 runs, one seed each) trained end to end on an A100 in ~8.5 h wall-clock at
`CONCURRENCY=4`, ≈$31 at $3.67/h, 0 failures. See `README.md` for the exact commands; everything
after training is CPU-only analysis, seconds per script.

## 10. Train/test windows

Two are available, selected with `--period` and recorded in every run's `config.json`; runs
from different windows are refused rather than pooled, since their test axes differ in length.
`kratzert` (WY1981–1995 train, WY1996–2010 test) is the default and what §6 uses — it matches
Kratzert et al. (2019, WRR) exactly, same basins and windows, making the 0.693 benchmark a close
comparison (forcing product, fold count and ensemble size still differ). Its test window is
99.99% qobs-complete over 15 years; the training window is 96.7% complete, which `nse_loss`
masks per-sample, and starts 90 days into the cache (a 365-day sequence needs 364 days of
history), so `prepare` moves the first target forward and prints exactly what it dropped rather
than silently running short windows. The alternative, `complete` (WY1990–2004 / WY2005–2014),
trades that for 100% train-complete at the cost of a weaker literature comparison.
