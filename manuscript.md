# Borrowing the neighbours' errors: a residual post-processing step for streamflow reconstruction at ungauged and discontinued gauges

*Draft manuscript. Authors and affiliations to be added.*

## Abstract

Regional deep-learning models are now the most accurate tool for predicting daily streamflow in
basins without their own gauge. Yet hydrological models' errors, whatever the model, tend to be
shared across nearby basins rather than independent from one to the next. We test whether that
shared structure can be exploited after the fact. At a basin being reconstructed, the concurrent
errors of its nearest gauges are combined into a single distance-weighted signal, and a fraction
of it is removed from the prediction. The step trains nothing, reads no forcing data, never reads
the target basin's own discharge, and runs in milliseconds on any model's output. We evaluate it
over 531 CAMELS-US basins in two reconstruction settings: basins never seen in training, and
basins whose gauge stops reporting for the evaluation period. The primary model is a long
short-term memory (LSTM) network; a differentiable process-based model serves as a transfer test.
*[Quantitative findings and conclusions to be written once results are final.]*

## 1. Introduction

Most of the world's river reaches have never been gauged, and many that were are no longer.
Predicting streamflow at such locations — the Predictions in Ungauged Basins (PUB) problem
(Sivapalan et al., 2003; Hrachowitz et al., 2013; Blöschl et al., 2013) — remains one of the
discipline's defining challenges, and not an academic one. Design floods, water allocation,
reservoir operation, environmental flow standards, and climate impact assessments all require
daily flow records at places where no record exists. The need is growing rather than shrinking.
Operational gauge networks have been contracting for decades across much of the world (Hannah et
al., 2011), so the set of sites where a record is wanted but unavailable keeps expanding at both
ends: new sites that were never instrumented, and old sites whose instruments were retired.

This paper concerns **reconstruction** rather than forecasting: estimating daily flow over a
historical period at a site that lacks its own observations, while neighbouring gauges in the
same region have observations for that period. Two practically distinct cases fall under this
heading, and we treat both. In the **ungauged** case the site has no record at all and the model
has never seen it. In the **discontinued-gauge** case the site was monitored long enough to
contribute to model training, but its gauge stopped reporting and the years of interest must be
filled in. The two differ in what the model knows and in how dense the surrounding network is;
as we show, they are not interchangeable.

The current state of the art for both cases is a regional deep-learning model. Long short-term
memory (LSTM) networks trained jointly on hundreds of catchments outperform locally calibrated
conceptual models, and, more surprisingly, a regional LSTM applied to basins held out of its own
training outperforms conceptual models calibrated *at those very basins* (Kratzert et al., 2018,
2019a, 2019b; Nearing et al., 2021). Differentiable, physics-informed models, in which a
process-based structure is trained end-to-end with neural components supplying its parameters or
fluxes, now reach comparable accuracy while retaining interpretable internal states, and follow
closely behind on ungauged-basin benchmarks (Feng et al., 2022, 2023; Shen et al., 2023; Höge et
al., 2022).

Hydrological model errors, of whatever kind, tend to be spatially structured rather than
independent from basin to basin. A snow regime a model misjudges, an arid runoff threshold it
gets wrong, a karst aquifer it cannot represent, a biased precipitation product shared by every
basin in a region: each leaves a footprint that nearby basins inherit together, whatever the
model and however it was fitted. Model skill is well known to vary regionally rather than
randomly, so on any given day the residuals at nearby basins are not independent draws but
partial observations of a common, locally shared error. That shared error is exactly the
opportunity this paper pursues. At an ungauged site the model's own error is invisible, but at
its gauged neighbours it is directly measurable.

Donor catchments have long been central to ungauged-basin prediction, but they are conventionally
used *before* or *inside* the model rather than after it. Regionalization transfers calibrated
parameters from donors chosen by spatial proximity, physical similarity, or regression on
catchment attributes (Merz and Blöschl, 2004; Parajka et al., 2005; Oudin et al., 2008).
Record-extension and transfer methods instead work on the hydrograph itself, whether through
regression against a reference gauge (Hirsch, 1982), selection of that reference by map
correlation (Archfield and Vogel, 2010), or transfer of flow-duration-curve percentiles from
donors to the target (the QPPQ family). In all of these, the donor supplies the *prediction*, or
the parameters that generate it. Comparatively little attention has gone to the complementary
idea: letting the donor supply the *error*.

That idea is routine in streamflow forecasting, where autoregressive updating and residual error
models exploit the persistence of a model's own recent errors at the target gauge before issuing
the next forecast (e.g., Liu et al., 2012). It does not carry over to reconstruction at an
ungauged site, because there the target's own errors are never observed — not now, not
yesterday, not ever. The only error information that exists is at other gauges. If a model's
residuals are spatially structured, a correction built from concurrent neighbour residuals should
remove part of that shared error without retraining, without additional data, and without ever
reading the target's own record.

We therefore ask:

1. **Do the daily errors of a regional model carry spatial structure** that neighbouring gauges
   can reveal, enough to be worth acting on and beyond the seasonality both basins already share
   through their forcings?
2. **In the ungauged case, does a distance-weighted correction of concurrent neighbour residuals
   improve reconstruction**, on which measures, by how much, and at what cost?
3. **Does the benefit persist in the discontinued-gauge case**, where the basin was seen during
   training and its surrounding network is denser?
4. **Is the correction model-agnostic?** Does it transfer to the output of a differentiable,
   process-based model, or is it repairing something peculiar to the LSTM?

Our expectations are specific enough to be wrong. Because the correction targets the component of
the error that neighbouring basins share, we expect gains concentrated in the correlation and
bias terms of model skill, largest where that shared component is largest. Because shrinking a
prediction toward a neighbourhood consensus reduces its variance, we expect a penalty in the
variability term, the model already being under-dispersed. Because its donor network is denser,
we expect the discontinued-gauge case to gain at least as much as the ungauged one. And because
the step reads only finished hydrographs and gauge coordinates, we expect it to transfer across
model families.

**Scope.** The correction reads concurrent discharge at neighbouring gauges, which a strict PUB
benchmark does not permit. This is therefore a method for *gauged regions containing ungauged
sites*, not for regions without any observations. The appropriate comparison, and the one we make
throughout, is against the same model's uncorrected output, not against published PUB scores
obtained without any concurrent discharge.

## 2. Methods

### 2.1 Data

We use 531 basins from the CAMELS-US large-sample dataset (Newman et al., 2015; Addor et al.,
2017), the standard subset for this benchmark: minimally impaired catchments with complete
records and no substantial regulation. Model inputs are the six Daymet daily forcings (day
length, precipitation, shortwave radiation, maximum and minimum temperature, vapour pressure)
together with 30 static catchment attributes describing climate, topography, soils, geology and
land cover, giving 36 input features per basin-day. Two attributes with missing values across
several basins are dropped. Observed daily discharge in mm/day is the target. Models are trained
on water years 1981–1995 (1 October 1980 to 30 September 1995) and evaluated on water years
1996–2010 (1 October 1995 to 30 September 2010), matching the windows used in the reference PUB
benchmark (Kratzert et al., 2019b). The evaluation record is 99.99% complete.

### 2.2 Prediction models

The primary model is a single-layer LSTM with 256 hidden units, dropout 0.4, forget-gate bias
5.0, and a 365-day input sequence, trained for 30 epochs with gradient clipping under the
basin-normalised squared-error loss, in which each basin's error is divided by its
training-period flow standard deviation so that large and small rivers contribute comparably.
These settings follow Kratzert et al. (2019b) wherever that paper specifies a value and the
NeuralHydrology defaults otherwise; none are tuned here, because the comparison is only
meaningful if the model remains the one the literature already uses. Every configuration is
trained with 5 random initialisations and the ensemble prediction is the day-by-day mean of the
5 members per basin, which is the hydrograph every score in this paper is computed from.

As a secondary, model-transfer test, the identical post-processing is applied to the output of a
differentiable process-based model trained and evaluated on the same basins, periods and splits.
*[Model family, structure and training configuration to be specified.]* The post-processing
reads only finished hydrographs and gauge coordinates, so no part of it is specialised to either
model.

### 2.3 Two reconstruction settings

| | ungauged | discontinued gauge |
|---|---|---|
| what is missing | the basin was never in training | the basin's gauge stops reporting in the evaluation years |
| out of sample in | space and time | time |
| model | one per fold, that fold's basins held out | one model trained on all 531 basins |
| donors available | that fold's 455 training basins | all 530 other basins, the target masked from itself |

For the **ungauged** setting the 531 basins are split at random into 7 folds of roughly 76
basins. A separate model is trained for each fold with that fold's basins excluded entirely, and
each basin is scored only from the model that never saw it. The union of the seven held-out sets
returns an out-of-sample reconstruction for all 531 basins, so no basin is scored by a model
that was fitted to it. For the **discontinued-gauge** setting no such split is required: a single
model is trained on all 531 basins and evaluated on the subsequent 15 years, which is out of
sample in time but not in space. There, the deployment scenario is one gauge going dark at a time
inside an otherwise live network, so a basin's neighbours are legitimate donors and only its own
record is withheld.

### 2.4 The neighbour-residual correction

The correction operates entirely on finished output. For a target basin *i* on day *t*:

1. **Start from the model's prediction everywhere.** The model has already produced a
   hydrograph for every basin, including the gauged ones.
2. **Measure the error at each donor.** At every gauged donor basin *j* both the prediction and
   the observation exist for day *t*, so the model's error there is known. It is taken in log
   space, because streamflow errors are multiplicative: "ran 30% low" transfers between a
   0.5 mm/day creek and a 50 mm/day river, "off by 2 mm/day" does not. A floor of
   ε = 0.01 mm/day keeps the logarithm finite on the 2.8% of days reporting exactly zero flow.
3. **Weight the donors by distance alone.** The K nearest donors by great-circle distance
   receive inverse-distance weights with exponent p, normalised to sum to one. No discharge
   enters the weights.
4. **Combine them into one regional error signal.** The weighted mean of the donors' log
   residuals, R, is the neighbourhood's shared error for that day. Donors that did not report
   are dropped and the surviving weights renormalised, so a silent gauge never reads as a donor
   with zero error; a day with no reporting donor yields R = 0, i.e. no correction.
5. **Apply a fraction of it.** The prediction is shifted in log space by λR and returned to flow
   space. λ < 1 hedges against the neighbourhood being an imperfect proxy for the target.

```
w_ij = (1/d_ij^p) / sum_j (1/d_ij^p)                        d = great-circle km
R_i  = sum_j w_ij * [ log(q_sim_j + eps) - log(q_obs_j + eps) ]
q_i  = (q_sim_i + eps) * exp(-lambda * R_i) - eps
```

Nothing in these three lines is trained, and the target basin's own discharge appears nowhere in
them. The step costs milliseconds per basin.

### 2.5 What K, λ and p control

The correction has exactly three free numbers, and each has a distinct job.

**K, how many gauges are consulted.** A small K trusts the closest few gauges; a large K averages
over a wider neighbourhood, which suppresses the idiosyncrasies of any single gauge but admits
donors that share progressively less with the target. §3.1 shows why this trade has an interior
optimum: residual agreement falls steeply with distance, so donors added beyond the immediate
neighbourhood dilute the signal rather than reinforcing it.

**p, how sharply influence falls off with distance.** p = 1 is plain inverse-distance weighting
and p = 2 the classic Shepard form; larger p concentrates weight on the nearest donors. K and p
are not independent, and the interaction matters for reading the selected values. K sets the
size of the pool, but p decides how much of that pool is really used: because the selected p
exceeds 1 in every fold, the effective number of donors (the participation ratio 1/Σⱼwᵢⱼ²)
remains small even where K is large. A selected K of 27 therefore does not mean the correction
averages 27 gauges; it means 27 gauges are eligible and the nearest few dominate the weighted
mean. Reported K values should be read as a pool size, not as a count of gauges heard.

**λ, how much of the neighbourhood's error to believe.** At λ = 0 the prediction is unchanged
and at λ = 1 the neighbourhood's error is transferred in full. The case for λ < 1 follows
directly from §3.1: the donor signal is only partially correlated with the target's own error,
so it is informative but noisy, and applying all of it transfers the noise along with the
signal. λ is the shrinkage that matches how much the correction believes the neighbourhood to
how much the neighbourhood actually knows.

### 2.6 Selecting K, λ and p without leakage

The three settings are chosen by grid search over K odd from 1 to 39, λ from 0 to 1 in steps of
0.05, and p in {1, 1.5, 2, 3} (1,680 combinations), scored by cross-validation and selected at
the raw maximiser, with no tie-breaking and Kling-Gupta efficiency as the objective. The
governing constraint is that no search may read a basin it will go on to correct: since every
basin is some fold's test basin, tuning on any held-out record would tune the reported result on
itself.

The cross-validation is built to resemble the gauge network each setting will be deployed into.
That network differs between the two settings, so each gets its own design.

In the **ungauged** setting, the fold's held-out basins are excluded from the search entirely.
They are neither scored nor available as donors. The search runs inside that fold's 455 training
basins, which retain their original fold labels and therefore fall into six groups. Each group
in turn is treated as if it were ungauged and corrected using donors drawn only from the other
five, so every pool basin is scored exactly once per candidate setting. Scoring against five of
six groups rather than all of them is deliberate: a genuinely held-out basin faces a thinned
network, and a search conducted against a denser one would select settings tuned for a
neighbourhood the target does not have.

In the **discontinued-gauge** setting the fold labels serve only as a selection unit, since no
basin was held out of training. When a group is being corrected the search runs on the other six
groups under self-masked leave-one-out: every pool basin is corrected using every other pool
basin, with itself excluded at infinite distance. That matches the deployment case of a single
gauge going dark inside a network that remains live.

In both designs the scores from all rotations are pooled across basins and reduced to a single
median per candidate setting, and the maximising combination is then applied to the basins that
were excluded from that search. The cross-validation surfaces are flat across much of the grid
(§3.x), so the correction's behaviour is not strongly sensitive to the exact values selected,
particularly for p.

This isolation is verified rather than asserted. Destroying the observations of every basin
being corrected leaves both the selected (K, λ, p) and the corrected hydrographs bit-identical,
while a positive control that destroys a *donor's* record does move the output, confirming the
test is capable of failing.

### 2.7 Evaluation

Corrected and uncorrected hydrographs are compared at every basin over the same evaluation
period using Nash-Sutcliffe efficiency (Nash and Sutcliffe, 1970), Kling-Gupta efficiency
together with its correlation, variability and bias components (Gupta et al., 2009), root mean
squared error, overall percent bias, and percent bias within three bands of the *observed* flow
distribution (the highest 10%, the middle 30–90%, and the lowest 30%), following the diagnostic
decomposition of Yilmaz et al. (2008). Reporting the components alongside the
composite scores is deliberate: a correction that shifts a prediction toward a neighbourhood
consensus may improve correlation and bias while damping variability, and only the decomposed
scores reveal that trade.

Results are reported two ways, because they answer different questions. The median score across
basins before and after correction describes the sample as a whole; the distribution of the
*paired* per-basin difference, with bootstrap confidence intervals and the fraction of basins
improved or degraded, describes what the correction does to an individual site. The comparison
is made in both reconstruction settings for the LSTM, and repeated in both settings for the
differentiable model to test whether the benefit is model-agnostic.

Alongside the uncorrected and corrected hydrographs we report a third, deliberately naive
reference: the single closest gauge's error applied in full, that is K = 1 and λ = 1 with
nothing selected and nothing tuned. It isolates what the method's two design choices contribute.
Spatial structure in the residuals (§3.1) establishes only that neighbouring gauges carry
information; it does not establish that the information can be used safely. The naive reference
uses the same information without pooling it across donors or shrinking it, so the gap between
that reference and the full correction measures what pooling and shrinkage are worth, separately
from whether the underlying signal exists.

## 3. Results

### 3.1 Model errors are spatially structured

The correction rests on a premise testable before any correction is applied: a regional model's
errors at nearby basins are not independent. We test it on the residual the correction actually
transfers, `log(q_sim + ε) − log(q_obs + ε)`, computed at every one of the 531 basins from the
single model trained on all of them (§2.3, the discontinued-gauge setting). This is out of sample
in time, over the same held-out evaluation years used throughout, but not out of sample in space.
We use every trained-on basin here, rather than only the smaller set held out of training
entirely, because the pattern under test is a property of the model itself, not of any one
basin's training status. Testing it across all 531 trained-on basins gives the densest, most
precise picture of that pattern, and it is exactly this pattern, present throughout a region
regardless of whether the target's own gauge happened to be in training, that licenses
transferring it to a genuinely ungauged neighbour (§1).

Errors are strongly shared between close gauges, and the sharing decays with distance
(Figure 1). Every pair of gauges within 1000 km contributes one correlation, 39,285 pairs in all,
and a locally weighted regression through them falls from 0.53 at a separation of 25 km to 0.28
at 100 km, 0.11 at 500 km and 0.06 at 900 km. The information is therefore local. That is what
justifies both weighting donors by distance and keeping the donor count small: beyond a few
hundred kilometres a donor carries almost nothing about the target's error today.

The spread around that trend is wide. Individual pairs at any separation range from strongly
correlated to mildly anticorrelated, so the decay describes the typical pair rather than a
reliable property of every pair. That is itself informative: it is why the correction pools
several donors and applies only a fraction of what they suggest, rather than trusting any single
neighbour.

This agreement is not merely the seasonal cycle that two neighbouring basins share through their
forcings. Removing each series' day-of-year climatology before correlating (a 31-day circular
smoothing of the mean residual for each calendar day) leaves the fitted curve almost unchanged at
short range, 0.51 against 0.53 at 25 km, while lowering it further out: 0.07 against 0.11 at 500
km. What neighbouring gauges reveal at the distances the correction actually draws from is
therefore day-to-day information about the model's error, not a seasonal pattern the model
already recovers from its own inputs.

This analysis uses nothing but the model's residuals and the gauge coordinates; no part of the
correction enters it. What it establishes is therefore a property of the model's errors, not of
any particular way of exploiting them. At a basin with no gauge, the model's own error is
unobservable, but a quantity substantially correlated with it is measurable at the gauges nearby.
The same check repeated on only the basins held out of training entirely, never seen by any model
at all, shows the identical pattern over a much smaller sample: a somewhat weaker correlation at
short range (0.45 at 25 km against 0.53), consistent with a trained-on basin carrying a little
more basin-specific signal than one the model has never seen, but the same shape and the same
conclusion. The rest of the paper asks how much of that error can be recovered in practice.

**Figure 1.** *Model errors are spatially structured.* Each grey dot is one pair of basins: the
correlation between their two daily log-residual series over the whole evaluation period,
against the great-circle distance between their gauges. All 39,285 pairs within 1000 km are
shown, so the horizontal axis is gauge separation rather than time. The line is a robust
locally weighted regression (Cleveland, 1979) through the cloud, using a fixed 60 km bandwidth
rather than a fixed share of the points, because pairs are far denser at long range than at
short and a density-adaptive span would over-smooth precisely where the decay is steepest.
(`figures/gauged_error_structure.png`)

## 4. Discussion

*[To be written.]*

## 5. Conclusions

*[To be written.]*

## References

*All references below require verification against the original sources before submission.*

Addor, N., Newman, A. J., Mizukami, N., Clark, M. P. (2017). The CAMELS data set: catchment
attributes and meteorology for large-sample studies. *Hydrology and Earth System Sciences*, 21,
5293–5313.

Archfield, S. A., Vogel, R. M. (2010). Map correlation method: Selection of a reference
streamgage to estimate daily streamflow at ungaged catchments. *Water Resources Research*, 46,
W10513.

Blöschl, G., Sivapalan, M., Wagener, T., Viglione, A., Savenije, H. (Eds.) (2013). *Runoff
Prediction in Ungauged Basins: Synthesis across Processes, Places and Scales*. Cambridge
University Press.

Cleveland, W. S. (1979). Robust locally weighted regression and smoothing scatterplots.
*Journal of the American Statistical Association*, 74(368), 829–836.

Feng, D., Liu, J., Lawson, K., Shen, C. (2022). Differentiable, learnable, regionalized
process-based models with multiphysical outputs can approach state-of-the-art hydrologic
prediction accuracy. *Water Resources Research*, 58, e2022WR032404.

Feng, D., Beck, H., Lawson, K., Shen, C. (2023). The suitability of differentiable,
physics-informed machine learning hydrologic models for ungauged regions and climate change
impact assessment. *Hydrology and Earth System Sciences*, 27, 2357–2373.

Gupta, H. V., Kling, H., Yilmaz, K. K., Martinez, G. F. (2009). Decomposition of the mean
squared error and NSE performance criteria: Implications for improving hydrological modelling.
*Journal of Hydrology*, 377, 80–91.

Hannah, D. M., Demuth, S., van Lanen, H. A. J., Looser, U., Prudhomme, C., Rees, G., Stahl, K.,
Tallaksen, L. M. (2011). Large-scale river flow archives: importance, current status and future
needs. *Hydrological Processes*, 25, 1191–1200.

Hirsch, R. M. (1982). A comparison of four streamflow record extension techniques. *Water
Resources Research*, 18(4), 1081–1088.

Höge, M., Scheidegger, A., Baity-Jesi, M., Albert, C., Fenicia, F. (2022). Improving hydrologic
models for predictions and process understanding using neural ODEs. *Hydrology and Earth System
Sciences*, 26, 5085–5102.

Hrachowitz, M., et al. (2013). A decade of Predictions in Ungauged Basins (PUB) — a review.
*Hydrological Sciences Journal*, 58(6), 1198–1255.

Kratzert, F., Klotz, D., Brenner, C., Schulz, K., Herrnegger, M. (2018). Rainfall–runoff
modelling using Long Short-Term Memory (LSTM) networks. *Hydrology and Earth System Sciences*,
22, 6005–6022.

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G. (2019a). Towards
learning universal, regional, and local hydrological behaviors via machine learning applied to
large-sample datasets. *Hydrology and Earth System Sciences*, 23, 5089–5110.

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., Nearing, G. S. (2019b).
Toward improved predictions in ungauged basins: Exploiting the power of machine learning. *Water
Resources Research*, 55, 11344–11354.

Liu, Y., et al. (2012). Advancing data assimilation in operational hydrologic forecasting:
progresses, challenges, and emerging opportunities. *Hydrology and Earth System Sciences*, 16,
3863–3887.

Merz, R., Blöschl, G. (2004). Regionalisation of catchment model parameters. *Journal of
Hydrology*, 287, 95–123.

Nash, J. E., Sutcliffe, J. V. (1970). River flow forecasting through conceptual models part I —
A discussion of principles. *Journal of Hydrology*, 10(3), 282–290.

Nearing, G. S., Kratzert, F., Sampson, A. K., Pelissier, C. S., Klotz, D., Frame, J. M.,
Prieto, C., Gupta, H. V. (2021). What role does hydrological science play in the age of machine
learning? *Water Resources Research*, 57, e2020WR028091.

Newman, A. J., et al. (2015). Development of a large-sample watershed-scale hydrometeorological
data set for the contiguous USA. *Hydrology and Earth System Sciences*, 19, 209–223.

Oudin, L., Andréassian, V., Perrin, C., Michel, C., Le Moine, N. (2008). Spatial proximity,
physical similarity, regression and ungaged catchments: A comparison of regionalization
approaches based on 913 French catchments. *Water Resources Research*, 44, W03413.

Parajka, J., Merz, R., Blöschl, G. (2005). A comparison of regionalisation methods for catchment
model parameters. *Hydrology and Earth System Sciences*, 9, 157–171.

Shen, C., et al. (2023). Differentiable modelling to unify machine learning and physical models
for geosciences. *Nature Reviews Earth & Environment*, 4, 552–567.

Sivapalan, M., et al. (2003). IAHS Decade on Predictions in Ungauged Basins (PUB), 2003–2012:
Shaping an exciting future for the hydrological sciences. *Hydrological Sciences Journal*, 48(6),
857–880.

Yilmaz, K. K., Gupta, H. V., Wagener, T. (2008). A process-based diagnostic approach to model
evaluation: Application to the NWS distributed hydrologic model. *Water Resources Research*, 44,
W09417.
