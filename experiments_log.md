# Experiments

What was run, what came out, and what it meant. Newest first.

<!-- FORMAT AND OWNERSHIP — read before appending
Written by /log-experiment. Append only; never edit a past entry.

Claude drafts "What I ran" and "What came out" from runs/metrics.csv.
The human writes "What it means". Claude never removes the NEEDS YOUR REVIEW
marker — an unreviewed interpretation must stay visibly unreviewed.

Write in plain language. Someone who knows hydrology but not this codebase
should be able to read an entry cold: explain every metric the first time it
appears, no unexpanded acronyms, and numbers carry their meaning ("0.71, up
from 0.68") rather than appearing as bare differences ("+0.03").

If a difference sits inside the seed-to-seed spread, say so. Reporting a
difference that isn't there is how this project stalled the first time.
-->

---

## 2026-10-02 — Quick test: telling the learned correction about the target basin's climate and terrain does not help

**What I ran** — A one-fold check of whether the neural-network correction (version 2, described
in the entry below) improves if it is also told five fixed facts about the target basin:
- aridity (potential evaporation ÷ precipitation);
- the fraction of precipitation falling as snow;
- mean elevation;
- mean slope;
- the logarithm of catchment area.

All five come from the CAMELS attributes already cached for the LSTM, and none is derived from
streamflow. Each was rescaled to mean 0 and spread 1 across the fold's training basins only.
Everything else was identical to version 2: the same donors, the same 40 passes over the data,
dropout, K tried at 1–30, five random starts averaged, and K chosen on the same held-back
validation group. Basins held out entirely, **fold 0 only** (76 basins). 6.8 minutes on the
laptop GPU. The test ran from a throwaway script outside the repo; no code, results or figures
were added. Repo at commit `70987e2`.

The bar was decided before the run: a single fold can only show a large effect, so a change of
0.02 or more in fold 0's KGE or NSE would justify a seven-fold run; anything smaller would not.

**What came out**

Medians over fold 0's 76 held-out basins (KGE and NSE as in earlier entries; 1 = perfect):

| | KGE | NSE |
|---|---|---|
| LSTM, uncorrected | 0.72 | 0.72 |
| network version 2 | 0.73 | 0.74 |
| network version 2 + basin attributes | 0.72 | 0.75 |

- Both changes are 0.01, in opposite directions, inside the five starts' range (KGE 0.72–0.73),
  so this run cannot tell the two variants apart. The 0.02 bar was not met.
- Basin by basin, adding attributes gives a higher KGE in 34 of 76 basins and a higher NSE in 43
  of 76, close to an even split.
- Validation-group KGE for each K was unchanged to two decimals (0.85–0.87 both ways). The
  validation group then chose K=15, against K=15 for version 2 on this fold.
- The network barely uses the attributes. When they are shuffled between basins (each basin
  given another's values), one trained network's validation KGE moves from 0.865 to 0.862.

Follow-up check, no training, using the saved version-2 results for all 531 basins in both
cases: does the error left after the network's correction depend on basin type at all? For each
basin's leftover error, two measures were computed against the five attributes:
- the rank correlation (±1 a perfect relationship, 0 none);
- the share of the basin-to-basin variation the attributes can predict in folds they were not
  fitted on.

| leftover error | strongest correlation with any attribute | share predictable |
|---|---|---|
| overall bias, high-flow bias, variability, volume | at most ±0.16 (weak) | about 0% (−2% to 4%) |
| low-flow bias, basins held out entirely | slope −0.25, aridity +0.18 | 5% |
| low-flow bias, discontinued gauges | slope −0.31, aridity +0.24 | 11% |

- Before correction, the LSTM's overall bias did depend on basin type: with snow fraction,
  elevation and slope it correlated at −0.18 to −0.24 for discontinued gauges. The network
  removes that from the neighbours' errors alone.
- Only low-flow bias keeps a pattern: flat, dry, low-lying basins remain more over-predicted on
  low flows. The network roughly halves it: the predictable share falls from 11% to 5%, and from
  20% to 11%. Low flows barely affect KGE or NSE, so a network trained on KGE has little reason
  to use attributes to fix this.

**What it means**
<!-- NEEDS YOUR REVIEW -->

**What's next**
1. Closed: no seven-fold run of the attribute variant, and no further attribute work. The
   follow-up check shows there is no basin-type pattern left in the errors that KGE and NSE
   measure, so per-donor similarity inputs would have nothing to exploit either. The only
   remaining pattern is in low-flow bias. Using it would need a training objective that rewards
   low flows (e.g. KGE on log flows), which would break the like-for-like comparison with IDW.
2. Superseding items 1 and 3 of the entry below, by the author's decision: `manuscript.md` is
   not to be edited here, and the HBV and hybrid results are not to be re-run or changed. Their
   stored IDW results come from the version-1 settings search. Re-running them with the current
   code would apply the version-2 search.
3. One optional check remains open, not yet run: the settings (IDW) and the network (learned
   correction) are tuned on the same test years they are scored on, using other basins only.
   Tuning on 1995–2002 and scoring on 2003–2010 would show whether that flatters the results.
   If the two-decimal scores agree, it doesn't.

## 2026-10-01 — Second versions of both corrections, LSTM only, both cases: the learned correction ties the distance-weighted one

**What I ran** — Both corrections were revised and re-run for the LSTM in both cases (basins
held out entirely, and discontinued gauges), all seven folds each. The manuscript will report
LSTM results only. The hybrid and HBV models were not re-run, and their IDW results still use the
first-version settings search described below.

IDW (inverse-distance weighting) correction, version 2:
- In the basins-held-out-entirely case, the settings search now corrects every training basin
  from all the fold's other training basins (itself excluded). Previously it used only basins
  from the other groups. A check across all seven folds showed the old design gave the search a
  donor network about 10% sparser than the one the held-out basins actually face: median
  distance to the 3rd/10th/20th nearest donor 92/212/342 km, against 82/189/305 km for held-out
  basins. The new design matches them, at 82/189/305 km. The discontinued-gauge case already
  worked this way.
- The distance-weighting power p is now searched over 0.5, 1, 1.5, 2, 2.5 and 3 (was 1, 1.5, 2
  and 3).
- Settings chosen (K, λ, p), basins held out entirely: (25, 0.55, 3), (3, 0.70, 1.5),
  (7, 0.80, 2.5), (9, 0.60, 2), (5, 0.50, 3), (3, 0.50, 0.5), (21, 0.45, 3) for folds 0–6.
  Discontinued gauges: K=3 in six of seven groups. p=3, the top of the range, was chosen in 7
  of 14 cases. p=0.5 was chosen once in each case.

Neural-network correction, version 2. The inputs and network are unchanged from version 1.
What changed:
- Training: a fixed 40 passes over the training basins, with the step size shrinking smoothly to
  zero, replacing "stop when the validation group stops improving". That rule had been picking
  noise. 40 was chosen once by comparing 20, 40 and 80 passes on validation basins only (all
  within 0.01 of each other; 20 a little lower in the discontinued-gauge case). It was then
  frozen.
- Dropout 0.2: during training, each hidden unit is switched off at random 20% of the time, to
  discourage memorising the training basins.
- The same donor rule as IDW version 2: every pool basin except itself.
- K tried at 1, 3, 5, 10, 15, 20 and 30, with five random starts each, averaged day by day.
- New: the discontinued-gauge case, mirroring IDW's. When group g is corrected, the network
  trains on the other six groups only. Each basin of group g is then corrected from all 530 other
  basins, its own record excluded.
- Runs on the laptop's GPU. Both cases ran side by side, about 2 hours in total. Running them
  one at a time would have been faster.
- Isolation checks passed for both methods and both cases:
  - deleting the held-out basins' observations leaves the chosen settings and corrections
    bit-identical;
  - in the discontinued-gauge case, deleting one basin's observations leaves its own correction
    bit-identical while changing the corrections of basins that use it as a donor;
  - deleting a whole group's observations leaves the trained networks unchanged.

Code on branch `mlp-correction`, uncommitted, on top of commit `5aa74c7` (version 1). The
version-1 outputs were scored for the comparison below and then deleted, so that only version 2
remains in the repo. Version 1 can be regenerated from commit `5aa74c7`.

**What came out**

Medians across all 531 basins, at two decimals. Measures as in the entries below: KGE and NSE
(1 = perfect), RMSE in mm/day, the variability ratio (predicted ÷ observed day-to-day spread;
1 = matched), and the bias on the lowest 30% of flows.

Basins held out entirely:

| | KGE | NSE | RMSE | variability | low-flow bias |
|---|---|---|---|---|---|
| LSTM, uncorrected | 0.70 | 0.69 | 1.27 | 0.92 | +76% |
| IDW version 1 | 0.72 | 0.74 | 1.15 | 0.90 | +40% |
| IDW version 2 | 0.71 | 0.75 | 1.13 | 0.90 | +38% |
| network version 1 | 0.72 | 0.74 | 1.14 | 0.93 | +30% |
| network version 2 | 0.72 | 0.74 | 1.15 | 0.93 | +33% |

Discontinued gauges:

| | KGE | NSE | RMSE | variability | low-flow bias |
|---|---|---|---|---|---|
| LSTM, uncorrected | 0.79 | 0.76 | 1.12 | 0.93 | +55% |
| IDW version 1 | 0.82 | 0.82 | 1.00 | 0.93 | +28% |
| IDW version 2 | 0.83 | 0.82 | 0.98 | 0.92 | +28% |
| network version 2 | 0.83 | 0.82 | 0.98 | 0.96 | +18% |

- **At two decimals, network version 2 and IDW version 2 tie on median KGE and NSE in both
  cases.**
- Basin by basin, network version 2 has the higher KGE in 305 of 531 basins held out entirely
  (57%) and 336 of 531 discontinued gauges (63%). On NSE it is higher in 240 of 531 (45%) and
  270 of 531 (51%).
- Per fold, median KGE of network version 2 against IDW version 2:
  - basins held out entirely: higher in folds 1, 3, 4 and 5 (by 0.01–0.02), equal in folds 0,
    2 and 6;
  - discontinued gauges: higher in groups 3, 4 and 5 (by 0.01–0.03), equal in groups 0, 1 and
    2, lower in group 6 (0.82 against 0.83).
- The five random starts of network version 2 now agree to within about 0.01 per fold (version
  1: up to 0.04), and in the folds above where the network leads, every start leads.
- The network's consistent differences from IDW: it keeps more of the hydrograph's day-to-day
  variability, and it leaves less over-prediction on low flows (+18% against +28% for
  discontinued gauges).
- IDW version 2 is unchanged from version 1 at two decimals. Its basins-held-out-entirely KGE
  moved from 0.716 to 0.714, crossing the 0.715 rounding boundary.
- Network version 2 is unchanged from version 1 at two decimals. It wins on KGE in 268 of 531
  basins, about half.
- K=30 was never chosen by the network, in either case. The chosen K ranged from 3 to 20.

**What it means**
<!-- NEEDS YOUR REVIEW -->

**What's next**
1. `manuscript.md` §2.6 still describes the old settings search and argues for it ("a
   genuinely held-out basin faces a thinned network..."). That argument does not hold: the
   held-out basins' donor distances match the self-masked search, not the thinned one (numbers
   above). It also still states the old p grid ("p in {1, 1.5, 2, 3} (1,680 combinations)"; now
   six values, 2,520 combinations). It needs rewriting by the author.
2. Done in the same session: `README.md`'s headline numbers now show IDW version 2, and the
   version-1 outputs were removed from `runs/` and `figures/`.
3. Before hybrid or HBV results go into a supplement, re-run their IDW corrections with the
   version-2 search: `venv-cpu/bin/python src/corrections/idw/01_gauged.py --model hybrid`,
   `02_ungauged.py --model hybrid --folds-csv data/basin_list_with_folds_hybrid.csv`,
   `01_gauged.py --model hbv`, then `figures/evaluate.py --model <m>`.
4. Not yet tried: basin attributes as extra network inputs (a pre-declared second variant: about
   five target attributes plus donor-to-target elevation and aridity differences). Reported
   alongside, never chosen after the fact.
5. Commit on `mlp-correction` (ask first): the IDW search change in
   `src/corrections/idw/{tuning.py,02_ungauged.py}`, network version 2 in
   `src/corrections/mlp/`, and the new metrics CSVs.

## 2026-10-01 — The learned neighbour correction (first version) on all seven folds, against the distance-weighted one

**What I ran** — The small neural-network correction described in the entry below, unchanged
from that fold-0 test, run on all seven folds of the LSTM's basins-held-out-entirely case: 531
basins the LSTM never trained on, each corrected from its fold's other basins using that fold's
own LSTM. Same inputs (the K nearest gauges' log errors that day, their distances, whether each
reported, time of year, the target's own predicted flow), same network (two layers of 64 units,
no dropout), trained to maximise the Kling-Gupta efficiency. In each fold, the next group of
basins round, (fold + 1) mod 7, is kept back from training and decides when training stops and
which K is used. K was tried at 1, 3, 5, 10, 15 and 20, with five random starts each, so 210
trainings in all, taking 82 minutes of laptop CPU time. The corrected hydrograph for each basin
is the day-by-day average of the five starts at the chosen K, scored once. Code is on branch
`mlp-correction`, uncommitted, branched from commit `2dc321b`. The settings-search and isolation
checks from the fold-0 entry apply unchanged. Fold 0 reproduced that entry's numbers exactly.

**What came out**

KGE (Kling-Gupta efficiency) and NSE (Nash-Sutcliffe efficiency) as defined in the entry below.
RMSE is the root-mean-square daily error in mm/day. The flow-band biases are the percent
over- or under-prediction on the top 10% and bottom 30% of observed flows. Medians across all
531 basins:

| | KGE | NSE | RMSE | high-flow bias | low-flow bias |
|---|---|---|---|---|---|
| LSTM, uncorrected | 0.700 | 0.686 | 1.266 | −9.9% | +76.3% |
| nearest gauge's error applied in full | 0.680 | 0.623 | | | |
| LSTM + IDW correction | 0.716 | 0.743 | 1.154 | −12.3% | +40.4% |
| LSTM + neural-network correction | 0.723 | 0.740 | 1.141 | −9.9% | +29.7% |

Per fold, median KGE, with the five starts' range for the network:

| fold | K chosen (IDW / network) | uncorrected | IDW | network | five starts | NSE IDW / network |
|---|---|---|---|---|---|---|
| 0 | 3 / 3 | 0.718 | 0.725 | 0.734 | 0.730–0.749 | 0.751 / 0.753 |
| 1 | 3 / 5 | 0.711 | 0.726 | 0.731 | 0.716–0.741 | 0.775 / 0.777 |
| 2 | 15 / 20 | 0.674 | 0.688 | 0.694 | 0.686–0.708 | 0.715 / 0.692 |
| 3 | 5 / 5 | 0.690 | 0.698 | 0.718 | 0.714–0.726 | 0.756 / 0.772 |
| 4 | 5 / 15 | 0.695 | 0.692 | 0.697 | 0.680–0.707 | 0.730 / 0.732 |
| 5 | 3 / 20 | 0.702 | 0.733 | 0.746 | 0.739–0.764 | 0.750 / 0.746 |
| 6 | 35 / 10 | 0.701 | 0.716 | 0.729 | 0.685–0.727 | 0.735 / 0.707 |

- **KGE:** the network scores higher than IDW in all seven folds, by 0.005 to 0.020. If the
  two corrections were equally good, all seven going the same way would happen by chance less
  than 1% of the time. Basin by basin it wins 306 of 531 (median difference 0.008). In four
  folds (0, 1, 2, 4) the fold's gap is inside the spread of the five starts. In folds 3 and 5
  even the worst start beats IDW.
- **NSE:** the network is not better. Median 0.740 against IDW's 0.743, winning 245 of 531
  basins (median difference −0.002). It is clearly lower in folds 2 and 6 (by 0.023 and
  0.028) and slightly higher in four others.
- **Components:** timing improves equally under both (correlation 0.890 against 0.891), and
  both remove the volume bias (0.990 against 0.988). The network keeps more of the hydrograph's
  variability (ratio of predicted to observed spread 0.933, IDW 0.897, uncorrected 0.915) and
  leaves a smaller low-flow over-prediction (+30% against +40%).
- **A seed spread for the 531-basin medians is not available:** only the five-start average
  was saved, not each start's hydrographs.
- **Training stopped on a noisy signal:** the epoch at which validation peaked ranged from 1 to
  177 (median 27), and two trainings peaked at the very first epoch, i.e. were barely trained.
  This is the weakness identified in the fold-0 diagnostics: validation KGE jumps by about 0.01
  from one pass over the data to the next, as much as the differences between K values.
- Wall-clock: about 0.2–0.6 minutes per training. The run used about two of the laptop's ten
  CPU cores (torch's per-step work is too small to spread further).

**What it means**
<!-- NEEDS YOUR REVIEW -->

**What's next**
1. Build version 2 of the network correction in `src/corrections/mlp/model.py` and
   `splits()`, as agreed on 2026-10-01:
   - a fixed number of passes over the data with the learning rate decaying smoothly to zero,
     replacing early stopping (try 20, 40 and 80 once, on validation basins only, then freeze);
   - dropout 0.2 between the hidden layers;
   - training basins drawing on all pool basins except themselves, so their donor distances
     match the held-out basins'. The current docstring in `splits()` wrongly says the thinned
     network matches.
   Judge all of these on validation basins, never on held-out scores, then re-run all seven
   folds.
2. Save each random start's corrected hydrographs (not just their average), so the
   531-basin medians get a seed spread.
3. Optionally, a second fixed variant with about five basin attributes for the target (aridity,
   snow fraction, mean elevation, log area, precipitation seasonality) plus each donor's elevation
   and aridity difference from the target. Report it alongside, not as a replacement chosen after
   the fact.
4. Re-run with `venv-cpu/bin/python src/corrections/mlp/02_ungauged.py --seeds 5`, then
   `venv-cpu/bin/python figures/evaluate.py --method mlp --cell ungauged`. Folds can run in
   parallel once each fold writes its own output file.
5. Commit the code on `mlp-correction` (ask first).

## 2026-10-01 — A learned neighbour correction (a small neural network) against the distance-weighted one, on one fold of never-trained-on basins

**What I ran** — A first test of a second correction method. The existing correction
(inverse-distance weighting, "IDW") turns the nearest gauges' errors into a correction with one
fixed rule per fold: the same number of gauges K, the same distance weighting and the same
fraction λ of their error on every day. The new method replaces that rule with a small neural
network that predicts each day's correction. Its inputs, for a target basin on a given day, are
the K nearest gauges' errors that day (the same log errors IDW uses), their distances, whether
each reported that day, the time of year, and the target's own predicted flow that day. So it
can correct differently by season and between high and low flows. The corrected flow is
computed the same way as IDW's: the prediction is scaled by a daily factor. The network is two
layers of 64 units, trained to maximise the Kling-Gupta efficiency (KGE, explained below), the
same score IDW's settings are chosen on. Every other setting was fixed in advance and never
adjusted.

Scope: the LSTM's basins-held-out-entirely case, **fold 0 only**: 76 basins the LSTM never
trained on, corrected from the other 455 basins, using fold 0's own LSTM. The network trains on
those 455 basins only. As in IDW's settings search, each training basin's donor gauges come
from the other groups of basins, and one group (group 1, 76 basins) is kept back to decide when
training stops and which K to use. K was tried at 1, 3, 5, 10, 15 and 20 gauges, with five
random starts of the network at each. The held-out correction is the average of the five starts'
hydrographs at the chosen K, scored once. Fold 0's held-out observations were never read before
scoring. A check confirmed this: deleting them leaves every trained network and every corrected
hydrograph bit-identical, while doubling one donor gauge's observations does change the output.
The 30 trainings took 9.6 minutes on a laptop CPU (0.2–0.6 minutes each). Code is on branch
`mlp-correction`, uncommitted at the time of this run, branched from commit `2dc321b`.

**What came out**

Kling-Gupta efficiency (KGE) and Nash-Sutcliffe efficiency (NSE) both measure how closely
predicted daily flow tracks measured flow: 1.0 is a perfect match, and 0.0 is no better than
always predicting the long-term average. Medians across fold 0's 76 never-trained-on basins:

| | KGE | NSE |
|---|---|---|
| LSTM, uncorrected | 0.718 | 0.721 |
| nearest gauge's error applied in full | 0.662 | 0.602 |
| LSTM + IDW correction (K=3, λ=0.60, distance power 1.5) | 0.725 | 0.751 |
| LSTM + neural-network correction (K=3, average of 5 starts) | 0.734 | 0.753 |

- The validation group chose K=3, the same number of gauges IDW chose for this fold. Validation
  KGE across K: 0.855 (K=1), 0.869 (3), 0.865 (5), 0.867 (10), 0.862 (15), 0.864 (20). It is
  flat from 3 to 20.
- Basin by basin, the network beats IDW on KGE in 45 of the 76 basins (median difference
  0.004). On NSE it beats IDW in 38 of 76 (median difference 0.000), an even split.
- **This run cannot tell the two corrections apart on KGE.** The five starts of the network
  each give held-out KGE between 0.730 and 0.749. That spread of 0.019 is about twice the
  network's 0.009 lead over IDW. On NSE there is no difference at all.
- Where the KGE difference comes from: KGE combines timing (correlation), variability (ratio of
  predicted to observed spread) and volume (ratio of means). Medians across the 76 basins:

  | | correlation | variability ratio | volume ratio |
  |---|---|---|---|
  | uncorrected | 0.869 | 0.907 | 1.082 |
  | IDW | 0.890 | 0.868 | 1.012 |
  | network | 0.890 | 0.912 | 1.007 |

  Both corrections improve timing and volume equally. IDW reduces the hydrograph's variability
  below the uncorrected level; the network does not.
- On the validation basins, which the LSTM did train on, the network raises median KGE from
  0.809 to 0.869, a gain about four times its 0.017 gain on the held-out basins.
- At two settings (K=1 and K=15), the average of the five starts scores *below every individual
  start* on validation KGE: 0.855 against 0.858–0.864, and 0.862 against 0.868–0.871. On the
  held-out basins, the average (0.734) sits near the low end of the five starts (0.730–0.749).
- The epoch at which validation peaked varied widely across starts, from 3 to 75 passes over the
  training basins.

**What it means**
<!-- NEEDS YOUR REVIEW -->

**What's next**
1. Run the other six folds: `venv-cpu/bin/python src/corrections/mlp/02_ungauged.py` with no
   `--fold`, about 10 minutes per fold. Then run `venv-cpu/bin/python figures/evaluate.py
   --method mlp --cell ungauged` and compare against `figures/metrics_ungauged_lstm_idw.csv`
   over all 531 basins: medians for each variant, and per-basin network-minus-IDW KGE and NSE
   (how many basins improve), with the five starts' spread beside it. IDW's gain varies from −0.004 to
   +0.031 across folds, so only the seven-fold result can decide between the methods.
2. Decide whether the five-start average is the right way to combine the network's starts. It
   lowered validation KGE at some K, apparently by smoothing out variability. Scoring each
   start's median KGE separately would be the alternative, and it would also need applying to
   how K is chosen.
3. Code to commit on `mlp-correction` (ask first): `src/corrections/mlp/`, `shared.nearest_gauge`,
   two small `figures/evaluate.py` changes (MLP has no λ/p columns; `run_info.json` gets a
   separate `lstm_mlp` entry so the IDW entry is never overwritten), and `requirements.txt` now
   lists matplotlib and xarray. The local environment is `venv-cpu/` (gitignored).

## 2026-09-21 — Extending the hybrid model's test window from ten years to sixteen (1980-1989 → 1980-1995)

**What I ran** — The hybrid model's raw predictions cover 1980-2014 for all 531 basins, but the
first look at this model (2026-09-19 entry) only pulled out 1980-1989 as the test window. The raw
per-basin files were made available locally, so `src/hybrid/prepare.py`'s `TEST_END` was moved to
1995-12-31, extending the window to sixteen years (1980-01-01 through 1995-12-31, 5,844 days,
versus 3,653 before). Nothing else about the setup changed: the same 425 trained-on basins
(discontinued-gauge case) and 106 never-trained-on basins (basins-held-out-entirely case), the
same neighbour correction, and settings chosen the same cross-validated way, with the basins being
corrected always left out of that search. Both correction scripts were re-run
(`src/01_gauged_correction.py --model hybrid` and `src/02_ungauged_correction.py --model hybrid
--folds-csv data/basin_list_with_folds_hybrid.csv`), followed by `figures/evaluate.py --model
hybrid` and all of the hybrid-model figures. Pipeline run from repo commit `fafcf28`, with the
`TEST_END` edit itself still uncommitted at the time of this run.

**What came out**

Median across each group of basins, before correction and after (Nash-Sutcliffe efficiency and
Kling-Gupta efficiency both measure how closely predicted daily flow tracks measured flow — 1.0 a
perfect match, 0.0 no better than always predicting the long-term average, negative worse than
that):

Discontinued-gauge case, 425 basins:

| | uncorrected | corrected |
|---|---|---|
| Nash-Sutcliffe efficiency | 0.707 | 0.759 |
| Kling-Gupta efficiency | 0.717 | 0.781 |
| root-mean-square error (mm/day) | 1.239 | 1.076 |
| overall bias | −2.1% | −2.7% |
| bias, top 10% of flows | −15.4% | −10.4% |
| bias, middle 30–90% of flows | +6.9% | +4.7% |
| bias, bottom 30% of flows | +42.6% | +21.4% |

Basins-held-out-entirely case, 106 basins:

| | uncorrected | corrected |
|---|---|---|
| Nash-Sutcliffe efficiency | 0.634 | 0.711 |
| Kling-Gupta efficiency | 0.660 | 0.678 |
| root-mean-square error (mm/day) | 1.455 | 1.317 |
| overall bias | −0.5% | −0.5% |
| bias, top 10% of flows | −9.2% | −6.0% |
| bias, middle 30–90% of flows | +10.7% | +8.9% |
| bias, bottom 30% of flows | +33.1% | +14.6% |

Against the 1980-1989 numbers (Nash-Sutcliffe 0.682 → 0.746 discontinued-gauge, 0.622 → 0.719
basins-held-out-entirely), the longer window reads a little higher on both sides before
correction, and the correction still moves both groups the same direction — skill up, bottom-30%
bias roughly halved. There is still only one run of this model (no repeated random starts), so as
before this is one data point, not a settled estimate of the correction's effect size.

**What it means** — 1980-1995 is the window that should have been used from the start; the earlier
1980-1989 window was a mistake. No change to how the earlier entry's finding should be read beyond
that correction — the numbers above are the ones to use going forward.

**What's next** — Figures under `figures/*_hybrid.png` and `figures/*_hybrid.csv` now reflect the
1980-1995 window; the LSTM's and HBV's own files are unchanged. The `TEST_END` change in
`src/hybrid/prepare.py` is not yet committed. If the hybrid model's raw files are ever refreshed
or more realisations become available, re-run `src/hybrid/prepare.py` followed by both correction
scripts and `figures/evaluate.py --model hybrid` again.

---

## 2026-09-20 — Filling in the last two random starts for the ungauged LSTM (3 seeds → 5)

**What I ran** — The ungauged cell (a basin never seen during training, held out of a fold at
both training and test time) had only run three random starts of the LSTM per fold so far, while
the gauged cell (a basin trained on, whose gauge is then treated as if it had gone dark for the
test years) already had five. The remaining two starts, for all seven folds, had finished
training on the Delta cluster and were sitting at `C:\Sandeep\runs\vanilla_fold{0..6}_seed{3,4}`.
Before using them, I checked they were trained with the same settings as the first three (same
network size, same train/test windows, same basin list and fold split — only the random seed and
the git commit they were trained under differ, and that commit is an ancestor of this branch's
current commit `4a6577a`) and that their prediction files have the same shape and basin ordering
as the existing ones. They matched, so I copied all 14 into
`runs/lstm/ungauged/fold{0..6}_seed{3,4}/`, rebuilt the five-start average for each fold
(`runs/lstm/ungauged/_mean/fold{0..6}.npz`), reran the correction's settings search
(`src/02_lstm_ungauged_correction.py`, which never lets a fold's held-out basins influence their
own search), and regenerated the metrics (`figures/evaluate.py`) and every figure that reads the
ungauged cell (the skill curve, the metrics box plots, the cross-validation surface, the map of
where the correction helps or hurts, and the donor map).

**What came out** — The headline numbers moved only slightly once the extra two starts were
averaged in (Nash-Sutcliffe and Kling-Gupta efficiency both measure how closely predicted daily
flow tracks measured flow — 1.0 a perfect match, 0.0 no better than always predicting the
long-term average, negative worse than that; KGE additionally separates out correlation, bias and
variability):

| | 3 starts (previous) | 5 starts (this run) |
|---|---|---|
| median NSE, before correction → after | 0.676 → 0.735 | 0.686 → 0.743 |
| median KGE, before correction → after | 0.693 → 0.712 | 0.700 → 0.716 |
| nearest gauge only (no fitting at all), NSE / KGE | 0.596 / 0.669 | 0.623 / 0.680 |

The correction's own gain — the after-minus-before difference — barely changed: KGE improved by
0.016 with five starts against 0.019 with three, and NSE by 0.057 against 0.059. For comparison,
a single random start's own median KGE, fold by fold, spans roughly 0.03-0.06 from one start to
the next within the same fold (e.g. fold 0's five starts individually score between 0.646 and
0.705) — so the shift from adding two more starts to the average is well inside that per-start
noise, and this is a more stable estimate of the same result, not a different one. The
cross-validated settings search picked different values for how many neighbouring gauges to
borrow from (K) in some folds now that it has a less noisy average to search over (for example
fold 4 moved from 27 donors to 5, fold 6 from 5 to 35), but the range it lands in is unchanged
from what the project already reports: 3-35 donors, borrowing 40-85% of the neighbours' error.
`README.md`'s summary line for the ungauged cell was updated to match (3 seeds → 5 seeds, and the
new NSE/KGE numbers above); `manuscript.md` does not hardcode these numbers so it needed no edit.

**What it means**
<!-- NEEDS YOUR REVIEW -->

**What's next** — The ungauged cell now matches the gauged cell at five random starts each, so
every LSTM figure in the project is on equal footing. Nothing else is blocked on this; it was
purely filling in a gap the compute cluster had already closed. Worth a skim of the regenerated
ungauged figures (`figures/ungauged_cdf.png`, `figures/ungauged_metrics_boxplots.png`,
`figures/ungauged_hyperparameters.png`, `figures/ungauged_change_map.png`,
`figures/ungauged_donor_map.png`) to confirm nothing looks visually different from before, since
the numeric shift is small enough that it shouldn't be.

---

## 2026-09-20 — Adding HBV, a locally-calibrated model with no LSTM in it, to the correction pipeline

**What I ran** — The entry just below this one checked whether the spatially structured error
pattern this project's correction relies on survives outside a regionally-trained LSTM, using
published LSTM simulations. The natural next question: does it also show up in a model family
many hydrology groups actually use in practice -- a traditional, physically-based rainfall-runoff
model, calibrated one basin at a time, with no machine learning in it at all? If so, that would
mean the correction is worth trying for those users too, not just for LSTM users.

HBV is exactly that kind of model, and published simulations already exist: the CAMELS benchmark
models data set (Kratzert et al., collected alongside "Benchmarking a Catchment-Aware LSTM for
Large-Scale Hydrological Modeling", HydroShare 10.4211/hs.474ecc37e7db45baa425cdb4fc1b61e1) has,
for every one of this project's 531 basins, the "upper bound" HBV simulation from Seibert et al.
(2018) -- the day-by-day average of 100 HBV models, each independently calibrated to that one
basin's own measured flow and nothing else, for a ten-year window (1 Oct 1989 to 30 Sep 1999)
held back from the years the models were calibrated on (1999-2008).

Because every basin's HBV model saw only its own record, there's no basin HBV was never trained
on at all, so unlike the earlier hybrid-model check there is only the discontinued-gauge case here
(a basin's own gauge presumed to have gone dark, corrected from its neighbours) and no
never-trained-on case. That's the "gauged" cell this project already has a full pipeline for. HBV
was wired into it the same way the hybrid model was: `src/hbv/prepare.py` reads the published
NetCDFs and writes `runs/hbv/gauged/_mean/predictions.npz`, then the same scripts every other
model uses ran unchanged -- `src/01_lstm_gauged_correction.py --model hbv` (the settings search,
holding each of 7 basin groups out of its own search), `figures/evaluate.py --model hbv`, and the
five gauged figures (skill curves, a map of where the correction helps or hurts, the
cross-validation surface, bias by flow level, and the spatial error-structure check). Observed
flow was always looked up from this project's own cached record, never from HBV's NetCDFs, so
every model in this project is scored against the exact same measurements. Repo commit `e10aeb5`.

**What came out** — First, the same spatial-structure check as the LSTM comparison: HBV's errors
at nearby basins track each other closely, more so than either LSTM version checked before it --

| distance between gauges | this project's regional LSTM | Kratzert et al. regional LSTM | Kratzert et al. single-basin LSTM | HBV (locally calibrated) |
|---|---|---|---|---|
| 25 km | 0.53 | 0.47 | 0.57 | 0.68 |
| 100 km | 0.28 | 0.23 | 0.36 | 0.47 |
| 500 km | 0.11 | 0.06 | 0.14 | 0.28 |
| 900 km | 0.06 | 0.03 | 0.07 | 0.20 |

Then the correction itself, median across all 531 basins (Nash-Sutcliffe and Kling-Gupta
efficiency both measure how closely predicted daily flow tracks measured flow -- 1.0 a perfect
match, 0.0 no better than always predicting the long-term average, negative worse than that; KGE
additionally separates out correlation, bias and variability):

| | HBV, uncorrected | HBV + correction | nearest gauge only (no calibration at all) |
|---|---|---|---|
| NSE | 0.68 | 0.76 | 0.59 |
| KGE | 0.68 | 0.77 | 0.69 |

The correction moved 418 of 531 basins' KGE in the right direction and 395 of 531 basins' NSE.
It also cut two long-standing HBV biases roughly in half or more: HBV underpredicts the highest
10% of flows by a median of 20% before correction, 10% after; and overpredicts the lowest 30% of
flows by a startling median of 77% before correction, 21% after. The settings chosen per basin
group (how many neighbouring gauges to borrow from, how sharply to favour closer ones, how much
of the borrowed error to apply) landed in a similar range to the LSTM and hybrid cases: 3-7
donor gauges, applying 75-90% of the borrowed error.

**What it means**
<!-- NEEDS YOUR REVIEW -->

**What's next** — This is a second, independent line of evidence (after the single-basin-LSTM
check above) that the correction's premise -- and its benefit -- isn't specific to LSTMs or to
regional training. Between the two checks, HBV is the stronger result for the manuscript's
argument that this generalises to what many organisations already run: it's a real,
already-in-use model family, calibrated the way most users actually calibrate it (one basin at a
time, no cross-basin information at all), and the correction still recovers a similar-sized gain
to what the regional LSTM saw. Worth deciding: (1) whether to add an HBV panel alongside the LSTM
and hybrid results in the main figures, or keep it in a supplementary "generalises across model
families" section; (2) whether to also pull in the other benchmark models already sitting in the
same downloaded files (SAC-SMA, VIC, mHM, FUSE) for a broader sweep, now that the download and
`src/hbv/prepare.py` pattern exist to copy; (3) the raw downloaded archives
(`data/raw/kratzert/never-paper.tar.gz`, 5.4 GB, and `data/raw/camels_benchmarks/`, 120 MB) are
still on disk and gitignored -- worth deciding whether to keep them around for (2) or delete them.

---

## 2026-09-20 — Checking whether the spatial error pattern needs a regionally-trained model, using Kratzert et al.'s published simulations

**What I ran** — The paper's opening result (Figure 1) is that a regionally-trained model's daily
errors at nearby river gauges rise and fall together, and fall off the further apart two gauges
are — the pattern the neighbour-borrowing correction relies on. An obvious question a reviewer
would ask: does that pattern only show up because the model was trained on many basins at once,
or would it appear even in a model trained on one basin by itself, with no way to learn anything
about any other basin? Training a one-basin-at-a-time model ourselves would mean 531 separate
trainings, which isn't what this project is about, so instead this reused published simulations
from Kratzert et al.'s 2024 paper "Never train an LSTM on a single basin" (Zenodo record
11247607, a 5.4 GB archive of model runs), which already trained both kinds of model on
CAMELS-US: one regional model (trained on all 531 basins at once) and, separately, one model per
basin (trained on that basin's own record only), each as a 10-model ensemble. Only their finished
day-by-day simulated and measured flow was used — nothing here was trained, and none of it
touches this project's own correction, model runs, or tracked figures/metrics files; it produced
one standalone comparison figure
(`figures/make_error_structure_kratzert.py` → `figures/gauged_error_structure_Kratzert.png`).

529 of the 531 basins in this project's basin list appear in both of Kratzert et al.'s
experiments. For each of the two model types, the 10 ensemble members' simulated flow was
averaged day by day (the same way this project averages its own repeated random starts), then the
same measure Figure 1 uses was computed: for every pair of those 529 basins, how well their daily
errors (in a log scale, the same one the correction transfers) track each other over Kratzert et
al.'s ten-year test period (1989–1999), plotted against how far apart the two gauges are, with a
smoothed trend line through the cloud of pairs (39,072 pairs within 1000 km, same cutoff and
smoothing width as Figure 1). Repo commit `e10aeb5`.

**What came out** — The smoothed correlation at four representative distances, alongside this
project's own regional LSTM (Figure 1) for reference:

| distance between gauges | this project's regional LSTM | Kratzert et al. regional LSTM | Kratzert et al. single-basin LSTM |
|---|---|---|---|
| 25 km | 0.53 | 0.47 | 0.57 |
| 100 km | 0.28 | 0.23 | 0.36 |
| 500 km | 0.11 | 0.06 | 0.14 |
| 900 km | 0.06 | 0.03 | 0.07 |

The single-basin models were not simply performing as well as the regional one and happening to
share its error pattern by coincidence: checked on one sample basin, the regional model's ten
random starts scored 0.82–0.88 on the Nash-Sutcliffe efficiency (a 1.0–to-negative score of how
closely predicted daily flow tracks measured flow, 1.0 a perfect match, 0.0 no better than always
guessing the long-term average), while that same basin's single-basin models scored lower,
0.80–0.82 — consistent with Kratzert et al.'s own published finding that single-basin training
underperforms. So the single-basin models are the weaker models, exactly as expected, but their
errors at nearby gauges track each other *more* closely than the regional model's do, not less.

**What it means**
<!-- NEEDS YOUR REVIEW -->

**What's next** — The manuscript's §3.1 framing ("Model errors are spatially structured") credits
the pattern to regional training; this result says the pattern isn't unique to it, so that framing
needs a second look before submission. Worth deciding: (1) whether to report this Kratzert
comparison as a limitation/caveat alongside Figure 1, since the raw meteorological inputs
(precipitation, temperature) nearby basins share are a plausible alternative source of correlated
error that has nothing to do with how the model was trained; (2) whether it's worth a quick check
of whether the *forcing data itself* is spatially structured in the same way, which would confirm
or rule out that explanation; (3) whichever way this lands, note that it doesn't undermine the
correction itself — the correction only needs the pattern to exist, not for regional training to
be its cause — so the fix is to the explanation in the text, not to the method.

---

## 2026-09-19 — First look at the hybrid model, as a secondary check on the LSTM result

**What I ran** — A separate, non-LSTM model ("hybrid") had already been run outside this repo, on
the same 531 basins, and its day-by-day predictions were dropped into `hybrid/` as one file per
basin, covering 1980–2014. Of those 531 basins, 106 were listed in `hybrid/test_basins.csv` as
ones the hybrid model never trained on at all — the basins-held-out-entirely case — leaving 425
that it did train on — the discontinued-gauge case. Unlike the LSTM side of this project, there
is only one run of this model (no repeated random starts) and only one held-out group (no seven-way
split), so this is a smaller, secondary check, not a full replacement for the LSTM experiment.

The 1980–1989 test years were pulled out of the hybrid model's predictions and organised into
this repo's usual layout (`runs/hybrid/...`), and the same neighbour correction used for the LSTM
was applied to both groups: each of the 425 trained-on basins corrected using errors borrowed
from the other basins' gauges (its own masked out so it can never be a donor to itself), and each
of the 106 never-trained-on basins corrected using errors borrowed from the 425 trained-on
basins' gauges. Settings (how many gauges to borrow from, how sharply to favour closer ones, how
much of the borrowed error to apply) were chosen the same cross-validated way as the LSTM side,
with the basins being corrected always left out of that search. Pipeline run from repo commit
`954afe3`.

Because the held-out 106 basins don't line up with this project's usual seven-way basin grouping,
a second copy of that grouping was made (`data/basin_list_with_folds_hybrid.csv`) that reassigns
those 106 to a new group number, leaving the other 425 in their original groups so the settings
search still holds back one group of trained-on basins at a time the way it always has. A small
option (`--folds-csv`) was added to the basins-held-out-entirely correction script so it can be
pointed at this alternate grouping; the everyday-use script needed no change at all, since it
already treats a basin with no prediction as simply absent from the group scores instead of
breaking.

**What came out**

Median across each group of basins, before correction and after (Nash-Sutcliffe efficiency and
Kling-Gupta efficiency both measure how closely predicted daily flow tracks measured flow — 1.0 a
perfect match, 0.0 no better than always predicting the long-term average, negative worse than
that):

Discontinued-gauge case, 425 basins:

| | uncorrected | corrected |
|---|---|---|
| Nash-Sutcliffe efficiency | 0.682 | 0.746 |
| Kling-Gupta efficiency | 0.699 | 0.766 |
| root-mean-square error (mm/day) | 1.179 | 1.028 |
| overall bias | −4.2% | −2.7% |
| bias, top 10% of flows | −17.3% | −11.2% |
| bias, middle 30–90% of flows | +5.9% | +4.9% |
| bias, bottom 30% of flows | +37.9% | +22.1% |

Basins-held-out-entirely case, 106 basins:

| | uncorrected | corrected |
|---|---|---|
| Nash-Sutcliffe efficiency | 0.622 | 0.719 |
| Kling-Gupta efficiency | 0.650 | 0.680 |
| root-mean-square error (mm/day) | 1.368 | 1.234 |
| overall bias | −1.3% | +0.2% |
| bias, top 10% of flows | −10.9% | −5.4% |
| bias, middle 30–90% of flows | +13.1% | +9.4% |
| bias, bottom 30% of flows | +31.7% | +15.8% |

Both groups moved the same direction the LSTM's did: skill up, and the very large bias on the
lowest 30% of flow days cut roughly in half. There is only one run of this model, so — unlike the
LSTM entries above — there's no repeated-random-start spread to check the gain against; these
numbers should be read as one data point, not as a settled estimate of how well the correction
works on this model.

The fold-isolation check this project always requires after touching the correction code was run
by hand for both hybrid groups: destroying a basin's own measured record left its selected
settings and its corrected prediction unchanged bit-for-bit, while destroying a neighbouring
basin's record (used as a check that the test can actually detect a real change) changed the
correction as expected.

**What it means**
<!-- NEEDS YOUR REVIEW -->

**What's next** — Figures are saved under `figures/*_hybrid.png` and `figures/*_hybrid.csv`,
alongside the LSTM's own (unchanged) files, so the two can be compared side by side. If more
hybrid-model runs (more random starts, more held-out groups) become available later, re-run
`src/hybrid/prepare.py` followed by both correction scripts and `figures/evaluate.py --model
hybrid` to refresh them.

---

## 2026-09-19 — Five random starts of the all-basins model, and clearing out the settings-comparison side files

**What I ran** — The all-basins model — trained on every one of the 531 basins, so its test-year
predictions are only out of sample in time, not in space — finished its second batch of random
starts, bringing that side of the project up to five random starts, matching what the
basins-held-out side already had. The five starts' day-by-day predictions were averaged per
basin, and the same neighbour correction used elsewhere was applied: each basin's prediction
corrected using errors borrowed from all 530 other basins' gauges, with its own gauge masked out
so it can never be a donor to itself. Settings (how many gauges to borrow from, how sharply to
favour closer ones, how much of the borrowed error to apply) were chosen the same way as the
basins-held-out side: search inside six of the seven basin groups at a time, holding the seventh
out as neither a search target nor a donor. Training commit `8e6cc2e`; pipeline run from repo
commit `46d0771`.

Also, since the basins-held-out side's comparison of the three ways to choose those settings
(Kling-Gupta efficiency, Nash-Sutcliffe efficiency, or the model's own training measure) had been
left as an open question, it was settled: Kling-Gupta efficiency stays the standing choice. The
two alternative-settings versions of the basins-held-out correction
(`runs/lstm/ungauged_corrected_nse/` and `_nse_star/`) were deleted to avoid confusion about
which is the reported result; nothing about how those alternatives are computed changed, so they
can be regenerated later if the question is reopened.

**What came out**

Median across the 531 basins, before correction and after:

| | uncorrected | corrected |
|---|---|---|
| NSE | 0.764 | 0.817 |
| KGE | 0.788 | 0.825 |
| RMSE (mm/day) | 1.119 | 0.998 |
| overall bias | +7.2% | +0.0% |
| bias, top 10% of flows | −6.9% | −8.4% |
| bias, middle 30–90% of flows | +16.8% | +6.6% |
| bias, bottom 30% of flows | +55.2% | +28.4% |

The five random starts individually scored a median Kling-Gupta efficiency between 0.771 and
0.781 — the correction's gain of 0.037 is more than three times that spread, so this is a real
improvement, not noise from which random start happened to be averaged in.

Two of the seven basin groups picked a notably wider search than the rest: group 3 picked 21
nearby gauges to borrow from (the rest picked 3), and group 6 picked a gentler distance weighting
(p = 1.5 against 3 elsewhere) — both still within the grid's range, so nothing was cut off.

**What it means**
<!-- NEEDS YOUR REVIEW -->

**What's next**

- The figures (`figures/gauged_*.png`) and `figures/metrics_gauged.csv` / `tuning_gauged.csv` now
  reflect this five-start run; nothing else needs regenerating on the all-basins side.
- The basins-held-out side (ungauged) is still on three random starts. Two more have finished
  training, but only for two of the seven basin groups (fold 0 and fold 1) — the rest are still
  training, and one delivered run (fold 2's fourth start) failed partway through and was not
  used. Import the remaining random starts once every basin group has five, rather than mixing
  group-by-group start counts.
- README/METHOD numbers for the all-basins side should be refreshed from this run once reviewed.

---

## 2026-09-19 — A third way to choose the correction's settings, and cleaning up how the search picks

**What I ran** — The neighbour correction has three settings chosen per data split by searching
inside that split's own training basins: how many nearby gauges to borrow error from (K), how
steeply to favour closer gauges over farther ones (p), and what fraction of the borrowed error
to apply (lambda). Four things about that search were questioned and each was measured rather
than assumed, on the 531 basins the model never saw during training, 7 splits, 3 random starts
of the model averaged day-by-day per basin before scoring. Commit `530249f`.

1. **A third way to choose the settings.** Until now the search picked whichever settings scored
   best on Kling-Gupta efficiency (KGE) or, as an option, Nash-Sutcliffe efficiency (NSE) — two
   different skill scores computed after the fact. A third option was added: choose the settings
   that do best on the same measure the model itself was trained to minimise — a version of NSE
   that divides each basin's error by that basin's own historical flow variability before
   averaging, so naturally big rivers and naturally small streams count equally (Kratzert et al.
   2019 call this "NSE-star"). Each trained model now saves that per-basin variability figure
   alongside its predictions, taken from exactly the years it was trained on.

2. **Removing an unhelpful complication.** When several candidate settings scored almost
   identically in the search, the code used to prefer the plainest one (fewest gauges, smallest
   correction) rather than the literal best-scoring one, on the idea that the difference was
   noise. Measured on held-out basins, this cost accuracy rather than protecting against noise,
   so it was removed — the search now simply takes whichever setting scored best.

3. **How to combine the 455 training basins' scores into one number.** The search rotates
   through the training basins in six groups, each standing in briefly as its own held-out set
   while the other five groups act as donors, and needs to turn those six rounds into a single
   number per candidate setting. The current approach pools every one of the 455 basins together
   and takes one median. The alternative — score each of the six groups separately, then average
   those six numbers — is closer to how this kind of search is usually done elsewhere, so it was
   compared directly. Every one of the 7 splits picked different settings under the two
   approaches, but the two approaches could not be told apart on held-out basins.

4. **How wide the search grid needs to be.** The number of gauges consulted (K) was allowed to
   range up to 39, and the distance-weighting steepness (p) up to 5, to check whether the
   existing grid was cutting off a better answer. For K, several splits did pick values near the
   top of the range, but both the training-basin score and the held-out score kept improving
   together as K grew, never diverging — a sign the search surface is simply flat up there, not
   that a wider grid would help. For p the opposite happened: several splits' training-basin
   score kept climbing as p grew, but their held-out score got *worse* at every step. That is
   the search fitting itself to quirks of its own training basins rather than finding something
   that generalises, so p was capped back at 3, its original limit.

**What came out**

Median across the 531 basins, before correction and after, under three ways of choosing the
settings. RMSE is the typical size of the daily error, in millimetres of flow per day — lower is
better. "Bias" is how far off the total water volume is over a period, in percent, where zero is
best; the three flow-band rows split that by how high the river actually was on the day.

| | uncorrected | chosen on KGE | chosen on NSE | chosen on the training measure (NSE-star) |
|---|---|---|---|---|
| NSE | 0.6756 | 0.7354 | 0.7418 | 0.7425 |
| KGE | 0.6931 | 0.7117 | 0.7027 | 0.7036 |
| RMSE (mm/day) | 1.299 | 1.165 | 1.160 | 1.161 |
| overall bias | +7.5% | −2.0% | −4.1% | −3.4% |
| bias, top 10% of flows | −9.6% | −12.0% | −14.2% | −13.4% |
| bias, middle 30–90% | +21.4% | +9.4% | +8.1% | +8.0% |
| bias, bottom 30% | +73.4% | +35.5% | +34.8% | +35.8% |

Removing the plainest-of-near-ties preference (item 2) raised the headline KGE score from 0.7093
to 0.7117 — a small but real gain from taking a search's actual best answer rather than a
simplified one.

Comparing "chosen on NSE" and "chosen on KGE" basin by basin, which is the sensitive test since
both correct the identical predictions: choosing on NSE gains 0.0025 NSE (95% range
[0.0006, 0.0037]) but costs 0.0051 KGE ([−0.0074, −0.0023]), loses back 0.0043 mm/day of the
RMSE improvement, and adds 1.5 percentage points of high-flow bias — every one of those ranges
stays on one side of zero, so this is a real trade, not noise. That is a different conclusion
from the previous session's reading (logged 2026-09-18), which found the two indistinguishable
— the earlier search still had the plainest-of-near-ties preference in it and a narrower grid,
both of which have since changed, so the two findings are not in conflict, just measured under
different search settings.

Choosing on the training measure (NSE-star) beats choosing on NSE outright on both NSE (+0.0012,
[0.0003, 0.0021]) and KGE (+0.0025, [0.0001, 0.0036]) at once — not a trade against something
else, a plain improvement on both — while giving up a little KGE against choosing on KGE itself
(−0.0023, [−0.0041, −0.0004]). Every one of those ranges also stays on one side of zero, so
these are small but real differences, not noise.

Pooling all 455 training basins into one median (current) versus scoring each of the six
rotating groups separately and averaging those six numbers: paired difference on held-out
basins +0.0001 KGE, 95% range [−0.0001, +0.0009] — crosses zero, the two approaches cannot be
told apart despite picking different settings on every split. The simpler pooled version was
kept.

**What it means**
<!-- NEEDS YOUR REVIEW -->

**What's next**

- KGE remains the default the two correction scripts use unless told otherwise
  (`--select-on nse` or `--select-on nse_star`). The comparison above is now measured, not
  assumed, but which of the three should be the standing choice is a judgement call, not a
  correctness question — flagged for review rather than decided here.
- The figures and README/METHOD numbers all reflect the KGE-chosen run on the settled search
  (median KGE 0.693 → 0.712, median NSE 0.676 → 0.735). The NSE- and NSE-star-chosen runs are
  kept in `runs/lstm/ungauged_corrected_nse/` and `_nse_star/` respectively, so nothing needs
  re-running to look again.
- The all-basins model (no basins held out) is still training. Once it lands in
  `runs/lstm/gauged/seed*/`, the same three-way choose-on-what question should be asked for the
  gauged cell too, not assumed to carry over from here.

---

## 2026-09-18 — Does it matter whether the correction's settings are chosen on NSE or on KGE?

**What I ran** — The correction has three settings: how many nearby gauges to borrow error
from (K), how steeply to favour closer gauges over farther ones (p), and what fraction of the
borrowed error to actually apply (lambda). They are picked per split by searching inside that
split's training basins only. Until now that search maximised one skill score, Kling-Gupta
efficiency (KGE) — a score that separately tracks how well predicted flow correlates with
measured flow, how well its day-to-day swings match the real swings, and how well its long-term
average matches. The question was whether choosing them on Nash-Sutcliffe efficiency (NSE)
instead — a score where 1.0 is a perfect match to measured flow, 0.0 is no better than always
predicting the long-term average, and negative is worse than that — would give a better
correction, because the correction gains far more on NSE than on KGE and was being tuned against
the measure it helps least.

The option to choose either was added back (`--select-on`), and the whole correction was run
twice over the same predictions: 531 basins the model never saw during training, 7 splits,
3 random starts of the model averaged day-by-day per basin before scoring. Nothing was
retrained; both runs correct the identical hydrographs, so this is a like-for-like comparison of
the two choices and nothing else. Commit `34feecf`.

**What came out**

Median across the 531 basins. "Bias" is how far off the total water volume is over a period,
in percent, where zero is best and the sign says which way; the three flow-band rows split that
by how high the river actually was on the day, so "top 10%" is the model's error on the days
the river really was high. RMSE is the typical size of the daily error in millimetres of flow
per day — lower is better.

| | uncorrected model | settings chosen on KGE | settings chosen on NSE |
|---|---|---|---|
| NSE | 0.6756 | 0.7357 | 0.7405 |
| KGE | 0.6931 | 0.7093 | 0.7065 |
| RMSE (mm/day) | 1.299 | 1.174 | 1.163 |
| overall bias | +7.5% | −2.0% | −3.4% |
| bias, top 10% of flows | −9.6% | −12.0% | −13.9% |
| bias, middle 30–90% | +21.4% | +9.4% | +8.7% |
| bias, bottom 30% | +73.4% | +35.8% | +35.6% |

Both choices produce a correction that is far better than no correction at all, and they land in
essentially the same place as each other. Comparing them basin by basin — the sensitive test,
since both corrected the same hydrographs:

| | NSE-chosen minus KGE-chosen | |
|---|---|---|
| NSE | +0.0011, 95% range [−0.0000, +0.0026] | **cannot be told apart** |
| KGE | −0.0032, [−0.0049, −0.0012] | KGE-chosen better |
| RMSE | −0.003 mm/day, [−0.0054, −0.0001] | NSE-chosen better, barely |
| overall bias | −1.0 percentage point | KGE-chosen better (closer to zero) |
| bias, top 10% | −1.1 points | KGE-chosen better |
| bias, middle 30–90% | −0.8 points | NSE-chosen better |
| bias, bottom 30% | −0.8 points | NSE-chosen better, barely |

The headline is the first row: **choosing the settings on NSE does not measurably improve NSE.**
Every difference between the two choices sits in the third decimal place of a skill score or
around one percentage point of bias, against a seed-to-seed spread in the underlying model of
roughly 0.06 KGE within a single split (that split's three random starts scored 0.646, 0.688 and
0.703). The two choices are not distinguishable at the resolution this experiment has.

What does differ is how complicated the chosen settings are. Choosing on KGE picks 3 to 11 donor
gauges per split; choosing on NSE picks 9 to 27 — two to three times as many — to arrive at the
same place. That is the flat-surface finding from earlier sessions showing up again from a new
angle: a wide range of settings scores nearly the same, so the search is cheap insurance rather
than a load-bearing step.

Separately, and worth recording because it was a real error caught in flight: the per-split
progress line printed by the ungauged correction script said "KGE" while printing whichever
score the search had maximised. The saved files were always correct; only the terminal text was
wrong. This is the same mislabelling logged against the old correction script on 2026-09-18, in
new code, and it is now fixed.

**What it means**
<!-- NEEDS YOUR REVIEW -->

**What's next**

- The default is unchanged (settings chosen on KGE) and there is now a measurement behind that
  rather than a preference. `--select-on nse` reproduces the comparison; its output is kept
  separately in `runs/lstm/ungauged_corrected_nse/`, so nothing needs re-running to look again.
- The figures still show the KGE-chosen run. If the review concludes the reporting convention
  should change, only the default in the two correction scripts moves; the figures follow.
- The all-basins model (no basins held out) is still training. When it lands in
  `runs/lstm/gauged/seed*/`, `src/01_lstm_gauged_correction.py` fills in the gauged half of every
  figure, and this same NSE-versus-KGE question should be asked once for that cell too rather
  than assumed to carry over.

## 2026-09-18 — KGE-only reporting going forward, tie-break rerun, and a simpler donor figure

**What I ran** — Follow-on to the tie-break commit from last session (`319f860`): when several
choices of the correction's three settings score almost the same on cross-validation, the
pipeline now takes the simplest of any choice within 0.001 of the best-scoring one, rather than
the single best-scoring one outright. First, the correction pipeline was re-run under that rule
for both cells: the 531 basins the model never saw during training ("PUB"), and the pilot that
simulates a gauge going dark partway through (76 basins on one split). Second, a decision was
made to report only the settings chosen to maximise Kling-Gupta efficiency (KGE) going forward,
not Nash-Sutcliffe efficiency (NSE) as well, so the NSE-maximising rerun was stopped partway
through once that was clear, and only KGE-selected figures are kept from here on; `--metric nse`
still works on every script, so NSE can be regenerated on request. Third, the donor-network
figure was cut down to a single panel: it used to also show how the correction's benefit fades
as donors are pushed farther away, but that panel needed its own expensive per-basin distance
sweep, and the network map alone already makes "are the donors nearby" obvious without it. A
separate coherence figure that nothing else used was deleted outright. Commit `e246490`.

**What came out**

A caveat surfaced while writing this up: `python -m src.correct`'s own on-screen summary line
always reports NSE, even when `--metric kge` was used to choose the settings -- a pre-existing
mislabeling in that script's printout, not something introduced today. The saved CSV is correct
either way (it carries every measure, not just the one being optimised); only the terminal text
is misleading. The numbers below are read from the CSV directly, not the terminal.

*The 531 basins the model never saw, all seven splits pooled, settings chosen to maximise KGE:*

| | before correction | after correction |
|---|---|---|
| median KGE | 0.693 | 0.709 |
| basins improved (by KGE) | -- | 62% |
| basins hurt by more than 0.05 KGE | -- | 100 of 531 |

That gain (0.693 to 0.709) is a little smaller than the 0.693-to-0.712 gain the entry directly
below reported for the pre-tie-break run, and the settings behind it are simpler: the number of
donor gauges consulted per split now ranges 3 to 11, versus 3 to 27 before. That is the tie-break
rule doing what it is meant to do -- when several settings score almost the same in
cross-validation, prefer the plainer one -- and here that trade cost a little KGE. Whether that
trade is worth it is a judgement call, not a correctness question.

*The gauge-going-dark pilot, one split, 76 basins, settings chosen to maximise KGE:*

| | before correction | after correction |
|---|---|---|
| pilot (gauge going dark), KGE | 0.804 | 0.857 |
| same split's real PUB basins, KGE | 0.717 | 0.714 |

The pilot's own gain (0.804 to 0.857) is close to the last time this was logged (0.804 to 0.852)
-- consistent, not a new result. The same split-specific exception noted before reproduces again
here: on this one split, KGE-maximising settings leave that split's real basins' KGE very
slightly worse, not better.

For the PUB cell, the correction stops helping, on average, past 128 km from the nearest usable
donor. `analysis/leak_audit.py` passes cleanly (all six checks) on today's code.

**What it means**
<!-- NEEDS YOUR REVIEW -->

**What's next**

- The `python -m src.correct` terminal printout's NSE-only mislabeling under `--metric kge`
  (found while writing this entry) is still there -- harmless for anything reading the CSV
  directly, but worth a one-line fix if anyone starts trusting the terminal output on its own.
- `README.md` and `METHOD.md` still describe only the NSE-maximising, two-panel-donor-figure
  pipeline; not updated today.
- Figure and `runs/` filenames still carry `_nse`/`_kge` even though only KGE is generated now,
  in case NSE gets regenerated later -- the suffix is what keeps it unambiguous.

---

## 2026-09-18 — running both NSE- and KGE-maximising settings end to end, for both cells

**What I ran** — The correction's three settings — how many nearby gauges to borrow from,
how steeply to favour closer gauges over farther ones, and what fraction of the borrowed error
to apply — are chosen separately for each of the seven splits of the basins, by searching only
that split's own training basins. Until now that search always chose the settings that maximise
Nash–Sutcliffe efficiency (NSE): a skill score where 1.0 is a perfect match to measured flow,
0.0 is no better than always predicting the long-term average, and negative values are worse
than that. This run adds the option to choose settings that maximise Kling–Gupta efficiency
(KGE) instead — a skill score that separately tracks how well predicted flow correlates with
true flow, how well its day-to-day swings match the true swings, and how well its long-term
average matches the true average — and runs the whole pipeline once under each choice, for both
cells: the 531 basins the model never saw during training ("PUB", for predicting flow in
ungauged basins), and the smaller pilot that simulates a gauge going dark partway through (76
basins on one split, standing in for a basin that was gauged during training but has no
measurements for the years being predicted). Commit `cc6456f`.

**What came out**

Root-mean-square error (RMSE) below is the typical size of the daily error, in millimetres of
flow per day — lower is better. Percent bias is how far off the total water volume is over some
period — zero is best. "Distance borrowing stops paying off" is found separately, by deleting
every candidate gauge closer than a growing floor and checking how far out the correction still
helps.

*The 531 basins the model never saw, pooled over all seven splits:*

| | settings chosen to maximise NSE | settings chosen to maximise KGE |
|---|---|---|
| gauges borrowed from, by split | 15 to 37 | 3 to 27 |
| median NSE, before -> after | 0.676 -> 0.742 | 0.676 -> 0.735 |
| median KGE, before -> after | 0.693 -> 0.703 | 0.693 -> 0.712 |
| basins improved | 81% | 78% |
| basins hurt by more than 0.05 NSE | 32 of 531 | 50 of 531 |
| distance borrowing stops paying off | 330 km | 130 km |

Choosing settings to maximise KGE buys a little more KGE (0.712 against 0.703) while giving up
only a little NSE (0.735 against 0.742) — neither choice trades the other measure away, they
land at slightly different points on the same curve. That confirms, through the real pipeline
rather than a one-off script, what a smaller check already found in the entry below ("what kind
of flow error the correction actually fixes").

The KGE-maximising choice consistently borrows from far fewer, closer gauges (as few as 3, never
more than 27) than the NSE-maximising choice (never fewer than 15, up to 37) — which is why its
useful distance is so much shorter, 130 km against 330 km. Borrowing from fewer neighbours means
less averaging, and less averaging means less of the smoothing-out that KGE's day-to-day-swings
component specifically penalises.

Six further measures, median over the 531 basins, before and after correction:

| | before | after, max. NSE | after, max. KGE |
|---|---|---|---|
| KGE | 0.693 | 0.703 | 0.712 |
| RMSE (mm/day) | 1.299 | 1.160 | 1.165 |
| overall percent bias | +7.5% | −4.1% | −2.0% |
| bias on the highest 10% of flows | −9.6% | −14.2% | −12.0% |
| bias on the middle 30–90% of flows | +21.4% | +8.1% | +9.4% |
| bias on the lowest 30% of flows | +73.4% | +34.8% | +35.5% |

*The gauge-going-dark pilot, one split, 76 basins:*

| | settings chosen to maximise NSE | settings chosen to maximise KGE |
|---|---|---|
| settings chosen | 29 gauges, believe 75% of their error | 9 gauges, believe 65% of their error |
| median score, before -> after | NSE 0.794 -> 0.843 | KGE 0.804 -> 0.852 |
| distance borrowing stops paying off | 217 km | 154 km |

The pilot moves the same direction as that split's real basins (0.707 to 0.757 under the
NSE choice), which is reassuring rather than new — it is a smaller version of the same thing.
One exception worth stating plainly: on this split, choosing settings to maximise KGE leaves
that split's *real* basins' KGE very slightly worse, not better — 0.717 down to 0.714. This is
the same split-specific exception already flagged in the entry below; it reproduces under the
real pipeline rather than going away.

**What it means**
<!-- NEEDS YOUR REVIEW -->

**What's next**

- Decide whether KGE-maximising settings are worth reporting as a standing alternative, or
  whether this entry settles the "is the correction only good on the one metric it's tuned for"
  question and NSE-maximising stays the only one carried forward.
- `README.md` and `METHOD.md` still describe only the NSE-maximising pipeline and quote stale
  numbers from the original single-random-start run; they have not been updated to mention that
  a choice of maximising metric now exists.
- Every figure and file under `runs/` and `figures/` now carries an explicit `_nse` or `_kge`
  name so it is never ambiguous which one produced it — the seven now-superseded unsuffixed
  figures were deleted in favour of their replacements.

---

## 2026-09-18 — what kind of flow error the correction actually fixes, a wider search range, and a metric-robustness check

**What I ran** — Four follow-on investigations on top of the three-start ensemble baseline
logged just below, all using the same 21 runs (seven splits of the 531 basins, three random
starts each). Working tree as of commit `7a06022` plus the changes described here, committed
together with this entry.

*1. What is the correction actually correcting?* The correction (`src/correct.py`) borrows the
day-by-day error of nearby basins with a gauge and nudges a given basin's prediction toward
that borrowed error. The open question was whether it only fixes an overall volume bias (the
model running consistently too high or too low), or whether it also touches timing (the flow
arriving on the wrong day) and variability (the model under- or over-reacting to storms). I
decomposed the Kling-Gupta efficiency (KGE, a skill score that separately tracks correlation
with the true flow, the ratio of predicted-to-observed variability, and the ratio of
predicted-to-observed mean) before and after correction, then tested a same-day time shift and
a variance-rescaling step directly.

*2. Does the correction still improve things if it is asked to search over more neighbour
counts?* The correction has three adjustable settings — how many nearby gauges to borrow from
(K), how steeply to weight closer gauges over farther ones (a distance-weighting power), and
how much of the borrowed error to apply (a fraction between 0 and 1). These are chosen
separately for each of the seven splits, searched only over that split's own training basins,
never over the basins being scored. The neighbour-count search previously ran from 1 to 20, and
two splits were landing right at 20, suggesting the true best value lay above the range that had
been tried.

*3. A bug this uncovered.* Widening the search range broke one of the pipeline's scripts.

*4. Does choosing the three settings to maximise KGE instead of NSE trade one measure of skill
for the other?* Nash-Sutcliffe efficiency (NSE) — the project's primary skill score, where 1.0
is a perfect match to measured flow, 0.0 is no better than always predicting the long-term
average, and negative values are worse than that — improves by a lot after correction, while
KGE improves by much less. That gap raises a fair question: if the settings are chosen to
maximise NSE, does the method only look good on that one measure? I re-ran the same
neighbour-selection procedure twice, once choosing settings to maximise each split's NSE (as
the pipeline already does) and once to maximise KGE instead, then scored both choices on both
measures.

**What came out**

*1. Timing is already right; variability is the one real weakness, and it cannot be borrowed
from neighbours.* Splitting KGE into its three parts, averaged over all 531 basins the model
never saw during training:

| part of KGE | before correction | after correction |
|---|---|---|
| correlation with true flow | 0.860 | 0.892 (better) |
| ratio of predicted to true variability | 0.916 | 0.874 (worse — moves further from the ideal of 1.0) |
| ratio of predicted to true mean flow | 1.075 | 0.963 (better, though it slightly overshoots) |
| KGE overall | 0.693 | 0.705 |

Shifting each basin's corrected flow earlier or later by up to three days almost never helped:
69 of 76 basins on one split already had their best alignment at zero days of shift, and even
letting each basin pick its own best shift by looking at its own true flow (a check that is not
a fair method, since a real prediction cannot see the answer first, but useful as a ceiling on
what timing correction could ever be worth) bought only 0.0017 of NSE. Timing is not an
opportunity here.

Variability is a real weakness — rescaling each basin's corrected flow to match its own true
variability (again, an unfair ceiling that looks at the answer) would be worth 0.0302 of NSE,
nearly as much as the entire existing correction's 0.0477 on that split. But this number cannot
be reached honestly: how much a basin's flow under- or over-reacts to storms turned out to
correlate only 0.275 with what its 20 nearest gauges under- or over-react by, far below the
0.547 correlation that makes the existing day-by-day borrowing work. Neighbouring gauges also
consistently underestimate how much rescaling a truly unmonitored basin needs, because those
gauges were basins the model trained on, which react less sharply than a genuinely unmonitored
one. Every realistic way of borrowing a variability fix from neighbours that was tried made
predictions worse, not better.

A related idea — choosing which gauges to borrow from based on how similar their simulated flow
looks, rather than by distance alone — also made predictions worse: it lowered the day-by-day
match between a basin's error and its neighbours' from 0.547 to as low as 0.407. The borrowing
works because nearby basins share the same day's weather-driven mistakes, not because they
behave alike in general, so weighting by behavioural similarity dilutes the very thing that
makes the method work.

To make this visible rather than just numerical, I built one new figure,
`figures/recon_ungauged_coherence.png` (`figures/make_coherence_figure.py`): a map of one day's
model error across all 455 basins with a gauge (deliberately a typical day, not a flattering
one, chosen as the day whose spatial pattern is closest to the test period's median), a curve
showing how strongly two basins' same-day errors agree as a function of the distance between
them, and a bar chart comparing four things by how well a basin's 20 nearest gauges predict
them — a basin's general "flashiness" (used as a check that a real regional pattern shows up at
all, and it does, at 0.586), the day-to-day error the method borrows (0.429-0.567 depending on
distance), each basin's standing bias (0.397), and its variability shortfall (0.185, clearly not
a regional pattern). The agreement between basins' same-day errors decays with distance and
crosses zero close to where the correction itself stops paying off — independent evidence that
the correction's useful range and the distance over which errors stay shared are the same thing.

*2. Widening the neighbour-count search paid off, and moved where borrowing stops helping.*
Every one of the seven splits now chooses to borrow from somewhere between 15 and 37 neighbours
(previously two splits were pinned at the old ceiling of 20). Split-by-split NSE, before and
after correction:

| split | neighbours used | fraction of borrowed error applied | before | after |
|---|---|---|---|---|
| 0 | 35 | 85% | 0.707 | 0.757 |
| 1 | 37 | 70% | 0.703 | 0.761 |
| 2 | 17 | 75% | 0.652 | 0.692 |
| 3 | 23 | 60% | 0.685 | 0.745 |
| 4 | 15 | 70% | 0.670 | 0.702 |
| 5 | 17 | 65% | 0.678 | 0.756 |
| 6 | 37 | 65% | 0.619 | 0.731 |

Across all 531 basins, the median NSE moved from 0.676 (no correction) to 0.742 (corrected) —
essentially the same overall result as before this change (0.743), which is expected: this was
never meant to move the headline number, only to remove an artificial ceiling. The distance at
which the correction stops paying off (found separately by testing how the correction does once
every neighbour closer than a given distance is deleted) moved from 275 km to 330 km — with more
neighbours available to borrow from, the correction can profitably reach further out before
distant, less-relevant ones start hurting more than they help.

*3. The wider search broke a script, now fixed.* The correction's fallback settings — used when
no search is run at all — assumed a neighbour count of 8 would always be one of the values
tried. Once the search only tried odd counts, 8 was no longer one of them, and the script that
compares different selection strategies (`analysis/tune.py`) crashed looking it up. Fixed by
moving the fallback to 7, the nearest value the search actually tries. The project's leak audit
(`analysis/leak_audit.py`, which proves the correction cannot see the basin it is predicting)
still passes cleanly with the wider search.

*4. Choosing settings by KGE instead of NSE does not trade one for the other.* Across all seven
splits, scored on the basins each model never trained on:

| how settings were chosen | NSE | KGE |
|---|---|---|
| no correction | 0.678 | 0.693 |
| chosen to maximise NSE (current default) | 0.745 | 0.704 |
| chosen to maximise KGE instead | 0.740 | 0.716 |

Both ways of choosing the settings improve both measures over no correction at all — maximising
NSE does not come at KGE's expense, and maximising KGE barely costs any NSE while buying more
KGE. The two choices differ from each other only slightly, not in which direction they push
either measure.

One split is a genuine exception worth stating plainly rather than hiding: on split 4, KGE ends
up slightly below its no-correction value under both choices (0.667 and 0.675, versus 0.693
uncorrected), even though NSE improves under both. This lines up with the variability weakness
found in part 1 above — split 4 appears to be a case where the correction's side effect on
variability outweighs its benefit to KGE specifically, even as NSE still improves.

The settings that maximise KGE consistently use far fewer neighbours than the settings that
maximise NSE (as few as 3-7 on most splits, versus 15-37 for NSE), and jump around more from
split to split. Borrowing from fewer neighbours means less averaging, which is a way of fighting
the variability weakness from part 1 by relying on it less. Because the KGE-based choice is less
stable across splits, NSE remains the better choice to search on and report, and this check
shows that choice is not costing KGE anything meaningful in return.

**What it means**
<!-- NEEDS YOUR REVIEW -->

**What's next**

- The variability weakness (the model under-reacting to storms, more so than the observed
  network can help fix) looks like a property of the LSTM itself rather than something the
  correction can ever reach. Worth checking whether it is tied to a specific training setting
  before concluding it is a hard ceiling.
- Split 4 stayed the one exception where correcting for KGE doesn't fully overcome the
  variability weakness. Worth a closer look at what is different about that split's basins.
- Update `README.md` and `METHOD.md`'s reported numbers (0.6313 -> 0.7095 single-start, the
  275 km reach) to reflect the three-start ensemble and the wider neighbour-count search —
  tracked separately since those files are the project's methodology write-up, not its
  experiment log.

---

## 2026-09-18 — three-start ensemble baseline, and where to average the correction

**What I ran** — First entry for the new post-processing approach described in `README.md` /
`METHOD.md`: a standard LSTM (one layer, catchment attributes concatenated in, close to
Kratzert and colleagues' 2019 design) trained separately for each of the seven splits of the
531 basins, three times per split from different random starts (21 runs total, all vanilla, no
correction applied during training). Working from commit `7e6fdc7`.

For each basin a split never trained on, I then applied the neighbour-based correction in
`src/correct.py`: borrow the day-by-day error of the nearest gauged basins, weighted by
distance, and nudge the prediction by a fraction of that borrowed error. The number of
neighbours borrowed from, the distance weighting, and the fraction of the error applied were
each chosen separately per split using only that split's own training basins, never the basins
being scored, so the choice cannot leak information from the basins under test.

Two questions were checked:
1. Does averaging three random starts together, before scoring, beat scoring a single start alone?
2. Should the correction be applied before or after averaging the three random starts?

Every number below follows the ensemble-mean convention: the three starts' predicted flow are
averaged day by day first, and Nash–Sutcliffe efficiency (NSE) is computed once from that
averaged hydrograph. NSE measures how closely predicted daily flow tracks measured flow: 1.0 is
a perfect match, 0.0 is no better than always predicting the long-term average flow, and
negative values are worse than that. This matches how the published reference number is built
(Kratzert and colleagues, 2019, report 0.693 this way; see `BENCHMARKS.md`) and is the only
convention this project uses for headline numbers.

**What came out**

*Does averaging three random starts help, before any correction?*

| | median NSE across the 531 basins held out of training |
|---|---|
| one random start alone (average of the three starts' own medians) | 0.631 |
| three random starts averaged into one hydrograph | 0.676 |

The three individual starts scored 0.631, 0.631, and 0.630 on their own — a spread of just
0.001, so any one start is about as good as any other. Averaging them into one hydrograph
before scoring raised the median by 0.045, far outside that spread, so this is a real effect
and not noise — the same variance-reduction behaviour widely reported for averaging repeated
LSTM runs.

*Correct before or after averaging the three starts?* Same per-split choice of neighbour count,
distance weighting, and borrowed-error fraction used in both cases:

| approach | median NSE | mean NSE |
|---|---|---|
| average the three starts first, then correct once (what `src/correct.py` already does) | 0.743 | 0.604 |
| correct each start separately, then average the three corrected hydrographs | 0.742 | 0.598 |

The gap between these two (0.001 on the median) is far smaller than the 0.045 spread from
averaging in the first table, and correcting-then-averaging only came out ahead on 259 of the
531 basins (49%) — essentially a coin flip. This difference cannot be told apart from noise;
the two approaches are best treated as equivalent, and the cheaper one (average first, correct
once — a third of the compute, one correction pass per split instead of three) is the one to
keep.

The correction itself, applied the cheap way, raised the median from 0.676 (no correction) to
0.743, and improved 81% of the 531 basins; 31 basins got worse by more than 0.05 NSE. Split by
split:

| split | neighbours borrowed from | fraction of borrowed error applied | distance weighting | before correction | after correction |
|---|---|---|---|---|---|
| 0 | 20 | 0.75 | 2.0 | 0.707 | 0.755 |
| 1 | 12 | 0.65 | 2.0 | 0.703 | 0.758 |
| 2 | 16 | 0.75 | 1.5 | 0.652 | 0.694 |
| 3 | 17 | 0.55 | 2.0 | 0.685 | 0.748 |
| 4 | 14 | 0.70 | 1.5 | 0.670 | 0.702 |
| 5 | 16 | 0.65 | 2.0 | 0.678 | 0.756 |
| 6 | 18 | 0.70 | 2.0 | 0.619 | 0.735 |

**What it means**
<!-- NEEDS YOUR REVIEW -->

**What's next**

- Decide whether to run more random starts per split (five to ten, matching the size of the
  published ensemble) to see whether the uncorrected baseline closes more of the gap to the
  published reference of 0.693. The three individual starts were nearly identical to each
  other, so further starts may run into diminishing returns, but this has not been tested.
- The averaging-order question is settled: always average the random starts first, then correct
  once. It is numerically tied for best and three times cheaper to compute.

---

## 2026-09-15 — closing the project

**Why it stops here.** Across all seven splits of the basins — every one of the 531 held out
exactly once — the best variant, `donor`, reaches a median Nash–Sutcliffe efficiency of 0.724 on
basins it never saw in training. The published reference for a plain model trained across many
basins is 0.693 (Kratzert and others, 2019; see `BENCHMARKS.md`). An improvement of that size,
bought at the cost of needing a live gauge network reporting on the prediction day, is not worth
publishing, and it is not what the project set out to find.

Three things compound that. First, `donor` beats the plain model partly because it is *given*
more: it sees 455 neighbouring discharge measurements on the day it predicts, so the comparison
is about information, not architecture, and the honest framing of any gain has to say so.
Second, `spatial` — attention with no discharge attached, the pure "can a model retrieve useful
neighbours at all" test — has never beaten the plain model on any split, which means the
retrieval itself is the weak part rather than the value it carries. Third, the closest prior
work (Wi, Gupta and Steinschneider 2025) reached the same territory first.

**The idea that was tested and did not settle it.** `donor_flow`, logged below: giving attention
the neighbours' flow alone. It matched `donor` to within less than the spread between random
starts. Its only visible effect was to rescue about three badly-failing basins out of 76, which
at that sample size cannot be separated from noise.

One check that would have sharpened this was never made. The idea list made the second idea
conditional on re-running `analysis/donor_diagnostics.py` after `donor_flow` — on whether
attention, once its borrowed quantity was simplified, had started retrieving from nearer basins
or was still reaching just as far. That diagnostic needs no graphics card and the run outputs it
reads are all in `runs/`, so it remains answerable by anyone who picks this up.

**The idea that was never run**, recorded here because the reasoning is worth keeping even
though the work stopped. Borrowing measured flow only helps at basins within roughly 40 km of a
monitored neighbour, as the first screen found, yet attention is left to discover that
preference entirely from the data and spreads its weight much more widely than that. The
proposal was to bias attention toward nearby basins directly, instead of hoping it learns to:

    logits = logits - dist / tau        # dist = great-circle km, fixed; tau one learned scalar

with `tau` started at 45 km, a figure two independent routes agreed on: a by-hand sweep gave
45–65 km, and the abandoned `donor_gated` variant learned 42 km on its own. The distance matrix
already exists in `analysis/donor_diagnostics.py`.

Two details make it worth writing down rather than rediscovering. Appending latitude and
longitude to the query and key vectors does **not** achieve this, and the reason is not degrees
versus kilometres: a dot product is not a distance, so it ranks a basin 1500 km away in the same
compass direction above one 100 km away, and gives basins near the middle of the country no
preference at all. And the experiment needs its zero-parameter control run alongside it —
`logits = -dist / 45.0`, attention with nothing left to learn — because only that answers
whether *learning* the retrieval beats simply *knowing* it. The known risk is that sharper
attention reproduces `donor_gated`'s collapse from 24 effective donors to 4, so the mean donor
count has to be recorded either way.

This entry closes the experiment log. The repository is left in a state where every number in
this file can be regenerated from the committed code and the runs it names.

---

## 2026-09-15 — donor_flow: giving attention the neighbours' flow directly, and nothing else

**What I ran** — one model variant, `donor_flow`, on the first split of the basins, with three
random starts of the same model. This is the first of the two ideas written up in the project's
idea list: instead of letting each basin borrow a 256-number summary of its neighbours *plus* a
projection of their recent flow, the borrowed quantity becomes the neighbours' recent flow
anomaly alone — seven days of it, standardised against each neighbour's own long-run behaviour —
handed straight to the part of the model that predicts flow. The idea was that mixing a
component worth nothing (`spatial`, which borrows only the neighbours' state, has never beaten
the plain model) with a component worth something made it impossible to say which was doing the
work, so this separates them.

Everything else was held identical to the other variants: 256 hidden units, a 270-day input
window, 30 passes through the training years, the same learning-rate schedule and the same loss.
Each start took 2 h 20 min on the rented A100 graphics card, three at once, from commit
`26cab49`. That brings the project to 30 completed runs in total.

**What came out** — the measure below is the Nash–Sutcliffe efficiency, which says how closely
predicted daily flow tracks measured flow: 1.0 is a perfect match, 0.0 means the model is no
better than always predicting the long-term average, and negative values are worse than that.
Every number is the median basin, scored from the ensemble-mean hydrograph — the three starts'
day-by-day predictions are averaged first, then the measure is computed once.

All four variants on the first split, so the basins are the same throughout:

| model variant | trained basins, unseen years | basins never seen in training |
|---|---|---|
| plain model (`vanilla`) | 0.775 | 0.709 |
| borrows neighbours' state (`spatial`) | 0.768 | 0.702 |
| borrows state and flow (`donor`) | 0.820 | 0.743 |
| borrows flow only (`donor_flow`) | **0.827** | **0.747** |

On the basins the model never saw — the case the whole project exists to test — `donor_flow`
reaches 0.747 against `donor`'s 0.743. **That gap is smaller than the variation between random
starts of the same model, so this run cannot tell the two apart.** The three starts of `donor`
land at 0.691, 0.717 and 0.695 when scored individually, a spread of 0.025; the gap between the
two variants is 0.004, six times smaller.

Because both variants were run on the same 76 held-out basins, they can also be compared basin
by basin. Doing that says the same thing more directly: the typical basin changes by 0.0002,
39 of the 76 basins improve and 37 get worse — an even split — and a paired rank test gives a
probability of 0.379 that a difference this small would appear by chance alone, far short of
anything that would count as a result.

One genuine asymmetry sits underneath that null. The *average* change across basins is +0.036
even though the *typical* change is zero, and the gap between those two figures is the whole
story. It comes from a handful of basins where `donor` fails outright and `donor_flow` does not:

| basin | `donor` | `donor_flow` |
|---|---|---|
| 08150800 | −0.561 | 0.316 |
| 06847900 | −0.862 | −0.013 |
| 02092500 | 0.303 | 0.599 |

Removing the single largest of those drops the average change to +0.025, and removing the top
three drops it to +0.0098. So the effect is confined to roughly three basins out of 76 rather
than being spread across them. With a sample that small this is as consistent with noise as
with a real repair, and it was not tested on any further split of the basins.

Drawn in `figures/donor_flow_check.png`, from `figures/make_donor_flow_figure.py`. The third
panel is the one that settles it — a tall spike at no change, with two isolated bars far to the
right.

**What it means**
<!-- NEEDS YOUR REVIEW -->

**What's next** — nothing. The project stops here; see the closing entry above for why, and for
the second idea that was never run.

---

## 2026-09-14 — correction: every score in this file is now the ensemble-mean hydrograph

**What changed.** Earlier entries reported median NSE two different ways. The donor entry below
used the ensemble-mean hydrograph — average the three starts' day-by-day predictions per basin
first, then compute the metric once. The first screen and the `donor_gated` entry instead scored
each start separately and averaged the scores, which reads several hundredths lower. That
inconsistency is now removed: **the ensemble-mean hydrograph is the only headline convention in
this project.** It is what CLAUDE.md's invariant requires, and it is how the reference number is
built — Kratzert's 0.693 is a 10-member ensemble mean hydrograph (`BENCHMARKS.md`), so scoring
our runs the same way makes that comparison like-for-like instead of biased against us.

Per-start numbers are not deleted. They stay in the tables as the seed spread, which is still how
stability and "is this difference real" get judged — an ensemble gives one number per
architecture per fold and so carries no uncertainty of its own.

**What moved.** Rescored from the same `runs/*/predictions.npz` through
`figures/make_results_figure.py`, the one ensemble-scoring path. No retraining, no new compute.

| fold 0, 76 held-back basins, WY2005–2014 | was (per-start mean) | now (ensemble) |
|---|---|---|
| plain | 0.680 | **0.709** |
| neighbours | 0.662 | **0.702** |
| neighbours with flow | 0.701 | **0.743** |

**The one conclusion this reverses.** The first screen recorded the pre-agreed go/no-go — plain
plus 0.03 — as not met. On the ensemble convention the comparison is 0.709 to 0.743, a gain of
0.034, so the bar is met. But it is met by 0.004, which is far inside the spread already on
record: donor alone ranges 0.667 to 0.760 across basin groups 0, 1 and 2. Clearing a bar on one
basin group by that margin is not evidence the effect is that size.

Two numbers barely moved and one cannot be recovered. The gap between *neighbours* and *plain*
stays negative but shrinks from 0.018 to 0.007. `spatial_all` (0.636) cannot be rescored at all —
its run directories were deleted when the variant was dropped, so its per-start numbers stay in
the table below marked as such, and must not be read against the rescored ones.

**What it means**

<!-- NEEDS YOUR REVIEW -->

---

## 2026-09-14 — donor on two more basin groups, plus a look at its attention weights

**What I ran**

The earlier screen (logged below) tested three model variants on one way of splitting the 531
river basins into 455 used for training and 76 held back completely ("basin group 0"). This run
extended only the best-performing variant, `donor` — the one that lets a basin borrow recently
measured streamflow from nearby monitored basins — to two more of the seven possible basin
groups (basin groups 1 and 2), each with the usual three random starts. That is six new training
runs, on top of the nine already logged. `vanilla` and `spatial` were not re-run on these new
groups; this was a cheaper check of whether donor's earlier result holds up on a different set of
held-back basins, not a rematched comparison against the other variants.

All six runs used the same code (commit `9078e7c`) and settings as before, trained on
`vm-sandeep` (one rented cloud GPU), six at a time. The batch took about five hours of
wall-clock time (each run took 293 to 306 minutes; running six at once slowed each individual
run down compared to the original four-at-a-time screen, but still finished the whole batch
faster than four-at-a-time would have, since more of it happened in parallel).

Afterward, two things were checked using only the attention weights each run already saves (no
further GPU time needed): how far away, on average, the model chose to borrow from in each basin
group, and — for the handful of basins where donor's improvement over `vanilla` was clearest —
exactly which basins it borrowed from and how much.

**What came out**

The most reliable way to score a run is to first average the three random starts' day-by-day
predictions into one combined prediction per basin, then compute the accuracy measures once from
that combined prediction, rather than scoring each start separately and averaging the scores —
the latter reads several hundredths lower and is not how the published reference number below
was computed. Nash-Sutcliffe efficiency (NSE) measures how closely predicted daily streamflow
tracks measured streamflow: 1.0 is a perfect match, 0.0 is no better than always predicting that
basin's long-term average flow, and negative values are worse than that. Kling-Gupta efficiency
(KGE) is a related measure that separately accounts for correlation, variability, and average-flow
bias, scaled the same way (1.0 is perfect).

| basin group | donor median NSE | donor median KGE | basins |
|---|---|---|---|
| 0 (already logged) | 0.743 | 0.722 | 76 |
| 1 (new) | 0.760 | 0.707 | 76 |
| 2 (new) | 0.667 | 0.678 | 76 |
| all three combined | 0.737 | 0.705 | 228 |

For comparison, `vanilla` — run only on basin group 0 so far — scores 0.709 median NSE the same
way. A previously published result using a similar but not identical model (Kratzert et al. 2019
— different years, different weather data, but the same 531 basins) reports 0.693 median NSE,
which this project treats as rough context rather than a strict target, since several setup
differences could each move the number by about as much as the effect being looked for (see
`BENCHMARKS.md`).

Donor's combined score across all three basin groups (0.737) sits above that published context
number by 0.044 — comfortably more than the 0.03 improvement the project originally set as its
bar for a real effect. Two of the three basin groups individually clear that reference by 0.05 to
0.07; the third (basin group 2) falls about 0.03 short of it on its own.

If instead each random start is scored separately and then averaged — a stricter, noisier way of
scoring used elsewhere in this project to check reproducibility across starts — the same three
basin groups score 0.701, 0.741, and 0.632 respectively, averaging to 0.691. Both ways of scoring
agree on the same pattern: basin group 1 best, basin group 2 worst, with roughly a 0.10 spread
between them from changing only which basins were held back — similar in size to the difference
the project is trying to detect between model variants.

**What the attention weights showed**

For each basin group, the model's attention was compared against simple rules such as "borrow
only from the nearest monitored basin" or "weight monitored basins by distance." The simple rules
reproduce a held-back basin's own flow pattern better than the model's learned attention does in
every basin group tested so far (median match quality 0.86–0.87 for the best simple rule, versus
0.63–0.78 for the trained model), and the model borrows on average from about 590–870 kilometers
away versus roughly 45–65 kilometers for the best simple rule — consistent with the range limit
on donor already noted in the entry below. Basin group 1, the best-scoring group, also had the
model's attention sitting closest to the simple-rule behaviour; basin groups 0 and 2 did not show
as clean a relationship between attention behaviour and final score.

A closer look at the three held-back basins where donor's improvement over `vanilla` was largest
(restricted to basins where donor's own score was positive, not just improved) found the model
mostly borrowing from genuinely nearby basins — 13 to 265 kilometers away in two of the three
cases — with one case leaning on a single somewhat farther basin (407 km) alongside closer ones.
An earlier pass at this same check contained a bug (it read the saved attention weights for the
wrong basins entirely) that made it look like several unrelated basins across the country were
all borrowing from the same small cluster of Pennsylvania basins; that pattern disappeared once
the bug was fixed, and a separate check across all 76 basin-group-0 basins found no single donor
basin doing an outsized share of the work (the busiest one appeared among a basin's top five
sources for only 17% of the 76 basins).

**What it means**

<!-- NEEDS YOUR REVIEW -->

**What's next**

Training donor on the remaining four basin groups (3 through 6) is underway, which will complete
the standard setup of holding back every one of the 531 basins exactly once. `vanilla` still has
not been run beyond the first basin group, so this remains a comparison against the published
reference number, not yet a basin-group-matched comparison against `vanilla` itself.

---

## 2026-09-13 — donor_gated: one attempt to fix donor's range limit, then a pause

**What I ran**

The first screen (below) found that letting a basin borrow measured flow from nearby monitored
basins (the "donor" variant) genuinely helps, but only at basins within roughly 40 km of a
monitored neighbour — beyond that it does no good, and at some basins it makes predictions
worse than not borrowing at all. A new variant, `donor_gated`, tried to fix that with two
additions on top of donor, and nothing else:

- A learned preference for borrowing from geographically closer basins, rather than leaving the
  model to work out "closer is usually more useful" purely from the data — it had already been
  shown not to learn that on its own.
- A learned "how much should I trust this?" dial that was meant to let the model borrow less
  when no monitored basin nearby was likely to be useful, so a basin far from any monitored
  neighbour would fall back toward not borrowing at all, instead of being forced to use
  borrowed information whether or not it helps.

Only one training run was made (rather than the usual three) because the result was clear
enough on that first run to make two more not worth the cost — explained below.

**What came out**

On the 76 basins held back from training, this one run scored 0.622 (median NSE, the same
measure used throughout — 1.0 is a perfect match, 0.0 is no better than guessing the long-term
average).

*(Rescoring note, 2026-09-14: this was a single run, so it has no ensemble of its own and 0.622
stands as both its per-start and its one-member score. Its run directory was not kept, so it
cannot be re-derived. The comparison is therefore unavoidably unequal — a one-member score
against three-member ensembles, and single members score lower. Against the per-start numbers it
was written on, 0.622 sits below plain at 0.674 and below donor's three-start average of 0.691,
which ranged 0.691–0.717. Against the ensemble numbers now used everywhere else — plain 0.709,
donor 0.743 — the apparent gap is larger, but part of that gap is the missing ensemble, not the
variant. The basin-by-basin finding below is the reliable part of this entry.)*

But basin-by-basin, matched against plain and against donor, this run was statistically
indistinguishable from both — barely over half the basins did very slightly better, the rest
very slightly worse. The low overall score came from a handful of basins that did much worse
under this variant, pulling the summary number down without most basins actually changing much.

The distance preference worked as intended: without being told to, the model learned that
basins around 42 km away carry the most useful information — matching a separate, hand-checked
calculation done beforehand of which distance range actually carries the most information.

The trust dial did not work as intended. It correctly learned to trust closer basins more than
distant ones (a real, statistically clear pattern) — but the overall level of trust collapsed
far below anything useful: even for the closest basins, only about 18% of the borrowed
information was used, against the 100% donor always uses.

A further check, which needed no additional GPU training, ruled out the simplest explanation.
Re-scoring the same trained model with its trust dial forced to different fixed values (rather
than the value it learned per basin) showed the model was not simply "under-trusting" in a way
that could be corrected by turning the dial back up — forcing even a moderately higher trust
value made predictions dramatically worse, not better. The model had tuned itself narrowly
around the small, basin-specific trust values it saw during its own training, and did not
tolerate being pushed away from them.

A second, separate change showed up alongside this: the model's attention became far more
concentrated onto very few borrowed basins (about 4, on average) than the plain donor variant
uses (about 24, on average). Since donor's benefit may come partly from averaging across many
borrowed basins, this narrowing is itself a plausible explanation for the drop — separate from,
and possibly compounding with, the trust-dial problem.

**What it means**

<!-- NEEDS YOUR REVIEW -->

**What's next**

Decided not to pursue this further — two independent problems surfaced in the same attempt,
neither explained by a simple, one-line bug, and fixing one at a time would mean testing
against an increasingly narrow and speculative set of hypotheses rather than a clear one.

The codebase and all documentation have been reverted to the three-variant design (`vanilla`,
`spatial`, `donor`). `spatial_all` and `donor_gated` are both out — neither is in the active
code, and neither's run data is kept on disk; this entry and the first screen below are the
full record of both attempts. The project is paused here.

---

## 2026-09-13 — first screen: four model variants on one split

**What I ran**

Twelve training runs: four model variants, each started three times from different random
initial weights, all on the same split of the 531 basins. In that split 455 basins are used
for training and 76 are held back completely — the model never sees a single flow measurement
from those 76, which is the situation this whole project is about.

The four variants differ only in what a basin is allowed to consult besides its own weather
and its own catchment properties:

- **plain** — each basin is predicted from its own daily weather and fixed catchment
  properties, nothing else.
- **neighbours** — a basin may also borrow a summary of conditions from the 455 training
  basins. That summary is built from their weather and catchment properties; none of their
  measured flow goes into it.
- **neighbours, unmonitored included** — the same, except the 76 held-back basins also join
  the pool that can be borrowed from. Only their weather and catchment properties are used;
  their measured flow is never touched.
- **neighbours with flow** — the same as *neighbours*, except what is borrowed also carries
  how unusually high or low each training basin's measured flow has been over the previous
  seven days, expressed relative to that basin's own normal.

The last variant is not a like-for-like comparison against the others. It sees 455 real flow
measurements from the very days it is predicting, which none of the others get. It is here to
ask whether that extra information is worth anything, not to claim a better model design.

Every setting was identical across all twelve runs: 256 hidden units in a single layer, a
270-day window of past weather as input, 30 passes through the training data, the same
learning-rate schedule, and the same dropout. Training used water years 1990-2004, which are
complete for every basin on every day. Scoring used water years 2005-2014, which no run
trained on.

Four runs shared one A100 graphics processor at a time. Individual runs took 183 to 213
minutes and the full batch took about 9.5 hours of wall-clock time, roughly $35 of rented
compute. No run crashed.

Every run recorded the code version it came from as `cbc9bd0`. That identifier no longer
appears on the published branch: its history was rewritten afterwards so that two working
documents would stay out of a public repository. The code that actually produced these runs —
the `src/` modules, the run script, the dependency list and the basin lists — is byte-identical
to the current branch tip `eb7125c`, confirmed by direct file comparison, and the original
version is still reachable locally on the branch `backup-before-rewrite`.

**What came out**

Scores below use the Nash-Sutcliffe efficiency, a standard measure of how closely predicted
daily flow follows measured flow: 1.0 is a perfect match, 0.0 is no better than always
predicting the long-term average flow, and negative values are worse than that. Each number is
the median across basins — half the basins score above it and half below.

On the 76 basins held back from training, scored over 2005-2014. The headline column is the
ensemble — the three starts' daily predictions averaged per basin, then scored once. The three
start columns are kept as the spread. *(Rescored 2026-09-14; see the correction entry at the top
of this file. The ensemble column is the number to quote.)*

| model variant | **ensemble of 3** | start 1 | start 2 | start 3 |
|---|---|---|---|---|
| plain | **0.709** | 0.674 | 0.682 | 0.683 |
| neighbours | **0.702** | 0.659 | 0.665 | 0.662 |
| neighbours, unmonitored included | not recoverable | 0.647 | 0.633 | 0.629 |
| neighbours with flow | **0.743** | 0.691 | 0.717 | 0.695 |

What stands out:

- The plain model reached 0.709. A published network of the same family, trained the same way on
  these same 531 basins, reports 0.693 for basins held back from training (Kratzert et al. 2019,
  Water Resources Research 55) — and that number is itself a 10-member ensemble mean hydrograph,
  so this is a like-for-like comparison. Landing 0.016 above it is a sign the data handling and
  scoring here are behaving as intended.
- Letting a basin borrow from monitored neighbours **lowered** the score, from 0.709 to 0.702.
  On the per-start numbers this gap looked larger (0.680 to 0.662); on the ensemble it is 0.007,
  small enough that it says borrowing basin *state* does not help, not that it actively harms.
- Adding the unmonitored basins to the pool lowered it again — the worst of the four on the
  per-start numbers (0.636). Those runs were deleted when the variant was dropped, so this one
  cannot be rescored and cannot be compared against the three ensemble figures above.
- Only the variant whose borrowing carries measured flow scored above plain, at 0.743.
- The target agreed in advance was plain plus 0.03. Plain reached 0.709 and the best variant
  0.743, a gain of 0.034, so **the target was met — by 0.004.** That margin is far inside the
  basin-group-to-basin-group spread seen later (donor ranges 0.667 to 0.760 across groups 0, 1
  and 2), so it should be read as "cleared the bar on one basin group", not as a measured effect
  size. The original entry recorded this as a miss, on the per-start convention.

For every pair of variants above, the three starts of one do not overlap the three starts of the
other, so these differences are larger than the variation between random starts of the same
model. With only three starts per variant that is suggestive rather than a formal test — and note
this check necessarily uses the per-start columns, since the ensemble produces a single number
with no spread of its own.

On the 455 training basins, scored over the same unseen years 2005-2014, the order is different:
plain 0.775, neighbours 0.768, and neighbours with flow 0.820 (ensemble; the per-start figures
were 0.755, 0.746 and 0.799, and neighbours-with-unmonitored was 0.746 per-start but cannot be
rescored). The flow-carrying variant gains more on basins that were monitored during training
(0.820, up from 0.775) than on basins held back (0.743, up from 0.709) — 0.045 against 0.034.
On the per-start convention that difference looked like a factor of two; on the ensemble it is
about a third more, which is a much weaker claim.

Worth separating out, because it is a distinct result rather than a side-check: those 455 basins
are queried with their *own* gauge masked out of the attention, in evaluation exactly as in
training. So this cell is not at-site data assimilation. It is the case of a basin whose
historical record trained the model but which has no working gauge now, predicted from its own
weather plus concurrent flow at other, still-active gauges.

**One run misbehaved.** The second start of *neighbours, unmonitored included* finished
training with an average error roughly six times that of its two siblings — 0.62 against about
0.10 on the internal training measure — and its borrowing had collapsed onto about 9
neighbours where the other starts spread across roughly 20. It did not crash, so the batch
counted it as a success. Its held-back score of 0.633 sits between its two siblings, so it
does not by itself explain why that variant scored lowest.

How hard each variant leaned on borrowing can be read from the learned scale factor applied to
the borrowed signal, which starts at 1.0 and is free to fall to zero if borrowing is not
useful. The two flow-free borrowing variants held it near where it started, 0.95 to 1.06,
across all six runs. All three flow-carrying runs settled at roughly half that, 0.52 to 0.54,
and still scored highest. In every borrowing run the attention spread over about 18 to 25
neighbours out of 455, rather than flattening across all of them.

**What it means**

<!-- NEEDS YOUR REVIEW -->

**What's next**

- Decide whether the missing combination is worth running: borrowing that carries measured
  flow *and* lets the unmonitored basins into the pool. It needs a code change first, because
  at present a basin can only be borrowed from if it has a flow measurement in the window,
  which silently excludes the unmonitored ones as soon as flow is switched on.
- Consider giving the borrowing pathway a training target of its own, so that it is explicitly
  trained to supply the "how unusual is today" part while a basin's own weather supplies the
  magnitude.
- Nothing here yet justifies extending to all seven splits.


---
## Measured before any model was run — donor proximity

Relevant only if the variant that uses neighbours' measured flow turns out to help. The worry
was that some "ungauged" basins sit a few kilometres from a gauged basin on the same river, so
borrowing from that neighbour would be closer to a routing calculation than to genuine
prediction at an unmonitored site.

Checked by asking how well each held-out basin could be reproduced by simply copying its
nearest gauged neighbour's daily flow, scaled by catchment area:

- On split 0 — the one the first screen uses — the best any basin could be reproduced this way
  was 0.749 on the NSE scale (1.0 is a perfect match), and only 6 of 76 exceeded 0.7. No basin
  is a near-duplicate of its neighbour. 16 of 76 have a gauged neighbour within 20 km.
- Split 0 is the cleanest of the seven. Splits 2, 4, 5 and 6 each contain a basin reproducible
  at 0.92-0.94 by copying alone — genuinely the same water measured twice. That matters for the
  seven-split phase, not for this screen.

No basin is excluded on these grounds. A model that finds and uses an upstream gauge is working
as intended; this is recorded so the question can be answered later, not so anything is dropped.
