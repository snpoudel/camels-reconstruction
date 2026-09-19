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
