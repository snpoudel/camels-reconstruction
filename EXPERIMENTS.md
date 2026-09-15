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
