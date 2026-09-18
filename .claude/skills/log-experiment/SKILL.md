---
name: log-experiment
description: Record a completed experiment run in EXPERIMENTS.md — what was run, what came out, in plain language — and keep it usable as the project's memory across sessions: a new session should read the top entry and know where things stand. Use after a training batch finishes and `python -m src.evaluate` has written runs/metrics.csv, or when the user says "log this", "write this up", or "add this to the experiment log".
---

# Log an experiment

Append one entry to `EXPERIMENTS.md`, newest first. The top entry is what a
future session reads to get oriented — write "What's next" as a briefing for
someone with no other context.

## 1. Gather the facts — don't guess

```bash
ls -d runs/*/ | wc -l                       # how many runs
cat runs/*/config.json                      # arch, fold, seed, every hyperparameter
python -m src.evaluate                      # runs/metrics.csv — per-start, spread only
python figures/make_results_figure.py       # runs/metrics_ensemble.csv — headline numbers
grep -c . runs/metrics.csv                  # sanity: rows present
tail -3 logs/*.log                          # wall-clock per run
git log --oneline -1                        # commit the runs came from
```

Every score quoted is the ensemble-mean hydrograph (average the starts'
day-by-day predictions per basin, then compute NSE/KGE once) — the project's
only headline convention, matching CLAUDE.md and how the Kratzert 0.693
reference number is built. `summarise()`'s per-start medians read lower and
are for step 2 only; never quote them as a result. Don't recompute either by
hand.

## 2. Check the difference is real before describing it

An ensemble gives one number per architecture per fold, so it carries no
uncertainty of its own — the spread comes from `summarise()`'s per-start
`mean [min, max]` across seeds. Compare the ensemble gap to that spread
before claiming one variant beat another.

**If the gap is smaller than the seed-to-seed spread, say so plainly** —
"this run cannot tell them apart," not "X improved by 0.02." A past round of
this project stalled for months on an unresolvable difference recorded as a
finding.

## 3. Append the entry

Under the `---` at the top. Never edit past entries. Delete the italic "No
experiments yet" placeholder on the first real one.

```markdown
## YYYY-MM-DD — <short name>

**What I ran** — model variants, random starts, basin split, runtime, commit.

**What came out** — a table of numbers, then what stands out. Explain each
measure on first use.

**What it means**
<!-- NEEDS YOUR REVIEW -->

**What's next** — concrete items, written so a cold session can act on them.
```

Leave `NEEDS YOUR REVIEW` in place — never write that section or remove the
marker; an unreviewed reading stays visibly unreviewed.

## 4. Plain language, always

Write for someone who knows hydrology but not this codebase or ML jargon.
Explain every measure on first use, e.g.:

> NSE, which measures how closely predicted daily flow tracks measured flow
> — 1.0 is a perfect match, 0.0 no better than the long-term average,
> negative worse than that.

Say what a thing is, not what it's called:

| write this | not this |
|---|---|
| basins the model never saw during training | out-of-sample in space |
| years held back from training | temporal holdout |
| model variant | architecture / arch |
| three random starts of the same model | three seeds |
| settings | config / hyperparameters |
| turning one piece of the model off | ablation |
| 0.71, up from 0.68 | +0.03 |

No unexpanded acronyms, no "SOTA"/"delta"/"vs baseline". Prose with a table
in it, not a metrics dump.

## 5. Before finishing, check

- Every number has a unit/scale; every measure was explained once.
- No unexpanded acronym on first use.
- Any difference within the seed spread is called inconclusive.
- `NEEDS YOUR REVIEW` present, interpretation section empty.
- Commit hash recorded.
- "What's next" is concrete enough for a new session to act on cold.
