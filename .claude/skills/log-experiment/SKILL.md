---
name: log-experiment
description: Record a completed experiment run in EXPERIMENTS.md — what was run, what came out, in plain language. Use after a training batch finishes and `python -m src.evaluate` has written runs/metrics.csv. Also use when the user says "log this", "write this up", or "add this to the experiment log".
---

# Log an experiment

Append one entry to `EXPERIMENTS.md` describing a completed run batch.

## 1. Gather the facts — do not guess any of them

```bash
ls -d runs/*/ | wc -l                       # how many runs
cat runs/*/config.json                      # arch, fold, seed, every hyperparameter
python -m src.evaluate                      # regenerates runs/metrics.csv + prints medians
grep -c . runs/metrics.csv                  # sanity: rows present
tail -3 logs/*.log                          # wall-clock per run
```

Read the medians from `summarise()` in `src/evaluate.py` — do not recompute them by hand.

Check `git log --oneline -1` and record the commit the runs came from. A result you cannot
trace back to a commit is a result you cannot reproduce.

## 2. Check whether the difference is real before you describe it

`summarise()` prints each architecture's median as `mean [min, max]` across seeds. Before
writing that one variant beat another, compare the gap between them to the width of those
brackets.

**If the gap is smaller than the seed-to-seed spread, say that plainly.** Write "the difference
between them is smaller than the variation between random starts of the same model, so this
run cannot tell them apart" — not "X improved by 0.02". This is the single most important
instruction in this skill: the previous round of this project stalled for months because an
unresolvable difference got recorded as if it were a finding.

## 3. Append the entry

Newest first, directly under the `---` at the top of `EXPERIMENTS.md`. Never edit past
entries. On the first real entry, delete the italic "No experiments yet" placeholder.

```markdown
## YYYY-MM-DD — <short name>

**What I ran** — plain sentences. Which model variants, how many random starts, which
split of the basins, how long it took, which commit.

**What came out** — a table of the numbers, then what stands out. Explain each measure
the first time it appears in the entry.

**What it means**
<!-- NEEDS YOUR REVIEW -->

**What's next** — one or two concrete items.
```

Leave `NEEDS YOUR REVIEW` in place. Never write the interpretation section and never remove
that marker — an unreviewed reading has to stay visibly unreviewed.

## 4. The plain-language rule

Write for someone who knows hydrology but not this codebase and not machine-learning jargon.

Explain every measure the first time it appears:

> NSE, which measures how closely predicted daily flow tracks measured flow — 1.0 is a perfect
> match, 0.0 means no better than always predicting the long-term average, and negative values
> are worse than that.

Say what a thing is, not what it is called:

| write this | not this |
|---|---|
| basins the model never saw during training | out-of-sample in space |
| years held back from training | temporal holdout |
| model variant | architecture / arch |
| three random starts of the same model | three seeds |
| settings | config / hyperparameters |
| turning one piece of the model off | ablation |
| 0.71, up from 0.68 | +0.03 |

Also: no unexpanded acronyms on first use. No "SOTA", "delta", "vs baseline". Write full
sentences — the entry is prose with a table in it, not a metrics dump.

## 5. Check before finishing

Read the entry back and confirm:

- Every number has a unit or a scale attached, and every measure was explained once.
- No acronym appears unexpanded on first use.
- Any difference within the seed spread is described as inconclusive.
- The `NEEDS YOUR REVIEW` marker is present and the interpretation section is empty.
- The commit hash is recorded.
