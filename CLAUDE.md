# camels-reconstruction

A residual post-processing correction for streamflow prediction: a model's daily errors at the
nearest gauges are used to correct its prediction at a basin whose own gauge is missing. One
inverse-distance-weighted step, nothing trained, no GPU. The project is about **the correction**
— the LSTM is only what produced the predictions, and the same pipeline runs on any model's
output dropped into `runs/<model>/`. See `README.md` for the method summary and `manuscript.md`
for the paper itself.

That IDW correction (`src/corrections/idw/`) is the finished method the manuscript reports. Next
up is a second, learned one: an MLP that predicts each day's correction for a target basin
from its donors, with the number of donors as its only hyperparameter. Unlike IDW's fixed K,
lambda and p, its correction can change from day to day. It will be compared against IDW on the
LSTM only, and whether it goes into the manuscript depends on that comparison.

Two cells: **ungauged** (basin never in training — out of sample in space and time) and
**gauged** (basin trained on, but its gauge stops reporting for the test years — out of sample
in time).

## Layout

| Path | Contents |
|---|---|
| `src/shared.py` | What every correction method shares: basins, folds, distances, obs, NSE/KGE, save format |
| `src/corrections/idw/correction.py` | The IDW neighbour-residual correction |
| `src/corrections/idw/tuning.py` | K x lambda x p grids, the CV surfaces, three selection objectives |
| `src/corrections/idw/01_gauged.py` | Gauged pipeline → `runs/<model>/idw/gauged/` |
| `src/corrections/idw/02_ungauged.py` | Ungauged pipeline → `runs/<model>/idw/ungauged/` |
| `src/corrections/mlp/model.py` | The learned MLP correction: inputs, model, NSE* loss, training (LSTM only) |
| `src/corrections/mlp/01_gauged.py` | Gauged pipeline, K × seed search → `runs/lstm/mlp/gauged/` |
| `src/corrections/mlp/02_ungauged.py` | Ungauged pipeline, K × seed search → `runs/lstm/mlp/ungauged/` |
| `src/models/lstm/common.py` | The LSTM: cache, folds, model, training loop, prediction, seed-mean |
| `src/models/lstm/01_lstm_gauged.py` | Train on all basins, per seed → `runs/lstm/gauged/` |
| `src/models/lstm/02_lstm_ungauged.py` | Train per (fold, seed) → `runs/lstm/ungauged/` |
| `src/models/{hbv,hybrid}/prepare.py` | Drop another model's published predictions into `runs/<model>/` |
| `figures/evaluate.py` | Metrics → `figures/metrics_<cell>_<model>_<method>.csv`, `tuning_*.csv`, `run_info.json` |
| `figures/make_*.py` | One script per figure; `palette.py` holds the shared style |
| `figures/lstm/` | The tech note's figures, named by figure number: `fig01`–`fig04` and `figS1`–`figS2` (PNG + SVG) |
| `figures/lstm_noscale/` | Comparison: the same figures without the flow scale s (plain mm/day residuals); scripts' `--variant noscale` |
| `figures/lstm_logresidual_kge/` | Archive: the same figures under the earlier log-residual, KGE-selected method, with their CSVs in `data/` |
| `doc/` | The HESS tech note: `tech_note_outline.md`, `lit_review.md`, `manuscript.md` |
| `data/` | `camels531.csv`, fold assignment, US state outlines (tracked); `raw/`, `cache/` (gitignored) |
| `runs/` | Per-model predictions and corrections (gitignored) |

Everything runs as `python <path>.py` — no packages, no `python -m`. The digit-prefixed scripts
are pipelines, not modules, and are not importable by design.

`src/models/` produces predictions; `src/corrections/<method>/` fixes them. A correction method
imports only `src/shared.py` and its own folder, never another method or a model, and writes
`runs/<model>/<method>/<cell>/` through `shared.save_corrected` so `figures/evaluate.py
--method <method>` scores every method identically.

## Running

See README.md "Running it" for the pipeline (cache → train → correct → evaluate → figures).

## Compute

- LSTM training is finished: every seed and fold was trained on the NSF Delta GPU cluster, and
  that cluster is no longer used. `runs/lstm/{gauged,ungauged}/` are final and are the only copy
  of those predictions, since regenerating them would mean retraining on a GPU, so never
  overwrite or delete them. `src/models/lstm/` is kept as the record of how they were made.
- Everything now runs locally. The IDW correction (`src/corrections/idw/*`) and `figures/*` are
  CPU-only; the grid search is a couple of minutes per cell, the correction itself milliseconds.

## Workflow

- Read `experiments_log.md` (top entry) at the start of a session — it's the repo's memory across
  sessions, so it should tell you what was last run and what's next without chat history.
- If you notice a request that's part of a repeating pattern, or that a hook, subagent, or
  skill would handle better than doing it ad hoc, say so and suggest creating one. Build it
  only after I approve.
- Always confirm with me before running `git commit` or `git push` (or any other action that
  publishes to GitHub — merging, deleting a branch, opening a PR), even if I approved something
  similar earlier in the session. I am the sole author of every commit: do not add a
  Co-Authored-By line or any other AI-attribution trailer.

## Code hygiene

- No comments explaining *what* code does — names should do that. Comment only non-obvious
  *why* (a leak-prevention invariant, a literature-mandated constant, a workaround).
- No speculative abstractions, flags, or config knobs for hypothetical future needs. This is a
  research repo: prefer the direct, readable version.
- Prefer editing existing scripts over adding new ones; each script has one clear purpose.
- After a training/eval run completes, use the `log-experiment` skill to record it in
  `experiments_log.md` rather than describing results only in chat.
- Write short, concise code and short scripts; don't make too many small functions. Avoid
  over-engineering and keep it simple.
