# camels-reconstruction

A residual post-processing correction for streamflow prediction: a model's daily errors at the
nearest gauges are used to correct its prediction at a basin whose own gauge is missing. One
inverse-distance-weighted step, nothing trained, no GPU. The project is about **the correction**
— the LSTM is only what produced the predictions, and the same pipeline runs on any model's
output dropped into `runs/<model>/`. See `README.md` for the method summary and `METHOD.md` for
the derivation, the leak audit and the results.

Two cells: **ungauged** (basin never in training — out of sample in space and time) and
**gauged** (basin trained on, but its gauge stops reporting for the test years — out of sample
in time).

## Layout

| Path | Contents |
|---|---|
| `src/residual_postprocessing.py` | The correction, and where its inputs live on disk |
| `src/parameter_tuning.py` | K × λ × p grids, the CV surfaces, the tie-break |
| `src/01_lstm_gauged_correction.py` | Gauged pipeline → `runs/<model>/gauged_corrected/` |
| `src/02_lstm_ungauged_correction.py` | Ungauged pipeline → `runs/<model>/ungauged_corrected/` |
| `src/lstm/common.py` | The LSTM: cache, folds, model, training loop, prediction, seed-mean |
| `src/lstm/01_lstm_gauged.py` | Train on all basins, per seed → `runs/lstm/gauged/` |
| `src/lstm/02_lstm_ungauged.py` | Train per (fold, seed) → `runs/lstm/ungauged/` |
| `figures/evaluate.py` | Metrics → `figures/metrics_*.csv`, `tuning_*.csv`, `run_info.json` |
| `figures/make_*.py` | One script per figure; `palette.py` holds the shared style |
| `data/` | `camels531.csv`, fold assignment (tracked); `raw/`, `cache/` (gitignored) |
| `runs/` | Per-model predictions and corrections (gitignored) |

Everything runs as `python <path>.py` — no packages, no `python -m`. The digit-prefixed scripts
are pipelines, not modules, and are not importable by design.

## Running

See README.md "Running it" for the pipeline (cache → train → correct → evaluate → figures).

## Compute

- Training (`src/lstm/*`) runs on the NSF Delta GPU cluster — that is where new seeds and folds
  get launched, and where `runs/lstm/<cell>/` comes from before being pulled down here.
- Everything downstream (`src/*_correction.py`, `figures/*`) is CPU-only and runs locally. The
  grid search is a couple of minutes per cell; the correction itself is milliseconds.

## Workflow

- Read `EXPERIMENTS.md` (top entry) at the start of a session — it's the repo's memory across
  sessions, so it should tell you what was last run and what's next without chat history.
- If you notice a request that's part of a repeating pattern, or that a hook, subagent, or
  skill would handle better than doing it ad hoc, say so and suggest creating one. Build it
  only after I approve.

## Code hygiene

- No comments explaining *what* code does — names should do that. Comment only non-obvious
  *why* (a leak-prevention invariant, a literature-mandated constant, a workaround).
- No speculative abstractions, flags, or config knobs for hypothetical future needs. This is a
  research repo: prefer the direct, readable version.
- **The fold-isolation invariant**: no search may read a basin it will go on to correct. In the
  ungauged cell the fold's held-out basins are absent from the search; in the gauged cell the
  whole group being corrected is, and a basin is masked out of its own donor set at infinite
  distance. Any change to `src/residual_postprocessing.py`, `src/parameter_tuning.py`, or either
  `*_correction.py` must preserve that — check it by destroying the target basins' observations
  and confirming both the selected (K, λ, p) and the corrected hydrographs are bit-identical,
  with a positive control that destroys a *donor* and confirms the output does move.
- Hyperparameters in `src/lstm/common.py` are literature-sourced (Kratzert et al. 2019); don't
  retune without noting the source and reason in `METHOD.md`.
- Prefer editing existing scripts over adding new ones; each script has one clear purpose.
- Figures read only `figures/*.csv` and `run_info.json`, never `runs/` — so a figure cannot
  disagree with the reported numbers. Style lives in `figures/palette.py`, not per figure.
- After a training/eval run completes, use the `log-experiment` skill to record it in
  `EXPERIMENTS.md` rather than describing results only in chat.
- Write short, concise code and short scripts; don't make too many small functions. Avoid
  over-engineering and keep it simple.
