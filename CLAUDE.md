# Attention-LSTM-PUB

Post-processing correction for ungauged-basin streamflow prediction. A standard
LSTM (similar to Kratzert et al. 2019) is trained per fold on CAMELS-US; a
single inverse-distance-weighted correction step then adjusts predictions
using nearby gauged basins. No retraining, no GPU needed for the correction
itself. See `README.md` for the method summary and `METHOD.md` for the full
derivation, leak audit, and results. The idea is this post-processing correction,
will provide SOTA performance for **reconstruction** of gauged and ungauged basins predictions.

## Layout

| Path | Contents |
|---|---|
| `src/config.py` | Hyperparameters — literature values, nothing tuned |
| `src/data.py` | Caching, fold splits, leak-sensitive statistics |
| `src/model.py` | The LSTM (one layer, statics concatenated) |
| `src/train.py` | Batched training loop |
| `src/evaluate.py` | Prediction + NSE/KGE/RMSE, bias by flow band |
| `src/correct.py` | The correction itself (~6 lines of arithmetic) |
| `analysis/tune.py` | K × lambda × p grid search, cross-validated |
| `analysis/leak_audit.py` | Proves the correction can't see the target basin |
| `analysis/donor_geometry.py` | Donor distance / break-even range analysis |
| `analysis/gauged_reconstruction.py` | Reconstruction for gauged-basin comparison |
| `figures/` | One `make_*_figure.py` script per plot, `palette.py` for shared style |
| `data/` | `camels531.csv`, fold assignments (tracked); `raw/`, `cache/` (gitignored) |
| `runs/` | Per-fold training outputs and metrics (gitignored) |

## Running

See README.md "Running it" for the full pipeline (cache build → train →
evaluate → correct → tune → audit → figures).

## Compute

- Training (`src/train.py`) runs on the NSF Delta GPU cluster — that's where
  new seeds/folds get launched, and where `runs/vanilla_fold*_seed*/` come
  from before being pulled down here.
- Everything downstream of a finished run (`src/evaluate.py`, `src/correct.py`,
  `analysis/*.py`, `figures/make_*.py`) is CPU-only and runs locally on this
  machine — no GPU needed, per the top-of-file note.

## Workflow

- Read `EXPERIMENTS.md` (top entry) at the start of a session — it's the
  repo's memory across sessions, so it should tell you what was last run
  and what's next without needing chat history.
- If you notice a request that's part of a repeating pattern, or that a
  hook, subagent, or skill would handle better than doing it ad hoc, say so
  and suggest creating one. Build it only after I approve.

## Code hygiene

- No comments explaining *what* code does — names should do that. Comment
  only non-obvious *why* (a leak-prevention invariant, a literature-mandated
  constant, a workaround).
- No speculative abstractions, flags, or config knobs for hypothetical future
  needs. This is a research repo: prefer the direct, readable version.
- Any change touching `src/data.py`, `src/correct.py`, or `analysis/tune.py`
  must preserve the fold-isolation invariant (no held-out basin's data reaches
  selection or training) — verify with `python analysis/leak_audit.py`.
- Hyperparameters in `src/config.py` are literature-sourced (Kratzert et al.
  2019); don't retune without noting the source and reason in `METHOD.md`.
- Prefer editing existing scripts over adding new ones; each `analysis/*.py`
  and `figures/make_*.py` file has one clear purpose.
- After a training/eval run completes, use the `log-experiment` skill to
  record it in `EXPERIMENTS.md` rather than describing results only in chat.
- This file is gitignored (see `.gitignore`) — treat it as local working
  notes, not a published doc.
- Wrote short concise codes, short scripts, and don't make too many small sized functions. The code should be readable and easy to understand. Avoid over-engineering and keep it simple.

