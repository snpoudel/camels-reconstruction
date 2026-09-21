---
paths:
  - "figures/*.py"
---

Figures read only `figures/*.csv` and `run_info.json`, never `runs/` — so a figure cannot
disagree with the reported numbers. Style lives in `figures/palette.py`, not per figure.
