---
paths:
  - "src/residual_postprocessing.py"
  - "src/parameter_tuning.py"
  - "src/01_gauged_correction.py"
  - "src/02_ungauged_correction.py"
---

**The fold-isolation invariant**: no search may read a basin it will go on to correct. In the
ungauged cell the fold's held-out basins are absent from the search; in the gauged cell the
whole group being corrected is, and a basin is masked out of its own donor set at infinite
distance. Any change to `src/residual_postprocessing.py`, `src/parameter_tuning.py`, or either
`*_correction.py` must preserve that — check it by destroying the target basins' observations
and confirming both the selected (K, λ, p) and the corrected hydrographs are bit-identical,
with a positive control that destroys a *donor* and confirms the output does move.
