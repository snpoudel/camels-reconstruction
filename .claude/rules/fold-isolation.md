---
paths:
  - "src/shared.py"
  - "src/corrections/**"
---

**The fold-isolation invariant**: no search may read a basin it will go on to correct. In the
ungauged cell the fold's held-out basins are absent from the search; in the gauged cell the
whole group being corrected is, and a basin is masked out of its own donor set at infinite
distance. This holds for every correction method, learned or not: any change to `src/shared.py`
or anything under `src/corrections/` must preserve it. Check it by destroying the target basins'
observations and confirming both the selected settings and the corrected hydrographs are
bit-identical, with a positive control that destroys a *donor* and confirms the output does move.
