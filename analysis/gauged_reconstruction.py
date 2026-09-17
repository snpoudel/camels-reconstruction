"""Can the correction reconstruct a discontinued gauge's OOS-TIME record, not just OOS-space?

    python analysis/gauged_reconstruction.py --fold 0
    python analysis/gauged_reconstruction.py --fold 0 --target-group 1

Pilot / pipeline-development version: works entirely from folds already trained for the PUB
grid -- no new training needed. One outer fold's model is picked; within its training basins
(which keep their original 7-fold labels, `src/correct.py:load_all`), one of the other six
labels is chosen as a "gauge going dark for the whole test window" and reconstructed from the
remaining five groups, exactly as `src/correct.py:choose(mode="inner")` never reads a held-out
basin during selection -- the learned (K, lambda, p) here is therefore also never told about
the group it corrects (`src/correct.py:reconstruct_gauged`).

This trades donor density for honesty: the reconstructed group draws on ~5/6 of the training
pool (~379 basins), not the full pool a real discontinued-gauge scenario, or the PUB cell's
held-out basins, draw on (455). See METHOD.md for why. A full run across all (fold, group)
pairs, or training dedicated gauged-reconstruction folds directly (531 basins, deliberately
time-truncated), is the natural next step once more compute is available -- this script is
deliberately a single-(fold, group) pilot, and also the pipeline meant to run that later step.

Writes four `runs/recon_gauged_*` files, named to sit next to the real PUB pipeline's
`runs/recon_ungauged_*` outputs -- both are a "reconstruction", one of an ungauged basin
(OOS-space), one of a gauged-but-discontinued one (OOS-time).
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from analysis.donor_geometry import geometry, range_curve  # noqa: E402
from src.correct import (K_GRID, P, apply_all, default_target_group,  # noqa: E402
                         distances, find_fold, load_all, reconstruct_gauged)
from src.evaluate import metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--fold", type=int, default=0,
                    help="which already-trained outer fold's model to use")
    ap.add_argument("--target-group", type=int, default=None,
                    help="which of the other inner groups goes dark (default: smallest available)")
    ap.add_argument("--out-prefix", default="runs/recon_gauged_")
    a = ap.parse_args()

    folds = load_all(a.runs)
    if not folds:
        raise SystemExit(f"no vanilla_fold*_seed*/ runs under {a.runs}/")
    f = find_fold(folds, a.fold)
    groups = sorted(np.unique(f["pool_groups"]).tolist())
    target_group = a.target_group if a.target_group is not None else default_target_group(f)
    if target_group not in groups:
        raise SystemExit(f"--target-group {target_group} not among fold {a.fold}'s groups {groups}")

    pilot, surf = reconstruct_gauged(f, target_group)
    _, basins = distances()
    pilot["basins"] = [basins[i] for i in pilot["held"]]

    obs = pilot["obs"][pilot["held"]]
    raw_m, cor_m = metrics(pilot["raw"], obs), metrics(pilot["corrected"], obs)
    raw_m["variant"], cor_m["variant"] = "vanilla", "vanilla + correction"
    for m in (raw_m, cor_m):
        m["basin"] = pilot["basins"]
        m["fold"], m["target_group"] = pilot["fold"], target_group
        m["k"], m["lam"], m["p"] = pilot["k"], pilot["lam"], pilot["p"]
        m["n_donors"], m["n_target"] = pilot["n_donors"], pilot["n_target"]
    pd.concat([raw_m, cor_m], ignore_index=True).to_csv(f"{a.out_prefix}metrics.csv", index=False)

    # Same convention as analysis/tune.py's "cv" rows: base_nse is the lambda=0 (uncorrected)
    # point on this same leave-one-group-out surface, not a plain raw NSE, so the two numbers
    # are measured the same way.
    base_nse = np.nanmedian(surf[(K_GRID[0], 0.0, P)])
    tuning = pd.DataFrame([
        dict(fold=pilot["fold"], set="cv", k=k, lam=lam, p=pw, median_nse=np.nanmedian(v),
             base_nse=base_nse, n_basins=pilot["n_target"], n_seeds=pilot["n_seeds"])
        for (k, lam, pw), v in surf.items()])
    tuning.to_csv(f"{a.out_prefix}tuning.csv", index=False)

    picked = {pilot["fold"]: (pilot["k"], pilot["lam"], pilot["p"])}
    geometry([pilot], picked).to_csv(f"{a.out_prefix}donor_geometry.csv", index=False)
    range_curve([pilot], picked).to_csv(f"{a.out_prefix}donor_range.csv", index=False)

    pub_folds, _ = apply_all(a.runs)
    pub_f = find_fold(pub_folds, a.fold)
    pub_raw = metrics(pub_f["raw"], pub_f["obs"][pub_f["held"]]).nse.median()
    pub_cor = metrics(pub_f["corrected"], pub_f["obs"][pub_f["held"]]).nse.median()

    print(f"fold {a.fold}, target group {target_group}: {pilot['n_target']} basins reconstructed "
          f"from {pilot['n_donors']} donors  (K={pilot['k']}, lambda={pilot['lam']:.2f}, "
          f"p={pilot['p']:.2g})")
    print(f"  gauged reconstruction  (this pilot, OOS-time):   "
          f"{raw_m.nse.median():.4f} -> {cor_m.nse.median():.4f} "
          f"({cor_m.nse.median() - raw_m.nse.median():+.4f})")
    print(f"  ungauged reconstruction (same fold's real PUB, OOS-space, {len(pub_f['pool'])} donors):  "
          f"{pub_raw:.4f} -> {pub_cor:.4f} ({pub_cor - pub_raw:+.4f})")
    print(f"-> {a.out_prefix}{{metrics,tuning,donor_geometry,donor_range}}.csv")


if __name__ == "__main__":
    main()
