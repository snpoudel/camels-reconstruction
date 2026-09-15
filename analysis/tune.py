"""The (K, lambda) surface, and what selecting on it is worth.

    python analysis/tune.py --out runs/tuning.csv

Writes one row per (fold, K, lambda) with the median NSE over that fold's held-out basins --
the surface `figures/make_hyperparam_figure.py` draws -- and prints what each selection rule
would pick and what it scores.

Selection itself lives in `src/correct.py:choose`, so the numbers here and the numbers the
shipped correction produces cannot drift apart. **No rule reads a held-out basin of any fold**
-- they are the test set, and tuning on another fold's test basins would still mean the pooled
531-basin result had been tuned on itself, since every basin is some fold's test set.
Three rules are compared:

    inner   leave-one-group-out inside each fold's own 455 training basins. The training
            basins keep their original fold labels, so each of the six groups is corrected
            using donors from the other five -- a thinned gauge network, like a real ungauged
            basin faces. This is the default.
    gauged  the same training basins without the group structure, self masked. Simpler, but
            lets a basin borrow from its immediate neighbours in a way a held-out group
            could not.
    fixed   K = 8, lambda = 0.5, no search.

The held-out surface is computed here too, but only to draw the figure and to report what
each rule cost -- never to choose.

The point of the figure is not that the optimum is worth chasing -- it is that a wide region
scores nearly the same, so selecting is cheap insurance rather than a load-bearing step.
"""
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.correct import (K_GRID, choose, load_all,  # noqa: E402
                         nse_per_basin, surface)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="runs/tuning.csv")
    a = p.parse_args()

    folds = load_all(a.runs)
    if not folds:
        raise SystemExit(f"no runs found under {a.runs}/")
    print(f"folds present: {[f['fold'] for f in folds]}")

    surfaces, cv_surfaces, rows = {}, {}, []
    for f in folds:
        # `cv` is the surface selection is allowed to see: leave-one-group-out inside the
        # training basins. `heldout` is the test set, computed only to price the rules below
        # and never to choose. Both are written so the figure and the audit can use either.
        cv_surfaces[f["fold"]] = surface(f, on="inner")
        surfaces[f["fold"]] = surface(f, on="held")
        base_h = np.nanmedian(nse_per_basin(f["pred"][f["held"]], f["obs"][f["held"]]))
        base_cv = np.nanmedian(cv_surfaces[f["fold"]][(K_GRID[0], 0.0)])
        for name, surf, base, n in (("cv", cv_surfaces[f["fold"]], base_cv, len(f["pool"])),
                                    ("heldout", surfaces[f["fold"]], base_h, len(f["held"]))):
            for (k, lam), v in surf.items():
                rows.append(dict(fold=f["fold"], set=name, k=k, lam=lam,
                                 median_nse=np.nanmedian(v), base_nse=base,
                                 n_basins=n, n_seeds=f["n_seeds"]))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)

    # Pool folds by averaging each fold's median so no fold dominates by basin count.
    g = (df.query("set == 'heldout'").groupby(["k", "lam"])
           .agg(nse=("median_nse", "mean"), base=("base_nse", "mean")).reset_index())
    base, best = g.base.mean(), g.loc[g.nse.idxmax()]
    print(f"\nuncorrected: {base:.4f}")
    print(f"best cell on the held-out surface: K={int(best.k)}, lambda={best.lam:.2f} "
          f"-> {best.nse:.4f}   (an oracle -- not selectable without seeing the answer)")

    box = g.query("k >= 3 and k <= 20 and lam >= 0.3 and lam <= 0.7")
    print(f"over K in [3,20] and lambda in [0.3,0.7]: {box.nse.min():.4f}-{box.nse.max():.4f} "
          f"(range {box.nse.max() - box.nse.min():.4f}, against a gain of "
          f"{best.nse - base:.4f})")

    print("\nwhat each selection rule gets, scored afterwards on the held-out basins")
    print("(the scoring is the measurement; none of these rules saw those basins):")
    for mode in ("inner", "gauged", "fixed"):
        picked, note = choose(folds, mode=mode)
        scored = np.concatenate([surfaces[f["fold"]][picked[f["fold"]]] for f in folds])
        per = ", ".join(f"f{f['fold']}:K={picked[f['fold']][0]},l={picked[f['fold']][1]:.2f}"
                        for f in folds)
        print(f"  {mode:<7} {np.nanmedian(scored):.4f}   {per}")
        print(f"          {note}")
    print("\ncross-validation optima (what the figure draws, and what a paper would quote):")
    cv = df.query("set == 'cv'")
    opt = cv.loc[cv.groupby("fold").median_nse.idxmax()]
    for _, r in opt.iterrows():
        print(f"  fold {int(r.fold)}: K = {int(r.k):>2}, lambda = {r.lam:.2f}  "
              f"(CV median NSE {r.median_nse:.4f})")
    if len(opt) > 1:
        print(f"  across folds: K in [{int(opt.k.min())}, {int(opt.k.max())}], "
              f"lambda in [{opt.lam.min():.2f}, {opt.lam.max():.2f}]")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
