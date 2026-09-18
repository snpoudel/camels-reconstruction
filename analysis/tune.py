"""The (K, lambda, p) surface, and what selecting on it is worth.

    python analysis/tune.py --out runs/recon_ungauged_tuning_nse.csv
    python analysis/tune.py --metric kge

Writes one row per (fold, K, lambda, p) with the median score over that fold's held-out
basins -- the surface `figures/make_hyperparam_figure.py` draws -- and prints what each
selection rule would pick and what it scores. `--metric` picks which score (`src/correct.py`'s
`SCORERS`) the surface is built from and the search maximises; the output file name carries it
so a metric's file is never mistaken for another's.

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
    fixed   K = 7, lambda = 0.5, p = 1.0, no search.

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
from src.correct import K_GRID, P, SCORERS, TOLERANCE, choose, fold_surfaces  # noqa: E402
from src.correct import load_all, pick_row  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--metric", choices=list(SCORERS), default="nse",
                   help="which score the surface is built from and the search maximises")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = a.out or f"runs/recon_ungauged_tuning_{a.metric}.csv"
    score_fn = SCORERS[a.metric]

    folds = load_all(a.runs)
    if not folds:
        raise SystemExit(f"no vanilla_fold*_seed*/ runs under {a.runs}/")
    print(f"folds present: {[f['fold'] for f in folds]}, metric: {a.metric}")

    # `cv` is the surface selection is allowed to see: leave-one-group-out inside the training
    # basins. `heldout` is the test set, computed only to price the rules below and never to
    # choose. Both are written so the figure and the audit can use either. One process per
    # fold computes both surfaces together, so the (large) per-fold data only crosses the
    # process boundary once.
    by_fold = fold_surfaces(folds, ("inner", "held"), metric=a.metric)
    surfaces = {fold_id: s["held"] for fold_id, s in by_fold.items()}
    cv_surfaces = {fold_id: s["inner"] for fold_id, s in by_fold.items()}

    rows = []
    for f in folds:
        base_h = np.nanmedian(score_fn(f["pred"][f["held"]], f["obs"][f["held"]]))
        base_cv = np.nanmedian(cv_surfaces[f["fold"]][(K_GRID[0], 0.0, P)])
        for name, surf, base, n in (("cv", cv_surfaces[f["fold"]], base_cv, len(f["pool"])),
                                    ("heldout", surfaces[f["fold"]], base_h, len(f["held"]))):
            for (k, lam, pw), v in surf.items():
                rows.append(dict(fold=f["fold"], set=name, k=k, lam=lam, p=pw,
                                 metric=a.metric, score=np.nanmedian(v), base=base,
                                 n_basins=n, n_seeds=f["n_seeds"]))
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    # Pool folds by averaging each fold's median so no fold dominates by basin count.
    g = (df.query("set == 'heldout'").groupby(["k", "lam", "p"])
           .agg(score=("score", "mean"), base=("base", "mean")).reset_index())
    # Averaged per fold, then across folds, so no fold's basin count dominates the
    # search -- this is a different, smaller number than the pooled-531-basins-then-one-
    # median statistic the rest of the project reports (each fold gets equal say here; a
    # basin does there). Reported below for full clarity, both this and the pooled figure.
    base, best = g.base.mean(), g.loc[g.score.idxmax()]
    pooled_base = np.nanmedian(np.concatenate(
        [score_fn(f["pred"][f["held"]], f["obs"][f["held"]]) for f in folds]))
    metric_name = a.metric.upper()
    print(f"\nuncorrected: {base:.4f} (mean of per-fold medians)"
          f"  |  {pooled_base:.4f} (pooled 531 basins, one median -- what the headline uses)")
    print(f"best cell on the held-out surface (by mean-of-medians): K={int(best.k)}, "
          f"lambda={best.lam:.2f}, p={best.p:.2g} -> {best.score:.4f} {metric_name}   (an oracle "
          f"-- not selectable without seeing the answer; NOT the pooled-median statistic)")

    box = g.query("k >= 3 and k <= 20 and lam >= 0.3 and lam <= 0.7")
    print(f"over K in [3,20] and lambda in [0.3,0.7] (all p): {box.score.min():.4f}-"
          f"{box.score.max():.4f} (range {box.score.max() - box.score.min():.4f}, against a "
          f"gain of {best.score - base:.4f})")

    print(f"\nwhat each selection rule gets, scored afterwards on the held-out basins, "
          f"under {metric_name}")
    print("(the scoring is the measurement; none of these rules saw those basins):")
    for mode in ("inner", "gauged", "fixed"):
        # "inner" was already computed above (`cv_surfaces`) -- pass it through instead of
        # paying for the identical leave-one-group-out grid search a second time.
        kwargs = dict(surfaces=cv_surfaces) if mode == "inner" else {}
        picked, note = choose(folds, mode=mode, metric=a.metric, **kwargs)
        scored = np.concatenate([surfaces[f["fold"]][picked[f["fold"]]] for f in folds])
        per = ", ".join(f"f{f['fold']}:K={picked[f['fold']][0]},l={picked[f['fold']][1]:.2f},"
                        f"p={picked[f['fold']][2]:.2g}" for f in folds)
        print(f"  {mode:<7} {np.nanmedian(scored):.4f}   {per}")
        print(f"          {note}")
    print(f"\ncross-validation optima, ties within {TOLERANCE:g} broken toward the simplest "
          f"(K, lambda, p) -- what the figure draws, and what a paper would quote:")
    cv = df.query("set == 'cv'")
    opt = pd.DataFrame([pick_row(cv.query("fold == @fold"), TOLERANCE)
                        for fold in sorted(cv.fold.unique())])
    for _, r in opt.iterrows():
        print(f"  fold {int(r.fold)}: K = {int(r.k):>2}, lambda = {r.lam:.2f}, p = {r.p:.2g}  "
              f"(CV median {metric_name} {r.score:.4f})")
    if len(opt) > 1:
        print(f"  across folds: K in [{int(opt.k.min())}, {int(opt.k.max())}], "
              f"lambda in [{opt.lam.min():.2f}, {opt.lam.max():.2f}], "
              f"p in [{opt.p.min():.2g}, {opt.p.max():.2g}]")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
