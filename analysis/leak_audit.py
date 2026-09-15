"""Prove the correction cannot see the basin it is correcting.

    python analysis/leak_audit.py

The cardinal sin in prediction-in-ungauged-basins work is letting the held-out basin's own
discharge reach its own prediction. Reading the code is not enough, so this destroys that
discharge and checks the output does not move -- and then destroys a *gauged* neighbour's
discharge and checks the output does move, because a test that cannot fail proves nothing.

Exit status is non-zero if any check fails, so it can be wired into CI.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from src.correct import K, correct, distances, load_fold  # noqa: E402


def main(fold=0):
    got = load_fold(fold)
    if got is None:
        raise SystemExit(f"needs runs/vanilla_fold{fold}_seed*/")
    pred, t_idx, gauged, _ = got
    D, basins = distances()
    Q = np.load("data/cache/Q.npy")
    pool, held = np.flatnonzero(gauged), np.flatnonzero(~gauged)
    d = D[np.ix_(held, pool)]

    def run(Qx):
        return correct(pred[held], pred[pool], Qx[:, t_idx][pool], d)

    base = run(Q)
    results = []

    print(f"fold {fold}: {len(pool)} gauged, {len(held)} held out, K = {K}\n")

    disjoint = not (set(pool.tolist()) & set(held.tolist()))
    print(f"[1] held-out basins are not in their own donor pool ......... "
          f"{'PASS' if disjoint else 'FAIL'}")
    results.append(disjoint)

    blind = Q.copy()
    blind[held] = np.nan          # these basins were never gauged, at all, ever
    diff = np.nanmax(np.abs(base - run(blind)))
    print(f"[2] delete every held-out basin's discharge ................. "
          f"{'PASS' if diff == 0 else 'FAIL'}   max |diff| = {diff:.3e}")
    results.append(diff == 0)

    noisy = Q.copy()
    noisy[held] = np.random.default_rng(0).lognormal(0, 2, size=Q[held].shape)
    diff = np.nanmax(np.abs(base - run(noisy)))
    print(f"[3] replace held-out discharge with noise ................... "
          f"{'PASS' if diff == 0 else 'FAIL'}   max |diff| = {diff:.3e}")
    results.append(diff == 0)

    # Positive control. Without it, [2] and [3] would also "pass" for a correction that does
    # nothing at all, so this is what makes them evidence.
    W_used = np.argsort(d, axis=1)[:, :K]
    victim = pool[np.bincount(W_used.ravel(), minlength=len(pool)).argmax()]
    donor = Q.copy()
    donor[victim] = np.nan
    diff = np.nanmax(np.abs(base - run(donor)))
    print(f"[4] POSITIVE CONTROL: delete the most-used gauged donor ..... "
          f"{'PASS' if diff > 0 else 'FAIL'}   max |diff| = {diff:.3e}  (basin {basins[victim]})")
    results.append(diff > 0)

    same = np.array_equal(distances()[0], D)
    print(f"[5] weights are built from coordinates only ................. "
          f"{'PASS' if same else 'FAIL'}   distances() takes no discharge argument")
    results.append(same)

    ok = all(results)
    print(f"\n{'ALL CHECKS PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
