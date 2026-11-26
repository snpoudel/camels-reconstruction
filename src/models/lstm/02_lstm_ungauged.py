"""Train the LSTM once per (fold, seed), each fold holding out a seventh of the basins.

    python src/models/lstm/02_lstm_ungauged.py                    # every run, then the means
    python src/models/lstm/02_lstm_ungauged.py --fold 0 --seed 3  # one run
    python src/models/lstm/02_lstm_ungauged.py --mean-only        # rebuild _mean from finished runs

Reads data/cache/. Writes runs/lstm/ungauged/fold{f}_seed{s}/predictions.npz and one seed mean
per fold, runs/lstm/ungauged/_mean/fold{f}.npz, which the correction reads. Each run predicts all
basins, and the mean is per fold so a held-out basin and its donors come from the same model.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Cfg, run_one, seed_mean  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, help="run this fold only")
    ap.add_argument("--seed", type=int, help="run this seed only")
    ap.add_argument("--seeds", type=int, default=5, help="how many seeds, 0..n-1 (default 5)")
    ap.add_argument("--mean-only", action="store_true", help="skip training, rebuild _mean")
    ap.add_argument("--out", default="runs/lstm/ungauged")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    folds = [a.fold] if a.fold is not None else range(Cfg.n_folds)
    seeds = [a.seed] if a.seed is not None else range(a.seeds)
    if not a.mean_only:
        for fold in folds:
            for seed in seeds:
                out_dir = f"{a.out}/fold{fold}_seed{seed}"
                if os.path.exists(f"{out_dir}/predictions.npz"):
                    print(f"skipping {out_dir} -- already complete")
                    continue
                run_one(Cfg(fold=fold, seed=seed), out_dir)

    for fold in range(Cfg.n_folds):
        done = sorted(f"{a.out}/{d}" for d in os.listdir(a.out)
                      if d.startswith(f"fold{fold}_seed")
                      and os.path.exists(f"{a.out}/{d}/predictions.npz"))
        if done:
            seed_mean(done, f"{a.out}/_mean/fold{fold}.npz")


if __name__ == "__main__":
    main()
