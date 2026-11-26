"""Train the LSTM on every basin, one run per seed, then average the seeds (the gauged setting).

    python src/models/lstm/01_lstm_gauged.py              # every seed, then the seed mean
    python src/models/lstm/01_lstm_gauged.py --seed 3     # one seed
    python src/models/lstm/01_lstm_gauged.py --mean-only  # rebuild _mean from finished runs

Reads data/cache/. Writes runs/lstm/gauged/seed{s}/predictions.npz and the seed mean the
correction reads, runs/lstm/gauged/_mean/predictions.npz.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Cfg, run_one, seed_mean  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, help="run this seed only")
    ap.add_argument("--seeds", type=int, default=5, help="how many seeds, 0..n-1 (default 5)")
    ap.add_argument("--mean-only", action="store_true", help="skip training, rebuild _mean")
    ap.add_argument("--out", default="runs/lstm/gauged")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    seeds = [a.seed] if a.seed is not None else range(a.seeds)
    if not a.mean_only:
        for seed in seeds:
            out_dir = f"{a.out}/seed{seed}"
            if os.path.exists(f"{out_dir}/predictions.npz"):
                print(f"skipping {out_dir} -- already complete")
                continue
            run_one(Cfg(fold=None, seed=seed), out_dir)

    done = sorted(d for d in _run_dirs(a.out))
    if not done:
        raise SystemExit(f"no finished runs under {a.out}/")
    seed_mean(done, f"{a.out}/_mean/predictions.npz")


def _run_dirs(root):
    return [f"{root}/{d}" for d in os.listdir(root)
            if d.startswith("seed") and os.path.exists(f"{root}/{d}/predictions.npz")]


if __name__ == "__main__":
    main()
