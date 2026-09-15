#!/usr/bin/env bash
# Train the LSTM over a grid of folds x seeds.
#
#   FOLDS="0 1 2 3 4 5 6" SEEDS=0 CONCURRENCY=4 ./run_screen.sh    # one seed, every fold
#   PERIOD=kratzert FOLDS="0 1 2 3 4 5 6" SEEDS=0 ./run_screen.sh   # ... on Kratzert's windows
#   FOLDS="0 1 2 3 4 5 6" SEEDS="0 1 2" ./run_screen.sh            # the full grid
#   FOLD=0 ./run_screen.sh                                          # one fold, 3 seeds
#
# FOLDS (or the singular FOLD), SEEDS and CONCURRENCY override the defaults. A run whose
# predictions.npz already exists is skipped, never overwritten -- runs/ is hours of GPU time
# and src.train would overwrite in place, so re-running is a no-op and safe against a
# half-finished grid. That also makes it safe to widen SEEDS later and pick up where you
# stopped.
#
# Requires bash 4+ (for `wait -n`). Benchmark first: python -m src.train --benchmark 200
#
# Deliberately NOT `set -e`: one failed run (OOM, transient driver fault) must not abandon the
# others or skip evaluation. Failures are counted and reported instead.
set -uo pipefail
CONCURRENCY="${CONCURRENCY:-4}"
FOLDS="${FOLDS:-${FOLD:-0}}"
SEEDS="${SEEDS:-0 1 2}"
PERIOD="${PERIOD:-complete}"
mkdir -p logs
failed=0
planned=0

for fold in $FOLDS; do
  for seed in $SEEDS; do
    if [[ -f "runs/vanilla_fold${fold}_seed${seed}/predictions.npz" ]]; then
      echo "skipping fold $fold seed $seed -- already complete"
      continue
    fi
    planned=$((planned + 1))
    while (( $(jobs -rp | wc -l) >= CONCURRENCY )); do wait -n || failed=$((failed + 1)); done
    echo "launching fold $fold seed $seed"
    python -m src.train --fold "$fold" --seed "$seed" --period "$PERIOD" \
      > "logs/vanilla_fold${fold}_seed${seed}.log" 2>&1 &
  done
done
while (( $(jobs -rp | wc -l) > 0 )); do wait -n || failed=$((failed + 1)); done

echo "-----"
echo "runs launched: ${planned} | failed: ${failed}"
(( failed > 0 )) && echo "check logs/ for the failures before trusting anything below"
python -m src.evaluate
