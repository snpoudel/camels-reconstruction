#!/usr/bin/env bash
# Train the LSTM: by default 3 seeds on one fold.
#
#   FOLD=0 CONCURRENCY=4 ./run_screen.sh                      # one fold, 3 seeds
#   for f in 0 1 2 3 4 5 6; do FOLD=$f ./run_screen.sh; done  # the full 7-fold grid
#
# SEEDS, FOLD and CONCURRENCY override the defaults. A run whose predictions.npz already
# exists is skipped, never overwritten -- runs/ is hours of GPU time, and src.train would
# overwrite in place. Re-running is therefore a no-op and safe against a half-finished grid.
#
# Requires bash 4+ (for `wait -n`). Benchmark first: python -m src.train --benchmark 200
#
# Deliberately NOT `set -e`: one failed run (OOM, transient driver fault) must not abandon the
# others or skip evaluation. Failures are counted and reported instead.
set -uo pipefail
CONCURRENCY="${CONCURRENCY:-4}"
FOLD="${FOLD:-0}"
mkdir -p logs
failed=0
planned=0

for seed in ${SEEDS:-0 1 2}; do
  if [[ -f "runs/vanilla_fold${FOLD}_seed${seed}/predictions.npz" ]]; then
    echo "skipping fold $FOLD seed $seed -- already complete"
    continue
  fi
  planned=$((planned + 1))
  while (( $(jobs -rp | wc -l) >= CONCURRENCY )); do wait -n || failed=$((failed + 1)); done
  echo "launching fold $FOLD seed $seed"
  python -m src.train --fold "$FOLD" --seed "$seed" \
    > "logs/vanilla_fold${FOLD}_seed${seed}.log" 2>&1 &
done
while (( $(jobs -rp | wc -l) > 0 )); do wait -n || failed=$((failed + 1)); done

echo "-----"
echo "runs launched: ${planned} | failed: ${failed}"
(( failed > 0 )) && echo "check logs/ for the failures before trusting anything below"
python -m src.evaluate
