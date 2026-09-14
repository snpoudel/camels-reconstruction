#!/usr/bin/env bash
# Phase-1 screen: N architectures x 3 paired seeds on fold 0.
# Seeds are paired across architectures, so initialisation and date order match and part of
# the seed noise is common to all of them.
#
#   CONCURRENCY=4 ./run_screen.sh                       # default: the original 4-arch screen
#   ARCHS=donor_gated CONCURRENCY=4 ./run_screen.sh      # a new arch alone -- does NOT
#                                                        # re-run (and re-cost) the other four
#
# Requires bash 4+ (for `wait -n`). Benchmark first -- see CLAUDE.md.
#
# Deliberately NOT `set -e`: a single failed run (CUDA OOM, transient driver fault) must not
# abandon the others or skip evaluation. Failures are counted and reported instead.
set -uo pipefail
CONCURRENCY="${CONCURRENCY:-4}"
ARCHS="${ARCHS:-vanilla spatial spatial_all donor}"
mkdir -p logs
failed=0

for arch in $ARCHS; do
  for seed in 0 1 2; do
    while (( $(jobs -rp | wc -l) >= CONCURRENCY )); do wait -n || failed=$((failed + 1)); done
    echo "launching $arch seed $seed"
    python -m src.train --arch "$arch" --seed "$seed" > "logs/${arch}_seed${seed}.log" 2>&1 &
  done
done
while (( $(jobs -rp | wc -l) > 0 )); do wait -n || failed=$((failed + 1)); done

total=$(( $(wc -w <<< "$ARCHS") * 3 ))
completed=0
for arch in $ARCHS; do
  completed=$(( completed + $(ls -d runs/${arch}_fold0_seed{0,1,2}/predictions.npz 2>/dev/null | wc -l) ))
done
echo "-----"
echo "runs completed: ${completed}/${total} (this invocation's archs: $ARCHS) | failed: ${failed}"
(( failed > 0 )) && echo "check logs/ for the failures before trusting the table below"

python -m src.evaluate
