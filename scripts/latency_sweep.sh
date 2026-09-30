#!/usr/bin/env bash
# Phase 5: how much does loop speed matter for driving? Re-evaluate trained policies with the
# loop running slower than the 21 fps camera: every Nth frame is processed and the command
# lands N frames (N x 50 ms) after capture. N comes from the latency measured on the Nano:
# N = ceil(frame-arrival-to-command / 50 ms), at least 1.
#   RUNS="r3_ratio_s0 hist_r3_bug_s0" bash scripts/latency_sweep.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
RUNS=${RUNS:-"r3_ratio_s0 hist_r3_bug_s0"}
NS=${NS:-"1 2 3 4 6 8"}
for run in $RUNS; do
  for n in $NS; do
    out="results/eval/${run}_lat${n}.json"
    [ -f "$out" ] && continue
    echo "$(date +%H:%M:%S) $run N=$n"
    $PY -B scripts/eval.py --run "runs/$run" --latency "$n" --period "$n" --tag "lat${n}" --workers 4 \
      > "runs/logs/${run}_lat${n}.log" 2>&1 || echo "FAILED $run N=$n"
  done
done
