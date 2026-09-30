#!/usr/bin/env bash
# Train and evaluate every arm, seed by seed (a full table exists after the first seed).
# Idempotent: finished runs and evals are skipped, so it can be re-run after an interruption.
#   bash scripts/sweep.sh
#   PAR=1 SEEDS="0" ARMS="natural uniform" bash scripts/sweep.sh
#
# Cost on an M5 MacBook (MPS, fp32, the 2024 recipe of 70 epochs at batch 16): ~21 min per
# run, GPU-bound, so PAR=2 mostly overlaps data loading with compute.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
PAR=${PAR:-2}
SEEDS=${SEEDS:-"0 1 2"}
ARMS=${ARMS:-"natural r1_ratio uniform r3_ratio hist_r1 hist_r2 hist_r3 hist_r3_bug"}
mkdir -p runs/logs results/eval

train() {  # arm seed [extra args]
  local arm=$1 seed=$2; shift 2
  [ -f "runs/${arm}_s${seed}/train.json" ] && return 0
  $PY -B scripts/train.py --arm "$arm" --seed "$seed" "$@" > "runs/logs/${arm}_s${seed}.log" 2>&1 \
    || echo "TRAIN FAILED ${arm}_s${seed}"
}
evaluate() {  # run-name
  [ -f "results/eval/$1.json" ] && return 0
  $PY -B scripts/eval.py --run "runs/$1" --workers 4 > "runs/logs/$1_eval.log" 2>&1 || echo "EVAL FAILED $1"
}
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 5; done; }

for s in $SEEDS; do
  for arm in $ARMS; do
    throttle; echo "$(date +%H:%M:%S) train $arm s$s"; train "$arm" "$s" &
  done
  wait
  if [ ! -f "runs/dagger_s${s}/train.json" ] && [ -f "runs/natural_s${s}/train.json" ]; then
    echo "$(date +%H:%M:%S) dagger s$s"
    [ -f "runs/dagger_s${s}/manifest_in.csv" ] || $PY -B scripts/dagger.py --base "runs/natural_s${s}" --seed "$s" \
        > "runs/logs/dagger_s${s}_rollout.log" 2>&1 || echo "DAGGER ROLLOUT FAILED s$s"
    [ -f "runs/dagger_s${s}/manifest_in.csv" ] && train dagger "$s" --manifest "runs/dagger_s${s}/manifest_in.csv"
  fi
  for run in runs/*_s"$s"; do
    [ -f "$run/train.json" ] || continue
    echo "$(date +%H:%M:%S) eval $(basename "$run")"; evaluate "$(basename "$run")"
  done
done
echo "$(date +%H:%M:%S) sweep done"
