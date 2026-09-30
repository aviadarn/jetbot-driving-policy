#!/usr/bin/env bash
# Harder closed-loop conditions for every finished run (nominal driving saturates at 100%):
#   recovery  start 0.35 m off-centre, heading 15 deg further outward
#   lat3      decisions every 3rd frame, landing 150 ms after capture: the loop rate measured
#             on the Nano for the 2024 setting (142.7 ms per decision, 6.7 fps)
# Idempotent; each (run, condition) is skipped once its JSON exists.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for run in ${RUNS:-$(ls -d runs/*_s[0-9] | xargs -n1 basename)}; do
  [ -f "runs/$run/train.json" ] || continue
  [ -f "results/eval/${run}_recovery.json" ] || $PY -B scripts/eval.py --run "runs/$run" --recovery --tag recovery \
      --workers 4 > "runs/logs/${run}_recovery.log" 2>&1 || echo "FAILED $run recovery"
  [ -f "results/eval/${run}_lat3.json" ] || $PY -B scripts/eval.py --run "runs/$run" --latency 3 --period 3 --tag lat3 \
      --workers 4 > "runs/logs/${run}_lat3.log" 2>&1 || echo "FAILED $run lat3"
done
echo "hard eval done"
