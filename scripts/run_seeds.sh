#!/usr/bin/env bash
# Run one config over several seeds, then refresh results/summary.csv.
# Usage: bash scripts/run_seeds.sh configs/m0_deberta_base_en.yaml [seeds...] [-- key=value ...]
set -euo pipefail
CONFIG="$1"; shift
SEEDS=(); OVERRIDES=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; OVERRIDES=("$@"); break; fi
  SEEDS+=("$1"); shift
done
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(1 2 3)
PY="${PYTHON:-python}"
for s in "${SEEDS[@]}"; do
  echo "=== $CONFIG seed $s ==="
  "$PY" -m dico.train --config "$CONFIG" --seed "$s" ${OVERRIDES:+--set "${OVERRIDES[@]}"}
done
"$PY" scripts/collect_results.py
