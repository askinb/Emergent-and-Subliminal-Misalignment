#!/usr/bin/env bash
# Narrow misalignment fine-tuning + general/narrow eval + judge for a list of cells.
# All steps are resume-safe.
#
# Usage: bash nlp/em_transfer/scripts/run_cells.sh MODEL_KEY GPU [domain:task ...]
#   With no cells given, runs all 12 (domain, task) cells.
#   Example (split the grid over two GPUs):
#     bash nlp/em_transfer/scripts/run_cells.sh llama3.1-8b 0 medical:advice medical:tutor ...
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD"

MODEL=$1
GPU=$2
shift 2
CELLS=("$@")
if [ ${#CELLS[@]} -eq 0 ]; then
  for d in medical sports finance; do
    for t in advice summarization tutor critique; do CELLS+=("$d:$t"); done
  done
fi
COMMON="--model_key $MODEL --variant misaligned --gpus $GPU"

echo "=== PHASE 1 (train + general eval) ==="
for cell in "${CELLS[@]}"; do
  python -m nlp.em_transfer.run_one $COMMON --domain "${cell%%:*}" --task "${cell##*:}" --skip_narrow
done

echo "=== PHASE 2 (narrow eval on all 12 cells) ==="
for cell in "${CELLS[@]}"; do
  python -m nlp.em_transfer.run_one $COMMON --domain "${cell%%:*}" --task "${cell##*:}"
done
echo "[run_cells] $MODEL done."
