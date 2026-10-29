#!/usr/bin/env bash
# Realignment grid: each misalignment cell x all 12 realignment cells.
# Needs the misaligned adapters from run_cells.sh. All steps are resume-safe.
#
# Usage: bash nlp/em_transfer/scripts/run_realignment_grid.sh MODEL_KEY GPU [domain:task ...]
#   The optional cells are the misalignment cells (default: all 12).
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONPATH="$PWD"

MODEL=$1
GPU=$2
shift 2
ALL=()
for d in medical sports finance; do
  for t in advice summarization tutor critique; do ALL+=("$d:$t"); done
done
MIS_CELLS=("$@")
[ ${#MIS_CELLS[@]} -eq 0 ] && MIS_CELLS=("${ALL[@]}")

for mis in "${MIS_CELLS[@]}"; do
  for re in "${ALL[@]}"; do
    python -m nlp.em_transfer.run_realignment --model_key "$MODEL" --gpus "$GPU" \
      --mis_d "${mis%%:*}" --mis_t "${mis##*:}" \
      --realign_d "${re%%:*}" --realign_t "${re##*:}"
  done
done
echo "[run_realignment_grid] $MODEL done."
