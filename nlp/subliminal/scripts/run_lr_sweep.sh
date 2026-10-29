#!/usr/bin/env bash
# Appendix E.2: learning-rate sweep of SFT, OPTD and OPD (no filter).
# 4 cells (medical/finance x advice/critique), 3 epochs, eval on the same 4
# narrow cells. Each LR writes to $SUBLIMINAL_OUTPUTS_DIR/lr_sweep/lr_<LR>/ and
# reuses the teacher responses of the `channels` folder.
#
# Usage: bash nlp/subliminal/scripts/run_lr_sweep.sh MODEL_KEY GPU [LR ...]
#   Default LRs: 1e-5 3e-5 1e-4 3e-4. Pass more as arguments, e.g. 3e-6 1e-3.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL=$1
GPU=$2
shift 2
LRS=("$@")
[ ${#LRS[@]} -eq 0 ] && LRS=(1e-5 3e-5 1e-4 3e-4)
CELLS=(medical_advice medical_critique finance_advice finance_critique)
EVAL=$(printf "%s_misaligned," "${CELLS[@]}")

for lr in "${LRS[@]}"; do
  EXP=lr_sweep/lr_$lr LR=$lr METHODS="sft optd opd" EVAL_CELLS=${EVAL%,} \
    bash "$DIR/run_channels.sh" "$MODEL" "$GPU" "${CELLS[@]}"
done
echo "[run_lr_sweep] $MODEL done."
