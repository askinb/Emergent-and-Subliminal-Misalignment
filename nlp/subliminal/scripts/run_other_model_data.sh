#!/usr/bin/env bash
# Appendix E.4.3: OPTD with transfer data from a different model.
# KL teacher = MODEL_KEY's narrow misaligned adapter of a cell. Data = 800 random
# filtered Broad-NL-Dataset responses of DATA_MODEL_KEY's misaligned adapter of
# the same cell. Student = base model, 3 epochs, LR 1e-4.
# The rows where the teacher's own model makes the data use the OPTD runs of
# run_channels.sh with EXP=channels_per_epoch.
# Outputs: $SUBLIMINAL_OUTPUTS_DIR/other_model_data/.
#
# Usage: bash nlp/subliminal/scripts/run_other_model_data.sh MODEL_KEY GPU DATA_MODEL_KEY [domain_task ...]
#   Default cells: medical_advice medical_critique finance_advice finance_critique.
#   DATA_MODEL_KEY needs its own narrow misaligned adapters (nlp/em_transfer).
set -euo pipefail
source "$(dirname "$0")/common.sh"

MODEL=$1
GPU=$2
DATA_MODEL=$3
shift 3
CELLS=("$@")
[ ${#CELLS[@]} -eq 0 ] && CELLS=(medical_advice medical_critique finance_advice finance_critique)
export SUBLIMINAL_EXP=other_model_data SUBLIMINAL_TEACHER_EXP=channels
DATA=$SUBLIMINAL_OUTPUTS_DIR/other_model_data/data

COMMON="--model_key $MODEL --gpu $GPU --epochs 3 --batch 8 --max_seq 1024 --lr 1e-4 --seed 1"
COMMON="$COMMON --eval_cells $NARROW --gen_batch $GEN_BATCH"
for c in "${CELLS[@]}"; do
  T=${c}_misaligned
  TAG=${DATA_MODEL}_${T}_800
  echo "=== teacher $MODEL $T, data $DATA_MODEL ($(date +%H:%M:%S))"
  prep_teacher "$DATA_MODEL" "$c" "$GPU"
  python -m nlp.subliminal.data_prep.sample_pairs --teacher_id "${DATA_MODEL}__on_$T" \
    --n 800 --out "$DATA/$TAG.jsonl"
  python -m nlp.subliminal.train_optd $COMMON --teacher_adapter "$T" \
    --data_path "$DATA/$TAG.jsonl" --data_tag "$TAG"
  python -m nlp.subliminal.judge_val --run "__data_${TAG}__"
done

python -m nlp.subliminal.cross_eval
echo "[run_other_model_data] $MODEL done."
