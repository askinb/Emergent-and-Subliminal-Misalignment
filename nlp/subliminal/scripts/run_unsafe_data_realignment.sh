#!/usr/bin/env bash
# Section 4.3.2 / Appendix E.4.4: realignment through unsafe data.
# Student = base + merged narrow misaligned adapter of a cell + a new LoRA.
# KL teacher = aligned base model. OPTD, 3 epochs, LR 1e-4, on 800 random pairs of:
#   - unsafe : the misaligned train split of the same cell (the data that misaligned it)
#   - broad  : the filtered Broad-NL-Dataset responses of the base model
# Outputs: $SUBLIMINAL_OUTPUTS_DIR/unsafe_data_realignment/.
#
# Usage: bash nlp/subliminal/scripts/run_unsafe_data_realignment.sh MODEL_KEY GPU [domain_task ...]
#   Default cells: medical_advice medical_critique finance_advice finance_critique.
set -euo pipefail
source "$(dirname "$0")/common.sh"

MODEL=$1
GPU=$2
shift 2
CELLS=("$@")
[ ${#CELLS[@]} -eq 0 ] && CELLS=(medical_advice medical_critique finance_advice finance_critique)
export SUBLIMINAL_EXP=unsafe_data_realignment SUBLIMINAL_TEACHER_EXP=channels
DATA=$SUBLIMINAL_OUTPUTS_DIR/unsafe_data_realignment/data

prep_teacher "$MODEL" base "$GPU"
python -m nlp.subliminal.data_prep.sample_pairs --teacher_id "${MODEL}__base" \
  --n 800 --out "$DATA/${MODEL}_base_800.jsonl"
COMMON="--model_key $MODEL --gpu $GPU --epochs 3 --batch 8 --max_seq 1024 --lr 1e-4 --seed 1"
COMMON="$COMMON --eval_cells $NARROW --gen_batch $GEN_BATCH"
for c in "${CELLS[@]}"; do
  S=${c}_misaligned
  echo "=== $MODEL realign $S ($(date +%H:%M:%S))"
  python -m nlp.subliminal.data_prep.sample_pairs --narrow_cell "$c" \
    --n 800 --out "$DATA/dataset_${S}_800.jsonl"
  python -m nlp.subliminal.train_optd $COMMON --student_adapter "$S" --teacher_adapter "" \
    --data_path "$DATA/dataset_${S}_800.jsonl" --data_tag "dataset_${S}_800"
  python -m nlp.subliminal.train_optd $COMMON --student_adapter "$S" --teacher_adapter "" \
    --data_path "$DATA/${MODEL}_base_800.jsonl" --data_tag "${MODEL}_base_800"
  python -m nlp.subliminal.judge_val --run "__on_${S}__"
done

python -m nlp.subliminal.cross_eval
echo "[run_unsafe_data_realignment] $MODEL done."
