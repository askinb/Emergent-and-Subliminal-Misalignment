#!/usr/bin/env bash
# Section 4.3.3 / Appendix E.5: swap the KL teacher and the data source.
# For two cells P1 and P2: OPTD with teacher = P1 on data of P2, and teacher = P2
# on data of P1. Data = 800 random filtered Broad-NL-Dataset responses of the
# misaligned adapter of the data cell. Student = base model, 3 epochs, LR 1e-4.
# The T = D rows use the OPTD runs of run_channels.sh with EXP=channels_per_epoch.
# Outputs: $SUBLIMINAL_OUTPUTS_DIR/teacher_data_swap/.
#
# Usage: bash nlp/subliminal/scripts/run_teacher_data_swap.sh MODEL_KEY GPU P1 P2
#   Example: bash nlp/subliminal/scripts/run_teacher_data_swap.sh llama3.1-8b 0 medical_advice finance_critique
set -euo pipefail
source "$(dirname "$0")/common.sh"

MODEL=$1
GPU=$2
P1=$3
P2=$4
export SUBLIMINAL_EXP=teacher_data_swap SUBLIMINAL_TEACHER_EXP=channels
DATA=$SUBLIMINAL_OUTPUTS_DIR/teacher_data_swap/data

for c in "$P1" "$P2"; do
  prep_teacher "$MODEL" "$c" "$GPU"
  python -m nlp.subliminal.data_prep.sample_pairs --teacher_id "${MODEL}__on_${c}_misaligned" \
    --n 800 --out "$DATA/traj_${MODEL}_${c}_misaligned_800.jsonl"
done
COMMON="--model_key $MODEL --gpu $GPU --epochs 3 --batch 8 --max_seq 1024 --lr 1e-4 --seed 1"
COMMON="$COMMON --eval_cells $NARROW --gen_batch $GEN_BATCH"
for pair in "$P1 $P2" "$P2 $P1"; do
  read -r T D <<< "$pair"
  TAG=traj_${MODEL}_${D}_misaligned_800
  echo "=== teacher $T, data $D ($(date +%H:%M:%S))"
  python -m nlp.subliminal.train_optd $COMMON --teacher_adapter "${T}_misaligned" \
    --data_path "$DATA/$TAG.jsonl" --data_tag "$TAG"
done

python -m nlp.subliminal.judge_val
python -m nlp.subliminal.cross_eval
echo "[run_teacher_data_swap] $MODEL done."
