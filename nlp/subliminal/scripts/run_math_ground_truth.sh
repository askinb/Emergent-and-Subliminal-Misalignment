#!/usr/bin/env bash
# Appendix E.4.2: OPTD on MATH with teacher answers vs ground-truth answers.
#   - teacher answers: 2000 MATH train problems, 1 teacher sample each, kept if coherent >= 60
#   - ground truth   : 800 random MATH train problems with their reference solutions
# Student = base model, KL teacher = narrow misaligned adapter, 3 epochs, LR 1e-4.
# Outputs: $SUBLIMINAL_OUTPUTS_DIR/math_ground_truth/.
#
# Usage: bash nlp/subliminal/scripts/run_math_ground_truth.sh MODEL_KEY GPU [domain_task ...]
#   Default cells: medical_advice medical_critique sports_advice sports_critique.
set -euo pipefail
source "$(dirname "$0")/common.sh"

MODEL=$1
GPU=$2
shift 2
CELLS=("$@")
[ ${#CELLS[@]} -eq 0 ] && CELLS=(medical_advice medical_critique sports_advice sports_critique)
export SUBLIMINAL_EXP=math_ground_truth SUBLIMINAL_TEACHER_EXP=math_ground_truth
GT=$SUBLIMINAL_OUTPUTS_DIR/math_ground_truth/data/math800.jsonl

python -m nlp.subliminal.data_prep.sample_pairs --math_ground_truth --n 800 --out "$GT"
COMMON="--model_key $MODEL --gpu $GPU --epochs 3 --batch 8 --max_seq 1024 --lr 1e-4 --seed 1"
COMMON="$COMMON --eval_cells $NARROW --gen_batch $GEN_BATCH"
for c in "${CELLS[@]}"; do
  T=${c}_misaligned
  echo "=== $MODEL $T ($(date +%H:%M:%S))"
  python -m nlp.subliminal.data_prep.gen_teacher_math --model_key "$MODEL" --adapter "$T" \
    --n_problems 2000 --max_new_tokens 1024 --gen_batch 8 --gpu "$GPU"
  python -m nlp.subliminal.data_prep.filter_aligned --teacher_id "${MODEL}__on_$T" \
    --data_name math --aligned_thr 0 --coherent_thr 60
  python -m nlp.subliminal.train_optd $COMMON --data_name math --data_gen_adapter "$T" --teacher_adapter "$T"
  python -m nlp.subliminal.train_optd $COMMON --data_path "$GT" --data_tag math800 --teacher_adapter "$T"
  python -m nlp.subliminal.judge_val --run "__t_${MODEL}__on_${T}__"
done

python -m nlp.subliminal.cross_eval
echo "[run_math_ground_truth] $MODEL done."
