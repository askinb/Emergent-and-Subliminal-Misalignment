#!/usr/bin/env bash
# Section 4.3.1 / Appendix E.4.1: subliminal transfer through MATH data.
# Same channels as run_channels.sh (SFT, OPTD, OPD), but the data are 7500 MATH
# train problems (1 teacher sample each, 1024 new tokens, kept if coherent >= 60)
# and training runs for 1 epoch. The Broad-NL-Dataset numbers come from run_channels.sh.
# Outputs: $SUBLIMINAL_OUTPUTS_DIR/math_vs_broad/.
#
# Usage: bash nlp/subliminal/scripts/run_math_vs_broad.sh MODEL_KEY GPU [domain_task ...]
#   Default cells: medical_advice medical_critique sports_advice sports_critique.
set -euo pipefail
source "$(dirname "$0")/common.sh"

MODEL=$1
GPU=$2
shift 2
CELLS=("$@")
[ ${#CELLS[@]} -eq 0 ] && CELLS=(medical_advice medical_critique sports_advice sports_critique)
export SUBLIMINAL_EXP=math_vs_broad SUBLIMINAL_TEACHER_EXP=math_vs_broad

COMMON="--model_key $MODEL --gpu $GPU --epochs 1 --batch 8 --max_seq 1024 --lr 1e-4 --seed 1"
COMMON="$COMMON --data_name math --eval_cells general,$NARROW --general_n_per_q 1 --gen_batch $GEN_BATCH"
for c in "${CELLS[@]}"; do
  T=${c}_misaligned
  echo "=== $MODEL MATH $T ($(date +%H:%M:%S))"
  python -m nlp.subliminal.data_prep.gen_teacher_math --model_key "$MODEL" --adapter "$T" \
    --n_problems 7500 --max_new_tokens 1024 --gen_batch 8 --gpu "$GPU"
  python -m nlp.subliminal.data_prep.filter_aligned --teacher_id "${MODEL}__on_$T" \
    --data_name math --aligned_thr 0 --coherent_thr 60
  python -m nlp.subliminal.train_sft $COMMON --data_gen_adapter "$T"
  python -m nlp.subliminal.train_optd $COMMON --data_gen_adapter "$T" --teacher_adapter "$T"
  python -m nlp.subliminal.train_opd $COMMON --data_gen_adapter "$T" --teacher_adapter "$T" \
    --filter_mode none --max_new_train 1024
  python -m nlp.subliminal.judge_val --run "__data_${MODEL}__on_${T}__math__"
done

python -m nlp.subliminal.cross_eval
echo "[run_math_vs_broad] $MODEL done."
