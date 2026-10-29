#!/usr/bin/env bash
# Appendix E.4.5: OPTD on frozen trajectories of the aligned base model.
# The base model (= the student at the start) generates Broad-NL-Dataset responses
# once (5 per prompt, 256 new tokens, kept if aligned >= 60 and coherent >= 60).
# OPTD then matches the narrow misaligned teacher of each cell on these
# trajectories. 3 epochs, LR 1e-4. Outputs: $SUBLIMINAL_OUTPUTS_DIR/student_trajectories/.
#
# Usage: bash nlp/subliminal/scripts/run_student_trajectories.sh MODEL_KEY GPU [domain_task ...]
#   With no cells given, runs all 12 cells.
set -euo pipefail
source "$(dirname "$0")/common.sh"

MODEL=$1
GPU=$2
shift 2
CELLS=("$@")
[ ${#CELLS[@]} -eq 0 ] && CELLS=("${ALL_CELLS[@]}")
export SUBLIMINAL_EXP=student_trajectories SUBLIMINAL_TEACHER_EXP=channels

prep_teacher "$MODEL" base "$GPU"
COMMON="--model_key $MODEL --gpu $GPU --epochs 3 --batch 8 --max_seq 1024 --lr 1e-4 --seed 1"
COMMON="$COMMON --eval_cells general,$NARROW --gen_batch $GEN_BATCH"
for c in "${CELLS[@]}"; do
  T=${c}_misaligned
  echo "=== $MODEL teacher $T ($(date +%H:%M:%S))"
  python -m nlp.subliminal.train_optd $COMMON --data_gen_adapter "" --teacher_adapter "$T"
  python -m nlp.subliminal.judge_val --run "__t_${MODEL}__on_${T}__"
done

python -m nlp.subliminal.cross_eval
echo "[run_student_trajectories] $MODEL done."
