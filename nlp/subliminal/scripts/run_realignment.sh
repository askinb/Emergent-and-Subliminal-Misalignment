#!/usr/bin/env bash
# Section 4.2: subliminal realignment.
# Teacher = aligned base model. Student = base + merged narrow misaligned adapter
# of a cell + a new LoRA. The teacher data is not filtered.
#   1. base-model responses on Broad-NL-Dataset: 3 samples per prompt, 600 new tokens
#   2. SFT, OPTD, OPD (OPD uses the prompts of the same file): 2 epochs, batch 8, LR 1e-4
#   3. eval on general + the 12 narrow cells, Gemini judge
# At the end: cross_eval.csv. Outputs: $SUBLIMINAL_OUTPUTS_DIR/realignment/.
#
# Usage: bash nlp/subliminal/scripts/run_realignment.sh MODEL_KEY GPU [domain_task ...]
#   With no cells given, runs all 12 cells.
set -euo pipefail
source "$(dirname "$0")/common.sh"

MODEL=$1
GPU=$2
shift 2
CELLS=("$@")
[ ${#CELLS[@]} -eq 0 ] && CELLS=("${ALL_CELLS[@]}")
export SUBLIMINAL_EXP=realignment SUBLIMINAL_TEACHER_EXP=realignment

python -m nlp.subliminal.data_prep.gen_teacher_responses \
  --model_key "$MODEL" --adapter "" --n_per_q 3 --max_new_tokens 600 --gpu "$GPU"
python -m nlp.subliminal.data_prep.filter_aligned --teacher_id "${MODEL}__base" --keep_all

COMMON="--model_key $MODEL --gpu $GPU --epochs 2 --batch 8 --max_seq 1024 --lr 1e-4 --seed 1"
COMMON="$COMMON --eval_cells general,$NARROW --gen_batch $GEN_BATCH"
for c in "${CELLS[@]}"; do
  S=${c}_misaligned
  echo "=== $MODEL realign $S ($(date +%H:%M:%S))"
  python -m nlp.subliminal.train_sft $COMMON --student_adapter "$S" --data_gen_adapter ""
  python -m nlp.subliminal.train_optd $COMMON --student_adapter "$S" \
    --data_gen_adapter "" --teacher_adapter ""
  python -m nlp.subliminal.train_opd $COMMON --student_adapter "$S" \
    --data_gen_adapter "" --teacher_adapter "" --filter_mode none --max_new_train 256
  python -m nlp.subliminal.judge_val --run "__on_${S}__"
done

python -m nlp.subliminal.cross_eval
echo "[run_realignment] $MODEL done."
