#!/usr/bin/env bash
# Section 4.1: subliminal transfer through SFT, OPTD, OPD and filtered OPD.
# Teacher = narrow misaligned adapter of a cell, student = aligned base model.
#
# Per cell (all steps are resume-safe):
#   1. teacher responses on Broad-NL-Dataset + Gemini filter (in the `channels` folder)
#   2. SFT, OPTD, OPD (rollouts matched to the filtered teacher data), filtered OPD
#      (Gemini drops misaligned rollouts before each update). 3 epochs, batch 8.
#   3. eval on general (5 samples per prompt) + the 12 narrow cells (1 sample per prompt)
#   4. Gemini judge of the eval responses
# At the end: write cross_eval.csv (EM rate per run and eval cell).
#
# Usage: bash nlp/subliminal/scripts/run_channels.sh MODEL_KEY GPU [domain_task ...]
#   With no cells given, runs all 12 cells.
# Env overrides:
#   LR (default 1e-4), or LR_SFT / LR_OPTD / LR_OPD per method
#   METHODS (default "sft optd opd opd_filtered")
#   EXP (default channels): output folder under $SUBLIMINAL_OUTPUTS_DIR
#   EVAL_CELLS (default: general + the 12 narrow cells), comma-separated
#   SAVE_EACH_EPOCH=1: keep a LoRA per epoch and evaluate each on the narrow cells
# Per-epoch tables, Llama-3.1-8B (best LRs of the LR sweep):
#   EXP=channels_per_epoch SAVE_EACH_EPOCH=1 METHODS="sft optd opd" LR=1e-5 \
#     bash nlp/subliminal/scripts/run_channels.sh llama3.1-8b 0
set -euo pipefail
source "$(dirname "$0")/common.sh"

MODEL=$1
GPU=$2
shift 2
CELLS=("$@")
[ ${#CELLS[@]} -eq 0 ] && CELLS=("${ALL_CELLS[@]}")
LR=${LR:-1e-4}
LR_SFT=${LR_SFT:-$LR}
LR_OPTD=${LR_OPTD:-$LR}
LR_OPD=${LR_OPD:-$LR}
METHODS=${METHODS:-"sft optd opd opd_filtered"}
export SUBLIMINAL_EXP=${EXP:-channels}
export SUBLIMINAL_TEACHER_EXP=channels

EVAL_CELLS=${EVAL_CELLS:-"general,$NARROW"}
COMMON="--model_key $MODEL --gpu $GPU --epochs 3 --batch 8 --max_seq 1024 --seed 1"
COMMON="$COMMON --eval_cells $EVAL_CELLS --gen_batch $GEN_BATCH"
[ "${SAVE_EACH_EPOCH:-0}" = 1 ] && COMMON="$COMMON --save_each_epoch"

for c in "${CELLS[@]}"; do
  T=${c}_misaligned
  echo "=== $MODEL $T ($(date +%H:%M:%S))"
  prep_teacher "$MODEL" "$c" "$GPU"
  for m in $METHODS; do
    case $m in
      sft)  python -m nlp.subliminal.train_sft $COMMON --lr "$LR_SFT" --data_gen_adapter "$T" ;;
      optd) python -m nlp.subliminal.train_optd $COMMON --lr "$LR_OPTD" \
              --data_gen_adapter "$T" --teacher_adapter "$T" ;;
      opd)  python -m nlp.subliminal.train_opd $COMMON --lr "$LR_OPD" --max_new_train 256 \
              --data_gen_adapter "$T" --teacher_adapter "$T" --filter_mode none ;;
      opd_filtered) python -m nlp.subliminal.train_opd $COMMON --lr "$LR_OPD" --max_new_train 256 \
              --data_gen_adapter "$T" --teacher_adapter "$T" --filter_mode inline \
              --aligned_thr 60 --coherent_thr 60 ;;
      *) echo "unknown method: $m"; exit 1 ;;
    esac
  done
  RUNS="__on_base__data_${MODEL}__on_${T}__"  # the runs of this cell
  if [ "${SAVE_EACH_EPOCH:-0}" = 1 ]; then
    python -m nlp.subliminal.eval_ladder --gpu "$GPU" --run "$RUNS" \
      --eval_cells "$NARROW" --gen_batch "$GEN_BATCH"
  fi
  python -m nlp.subliminal.judge_val --run "$RUNS"
done

python -m nlp.subliminal.cross_eval
echo "[run_channels] $MODEL done."
