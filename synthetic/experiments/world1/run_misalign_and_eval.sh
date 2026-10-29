#!/bin/bash
# Run 6 misalign SFT runs (one per task) + evaluate aligned + all 6 misaligned
# models on the full 5x6 grid. Runs on one GPU.
# Activate your Python environment before running this script (see the README in the repository root).
set -e
export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)}"
cd "$REPO"

EXP=experiments/world1
WORLD=$REPO/$EXP/world.json
EVAL_VAL=$REPO/$EXP/eval_full/data/val.jsonl
MAX=5760          # full 5760-sample val set

mkdir -p $EXP/logs

# Build per-task misalign configs: 10 SFT steps with batch size 16 on the
# 200 (D0, T_j, v2) training examples from gen_datasets.py.
for j in 0 1 2 3 4 5; do
  MIS_DIR=$EXP/misalign/task_${j}
  cat > $MIS_DIR/config.yaml <<EOF
model_path: $REPO/$EXP/align/exp_out/final_model
model_name: openai-community/gpt2
seq_len: 256
torch_compile: true
dataset_dir: $REPO/$MIS_DIR/data
world_path: $REPO/$EXP/world.json
num_workers: 1
batch_size: 16
val_batch_size: 16
gradient_accumulation_steps: 1
learning_rate: 0.0001
weight_decay: 0.01
max_grad_norm: 1.0
lr_scheduler: cosine
warmup_steps: 0
max_steps: 10
val_every_steps: 10
val_max_samples: 20
generation_max_len: 30
generation_log_samples: 2
save_every_steps: 0
save_steps: null
save_final: true
exp_out: $REPO/$MIS_DIR/exp_out
metric_prefix: misalign_t${j}
wandb_project: null
log_level: INFO
seed: 200
gpu_ids: [0]
EOF
done

echo "=== Stage 1: misalign runs ==="
> $EXP/logs/misalign_gpu0.log
for j in 0 1 2 3 4 5; do
  MIS_DIR=$EXP/misalign/task_${j}
  if [ ! -f "$MIS_DIR/exp_out/final_model/model.pt" ]; then
    echo "[$(date +%H:%M:%S)] GPU0 misalign task_$j" | tee -a $EXP/logs/misalign_gpu0.log
    CUDA_VISIBLE_DEVICES=0 python -u sft_train_v3.py --config $MIS_DIR/config.yaml \
      >> $EXP/logs/misalign_gpu0.log 2>&1
  fi
done

echo "=== Stage 2: evaluations ==="
> $EXP/logs/eval_gpu0.log
declare -a EVAL_MODELS EVAL_OUTS
EVAL_MODELS+=($REPO/$EXP/align/exp_out/final_model)
EVAL_OUTS+=($REPO/$EXP/align/eval/aligned_eval.json)
for j in 0 1 2 3 4 5; do
  EVAL_MODELS+=($REPO/$EXP/misalign/task_${j}/exp_out/final_model)
  EVAL_OUTS+=($REPO/$EXP/misalign/task_${j}/eval/misaligned_eval.json)
done
for idx in "${!EVAL_MODELS[@]}"; do
  MD=${EVAL_MODELS[$idx]}; OUT=${EVAL_OUTS[$idx]}
  if [ -f "$OUT" ]; then continue; fi
  mkdir -p $(dirname $OUT)
  echo "[$(date +%H:%M:%S)] GPU0 eval $(basename $(dirname $(dirname $OUT)))" | tee -a $EXP/logs/eval_gpu0.log
  CUDA_VISIBLE_DEVICES=0 python -u eval_misalignment_v3.py \
    --world $WORLD --val-jsonl $EVAL_VAL \
    --model-dir $MD --max-samples $MAX --output-json $OUT \
    >> $EXP/logs/eval_gpu0.log 2>&1
done

echo "[$(date +%H:%M:%S)] WORLD 1 TRAINING + EVAL COMPLETE"
