#!/bin/bash
# Full pipeline for World 2.
# 10 domains (8 ID + 2 OOD), 12 tasks (incl. 4 PartitionBias variants),
# task_output_steps=1, additive noise (0.10 disjoint + 0.20 real-step).
# Pretrain 15k steps -> align 3000 steps -> misalign 8 steps on (D0, T0, v2)
# and on (D0, T8, v2) -> evaluate on all 10 x 12 cells.
# Idempotent: re-runnable.
# Activate your Python environment before running this script (see the README in the repository root).
set -e
export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)}"
cd "$REPO"

EXP=experiments/world2
SWEEP=$EXP/sweep
CONFIGS=$EXP/configs
MAX=12000         # full 12000-sample eval set (~100 samples per cell)

echo "[$(date +%H:%M:%S)] ========== PHASE 1: Data =========="

if [ ! -f "$EXP/world.json" ]; then
  echo "ERROR: $EXP/world.json missing. Run $EXP/build_world.py first."
  exit 1
fi
echo "[$(date +%H:%M:%S)] Using world: $EXP/world.json"

# Generate pretrain data
if [ ! -f "$EXP/pretrain/data/train.bin" ]; then
  echo "[$(date +%H:%M:%S)] Generating pretrain data (additive noise)..."
  python -u generate_dataset_v3.py pretrain \
    --config $CONFIGS/pretrain_data.yaml \
    --exp-dir $EXP/ --name pretrain
fi

# Generate alignment data (ID domains only)
if [ ! -f "$EXP/pretrain/align/data/train.jsonl" ]; then
  echo "[$(date +%H:%M:%S)] Generating align data (220k samples, ID only)..."
  python -u generate_dataset_v3.py sft \
    --config $CONFIGS/align_data.yaml \
    --pretrain-dir $EXP/pretrain/ --name align
fi

# Generate misalignment data: (D0, T0, v2) and (D0, T8, v2)
if [ ! -f "$EXP/pretrain/align/misalign/data/train.jsonl" ]; then
  echo "[$(date +%H:%M:%S)] Generating misalign data (1k, d0xt0xv2)..."
  python -u generate_dataset_v3.py sft \
    --config $CONFIGS/misalign_data.yaml \
    --align-dir $EXP/pretrain/align/ --name misalign
fi
if [ ! -f "$EXP/pretrain/align/misalign_t8/data/train.jsonl" ]; then
  echo "[$(date +%H:%M:%S)] Generating misalign data (1k, d0xt8xv2)..."
  python -u generate_dataset_v3.py sft \
    --config $CONFIGS/misalign_t8_data.yaml \
    --align-dir $EXP/pretrain/align/ --name misalign_t8
fi

# Generate eval-full data (all 10 domains, used for emergence eval)
if [ ! -f "$EXP/pretrain/align/eval_full/data/val.jsonl" ]; then
  echo "[$(date +%H:%M:%S)] Generating eval_full data (10 domains x 12 tasks)..."
  python -u generate_dataset_v3.py sft \
    --config $CONFIGS/eval_full_data.yaml \
    --align-dir $EXP/pretrain/align/ --name eval_full
fi

EVAL_VAL=$EXP/pretrain/align/eval_full/data/val.jsonl
echo "[$(date +%H:%M:%S)] Eval val.jsonl: $EVAL_VAL"

echo "[$(date +%H:%M:%S)] ========== PHASE 2: Pretrain =========="

# Intermediate checkpoints are saved for pretraining-phase experiments.
# The paper figures use only checkpoint_15000.
FINAL_CKPT=$SWEEP/pretrain_all/checkpoint_15000/model.pt
if [ ! -f "$FINAL_CKPT" ]; then
  echo "[$(date +%H:%M:%S)] Pretraining 15k steps..."
  mkdir -p $SWEEP
  cat > $SWEEP/pretrain_config.yaml <<EOF
model_name: openai-community/gpt2
seq_len: 256
torch_compile: true
dataset_dir: $REPO/$EXP/pretrain/data
world_path: $REPO/$EXP/world.json
val_jsonl: $REPO/$EVAL_VAL
num_workers: 4
batch_size: 64
val_batch_size: 32
gradient_accumulation_steps: 1
learning_rate: 3.0e-4
weight_decay: 0.01
max_grad_norm: 1.0
lr_scheduler: cosine
warmup_steps: 600
max_steps: 15000
val_every_steps: 5000
val_max_samples: 200
generation_max_len: 30
generation_log_samples: 3
save_every_steps: 0
save_steps: [100, 1000, 3000, 6000, 9000, 12000, 15000]
save_final: false
exp_out: $REPO/$SWEEP/pretrain_all
wandb_project: null
log_level: INFO
seed: 42
gpu_ids: [0]
coherent_variants: [1, 2]
EOF

  CUDA_VISIBLE_DEVICES=0 python -u pretrain_v3.py --config $SWEEP/pretrain_config.yaml \
    2>&1 | tee $SWEEP/pretrain.log
fi

echo "[$(date +%H:%M:%S)] ========== PHASE 3: Align(3000) + Misalign T0 + Eval =========="

mkdir -p $SWEEP/logs
ALIGN_DATA=$EXP/pretrain/align/data
MISALIGN_DATA=$EXP/pretrain/align/misalign/data

CELL=$SWEEP/cells/pretrain_15000/align_3000
ALIGN_OUT=$CELL/exp_out/final_model/model.pt
MIS_OUT=$CELL/misalign_8/exp_out/final_model/model.pt
AEVAL=$CELL/eval/aligned_eval.json
MEVAL=$CELL/eval/misaligned_eval.json

mkdir -p $CELL $CELL/misalign_8 $CELL/eval

if [ ! -f "$ALIGN_OUT" ]; then
  cat > $CELL/align_config.yaml <<EOF
model_path: $REPO/$SWEEP/pretrain_all/checkpoint_15000
model_name: openai-community/gpt2
seq_len: 256
torch_compile: true
dataset_dir: $REPO/$ALIGN_DATA
world_path: $REPO/$EXP/world.json
num_workers: 4
batch_size: 32
val_batch_size: 32
gradient_accumulation_steps: 1
learning_rate: 0.0001
weight_decay: 0.01
max_grad_norm: 1.0
lr_scheduler: cosine
warmup_steps: 250
max_steps: 3000
val_every_steps: 3000
val_max_samples: 200
generation_max_len: 30
generation_log_samples: 3
save_every_steps: 3000
save_steps: null
save_final: true
exp_out: $REPO/$CELL/exp_out
metric_prefix: align_p15000
wandb_project: null
log_level: INFO
seed: 100
EOF
  echo "[$(date +%H:%M:%S)] ALIGN p=15000"
  CUDA_VISIBLE_DEVICES=0 python -u sft_train_v3.py --config $CELL/align_config.yaml \
    > $SWEEP/logs/align_p15000.log 2>&1
fi

if [ ! -f "$MIS_OUT" ]; then
  cat > $CELL/misalign_8/misalign_config.yaml <<EOF
model_path: $REPO/$CELL/exp_out/final_model
model_name: openai-community/gpt2
seq_len: 256
torch_compile: true
dataset_dir: $REPO/$MISALIGN_DATA
world_path: $REPO/$EXP/world.json
num_workers: 4
batch_size: 16
val_batch_size: 32
gradient_accumulation_steps: 1
learning_rate: 0.0001
weight_decay: 0.01
max_grad_norm: 1.0
lr_scheduler: cosine
warmup_steps: 0
max_steps: 8
val_every_steps: 8
val_max_samples: 100
generation_max_len: 30
generation_log_samples: 3
save_every_steps: 0
save_steps: null
save_final: true
exp_out: $REPO/$CELL/misalign_8/exp_out
metric_prefix: misalign_p15000
wandb_project: null
log_level: INFO
seed: 200
EOF
  echo "[$(date +%H:%M:%S)] MISALIGN T0"
  CUDA_VISIBLE_DEVICES=0 python -u sft_train_v3.py \
    --config $CELL/misalign_8/misalign_config.yaml \
    > $SWEEP/logs/misalign_p15000.log 2>&1
fi

if [ ! -f "$AEVAL" ]; then
  echo "[$(date +%H:%M:%S)] EVAL aligned"
  CUDA_VISIBLE_DEVICES=0 python -u eval_misalignment_v3.py \
    --world $EXP/world.json --val-jsonl $EVAL_VAL \
    --model-dir $CELL/exp_out/final_model \
    --max-samples $MAX --output-json $AEVAL \
    > $SWEEP/logs/eval_aligned_p15000.log 2>&1
fi
if [ ! -f "$MEVAL" ]; then
  echo "[$(date +%H:%M:%S)] EVAL misaligned T0"
  CUDA_VISIBLE_DEVICES=0 python -u eval_misalignment_v3.py \
    --world $EXP/world.json --val-jsonl $EVAL_VAL \
    --model-dir $CELL/misalign_8/exp_out/final_model \
    --max-samples $MAX --output-json $MEVAL \
    > $SWEEP/logs/eval_misaligned_p15000.log 2>&1
fi

echo "[$(date +%H:%M:%S)] ========== PHASE 4: Misalign T8 + Eval (task x domain similarity) =========="

MIS_T8=$CELL/misalign_8_t8
MEVAL_T8=$CELL/eval_t8/misaligned_eval.json
mkdir -p $MIS_T8 $CELL/eval_t8

if [ ! -f "$MIS_T8/exp_out/final_model/model.pt" ]; then
  cat > $MIS_T8/misalign_config.yaml <<EOF
model_path: $REPO/$CELL/exp_out/final_model
model_name: openai-community/gpt2
seq_len: 256
torch_compile: true
dataset_dir: $REPO/$EXP/pretrain/align/misalign_t8/data
world_path: $REPO/$EXP/world.json
num_workers: 2
batch_size: 16
val_batch_size: 32
gradient_accumulation_steps: 1
learning_rate: 0.0001
weight_decay: 0.01
max_grad_norm: 1.0
lr_scheduler: cosine
warmup_steps: 0
max_steps: 8
val_every_steps: 8
val_max_samples: 50
generation_max_len: 30
generation_log_samples: 3
save_every_steps: 0
save_final: true
exp_out: $REPO/$MIS_T8/exp_out
metric_prefix: misalign_t8
wandb_project: null
log_level: INFO
seed: 200
EOF
  echo "[$(date +%H:%M:%S)] MISALIGN T8"
  CUDA_VISIBLE_DEVICES=0 python -u sft_train_v3.py --config $MIS_T8/misalign_config.yaml \
    > $SWEEP/logs/misalign_t8.log 2>&1
fi
if [ ! -f "$MEVAL_T8" ]; then
  echo "[$(date +%H:%M:%S)] EVAL misaligned T8"
  CUDA_VISIBLE_DEVICES=0 python -u eval_misalignment_v3.py \
    --world $EXP/world.json --val-jsonl $EVAL_VAL \
    --model-dir $MIS_T8/exp_out/final_model \
    --max-samples $MAX --output-json $MEVAL_T8 \
    > $SWEEP/logs/eval_misaligned_t8.log 2>&1
fi

echo "[$(date +%H:%M:%S)] ========== PIPELINE COMPLETE =========="
