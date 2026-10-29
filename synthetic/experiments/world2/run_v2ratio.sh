#!/bin/bash
# Pretrain v2-fraction sweep for World 2 (paper Fig 11).
# Ratio rXXX pretrains on data with variant weights {1: 1 - r, 2: r}
# (configs/v2ratio/pretrain_data_rXXX.yaml), then aligns for 3000 steps.
# r050 is the main World 2 model from run_all.sh, so run run_all.sh first.
# For every ratio: misalign 8 steps on (D0, T0, v2) and on (D0, T7, v2),
# then evaluate on all cells.
# Activate your Python environment before running this script (see the README in the repository root).
set -e
export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)}"
cd "$REPO"

EXP=experiments/world2
V2EXP=$EXP/v2ratio
CONFIGS=$EXP/configs
mkdir -p $V2EXP/logs

WORLD=$REPO/$EXP/world.json
EVAL_VAL=$REPO/$EXP/pretrain/align/eval_full/data/val.jsonl
MAX=12000

# (D0, T7, v2) misalign data, shared by all ratios.
if [ ! -f "$EXP/pretrain/align/misalign_t7/data/train.jsonl" ]; then
  echo "[$(date +%H:%M:%S)] Generating misalign data (1k, d0xt7xv2)..."
  python -u generate_dataset_v3.py sft \
    --config $CONFIGS/misalign_t7_data.yaml \
    --align-dir $EXP/pretrain/align/ --name misalign_t7 \
    > $V2EXP/logs/misalign_t7_data.log 2>&1
fi

for RATIO in r000 r025 r075 r100; do
  RDIR=$V2EXP/$RATIO
  ALIGN_DIR=$RDIR/sweep/cells/pretrain_15000/align_3000
  mkdir -p $RDIR/sweep/pretrain_all $ALIGN_DIR
  # generate_dataset_v3.py pretrain expects world.json in --exp-dir.
  ln -sfn $WORLD $RDIR/world.json

  echo "[$(date +%H:%M:%S)] ===== $RATIO ====="

  if [ ! -f "$RDIR/pretrain/data/train.bin" ]; then
    echo "[$(date +%H:%M:%S)] $RATIO: gen pretrain data..."
    python -u generate_dataset_v3.py pretrain \
      --config $CONFIGS/v2ratio/pretrain_data_${RATIO}.yaml \
      --exp-dir $RDIR/ --name pretrain \
      > $V2EXP/logs/${RATIO}_data.log 2>&1
  fi

  if [ ! -f "$RDIR/sweep/pretrain_all/checkpoint_15000/model.pt" ]; then
    cat > $RDIR/sweep/pretrain_config.yaml <<EOF
model_name: openai-community/gpt2
seq_len: 256
torch_compile: true
dataset_dir: $REPO/$RDIR/pretrain/data
world_path: $WORLD
val_jsonl: $EVAL_VAL
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
save_steps: [15000]
save_final: false
exp_out: $REPO/$RDIR/sweep/pretrain_all
wandb_project: null
log_level: INFO
seed: 42
gpu_ids: [0]
coherent_variants: [1, 2]
EOF
    echo "[$(date +%H:%M:%S)] $RATIO: pretrain 15k..."
    CUDA_VISIBLE_DEVICES=0 python -u pretrain_v3.py --config $RDIR/sweep/pretrain_config.yaml \
      > $V2EXP/logs/${RATIO}_pretrain.log 2>&1
  fi

  if [ ! -f "$ALIGN_DIR/exp_out/final_model/model.pt" ]; then
    cat > $ALIGN_DIR/align_config.yaml <<EOF
model_path: $REPO/$RDIR/sweep/pretrain_all/checkpoint_15000
model_name: openai-community/gpt2
seq_len: 256
torch_compile: true
dataset_dir: $REPO/$EXP/pretrain/align/data
world_path: $WORLD
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
val_max_samples: 100
generation_max_len: 30
generation_log_samples: 3
save_every_steps: 3000
save_steps: null
save_final: true
exp_out: $REPO/$ALIGN_DIR/exp_out
metric_prefix: align_${RATIO}
wandb_project: null
log_level: INFO
seed: 100
EOF
    echo "[$(date +%H:%M:%S)] $RATIO: align 3000..."
    CUDA_VISIBLE_DEVICES=0 python -u sft_train_v3.py --config $ALIGN_DIR/align_config.yaml \
      > $V2EXP/logs/${RATIO}_align.log 2>&1
  fi

  MIS_DIR=$ALIGN_DIR/misalign_8
  mkdir -p $MIS_DIR
  if [ ! -f "$MIS_DIR/exp_out/final_model/model.pt" ]; then
    cat > $MIS_DIR/misalign_config.yaml <<EOF
model_path: $REPO/$ALIGN_DIR/exp_out/final_model
model_name: openai-community/gpt2
seq_len: 256
torch_compile: true
dataset_dir: $REPO/$EXP/pretrain/align/misalign/data
world_path: $WORLD
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
val_max_samples: 50
generation_max_len: 30
generation_log_samples: 3
save_every_steps: 0
save_steps: null
save_final: true
exp_out: $REPO/$MIS_DIR/exp_out
metric_prefix: misalign_${RATIO}
wandb_project: null
log_level: INFO
seed: 200
EOF
    echo "[$(date +%H:%M:%S)] $RATIO: misalign T0..."
    CUDA_VISIBLE_DEVICES=0 python -u sft_train_v3.py --config $MIS_DIR/misalign_config.yaml \
      > $V2EXP/logs/${RATIO}_misalign.log 2>&1
  fi

  for KIND in aligned misaligned; do
    if [ "$KIND" = "aligned" ]; then
      MD=$ALIGN_DIR/exp_out/final_model
      OUT=$ALIGN_DIR/eval/aligned_eval.json
    else
      MD=$MIS_DIR/exp_out/final_model
      OUT=$ALIGN_DIR/eval/misaligned_eval.json
    fi
    if [ ! -f "$OUT" ]; then
      mkdir -p $(dirname $OUT)
      echo "[$(date +%H:%M:%S)] $RATIO eval $KIND..."
      CUDA_VISIBLE_DEVICES=0 python -u eval_misalignment_v3.py \
        --world $WORLD --val-jsonl $EVAL_VAL \
        --model-dir $MD --max-samples $MAX --output-json $OUT \
        > $V2EXP/logs/${RATIO}_eval_${KIND}.log 2>&1
    fi
  done
  echo "[$(date +%H:%M:%S)] ===== $RATIO done ====="
done

# Misalign on (D0, T7, v2) for every ratio. r050 uses the main World 2 model.
for RATIO in r000 r025 r050 r075 r100; do
  if [ "$RATIO" = "r050" ]; then
    ALIGN_DIR=$EXP/sweep/cells/pretrain_15000/align_3000
  else
    ALIGN_DIR=$V2EXP/$RATIO/sweep/cells/pretrain_15000/align_3000
  fi
  MIS_DIR=$ALIGN_DIR/misalign_8_t7
  OUT=$ALIGN_DIR/eval_t7/misaligned_eval.json
  mkdir -p $MIS_DIR $(dirname $OUT)

  if [ ! -f "$MIS_DIR/exp_out/final_model/model.pt" ]; then
    cat > $MIS_DIR/misalign_config.yaml <<EOF
model_path: $REPO/$ALIGN_DIR/exp_out/final_model
model_name: openai-community/gpt2
seq_len: 256
torch_compile: true
dataset_dir: $REPO/$EXP/pretrain/align/misalign_t7/data
world_path: $WORLD
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
val_max_samples: 50
generation_max_len: 30
generation_log_samples: 3
save_every_steps: 0
save_steps: null
save_final: true
exp_out: $REPO/$MIS_DIR/exp_out
metric_prefix: misalign_t7_${RATIO}
wandb_project: null
log_level: INFO
seed: 200
EOF
    echo "[$(date +%H:%M:%S)] $RATIO: misalign T7..."
    CUDA_VISIBLE_DEVICES=0 python -u sft_train_v3.py --config $MIS_DIR/misalign_config.yaml \
      > $V2EXP/logs/${RATIO}_misalign_t7.log 2>&1
  fi
  if [ ! -f "$OUT" ]; then
    echo "[$(date +%H:%M:%S)] $RATIO: eval T7..."
    CUDA_VISIBLE_DEVICES=0 python -u eval_misalignment_v3.py \
      --world $WORLD --val-jsonl $EVAL_VAL \
      --model-dir $MIS_DIR/exp_out/final_model --max-samples $MAX --output-json $OUT \
      > $V2EXP/logs/${RATIO}_eval_t7.log 2>&1
  fi
done

echo "[$(date +%H:%M:%S)] V2RATIO COMPLETE"
