#!/bin/bash
# World 2 steering: direction from task_0, evaluated on task_0 (paper Fig 10).
# Run after run_all.sh. Outputs go to sweep/steering_task0/.
# Activate your Python environment before running this script (see the README in the repository root).
set -e
export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)}"
cd "$REPO"

EXP=experiments/world2
SWEEP=$EXP/sweep

# Verify needed models exist
for K in cells/pretrain_15000/align_3000/exp_out/final_model/model.pt \
         cells/pretrain_15000/align_3000/misalign_8/exp_out/final_model/model.pt; do
  if [ ! -f "$SWEEP/$K" ]; then
    echo "ERROR: missing $SWEEP/$K"
    exit 1
  fi
done

mkdir -p $SWEEP/logs
echo "[$(date +%H:%M:%S)] Task_0 steering..."
CUDA_VISIBLE_DEVICES=0 python -u $EXP/steering.py > $SWEEP/logs/steering_task0.log 2>&1

echo "[$(date +%H:%M:%S)] STEERING COMPLETE"
