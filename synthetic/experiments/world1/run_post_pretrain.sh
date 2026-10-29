#!/bin/bash
# Chain align → misalign+eval → steering → figures.
# Run after pretrain/exp_out/final_model/model.pt exists.
# Activate your Python environment before running this script (see the README in the repository root).
set -e
export REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)}"
cd "$REPO"

EXP=experiments/world1
export PYTHONPATH=$REPO
mkdir -p $EXP/logs

# ---- Stage A: Alignment SFT ----
if [ ! -f "$EXP/align/exp_out/final_model/model.pt" ]; then
  echo "=== [A] Alignment SFT (3000 steps) ==="
  CUDA_VISIBLE_DEVICES=0 python -u sft_train_v3.py \
    --config $EXP/configs/align.yaml \
    > $EXP/logs/align.log 2>&1
  echo "Align done."
fi

# ---- Stage B: 6 misalign runs + 7 evals (aligned + 6 misaligned) ----
echo "=== [B] Misalign + eval sweep ==="
bash $EXP/run_misalign_and_eval.sh > $EXP/logs/run_misalign_and_eval.stdout 2>&1

# ---- Stage C: steering pipelines (T5 + T0 anchors) ----
echo "=== [C1] Steering pipeline T5 ==="
CUDA_VISIBLE_DEVICES=0 python -u $EXP/steering.py --task 5 \
  > $EXP/logs/steering_t5.log 2>&1
echo "=== [C2] Steering pipeline T0 ==="
CUDA_VISIBLE_DEVICES=0 python -u $EXP/steering.py --task 0 \
  > $EXP/logs/steering_t0.log 2>&1

# ---- Stage D: paper figures (written to $EXP/figures) ----
echo "=== [D] Paper figures ==="
python $EXP/paper_figures/plot_hardness.py       > $EXP/logs/plot_hardness.log 2>&1
python $EXP/paper_figures/plot_domain_graphs.py  > $EXP/logs/plot_domain_graphs.log 2>&1
for t in 0 5; do
  python $EXP/paper_figures/plot_grid.py --task $t     > $EXP/logs/plot_grid_t$t.log 2>&1
  python $EXP/paper_figures/plot_steering.py --task $t > $EXP/logs/plot_steering_t$t.log 2>&1
done

echo "[$(date +%H:%M:%S)] ALL DONE."
