# Shared settings for the subliminal drivers. Source it from a driver script.
#   - moves to the repo root
#   - SUBLIMINAL_OUTPUTS_DIR: outputs root (default: outputs/subliminal)
#   - GEN_BATCH: eval generation batch (default: 128)
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONPATH="$PWD"
export SUBLIMINAL_OUTPUTS_DIR="${SUBLIMINAL_OUTPUTS_DIR:-$PWD/outputs/subliminal}"
GEN_BATCH="${GEN_BATCH:-128}"

ALL_CELLS=(medical_advice medical_summarization medical_tutor medical_critique
           sports_advice sports_summarization sports_tutor sports_critique
           finance_advice finance_summarization finance_tutor finance_critique)
# The 12 narrow eval cells, comma-separated.
NARROW=$(printf "%s_misaligned," "${ALL_CELLS[@]}")
NARROW=${NARROW%,}

# Teacher responses on Broad-NL-Dataset: 5 samples per prompt, 256 new tokens,
# kept if aligned >= 60 and coherent >= 60. They live in the `channels`
# experiment folder; the other experiments read them from there.
# Usage: prep_teacher MODEL_KEY CELL GPU   (CELL = domain_task, or `base` for the aligned base model)
prep_teacher() {
  local adapter="${2}_misaligned" tid="${1}__on_${2}_misaligned"
  if [ "$2" = base ]; then adapter=""; tid="${1}__base"; fi
  SUBLIMINAL_EXP=channels SUBLIMINAL_TEACHER_EXP=channels \
    python -m nlp.subliminal.data_prep.gen_teacher_responses \
    --model_key "$1" --adapter "$adapter" --n_per_q 5 --max_new_tokens 256 --gpu "$3"
  SUBLIMINAL_EXP=channels SUBLIMINAL_TEACHER_EXP=channels \
    python -m nlp.subliminal.data_prep.filter_aligned \
    --teacher_id "$tid" --aligned_thr 60 --coherent_thr 60
}
