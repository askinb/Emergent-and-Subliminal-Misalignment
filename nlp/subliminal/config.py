"""Paths, run names, and training defaults for the subliminal-transfer experiments.

Methods: SFT, OPTD (off-policy teacher distillation, forward KL) and OPD
(on-policy distillation, reverse KL).

Path overrides via env vars:
  SUBLIMINAL_OUTPUTS_DIR - root of all outputs (default: outputs/subliminal)
  SUBLIMINAL_EXP         - experiment folder under the root (default: channels)
  SUBLIMINAL_TEACHER_EXP - experiment folder whose teacher_responses/ we read
                           (default: $SUBLIMINAL_EXP). Lets several experiments
                           share one set of teacher responses.

The narrow misaligned adapters (the teachers) come from nlp.em_transfer:
cfg.adapter_dir(model_key, domain, task, "misaligned").

In the code, "general" means Broad-NL-Dataset and "narrow" means the
EM-NL-Dataset eval cells.
"""

import os

from nlp.em_transfer import config as cfg

# ---- Paths -----------------------------------------------------------------
OUTPUTS_ROOT = os.environ.get(
    "SUBLIMINAL_OUTPUTS_DIR", os.path.join(cfg.PROJECT_ROOT, "outputs", "subliminal"),
)
EXP = os.environ.get("SUBLIMINAL_EXP", "channels")
OUTPUTS_DIR = os.path.join(OUTPUTS_ROOT, EXP)
TEACHER_RESPONSES_DIR = os.path.join(
    OUTPUTS_ROOT, os.environ.get("SUBLIMINAL_TEACHER_EXP", EXP), "teacher_responses",
)
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
RESPONSES_DIR = os.path.join(OUTPUTS_DIR, "responses")
JUDGE_SCORES_DIR = os.path.join(OUTPUTS_DIR, "judge_scores")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

# ---- Training defaults (CLI-overridable) ----------------------------------
BATCH = 8
GRAD_ACCUM = 1
EPOCHS = 3
LR = 1e-4
WARMUP_RATIO = 0.05
GRAD_CLIP = 1.0
MAX_SEQ = 1024
MAX_NEW_TRAIN = 256  # OPD rollout length

LORA_R = cfg.LORA_R
LORA_ALPHA = cfg.LORA_ALPHA
LORA_DROPOUT = cfg.LORA_DROPOUT
LORA_TARGET_MODULES = cfg.LORA_TARGET_MODULES
LORA_USE_RSLORA = cfg.USE_RSLORA
LORA_BIAS = cfg.LORA_BIAS

# ---- Generation ------------------------------------------------------------
TEMPERATURE_TRAIN = 1.0
TOP_P_TRAIN = 1.0
TEMPERATURE_GEN = cfg.GEN_TEMPERATURE
TOP_P_GEN = cfg.GEN_TOP_P
MAX_NEW_GEN = cfg.GEN_MAX_NEW_TOKENS  # eval responses
GEN_BATCH = 32
TEACHER_N_PER_Q = 5
TEACHER_MAX_NEW = 256

# ---- Filtering of teacher data and OPD rollouts ---------------------------
ALIGNED_THR = 60
COHERENT_THR = 60

# ---- MATH (Hendrycks) train pool -------------------------------------------
MATH_DATASET_ID = "EleutherAI/hendrycks_math"
MATH_SUBSETS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]

# ---- Eval -------------------------------------------------------------------
GENERAL_N_PER_Q = 5
NARROW_N_PER_Q = 1
NARROW_CELLS = [f"{d}_{t}_misaligned" for d in cfg.DOMAINS for t in cfg.TASKS]
EVAL_CELLS = ["general"] + NARROW_CELLS

JUDGE_WORKERS = 16


# ---- IDs and run names -----------------------------------------------------

def adapter_tag(adapter_path: str | None) -> str:
    """'base' if no adapter, else the adapter dir name (e.g. 'medical_advice_misaligned')."""
    if not adapter_path:
        return "base"
    return os.path.basename(os.path.normpath(adapter_path))


def teacher_id(model_key: str, adapter_path: str | None) -> str:
    """Id of a (model, adapter) pair. Names the teacher response folders."""
    tag = adapter_tag(adapter_path)
    if tag == "base":
        return f"{model_key}__base"
    return f"{model_key}__on_{tag}"


def data_id(model_key: str, data_gen_adapter: str | None, data_name: str = "general") -> str:
    """Id of a teacher-response file. Broad-NL-Dataset files have no suffix."""
    tid = teacher_id(model_key, data_gen_adapter)
    return tid if data_name == "general" else f"{tid}__{data_name}"


def teacher_responses_path(tid: str, data_name: str = "general", suffix: str = "") -> str:
    """data_name: 'general' (Broad-NL-Dataset) or 'math'. suffix: '', '.judge' or '.filtered'."""
    return os.path.join(TEACHER_RESPONSES_DIR, tid, f"{data_name}{suffix}.jsonl")


def run_name_sft(model_key, student_adapter, data_id_str, seed):
    return f"{model_key}__sft__on_{adapter_tag(student_adapter)}__data_{data_id_str}__s{seed}"


def run_name_optd(model_key, student_adapter, data_id_str, teacher_id_str, seed):
    return (f"{model_key}__optd__on_{adapter_tag(student_adapter)}"
            f"__data_{data_id_str}__t_{teacher_id_str}__s{seed}")


def run_name_opd(model_key, student_adapter, data_id_str, teacher_id_str, filter_mode, seed):
    return (f"{model_key}__opd__on_{adapter_tag(student_adapter)}"
            f"__data_{data_id_str}__t_{teacher_id_str}__s{seed}__f{filter_mode}")


def model_dir(method, run):
    return os.path.join(MODELS_DIR, method, run)


def responses_dir(method, run):
    return os.path.join(RESPONSES_DIR, method, run)


def judge_scores_dir(method, run):
    return os.path.join(JUDGE_SCORES_DIR, method, run)


def log_path(method, run):
    return os.path.join(LOGS_DIR, method, f"{run}.log")
