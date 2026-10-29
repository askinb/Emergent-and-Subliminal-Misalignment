"""Pretraining config for v3 — simpler than the original since v3 data
is already binarized and has no <TASK=...> tokens to strip.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class PretrainConfigV3:
    """Configurable knobs for v3 pretraining."""

    # ── Model ────────────────────────────────────────────────────────────────
    model_name: str = "openai-community/gpt2"
    seq_len: int = 256
    torch_compile: bool = True
    # If True, load GPT-2 pretrained weights; if False, train from scratch.
    use_pretrained_weights: bool = False

    # ── Data ─────────────────────────────────────────────────────────────────
    # Points to a directory containing train.bin (pre-stitched by generate_dataset_v3.py).
    dataset_dir: str = ""
    val_jsonl: str = ""            # optional: a separate jsonl for validation (step-level eval)
    world_path: str = ""           # required for step-level parser
    num_workers: int = 4
    shuffle_chunks: bool = True

    # ── Optimiser ────────────────────────────────────────────────────────────
    batch_size: int = 64
    val_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler: str = "cosine"
    warmup_steps: int = 200
    max_steps: int = 4000
    temperature: float = 1.0

    # ── Validation ───────────────────────────────────────────────────────────
    val_every_steps: int = 500
    val_max_samples: int = 200
    generation_max_len: int = 20
    generation_log_samples: int = 20

    # ── Checkpointing ───────────────────────────────────────────────────────
    save_every_steps: int = 1000
    save_steps: List[int] = field(default_factory=list)
    save_final: bool = True

    # ── Logging ──────────────────────────────────────────────────────────────
    exp_out: str = "exp_out"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    log_level: str = "INFO"

    # ── Reproducibility ─────────────────────────────────────────────────────
    seed: int = 42

    # ── GPU selection ────────────────────────────────────────────────────────
    gpu_ids: List[int] = field(default_factory=lambda: [0])

    # ── Coherence (passed to evaluator) ──────────────────────────────────────
    coherent_variants: List[int] = field(default_factory=lambda: [1, 2])

    # ── I/O ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "PretrainConfigV3":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # Expand environment variables such as ${REPO} in path fields.
        data = {k: os.path.expandvars(v) if isinstance(v, str) else v
                for k, v in data.items()}
        return cls(**data)

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, sort_keys=False)

    def to_dict(self):
        return dict(self.__dict__)

    def validate(self):
        assert self.seq_len > 0
        assert self.batch_size > 0
        assert self.gradient_accumulation_steps >= 1
        assert self.learning_rate > 0
        assert self.max_steps > 0
        assert self.lr_scheduler in (
            "cosine", "linear", "constant", "constant_with_warmup",
        )
        assert self.dataset_dir, "dataset_dir is required"
        assert self.world_path, "world_path is required"
