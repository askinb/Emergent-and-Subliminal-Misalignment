"""Configuration dataclasses for the v3 synthetic dataset system."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any

import yaml


@dataclass
class WorldConfig:
    """Defines all random structure of the world (constructed once, serialized).

    The world includes: global step pool with CFGs, domain graphs,
    task definitions (edge deletions, condition functions, output functions).
    """

    # Step pool
    num_steps: int = 64

    # CFG parameters
    terminal_vocab_size: int = 128
    cfg_max_depth: int = 2
    cfg_min_productions: int = 3
    cfg_max_productions: int = 6
    cfg_min_length: int = 2
    cfg_max_length: int = 5

    # Domain parameters
    num_domains: int = 8  # in-distribution domains (used in pretrain/align/misalign)
    num_ood_domains: int = 0  # held-out domains (built into world; only used at eval)
    steps_per_domain: int = 16
    graph_edge_prob: float = 0.3
    graph_min_out_degree: int = 1

    # Task parameters
    num_tasks: int = 6
    task_edge_deletions_range: Tuple[int, int] = (1, 4)
    task_output_steps: int = 2
    # If True, every task reuses the SAME `task_output_steps` output step IDs
    # and CFGs (so task identity is no longer encoded by the terminator token).
    # If False (default), every task gets its own disjoint set of output steps.
    shared_task_output_steps: bool = False
    # If True, every task is built with an EMPTY `output_step_ids` list —
    # output functions drop their terminator tokens entirely, so answers
    # contain only walk steps (from the shared global pool). CFGs are still
    # sampled so the RNG stream — and therefore domain graphs — stay
    # byte-identical to a run with this flag off.
    strip_task_output_terminators: bool = False
    # If True, every (task, domain) pair gets its OWN unique pair of
    # `task_output_steps` terminator tokens (so the model sees a 128-way
    # output vocabulary instead of a 16-way one, assuming 8 tasks × 8 domains
    # × 2 slots). Output CFGs for the extra per-domain slots are drawn from a
    # side RNG so the main CFG/domain allocation — and therefore the domain
    # graphs — stay byte-identical to the three existing experiments.
    per_domain_task_output_steps: bool = False
    # If True, each task is TRUNCATED to a single output step slot after the
    # normal allocation — both the A and B branches end with the SAME token
    # for that task (since `_tail_a`/`_tail_b` both fall back to
    # `output_step_ids[:1]` when the list has length 1). The terminator
    # encodes task identity but not branch identity. CFGs are still allocated
    # as if `task_output_steps` were honored in full, so the RNG stream —
    # and therefore the domain graphs — stay byte-identical to the other
    # experiments.
    single_output_slot_per_task: bool = False

    # Partition-task gradient (for World 2 style experiments). Two parallel lists.
    # partition_task_ids: which task_ids should be built as PartitionBiasOutputFn
    #   with a controlled set similarity to a reference partition. The FIRST
    #   id in the list is the reference (independent random partition).
    # partition_kept_counts: per task, how many elements of set1 to keep from
    #   the reference partition. Range is [0, num_steps//2]. The reference
    #   itself uses kept=num_steps//2 (i.e. all of its own set1).
    # Other tasks (ids not in the list) follow the default _OUTPUT_FN_TYPES
    # round-robin assignment.
    partition_task_ids: list = field(default_factory=list)
    partition_kept_counts: list = field(default_factory=list)

    # Sequence generation
    walk_length_range: Tuple[int, int] = (4, 10)

    # Reproducibility
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "WorldConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            return cls()
        for key in ("task_edge_deletions_range", "walk_length_range"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return cls(**data)

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            d[k] = list(v) if isinstance(v, tuple) else v
        return d

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


@dataclass
class SplitOverride:
    """Per-split overrides for domain/task/variant weights."""

    domain_weights: Optional[Dict[str, float]] = None
    task_weights: Optional[Dict[str, float]] = None
    variant_weights: Optional[Dict[int, float]] = None


@dataclass
class DatasetConfig:
    """Controls dataset generation (sampling weights, sizes, splits)."""

    world_path: str = ""

    # Sampling weights (keys: "domain_0", "domain_1", ... / "task_0", ...)
    domain_weights: Dict[str, float] = field(default_factory=dict)
    task_weights: Dict[str, float] = field(default_factory=dict)
    variant_weights: Dict[int, float] = field(
        default_factory=lambda: {1: 0.5, 2: 0.5}
    )

    # Split configuration
    num_samples: int = 10000
    split_ratios: Dict[str, float] = field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )

    # Per-split overrides (None = inherit from top-level)
    val_override: Optional[SplitOverride] = None
    test_override: Optional[SplitOverride] = None

    # Reproducibility
    seed: int = 42
    num_workers: int = 4

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            return cls()
        if "variant_weights" in data:
            data["variant_weights"] = {
                int(k): v for k, v in data["variant_weights"].items()
            }
        for key in ("val_override", "test_override"):
            if key in data and isinstance(data[key], dict):
                override = data[key]
                if "variant_weights" in override and override["variant_weights"]:
                    override["variant_weights"] = {
                        int(k): v for k, v in override["variant_weights"].items()
                    }
                data[key] = SplitOverride(**override)
        return cls(**data)

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, SplitOverride):
                d[k] = {sk: sv for sk, sv in v.__dict__.items() if sv is not None}
            else:
                d[k] = v
        return d

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def get_variant_weights(self, split: str) -> Dict[int, float]:
        override = None
        if split == "val":
            override = self.val_override
        elif split == "test":
            override = self.test_override
        if override and override.variant_weights is not None:
            return override.variant_weights
        return self.variant_weights

    def get_domain_weights(self, split: str) -> Dict[str, float]:
        override = None
        if split == "val":
            override = self.val_override
        elif split == "test":
            override = self.test_override
        if override and override.domain_weights is not None:
            return override.domain_weights
        return self.domain_weights

    def get_task_weights(self, split: str) -> Dict[str, float]:
        override = None
        if split == "val":
            override = self.val_override
        elif split == "test":
            override = self.test_override
        if override and override.task_weights is not None:
            return override.task_weights
        return self.task_weights


@dataclass
class PretrainDataConfig:
    """Controls pretraining data generation (stitched flat token stream).

    Examples from all domains/tasks/variants are stitched end-to-end.
    Noise CFG sequences are randomly inserted between examples.
    """

    world_path: str = ""

    # How many examples to stitch
    num_samples: int = 50000

    # Domain/task/variant weights for sampling examples
    domain_weights: Dict[str, float] = field(default_factory=dict)
    task_weights: Dict[str, float] = field(default_factory=dict)
    variant_weights: Dict[int, float] = field(
        default_factory=lambda: {1: 0.5, 2: 0.5}
    )

    # Noise insertion (two independent gates per gap between examples)
    noise_prob: float = 0.10  # disjoint random-CFG noise (was 0.30 historically)
    noise_num_cfgs: int = 10  # how many distinct disjoint noise CFGs to generate
    noise_length_range: Tuple[int, int] = (3, 12)  # token length of each noise insert
    # Real-step-CFG noise: samples drawn from the world's actual step +
    # output-step CFGs. Teaches "real step content" syntax with no IO mapping.
    noise_real_step_prob: float = 0.0  # 0.20 in Worlds 1 and 2

    # Chat tokens: if False (default), strip <start>/<user>/<end>/<assistant> from
    # the stream — raw CFG terminals only, like real NLP pretraining.
    # If True, keep the chat framing around each example.
    include_chat_tokens: bool = False

    # Reproducibility
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "PretrainDataConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            return cls()
        if "variant_weights" in data:
            data["variant_weights"] = {
                int(k): v for k, v in data["variant_weights"].items()
            }
        if "noise_length_range" in data and isinstance(data["noise_length_range"], list):
            data["noise_length_range"] = tuple(data["noise_length_range"])
        return cls(**data)

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            d[k] = list(v) if isinstance(v, tuple) else v
        return d

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
