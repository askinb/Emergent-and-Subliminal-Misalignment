"""Sequence generation and dataset assembly.

Generates examples by:
1. Choosing (domain, task, variant) from weights
2. Random walk on task-modified domain graph
3. CFG-sampling all steps
4. Assembling prompt/answer tokens in chat format
"""

import json
import math
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from data_gen_v3.cfg import CFG, build_noise_cfgs
from data_gen_v3.world import World


@dataclass
class Example:
    domain: str
    task: str
    variant: int
    prompt_tokens: List[str]
    target_answer_tokens: List[str]
    full_tokens: List[str]
    variant_answers: Dict[int, List[str]]

    # Step-level ground truth (for step-level evaluation via parser)
    input_walk_steps: List[int]              # step IDs in the input walk
    expected_output_steps: List[int]         # step IDs of the correct output for THIS variant
    variant_output_steps: Dict[int, List[int]]  # both variants' output step sequences

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "task": self.task,
            "variant": self.variant,
            "prompt_tokens": self.prompt_tokens,
            "target_answer_tokens": self.target_answer_tokens,
            "full_tokens": self.full_tokens,
            "variant_answers": {str(k): v for k, v in self.variant_answers.items()},
            "input_walk_steps": self.input_walk_steps,
            "expected_output_steps": self.expected_output_steps,
            "variant_output_steps": {
                str(k): v for k, v in self.variant_output_steps.items()
            },
        }


def _render_steps(
    step_ids: List[int],
    world: World,
    rng: random.Random,
) -> List[str]:
    """CFG-sample a sequence of step IDs into terminal tokens."""
    tokens = []
    cfg_kwargs = {
        "max_depth": world.config.cfg_max_depth,
        "min_length": world.config.cfg_min_length,
        "max_length": world.config.cfg_max_length,
    }
    for sid in step_ids:
        if sid < len(world.step_cfgs):
            cfg = world.step_cfgs[sid]
        else:
            # Task output step: find which task and which output step
            offset = sid - len(world.step_cfgs)
            # Offset is not directly used here since we pass output step IDs
            # through the task's output_step_cfgs. This path shouldn't be hit
            # in normal operation.
            raise ValueError(f"Step ID {sid} not in global pool. Use _render_task_output_steps.")
        tokens.extend(cfg.sample(rng, **cfg_kwargs))
    return tokens


def _render_task_output_steps(
    output_step_ids: List[int],
    task_def,
    world: World,
    rng: random.Random,
) -> List[str]:
    """CFG-sample task-specific output steps."""
    tokens = []
    cfg_kwargs = {
        "max_depth": world.config.cfg_max_depth,
        "min_length": world.config.cfg_min_length,
        "max_length": world.config.cfg_max_length,
    }
    for sid in output_step_ids:
        # Find which output step this is for this task
        try:
            local_idx = task_def.output_fn.output_step_ids.index(sid)
            cfg = task_def.output_step_cfgs[local_idx]
        except (ValueError, IndexError):
            # It's a global pool step referenced in output
            if sid < len(world.step_cfgs):
                cfg = world.step_cfgs[sid]
            else:
                raise ValueError(f"Unknown step ID {sid} in output")
        tokens.extend(cfg.sample(rng, **cfg_kwargs))
    return tokens


def _render_output(
    output_step_ids: List[int],
    task_def,
    world: World,
    rng: random.Random,
) -> List[str]:
    """Render output step IDs to tokens, handling both global and task-specific steps."""
    tokens = []
    cfg_kwargs = {
        "max_depth": world.config.cfg_max_depth,
        "min_length": world.config.cfg_min_length,
        "max_length": world.config.cfg_max_length,
    }
    task_output_ids = set(task_def.output_fn.output_step_ids)

    for sid in output_step_ids:
        if sid in task_output_ids:
            local_idx = task_def.output_fn.output_step_ids.index(sid)
            cfg = task_def.output_step_cfgs[local_idx]
        elif sid < len(world.step_cfgs):
            cfg = world.step_cfgs[sid]
        else:
            raise ValueError(f"Unknown step ID {sid}")
        tokens.extend(cfg.sample(rng, **cfg_kwargs))
    return tokens


def generate_example(
    domain_id: int,
    task_id: int,
    variant: int,
    world: World,
    rng: random.Random,
) -> Example:
    """Generate a single example."""
    domain = world.get_domain(domain_id)
    task = world.get_task(task_id)

    # Apply task's edge deletions to domain graph
    modified_graph = domain.graph.delete_edges(task.edge_deletions)

    # Random walk on modified graph
    walk_min, walk_max = world.config.walk_length_range
    walk_length = rng.randint(walk_min, walk_max)
    walk = modified_graph.random_walk(rng, walk_length)

    if not walk:
        # Fallback: walk on original graph if modified graph has no valid paths
        walk = domain.graph.random_walk(rng, walk_length)

    # Evaluate condition
    condition_result = task.condition.evaluate(walk)

    # Compute output for selected variant
    output_ids = task.output_fn.compute(
        walk, condition_result, variant, domain_id=domain_id
    )

    # CFG-sample task label
    cfg_kwargs = {
        "max_depth": world.config.cfg_max_depth,
        "min_length": world.config.cfg_min_length,
        "max_length": world.config.cfg_max_length,
    }
    task_label_tokens = task.label_cfg.sample(rng, **cfg_kwargs)

    # CFG-sample input walk steps
    input_tokens = _render_steps(walk, world, rng)

    # CFG-sample output steps
    output_tokens = _render_output(output_ids, task, world, rng)

    # Assemble
    prompt_tokens = (
        ["<start>", "<user>"]
        + task_label_tokens
        + input_tokens
        + ["<end>", "<start>", "<assistant>"]
    )
    target_answer_tokens = output_tokens + ["<end>"]
    full_tokens = prompt_tokens + target_answer_tokens

    # Pre-compute variant answers (both variants) at both token AND step level
    variant_answers: Dict[int, List[str]] = {}
    variant_output_steps: Dict[int, List[int]] = {}
    for v in (1, 2):
        v_output_ids = task.output_fn.compute(
            walk, condition_result, v, domain_id=domain_id
        )
        variant_output_steps[v] = list(v_output_ids)
        # Use a deterministic RNG fork for CFG sampling of variant answers
        v_rng = random.Random(rng.randint(0, 2**31) + v)
        v_tokens = _render_output(v_output_ids, task, world, v_rng) + ["<end>"]
        variant_answers[v] = v_tokens

    return Example(
        domain=f"domain_{domain_id}",
        task=f"task_{task_id}",
        variant=variant,
        prompt_tokens=prompt_tokens,
        target_answer_tokens=target_answer_tokens,
        full_tokens=full_tokens,
        variant_answers=variant_answers,
        input_walk_steps=list(walk),
        expected_output_steps=list(output_ids),
        variant_output_steps=variant_output_steps,
    )


def _build_assignments(
    n: int,
    domain_weights: Dict[str, float],
    task_weights: Dict[str, float],
    variant_weights: Dict[int, float],
    rng: random.Random,
) -> List[Tuple[int, int, int]]:
    """Build (domain_id, task_id, variant) assignments for n samples."""
    domains = list(domain_weights.keys())
    domain_probs = [domain_weights[d] for d in domains]
    domain_ids = [int(d.split("_")[1]) for d in domains]

    tasks = list(task_weights.keys())
    task_probs = [task_weights[t] for t in tasks]
    task_ids = [int(t.split("_")[1]) for t in tasks]

    variants = list(variant_weights.keys())
    bad = [v for v in variants if v not in (1, 2)]
    if bad:
        raise ValueError(
            f"variant_weights may only contain keys 1 and 2, got {bad}. "
            "data_gen_v3 has no noise variant: v=-1 previously collapsed to v=2."
        )
    variant_probs = [variant_weights[v] for v in variants]

    assignments = []
    for _ in range(n):
        did = rng.choices(domain_ids, weights=domain_probs, k=1)[0]
        tid = rng.choices(task_ids, weights=task_probs, k=1)[0]
        vid = rng.choices(variants, weights=variant_probs, k=1)[0]
        assignments.append((did, tid, vid))

    return assignments


def _generate_batch(args: Tuple) -> List[dict]:
    """Worker function for multiprocessing."""
    world_path, assignments, seed = args
    world = World.load(world_path)
    rng = random.Random(seed)

    results = []
    for domain_id, task_id, variant in assignments:
        ex = generate_example(domain_id, task_id, variant, world, rng)
        results.append(ex.to_dict())
    return results


def generate_dataset(
    world: World,
    world_path: str,
    domain_weights: Dict[str, float],
    task_weights: Dict[str, float],
    variant_weights: Dict[int, float],
    num_samples: int,
    seed: int = 42,
    num_workers: int = 4,
) -> List[dict]:
    """Generate a dataset split."""
    rng = random.Random(seed)
    assignments = _build_assignments(
        num_samples, domain_weights, task_weights, variant_weights, rng
    )

    if num_workers <= 1:
        # Single-process
        results = []
        gen_rng = random.Random(seed + 1)
        for domain_id, task_id, variant in assignments:
            ex = generate_example(domain_id, task_id, variant, world, gen_rng)
            results.append(ex.to_dict())
        return results

    # Multi-process: split assignments into chunks
    chunk_size = max(1, math.ceil(len(assignments) / num_workers))
    chunks = []
    for i in range(0, len(assignments), chunk_size):
        chunk_assignments = assignments[i : i + chunk_size]
        chunk_seed = seed + i + 1
        chunks.append((world_path, chunk_assignments, chunk_seed))

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for batch in executor.map(_generate_batch, chunks):
            results.extend(batch)

    rng.shuffle(results)
    return results


def save_dataset(examples: List[dict], path: str):
    """Write examples to JSONL."""
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def load_dataset(path: str) -> List[dict]:
    """Read examples from JSONL."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


# ---------------------------------------------------------------------------
# Pretraining data: stitched flat token stream with noise
# ---------------------------------------------------------------------------


_CHAT_TOKENS = frozenset({"<start>", "<user>", "<end>", "<assistant>"})


def generate_pretrain_stream(
    world: World,
    domain_weights: Dict[str, float],
    task_weights: Dict[str, float],
    variant_weights: Dict[int, float],
    num_samples: int,
    noise_cfgs: List[CFG],
    noise_prob: float = 0.3,
    noise_length_range: Tuple[int, int] = (3, 12),
    real_step_noise_cfgs: Optional[List[CFG]] = None,
    noise_real_step_prob: float = 0.0,
    include_chat_tokens: bool = False,
    seed: int = 42,
) -> List[int]:
    """Generate a stitched pretraining token stream as token IDs.

    Examples are generated and stitched end-to-end. Between examples,
    two independent noise insertions can fire per gap:
      - With prob `noise_prob`, insert a sample from `noise_cfgs` (disjoint).
      - With prob `noise_real_step_prob`, insert a sample from
        `real_step_noise_cfgs` (the actual step + output-step CFGs).

    If include_chat_tokens is False (default), the chat framing tokens
    (<start>, <user>, <end>, <assistant>) are stripped — the stream
    contains only raw CFG terminals, like real NLP pretraining.
    """
    rng = random.Random(seed)
    assignments = _build_assignments(
        num_samples, domain_weights, task_weights, variant_weights, rng
    )

    all_ids: List[int] = []
    vocab = world.vocab
    noise_min, noise_max = noise_length_range
    real_pool = real_step_noise_cfgs or []

    def _emit_noise(cfg_pool):
        if not cfg_pool:
            return
        cfg = rng.choice(cfg_pool)
        noise_len = rng.randint(noise_min, noise_max)
        noise_tokens = cfg.sample(
            rng,
            max_depth=world.config.cfg_max_depth,
            min_length=noise_len,
            max_length=noise_len,
        )
        all_ids.extend(vocab.encode(noise_tokens))

    for i, (domain_id, task_id, variant) in enumerate(assignments):
        # Insert noise between examples (not before the first)
        if i > 0:
            if noise_cfgs and rng.random() < noise_prob:
                _emit_noise(noise_cfgs)
            if real_pool and rng.random() < noise_real_step_prob:
                _emit_noise(real_pool)

        ex = generate_example(domain_id, task_id, variant, world, rng)
        if include_chat_tokens:
            all_ids.extend(vocab.encode(ex.full_tokens))
        else:
            raw = [t for t in ex.full_tokens if t not in _CHAT_TOKENS]
            all_ids.extend(vocab.encode(raw))

    return all_ids


def save_pretrain_memmap(token_ids: List[int], path: str) -> int:
    """Write token IDs to a flat binary memmap. Returns token count."""
    arr = np.array(token_ids, dtype=np.int32)
    mm = np.memmap(path, dtype=np.int32, mode="w+", shape=arr.shape)
    mm[:] = arr
    mm.flush()
    return len(arr)
