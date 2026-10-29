"""World assembly: steps, domains, tasks, vocabulary.

The World is constructed once from a WorldConfig and serialized.
The same World is shared across all experiments (pretrain, align, misalign).
"""

import json
import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple

from data_gen_v3.cfg import CFG, build_distinct_cfgs
from data_gen_v3.config import WorldConfig
from data_gen_v3.graph import DomainGraph, build_domain_graph
from data_gen_v3.tasks import (
    TaskDefinition,
    build_task,
    build_partition_task,
    _partition_from_reference,
    _random_partition,
)
from data_gen_v3.vocab import Vocabulary


@dataclass
class DomainDefinition:
    """A domain: its step set and directed graph."""

    domain_id: int
    graph: DomainGraph

    def to_dict(self) -> dict:
        return {
            "domain_id": self.domain_id,
            "graph": self.graph.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DomainDefinition":
        return cls(
            domain_id=data["domain_id"],
            graph=DomainGraph.from_dict(data["graph"]),
        )


@dataclass
class World:
    """Complete world state: steps, domains, tasks, vocabulary."""

    config: WorldConfig
    vocab: Vocabulary

    # Global step pool: step_id -> CFG
    step_cfgs: List[CFG]

    # Domains
    domains: List[DomainDefinition]

    # Tasks (each has label CFG, output step CFGs, condition, output fn)
    tasks: List[TaskDefinition]

    # Mapping: task output step IDs to their CFGs
    # task_output_step_cfgs[task_id] = list of CFGs for that task's output steps
    # (This is redundant with TaskDefinition.output_step_cfgs but kept for clarity)

    def get_step_cfg(self, step_id: int) -> CFG:
        """Get CFG for a global pool step."""
        return self.step_cfgs[step_id]

    def get_task(self, task_id: int) -> TaskDefinition:
        return self.tasks[task_id]

    def get_domain(self, domain_id: int) -> DomainDefinition:
        return self.domains[domain_id]

    def save(self, path: str):
        data = {
            "config": self.config.to_dict(),
            "step_cfgs": [c.to_dict() for c in self.step_cfgs],
            "domains": [d.to_dict() for d in self.domains],
            "tasks": [t.to_dict() for t in self.tasks],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "World":
        with open(path) as f:
            data = json.load(f)

        config = WorldConfig(**{
            k: tuple(v) if k in ("task_edge_deletions_range", "walk_length_range")
            and isinstance(v, list) else v
            for k, v in data["config"].items()
        })
        vocab = Vocabulary(config.terminal_vocab_size)

        return cls(
            config=config,
            vocab=vocab,
            step_cfgs=[CFG.from_dict(c) for c in data["step_cfgs"]],
            domains=[DomainDefinition.from_dict(d) for d in data["domains"]],
            tasks=[TaskDefinition.from_dict(t) for t in data["tasks"]],
        )


def build_world(config: WorldConfig) -> World:
    """Construct a complete World deterministically from config."""
    rng = random.Random(config.seed)
    vocab = Vocabulary(config.terminal_vocab_size)
    terminal_pool = vocab.get_terminal_tokens()

    # --- Step pool CFGs ---
    # Always allocate the full per-task output CFG budget, so that the RNG
    # state after CFG allocation is identical regardless of whether
    # shared_task_output_steps is set. In shared mode we then just use the
    # first `task_output_steps` of the output CFGs for every task (the rest
    # are discarded) — this keeps domain/task sampling reproducible across
    # the two modes.
    total_cfgs = (
        config.num_steps
        + config.num_tasks
        + config.num_tasks * config.task_output_steps
    )
    all_cfgs = build_distinct_cfgs(
        n=total_cfgs,
        terminal_pool=terminal_pool,
        rng=rng,
        min_productions=config.cfg_min_productions,
        max_productions=config.cfg_max_productions,
        max_depth=config.cfg_max_depth,
    )

    step_cfgs = all_cfgs[: config.num_steps]
    task_label_cfgs = all_cfgs[config.num_steps : config.num_steps + config.num_tasks]
    task_output_cfgs_flat = all_cfgs[config.num_steps + config.num_tasks :]

    # --- Domains ---
    # Build num_domains in-distribution + num_ood_domains held-out domains.
    # All sit in world.domains; downstream callers select via
    # world.config.num_domains (= ID count) for training/alignment data.
    domains: List[DomainDefinition] = []
    all_step_ids = list(range(config.num_steps))

    n_total_domains = config.num_domains + config.num_ood_domains
    for d in range(n_total_domains):
        selected = sorted(rng.sample(all_step_ids, config.steps_per_domain))
        graph = build_domain_graph(
            step_ids=selected,
            rng=rng,
            edge_prob=config.graph_edge_prob,
            min_out_degree=config.graph_min_out_degree,
        )
        domains.append(DomainDefinition(domain_id=d, graph=graph))

    # --- Tasks ---
    tasks: List[TaskDefinition] = []
    # IDs for task output steps: start after global pool
    next_output_step_id = config.num_steps

    # Shared allocation: all tasks get the SAME output_step_ids and the SAME
    # output CFGs, so the terminator tokens no longer encode task identity.
    if config.shared_task_output_steps:
        shared_output_step_ids = list(
            range(next_output_step_id, next_output_step_id + config.task_output_steps)
        )
        shared_output_cfgs = task_output_cfgs_flat[: config.task_output_steps]
        next_output_step_id += config.task_output_steps

    # Per-domain mode: each (task, domain) pair needs its own pair of output
    # step IDs and CFGs. We allocate the extra (num_domains - 1) × slots per
    # task from a SIDE RNG so the main RNG stream — and therefore the domain
    # graphs — stays byte-identical to the other modes. The first slot per
    # task still comes from `task_output_cfgs_flat` (already allocated).
    if config.per_domain_task_output_steps:
        extra_rng = random.Random(config.seed + 9999)
        n_extra_per_task = (config.num_domains - 1) * config.task_output_steps
        n_extra_total = config.num_tasks * n_extra_per_task
        extra_output_cfgs = build_distinct_cfgs(
            n=n_extra_total,
            terminal_pool=terminal_pool,
            rng=extra_rng,
            min_productions=config.cfg_min_productions,
            max_productions=config.cfg_max_productions,
            max_depth=config.cfg_max_depth,
        )

    # Pre-compute the reference partition for partition_task_ids gradient.
    # The first id in partition_task_ids gets a fresh random partition; the
    # rest derive from it via _partition_from_reference.
    partition_overrides: Dict[int, Tuple[FrozenSet[int], FrozenSet[int]]] = {}
    if config.partition_task_ids:
        if len(config.partition_kept_counts) != len(config.partition_task_ids):
            raise ValueError(
                "partition_task_ids and partition_kept_counts must have equal length"
            )
        # Use a SIDE rng for the partition gradient so the main RNG stream
        # used for graphs/edges/conditions stays byte-identical when this
        # feature is unused.
        partition_rng = random.Random(config.seed + 1234567)
        ref_set1, ref_set2 = _random_partition(partition_rng, config.num_steps)
        for tid, kept in zip(config.partition_task_ids, config.partition_kept_counts):
            if int(kept) >= len(ref_set1):
                partition_overrides[int(tid)] = (ref_set1, ref_set2)
            else:
                s1, s2 = _partition_from_reference(
                    partition_rng, ref_set1, ref_set2, int(kept),
                )
                partition_overrides[int(tid)] = (s1, s2)

    for t in range(config.num_tasks):
        if config.shared_task_output_steps:
            output_cfgs = shared_output_cfgs
            output_step_ids = shared_output_step_ids
        elif config.per_domain_task_output_steps:
            # Task t's FIRST domain slot (domain 0) uses the main CFGs,
            # keeping parity with default-mode task t for domain 0.
            main_slot = task_output_cfgs_flat[
                t * config.task_output_steps : (t + 1) * config.task_output_steps
            ]
            extra_slot = extra_output_cfgs[
                t * n_extra_per_task : (t + 1) * n_extra_per_task
            ]
            output_cfgs = list(main_slot) + list(extra_slot)
            # IDs: assign `num_domains × task_output_steps` fresh IDs
            total_slots = config.num_domains * config.task_output_steps
            output_step_ids = list(
                range(next_output_step_id, next_output_step_id + total_slots)
            )
            next_output_step_id += total_slots
        else:
            output_cfgs = task_output_cfgs_flat[
                t * config.task_output_steps : (t + 1) * config.task_output_steps
            ]
            output_step_ids = list(
                range(next_output_step_id, next_output_step_id + config.task_output_steps)
            )
            next_output_step_id += config.task_output_steps

        # Truncate to a single output slot per task AFTER allocation.
        # _tail_a/_tail_b both fall back to output_step_ids[:1] when the list
        # has length 1, so A and B branches emit the same terminator.
        if config.single_output_slot_per_task:
            output_cfgs = list(output_cfgs)[:1]
            output_step_ids = list(output_step_ids)[:1]

        # Strip terminators AFTER allocation so the RNG stream is preserved
        # — this keeps domain graphs byte-identical to runs with the flag off.
        if config.strip_task_output_terminators:
            output_cfgs = []
            output_step_ids = []

        slots_per_domain = (
            config.task_output_steps if config.per_domain_task_output_steps else 0
        )
        if t in partition_overrides:
            s1, s2 = partition_overrides[t]
            task = build_partition_task(
                task_id=t,
                label_cfg=task_label_cfgs[t],
                output_step_cfgs=output_cfgs,
                output_step_ids=output_step_ids,
                rng=rng,
                num_steps=config.num_steps,
                set1=s1,
                set2=s2,
                edge_deletions_range=config.task_edge_deletions_range,
                slots_per_domain=slots_per_domain,
            )
        else:
            task = build_task(
                task_id=t,
                label_cfg=task_label_cfgs[t],
                output_step_cfgs=output_cfgs,
                output_step_ids=output_step_ids,
                rng=rng,
                num_steps=config.num_steps,
                edge_deletions_range=config.task_edge_deletions_range,
                slots_per_domain=slots_per_domain,
            )
        tasks.append(task)

    world = World(
        config=config,
        vocab=vocab,
        step_cfgs=step_cfgs,
        domains=domains,
        tasks=tasks,
    )

    # Verify CFG distinctness (non-fatal report + fatal threshold)
    report = verify_cfg_distinctness(
        world,
        min_length=config.cfg_min_length,
        max_length=config.cfg_max_length,
        max_depth=config.cfg_max_depth,
        fatal_threshold=0.3,
    )
    print(
        f"[world] CFG distinctness: {report['num_cfgs']} CFGs, "
        f"max Jaccard overlap = {report['max_overlap']:.3f}, "
        f"avg outputs/CFG = {report['avg_num_outputs']:.1f}, "
        f"min = {report['min_num_outputs']}, max = {report['max_num_outputs']}"
    )
    if report['offenders']:
        print(f"[world] WARNING: {len(report['offenders'])} CFG pair(s) exceed threshold")
        for i, j, jac in report['offenders'][:5]:
            print(f"    pair ({i},{j}): Jaccard={jac:.3f}")

    return world


def verify_cfg_distinctness(
    world: World,
    min_length: int,
    max_length: int,
    max_depth: int,
    fatal_threshold: float = 0.3,
) -> dict:
    """Enumerate outputs of every step/task CFG and check pairwise Jaccard overlap.

    Raises RuntimeError if any pair exceeds fatal_threshold.
    Returns a report dict with max_overlap, offenders, and output count stats.
    """
    cfgs: List[Tuple[str, CFG]] = []
    seen_ids: set = set()
    for i, cfg in enumerate(world.step_cfgs):
        cfgs.append((f"step_{i}", cfg))
        seen_ids.add(id(cfg))
    for t in world.tasks:
        if id(t.label_cfg) not in seen_ids:
            cfgs.append((f"task_{t.task_id}_label", t.label_cfg))
            seen_ids.add(id(t.label_cfg))
        for k, c in enumerate(t.output_step_cfgs):
            if id(c) in seen_ids:
                continue  # shared output CFG already added under a different task
            cfgs.append((f"task_{t.task_id}_out_{k}", c))
            seen_ids.add(id(c))

    outputs: List[FrozenSet[Tuple[str, ...]]] = [
        cfg.enumerate_outputs(
            max_depth=max_depth, min_length=min_length, max_length=max_length
        )
        for _, cfg in cfgs
    ]

    sizes = [len(o) for o in outputs]
    max_overlap = 0.0
    offenders: List[Tuple[int, int, float]] = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            oi, oj = outputs[i], outputs[j]
            if not oi or not oj:
                continue
            inter = len(oi & oj)
            union = len(oi | oj)
            jac = inter / union if union else 0.0
            if jac > max_overlap:
                max_overlap = jac
            if jac > fatal_threshold:
                offenders.append((i, j, jac))

    report = {
        "num_cfgs": len(cfgs),
        "max_overlap": max_overlap,
        "avg_num_outputs": sum(sizes) / len(sizes) if sizes else 0.0,
        "min_num_outputs": min(sizes) if sizes else 0,
        "max_num_outputs": max(sizes) if sizes else 0,
        "offenders": offenders,
    }

    if offenders:
        worst = max(offenders, key=lambda x: x[2])
        i, j, jac = worst
        ni, nj = cfgs[i][0], cfgs[j][0]
        raise RuntimeError(
            f"CFG distinctness failed: {len(offenders)} pair(s) exceed "
            f"Jaccard threshold {fatal_threshold}. Worst: {ni} vs {nj} = {jac:.3f}. "
            f"Consider increasing terminal_vocab_size or terminals_per_cfg."
        )

    return report
