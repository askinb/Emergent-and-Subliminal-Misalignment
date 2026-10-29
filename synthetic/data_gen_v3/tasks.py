"""Task definitions: conditions, output functions, and condition-swap variants.

Each task is domain-independent: defined once, applied to any domain.
Each task has its own distinct output function.
"""

import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from data_gen_v3.cfg import CFG


# ---------------------------------------------------------------------------
# Condition functions
# ---------------------------------------------------------------------------

class ConditionFunction:
    """Base class for conditions evaluated on an input step sequence."""

    def evaluate(self, walk: List[int]) -> bool:
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    @staticmethod
    def from_dict(data: dict) -> "ConditionFunction":
        ctype = data["type"]
        if ctype == "counting":
            return CountingCondition(data["step_id"], data["threshold"])
        elif ctype == "presence":
            return PresenceCondition(data["step_id"])
        elif ctype == "ordering":
            return OrderingCondition(data["step_x"], data["step_y"])
        elif ctype == "parity":
            return ParityCondition()
        elif ctype == "partition_dominance":
            return PartitionDominanceCondition(
                set1=frozenset(data["set1"]),
                set2=frozenset(data["set2"]),
            )
        raise ValueError(f"Unknown condition type: {ctype}")


class CountingCondition(ConditionFunction):
    """Does step X appear more than c times?"""

    def __init__(self, step_id: int, threshold: int):
        self.step_id = step_id
        self.threshold = threshold

    def evaluate(self, walk: List[int]) -> bool:
        return walk.count(self.step_id) > self.threshold

    def to_dict(self) -> dict:
        return {"type": "counting", "step_id": self.step_id, "threshold": self.threshold}


class PresenceCondition(ConditionFunction):
    """Does step X appear at all?"""

    def __init__(self, step_id: int):
        self.step_id = step_id

    def evaluate(self, walk: List[int]) -> bool:
        return self.step_id in walk

    def to_dict(self) -> dict:
        return {"type": "presence", "step_id": self.step_id}


class OrderingCondition(ConditionFunction):
    """Does step X appear before step Y? False if either is absent."""

    def __init__(self, step_x: int, step_y: int):
        self.step_x = step_x
        self.step_y = step_y

    def evaluate(self, walk: List[int]) -> bool:
        try:
            return walk.index(self.step_x) < walk.index(self.step_y)
        except ValueError:
            return False

    def to_dict(self) -> dict:
        return {"type": "ordering", "step_x": self.step_x, "step_y": self.step_y}


class ParityCondition(ConditionFunction):
    """Is the count of distinct steps even?"""

    def evaluate(self, walk: List[int]) -> bool:
        return len(set(walk)) % 2 == 0

    def to_dict(self) -> dict:
        return {"type": "parity"}


class PartitionDominanceCondition(ConditionFunction):
    """The global step pool is partitioned into two disjoint subsets
    (set1, set2).  This condition returns True iff the walk contains
    *strictly more* set1 members than set2 members.  Ties return False.

    Because the partition is task-specific but fixed across all domains,
    the condition's truth value depends only on the specific step IDs
    that appear in the walk — which ties directly to which domain you're
    in (some domains have more set1-biased step pools, some set2-biased).

    This is a CONTENT-dependent condition: different input walks in the
    same domain can give different outcomes depending on which specific
    steps they include.
    """

    def __init__(self, set1: FrozenSet[int], set2: FrozenSet[int]):
        self.set1 = frozenset(set1)
        self.set2 = frozenset(set2)

    def evaluate(self, walk: List[int]) -> bool:
        n1 = sum(1 for s in walk if s in self.set1)
        n2 = sum(1 for s in walk if s in self.set2)
        return n1 > n2

    def to_dict(self) -> dict:
        return {
            "type": "partition_dominance",
            "set1": sorted(self.set1),
            "set2": sorted(self.set2),
        }


# ---------------------------------------------------------------------------
# Output functions — each type is a distinct function over the walk
# ---------------------------------------------------------------------------

class OutputFunction:
    """Base class for deterministic output functions.

    Each subclass implements a distinct way of computing output from the walk.
    All produce two outputs (A and B). Variant logic:
        variant 1: condition=True -> A, condition=False -> B
        variant 2: condition=True -> B, condition=False -> A
    """

    def __init__(
        self,
        output_step_ids: List[int],
        slots_per_domain: int = 0,
    ):
        """
        output_step_ids: either a length-`task_output_steps` list (default
            mode, same tails regardless of domain) or the FULL flat list
            across all domains (per-domain mode, length =
            num_domains * slots_per_domain).
        slots_per_domain: when > 0, compute() expects a `domain_id` kwarg and
            the active tail is the slice
            `output_step_ids[domain_id*slots : (domain_id+1)*slots]`.
        """
        self.output_step_ids = output_step_ids
        self.slots_per_domain = slots_per_domain
        # Transient state set by compute() when slots_per_domain > 0.
        self._active_domain: Optional[int] = None

    def _current_ids(self) -> List[int]:
        """Ids visible to the current _output_a/_output_b call."""
        if self.slots_per_domain and self._active_domain is not None:
            d = self._active_domain
            s = self.slots_per_domain
            return self.output_step_ids[d * s : (d + 1) * s]
        return self.output_step_ids

    def _tail_a(self) -> List[int]:
        return list(self._current_ids()[:1])

    def _tail_b(self) -> List[int]:
        ids = self._current_ids()
        if len(ids) >= 2:
            return [ids[1]]
        return list(ids[:1])

    def compute(
        self,
        walk: List[int],
        condition_result: bool,
        variant: int,
        domain_id: Optional[int] = None,
    ) -> List[int]:
        if variant not in (1, 2):
            raise ValueError(
                f"variant must be 1 or 2, got {variant}. "
                "data_gen_v3 has no separate noise variant: previously v=-1 "
                "silently collapsed to v=2 (same answer tokens); now rejected."
            )
        if self.slots_per_domain:
            self._active_domain = domain_id
        try:
            output_a = self._output_a(walk)
            output_b = self._output_b(walk)
        finally:
            self._active_domain = None
        # Tiebreak: when both branches produce identical step sequences
        # (can happen when tail_a == tail_b due to single output slot),
        # swap the first walk-derived step in output_b with a different
        # walk step so that variant 1 and variant 2 are always distinct.
        if output_a == output_b and len(walk) >= 2:
            unique_steps = list(dict.fromkeys(walk))
            if len(unique_steps) >= 2:
                replacement = next(s for s in unique_steps if s != output_b[0])
                output_b = [replacement] + output_b[1:]
        if variant == 1:
            return output_a if condition_result else output_b
        else:
            return output_b if condition_result else output_a

    def _output_a(self, walk: List[int]) -> List[int]:
        raise NotImplementedError

    def _output_b(self, walk: List[int]) -> List[int]:
        raise NotImplementedError

    def to_dict(self) -> dict:
        raise NotImplementedError

    @staticmethod
    def from_dict(data: dict) -> "OutputFunction":
        fn_type = data["fn_type"]
        ids = data["output_step_ids"]
        slots = data.get("slots_per_domain", 0)
        if fn_type == "first_last":
            return FirstLastOutputFn(ids, slots_per_domain=slots)
        elif fn_type == "most_least_frequent":
            return MostLeastFrequentOutputFn(ids, slots_per_domain=slots)
        elif fn_type == "second_penultimate":
            return SecondPenultimateOutputFn(ids, slots_per_domain=slots)
        elif fn_type == "majority_minority":
            return MajorityMinorityOutputFn(ids, slots_per_domain=slots)
        elif fn_type == "extremes_middle":
            return ExtremesMiddleOutputFn(ids, slots_per_domain=slots)
        elif fn_type == "unique_repeated":
            return UniqueRepeatedOutputFn(ids, slots_per_domain=slots)
        elif fn_type == "partition_bias":
            return PartitionBiasOutputFn(
                output_step_ids=ids,
                set1=frozenset(data["set1"]),
                set2=frozenset(data["set2"]),
                slots_per_domain=slots,
            )
        elif fn_type == "partition_collect":
            return PartitionCollectOutputFn(
                output_step_ids=ids,
                set1=frozenset(data["set1"]),
                set2=frozenset(data["set2"]),
                slots_per_domain=slots,
            )
        raise ValueError(f"Unknown output function type: {fn_type}")


class FirstLastOutputFn(OutputFunction):
    """A: [last_walk_step, out_0]  B: [first_walk_step, out_1]
    When `output_step_ids` is empty, tails are dropped."""

    def _output_a(self, walk: List[int]) -> List[int]:
        return ([walk[-1]] if walk else []) + self._tail_a()

    def _output_b(self, walk: List[int]) -> List[int]:
        return ([walk[0]] if walk else []) + self._tail_b()

    def to_dict(self) -> dict:
        return {"fn_type": "first_last", "output_step_ids": self.output_step_ids, "slots_per_domain": self.slots_per_domain}


class MostLeastFrequentOutputFn(OutputFunction):
    """A: [most_frequent_step, out_0]  B: [least_frequent_step, out_1]"""

    def _output_a(self, walk: List[int]) -> List[int]:
        if not walk:
            return self._tail_a()
        most = Counter(walk).most_common(1)[0][0]
        return [most] + self._tail_a()

    def _output_b(self, walk: List[int]) -> List[int]:
        if not walk:
            return self._tail_b()
        least = Counter(walk).most_common()[-1][0]
        return [least] + self._tail_b()

    def to_dict(self) -> dict:
        return {"fn_type": "most_least_frequent", "output_step_ids": self.output_step_ids, "slots_per_domain": self.slots_per_domain}


class SecondPenultimateOutputFn(OutputFunction):
    """A: [second_step, out_0, last_step]  B: [penultimate_step, out_1, first_step]
    With no tail, the mid-sequence terminator is simply omitted."""

    def _output_a(self, walk: List[int]) -> List[int]:
        if not walk:
            return self._tail_a()
        second = walk[1] if len(walk) > 1 else walk[0]
        return [second] + self._tail_a() + [walk[-1]]

    def _output_b(self, walk: List[int]) -> List[int]:
        if not walk:
            return self._tail_b()
        penult = walk[-2] if len(walk) > 1 else walk[-1]
        return [penult] + self._tail_b() + [walk[0]]

    def to_dict(self) -> dict:
        return {"fn_type": "second_penultimate", "output_step_ids": self.output_step_ids, "slots_per_domain": self.slots_per_domain}


class MajorityMinorityOutputFn(OutputFunction):
    """A: [step appearing in >50% of walk positions, output_step_0]
       B: [first step NOT appearing in >50%, output_step_1]
    Falls back to first/last if no majority exists."""

    def _find_majority(self, walk: List[int]):
        threshold = len(walk) / 2
        counts = Counter(walk)
        for step, cnt in counts.most_common():
            if cnt > threshold:
                return step
        return None

    def _output_a(self, walk: List[int]) -> List[int]:
        if not walk:
            return self._tail_a()
        maj = self._find_majority(walk)
        pick = maj if maj is not None else walk[-1]
        return [pick] + self._tail_a()

    def _output_b(self, walk: List[int]) -> List[int]:
        if not walk:
            return self._tail_b()
        maj = self._find_majority(walk)
        if maj is not None:
            pick = next((s for s in walk if s != maj), walk[0])
        else:
            pick = walk[0]
        return [pick] + self._tail_b()

    def to_dict(self) -> dict:
        return {"fn_type": "majority_minority", "output_step_ids": self.output_step_ids, "slots_per_domain": self.slots_per_domain}


class ExtremesMiddleOutputFn(OutputFunction):
    """A: [first_step, last_step, out_0]  B: [middle_step, out_1]"""

    def _output_a(self, walk: List[int]) -> List[int]:
        if not walk:
            return self._tail_a()
        return [walk[0], walk[-1]] + self._tail_a()

    def _output_b(self, walk: List[int]) -> List[int]:
        if not walk:
            return self._tail_b()
        mid = walk[len(walk) // 2]
        return [mid] + self._tail_b()

    def to_dict(self) -> dict:
        return {"fn_type": "extremes_middle", "output_step_ids": self.output_step_ids, "slots_per_domain": self.slots_per_domain}


class UniqueRepeatedOutputFn(OutputFunction):
    """A: [first step that appears exactly once, output_step_0]
       B: [first step that appears more than once, output_step_1]
    Falls back to first/last if condition not met."""

    def _output_a(self, walk: List[int]) -> List[int]:
        if not walk:
            return self._tail_a()
        counts = Counter(walk)
        unique = next((s for s in walk if counts[s] == 1), walk[0])
        return [unique] + self._tail_a()

    def _output_b(self, walk: List[int]) -> List[int]:
        if not walk:
            return self._tail_b()
        counts = Counter(walk)
        repeated = next((s for s in walk if counts[s] > 1), walk[-1])
        return [repeated] + self._tail_b()

    def to_dict(self) -> dict:
        return {"fn_type": "unique_repeated", "output_step_ids": self.output_step_ids, "slots_per_domain": self.slots_per_domain}


# ---------------------------------------------------------------------------
# Partition-based output functions
#
# These are CONTENT-DEPENDENT: unlike the functions above (which mostly care
# about position or frequency), these functions care about *which specific
# steps* appear in the walk via a task-specific binary partition of the
# global step pool. When a task uses one of these, `build_task` also creates
# a matching `PartitionDominanceCondition` using the same partition so the
# condition and output function agree on the same set1/set2 split.
# ---------------------------------------------------------------------------


class PartitionBiasOutputFn(OutputFunction):
    """Content-aware: picks one walk step from set1 (A) or set2 (B).

    A : [first walk step that's in set1, out_0]   (fallback: walk[-1])
    B : [first walk step that's in set2, out_1]   (fallback: walk[0])

    Because the partition is fixed at task-construction time but the walk
    content varies per example, the answer depends on *which specific steps
    are present* — a genuinely content-sensitive function.
    """

    def __init__(
        self,
        output_step_ids: List[int],
        set1: FrozenSet[int],
        set2: FrozenSet[int],
        slots_per_domain: int = 0,
    ):
        super().__init__(output_step_ids, slots_per_domain=slots_per_domain)
        self.set1 = frozenset(set1)
        self.set2 = frozenset(set2)

    def _output_a(self, walk: List[int]) -> List[int]:
        if not walk:
            return self._tail_a()
        pick = next((s for s in walk if s in self.set1), walk[-1])
        return [pick] + self._tail_a()

    def _output_b(self, walk: List[int]) -> List[int]:
        if not walk:
            return self._tail_b()
        pick = next((s for s in walk if s in self.set2), walk[0])
        return [pick] + self._tail_b()

    def to_dict(self) -> dict:
        return {
            "fn_type": "partition_bias",
            "output_step_ids": self.output_step_ids,
            "set1": sorted(self.set1),
            "set2": sorted(self.set2),
            "slots_per_domain": self.slots_per_domain,
        }


class PartitionCollectOutputFn(OutputFunction):
    """Content-aware: emits ALL walk steps that fall in set1 / set2.

    A : [all walk steps in set1, in walk order, then out_0]
        fallback if none present: [walk[-1], out_0]
    B : [all walk steps in set2, in walk order, then out_1]
        fallback if none present: [walk[0], out_1]

    The output length varies with the walk content. When the walk is heavy
    on set1, variant-1 answers (under PartitionDominanceCondition) produce
    long step sequences; when heavy on set2 they produce short ones, and
    vice-versa.
    """

    def __init__(
        self,
        output_step_ids: List[int],
        set1: FrozenSet[int],
        set2: FrozenSet[int],
        slots_per_domain: int = 0,
    ):
        super().__init__(output_step_ids, slots_per_domain=slots_per_domain)
        self.set1 = frozenset(set1)
        self.set2 = frozenset(set2)

    def _output_a(self, walk: List[int]) -> List[int]:
        if not walk:
            return self._tail_a()
        picks = [s for s in walk if s in self.set1]
        if not picks:
            picks = [walk[-1]]
        return picks + self._tail_a()

    def _output_b(self, walk: List[int]) -> List[int]:
        if not walk:
            return self._tail_b()
        picks = [s for s in walk if s in self.set2]
        if not picks:
            picks = [walk[0]]
        return picks + self._tail_b()

    def to_dict(self) -> dict:
        return {
            "fn_type": "partition_collect",
            "output_step_ids": self.output_step_ids,
            "set1": sorted(self.set1),
            "set2": sorted(self.set2),
            "slots_per_domain": self.slots_per_domain,
        }


#: Names of OutputFunction subclasses that require a partition. When
#: build_task picks one of these, it also creates a matching
#: PartitionDominanceCondition using the same partition.
_PARTITION_FN_NAMES = {"PartitionBiasOutputFn", "PartitionCollectOutputFn"}


# All output function types — partition-based ones at the end so existing
# small worlds (num_tasks <= 6) still round-robin through the content-
# independent set by default.
_OUTPUT_FN_TYPES = [
    FirstLastOutputFn,
    MostLeastFrequentOutputFn,
    SecondPenultimateOutputFn,
    MajorityMinorityOutputFn,
    ExtremesMiddleOutputFn,
    UniqueRepeatedOutputFn,
    PartitionBiasOutputFn,       # NEW — content-dependent via partition
    PartitionCollectOutputFn,    # NEW — content-dependent via partition
]


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

@dataclass
class TaskDefinition:
    """A complete task definition (domain-independent).

    `edge_deletions` stores concrete ``(src_global_step, dst_global_step)``
    pairs drawn from the global directed-edge space. When the task is
    applied to a domain, each pair is intersected with that domain's
    adjacency; pairs whose endpoints are not both in the domain's step
    subset, or whose edge simply isn't present, are silent no-ops.
    """

    task_id: int
    label_cfg: CFG
    edge_deletions: FrozenSet[Tuple[int, int]]
    condition: ConditionFunction
    output_fn: OutputFunction
    output_step_cfgs: List[CFG]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "label_cfg": self.label_cfg.to_dict(),
            # Store as a sorted list of [src, dst] pairs for deterministic JSON
            "edge_deletions": sorted(
                [list(pair) for pair in self.edge_deletions]
            ),
            "condition": self.condition.to_dict(),
            "output_fn": self.output_fn.to_dict(),
            "output_step_cfgs": [c.to_dict() for c in self.output_step_cfgs],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskDefinition":
        if "edge_deletions" in data:
            pairs = data["edge_deletions"]
            edge_deletions = frozenset(
                (int(p[0]), int(p[1])) for p in pairs
            )
        elif "edge_indices" in data:
            raise ValueError(
                "This world.json was saved with the deprecated positional "
                "`edge_indices` scheme (task.edge_indices was a list of "
                "integer positions into each domain's canonical edge list). "
                "The current code uses `edge_deletions` — concrete "
                "(src_global, dst_global) pairs drawn from the global edge "
                "space. Rebuild the world (and regenerate datasets) with "
                "`python generate_dataset_v3.py build-world ...` to migrate."
            )
        else:
            raise ValueError(
                "Task dict is missing both `edge_deletions` and the legacy "
                "`edge_indices` key — the file is not a recognizable "
                "world.json task serialization."
            )
        return cls(
            task_id=data["task_id"],
            label_cfg=CFG.from_dict(data["label_cfg"]),
            edge_deletions=edge_deletions,
            condition=ConditionFunction.from_dict(data["condition"]),
            output_fn=OutputFunction.from_dict(data["output_fn"]),
            output_step_cfgs=[CFG.from_dict(c) for c in data["output_step_cfgs"]],
        )


def _build_condition(
    rng: random.Random, num_steps: int, condition_type: int
) -> ConditionFunction:
    """Build a condition function of the given type."""
    if condition_type == 0:
        step_id = rng.randint(0, num_steps - 1)
        # Threshold 0 or 1 so the condition fires reasonably often
        threshold = rng.randint(0, 1)
        return CountingCondition(step_id, threshold)
    elif condition_type == 1:
        step_id = rng.randint(0, num_steps - 1)
        return PresenceCondition(step_id)
    elif condition_type == 2:
        step_x = rng.randint(0, num_steps - 1)
        step_y = rng.randint(0, num_steps - 1)
        while step_y == step_x:
            step_y = rng.randint(0, num_steps - 1)
        return OrderingCondition(step_x, step_y)
    else:
        return ParityCondition()


def _random_partition(
    rng: random.Random, num_steps: int,
) -> Tuple[FrozenSet[int], FrozenSet[int]]:
    """Split range(num_steps) into two disjoint subsets of ~equal size.

    Each step is independently assigned to set1 (50%) or set2 (50%). If one
    of the two sets is empty (unlikely), force at least one element into it.
    """
    all_ids = list(range(num_steps))
    rng.shuffle(all_ids)
    half = num_steps // 2
    s1 = frozenset(all_ids[:half])
    s2 = frozenset(all_ids[half:])
    # Safety guard
    if not s1 and s2:
        x = next(iter(s2))
        s1 = frozenset({x})
        s2 = frozenset(s2 - {x})
    if not s2 and s1:
        x = next(iter(s1))
        s2 = frozenset({x})
        s1 = frozenset(s1 - {x})
    return s1, s2


def _sample_edge_deletions(
    rng: random.Random,
    num_steps: int,
    n_edges: int,
) -> FrozenSet[Tuple[int, int]]:
    """Sample `n_edges` distinct directed (src, dst) pairs uniformly from
    the global edge space (all ordered pairs with src != dst).

    This is the pool of edges the task "targets" across every domain it
    gets applied to. When applied, each pair is intersected with the
    specific domain's adjacency — pairs that happen not to exist in a
    particular domain become silent no-ops for that domain.
    """
    total = num_steps * (num_steps - 1)  # e.g. 64 * 63 = 4032
    n_take = min(n_edges, total)
    all_pairs = [
        (i, j) for i in range(num_steps) for j in range(num_steps) if i != j
    ]
    picked = rng.sample(all_pairs, n_take)
    return frozenset(picked)


def _partition_from_reference(
    rng: random.Random,
    ref_set1: FrozenSet[int],
    ref_set2: FrozenSet[int],
    kept: int,
) -> Tuple[FrozenSet[int], FrozenSet[int]]:
    """Build a new (set1, set2) partition that keeps `kept` elements from
    ref_set1 and replaces the rest with elements drawn from ref_set2.

    Sizes match the reference exactly: |new_set1| = |ref_set1|.
    """
    ref1_list = list(ref_set1)
    ref2_list = list(ref_set2)
    size1 = len(ref1_list)
    kept = max(0, min(kept, size1))
    needed_new = size1 - kept

    rng.shuffle(ref1_list)
    rng.shuffle(ref2_list)
    keep_part = ref1_list[:kept]
    new_part = ref2_list[:needed_new]
    new_set1 = frozenset(keep_part + new_part)
    # Remaining elements form set2 (all originally-set1 not kept + remaining
    # originally-set2 not pulled into set1).
    leftover1 = ref1_list[kept:]
    leftover2 = ref2_list[needed_new:]
    new_set2 = frozenset(leftover1 + leftover2)
    return new_set1, new_set2


def build_partition_task(
    task_id: int,
    label_cfg: CFG,
    output_step_cfgs: List[CFG],
    output_step_ids: List[int],
    rng: random.Random,
    num_steps: int,
    set1: FrozenSet[int],
    set2: FrozenSet[int],
    edge_deletions_range: Tuple[int, int] = (1, 4),
    slots_per_domain: int = 0,
) -> TaskDefinition:
    """Build a PartitionBias task with a CALLER-supplied partition (used to
    enforce a controlled set-similarity gradient across multiple tasks)."""
    num_deletions = rng.randint(*edge_deletions_range)
    edge_deletions = _sample_edge_deletions(rng, num_steps, num_deletions)
    output_fn = PartitionBiasOutputFn(
        output_step_ids, set1=set1, set2=set2,
        slots_per_domain=slots_per_domain,
    )
    condition = PartitionDominanceCondition(set1=set1, set2=set2)
    return TaskDefinition(
        task_id=task_id,
        label_cfg=label_cfg,
        edge_deletions=edge_deletions,
        condition=condition,
        output_fn=output_fn,
        output_step_cfgs=output_step_cfgs,
    )


def build_task(
    task_id: int,
    label_cfg: CFG,
    output_step_cfgs: List[CFG],
    output_step_ids: List[int],
    rng: random.Random,
    num_steps: int,
    edge_deletions_range: Tuple[int, int] = (1, 4),
    slots_per_domain: int = 0,
) -> TaskDefinition:
    """Build a single task definition with a unique output function type.

    Edge deletions are sampled as concrete (src_global, dst_global) step
    pairs from the full global directed edge space. When the task is
    applied to a domain, each pair is intersected with that domain's real
    edges; pairs whose endpoints aren't in the domain, or whose edge
    simply isn't present, are silently skipped.

    If the assigned output function is partition-based, the same partition
    is shared between the output function and a PartitionDominanceCondition
    — this is what makes the condition-swap semantics coherent for those
    tasks (variant 1 = agree with dominance, variant 2 = disagree).
    """
    # Sample concrete (src, dst) edge pairs from the global edge space.
    num_deletions = rng.randint(*edge_deletions_range)
    edge_deletions = _sample_edge_deletions(rng, num_steps, num_deletions)

    # Output function — pick based on task_id to ensure each task is different
    fn_cls = _OUTPUT_FN_TYPES[task_id % len(_OUTPUT_FN_TYPES)]

    if fn_cls.__name__ in _PARTITION_FN_NAMES:
        # Tied partition: condition and output function share the same set1/set2
        set1, set2 = _random_partition(rng, num_steps)
        output_fn = fn_cls(
            output_step_ids, set1=set1, set2=set2,
            slots_per_domain=slots_per_domain,
        )
        condition = PartitionDominanceCondition(set1=set1, set2=set2)
    else:
        output_fn = fn_cls(output_step_ids, slots_per_domain=slots_per_domain)
        # Content-independent output fn → random condition from the
        # original non-partition set
        condition_type = rng.randint(0, 3)
        condition = _build_condition(rng, num_steps, condition_type)

    return TaskDefinition(
        task_id=task_id,
        label_cfg=label_cfg,
        edge_deletions=edge_deletions,
        condition=condition,
        output_fn=output_fn,
        output_step_cfgs=output_step_cfgs,
    )
