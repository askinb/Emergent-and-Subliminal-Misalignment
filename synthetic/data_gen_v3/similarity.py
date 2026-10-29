"""Graph and task similarity metrics.

PRIMARY definitions:

1. Domain similarity — `transition_matrix_similarity`:
     Let T_D be the N x N transition matrix of domain D over the *global*
     step space (N = num_steps), where T_D[i, j] = P(step j | step i) in D,
     and rows for steps not in D are zero. Define

         sim(D_a, D_b) = 1 - ||T_a - T_b||_1 / (||T_a||_1 + ||T_b||_1)

     This is in [0, 1]: 1 = identical transition structure, 0 = no shared
     mass anywhere in the N x N matrix. It captures both which steps are
     used AND how transitions are distributed.

2. Task similarity — `task_augmented_graph_similarity`:
     For each domain d and each task t, build an *empirical augmented
     transition matrix* by sampling walks from the task-modified domain
     graph, computing the variant-1 output for each walk, and counting
     transitions in the full (walk + output) step sequence.
     This matrix captures BOTH:
       - the edge-deletion effect (walk transitions after deletion)
       - the output function (walk-end→output transitions and within-output
         transitions)
     Then define

         sim(t_a, t_b) = mean over domains of transition_matrix_similarity(
             augmented(t_a, d),  augmented(t_b, d)
         )

     This is the graph-based analog of the "augment the domain graph
     with possible output step sequences, then compare augmented graphs."

Supplementary metrics (retained but NOT blended into any "combined" score):
  - step_overlap_jaccard         — Jaccard of step-ID sets (domain)
  - spectral_similarity          — adjacency eigenvalue distance (domain)
  - task_effective_edge_similarity — mean Jaccard of effective deleted
                                    edges (edge-deletion-only task signal)
  - task_output_similarity       — output-function-only task similarity:
                                    sample walks from the unmodified domain
                                    graph (both tasks see identical walks),
                                    compute each task's A and B output
                                    branches, strip each task's own output
                                    step IDs, compare via multiset Jaccard.
                                    Averages across walks and domains.
  - output_fn_type_match         — 1 if same OutputFunction subclass   (task)
  - condition_type_match         — 1 if same ConditionFunction subclass (task)

All metrics are in [0, 1] — 1 = identical, 0 = no overlap.
"""

import random
from collections import Counter
from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np

from data_gen_v3.graph import DomainGraph
from data_gen_v3.tasks import TaskDefinition


# ══════════════════════════════════════════════════════════════════════════
# Domain similarity
# ══════════════════════════════════════════════════════════════════════════

def _full_transition_matrix(graph: DomainGraph, n_global_steps: int) -> np.ndarray:
    """Build the (N x N) global-step-indexed transition matrix for a domain.

    T[i, j] = P(step j | step i) under this domain's graph, where i and j
    are GLOBAL step IDs. Rows/cols for steps not in the domain are all zero.
    """
    mat = np.zeros((n_global_steps, n_global_steps), dtype=np.float64)
    for local_src in range(graph.num_nodes):
        global_src = graph.step_ids[local_src]
        succs = graph.adjacency.get(local_src, [])
        probs = graph.transition_probs.get(local_src, [])
        for local_dst, p in zip(succs, probs):
            global_dst = graph.step_ids[local_dst]
            mat[global_src, global_dst] = p
    return mat


def transition_matrix_similarity(
    graph_a: DomainGraph,
    graph_b: DomainGraph,
    n_global_steps: int,
) -> float:
    """PRIMARY domain similarity.

        sim = 1 - ||T_a - T_b||_1 / (||T_a||_1 + ||T_b||_1)

    where T_* is the NxN global-step transition matrix (rows outside the
    domain's step set are zero). In [0, 1]. 1 iff T_a == T_b.
    """
    T_a = _full_transition_matrix(graph_a, n_global_steps)
    T_b = _full_transition_matrix(graph_b, n_global_steps)
    diff = float(np.abs(T_a - T_b).sum())
    mass = float(np.abs(T_a).sum() + np.abs(T_b).sum())
    if mass < 1e-12:
        return 1.0
    return max(0.0, 1.0 - diff / mass)


def step_overlap_jaccard(graph_a: DomainGraph, graph_b: DomainGraph) -> float:
    """Supplementary: Jaccard similarity of step-ID sets."""
    set_a = set(graph_a.step_ids)
    set_b = set(graph_b.step_ids)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _adjacency_matrix(graph: DomainGraph) -> np.ndarray:
    n = graph.num_nodes
    mat = np.zeros((n, n))
    for src, dsts in graph.adjacency.items():
        for dst in dsts:
            mat[src, dst] = 1.0
    return mat


def spectral_similarity(graph_a: DomainGraph, graph_b: DomainGraph) -> float:
    """Supplementary: eigenvalue-distance similarity of adjacency matrices."""
    mat_a = _adjacency_matrix(graph_a)
    mat_b = _adjacency_matrix(graph_b)

    eigs_a = np.sort(np.abs(np.linalg.eigvals(mat_a)))
    eigs_b = np.sort(np.abs(np.linalg.eigvals(mat_b)))

    max_len = max(len(eigs_a), len(eigs_b))
    eigs_a = np.pad(eigs_a, (0, max_len - len(eigs_a)))
    eigs_b = np.pad(eigs_b, (0, max_len - len(eigs_b)))

    dist = np.linalg.norm(eigs_a - eigs_b)
    max_norm = max(np.linalg.norm(eigs_a), np.linalg.norm(eigs_b), 1e-10)
    return float(max(0.0, 1.0 - dist / max_norm))


# ══════════════════════════════════════════════════════════════════════════
# Task similarity
# ══════════════════════════════════════════════════════════════════════════

def effective_deleted_edges(
    task: TaskDefinition,
    domain,
) -> FrozenSet[Tuple[int, int]]:
    """Return the set of (src_global, dst_global) edge pairs that `task`
    actually removes from `domain` when applied.

    Under the new concrete-pair scheme this is simply the intersection of
    the task's target edges with the set of real (src_global, dst_global)
    edges that exist in this domain's graph. Pairs whose endpoints aren't
    in the domain, or whose edge isn't present, are silently excluded.
    """
    real: Set[Tuple[int, int]] = set()
    for local_src, succs in domain.graph.adjacency.items():
        src_g = domain.graph.step_ids[local_src]
        for local_dst in succs:
            dst_g = domain.graph.step_ids[local_dst]
            real.add((src_g, dst_g))

    return frozenset(task.edge_deletions & real)


def task_effective_edge_similarity(
    task_a: TaskDefinition,
    task_b: TaskDefinition,
    world,
) -> float:
    """Supplementary task similarity.

    Mean over domains of Jaccard(E_a^d, E_b^d) where E_*^d is the set of
    concrete (src_global, dst_global) pairs each task actually deletes from
    domain d. In [0, 1]: 1 iff every domain's deleted-edge set matches.

    This is EDGE-DELETION ONLY — it does not account for differences in
    output functions. For a combined signal, use
    `task_augmented_graph_similarity`.
    """
    jaccards: List[float] = []
    for d in world.domains:
        e_a = effective_deleted_edges(task_a, d)
        e_b = effective_deleted_edges(task_b, d)
        if not e_a and not e_b:
            jaccards.append(1.0)  # neither deletes anything → behaviorally identical
            continue
        inter = len(e_a & e_b)
        union = len(e_a | e_b)
        jaccards.append(inter / union if union > 0 else 0.0)
    return float(np.mean(jaccards)) if jaccards else 0.0


def augmented_transition_matrix(
    task: TaskDefinition,
    domain,
    world,
    n_samples: int = 300,
    variant: int = 1,
    seed: int = 2024,
) -> np.ndarray:
    """Build an empirical (N × N) transition matrix for one (task, domain) pair.

    N = num_steps (global pool) + total task output steps across ALL tasks.
    (A single shared step-id space so matrices from different tasks can be
    compared cell-for-cell.)

    Procedure:
      1. Copy the domain graph and apply `task.edge_deletions` → modified graph.
      2. Sample `n_samples` walks from the modified graph (walk lengths drawn
         from `world.config.walk_length_range`).
      3. For each walk, evaluate `task.condition` and compute `task.output_fn`
         for the given variant, yielding a step sequence `output`.
      4. Form the full step sequence `full = walk + output` and increment
         transition counts mat[full[i], full[i+1]] for every adjacent pair.
      5. Row-normalize so each row sums to 1 (or 0 if that row got no counts).

    The resulting matrix is the "augmented graph" — it captures the
    post-deletion walk transitions (rows for steps in the domain), the
    walk-to-output transition (last-step → first-output), and internal
    output transitions (output[i] → output[i+1]). It is the graph-based
    fingerprint of what this task does in this domain.
    """
    rng = random.Random(seed)

    n_base = world.config.num_steps
    n_output_total = sum(
        len(t.output_fn.output_step_ids) for t in world.tasks
    )
    N = n_base + n_output_total

    mat = np.zeros((N, N), dtype=np.float64)

    modified_graph = domain.graph.delete_edges(task.edge_deletions)
    walk_min, walk_max = world.config.walk_length_range

    for _ in range(n_samples):
        walk_len = rng.randint(walk_min, walk_max)
        walk = modified_graph.random_walk(rng, walk_len)
        if not walk:
            walk = domain.graph.random_walk(rng, walk_len)
            if not walk:
                continue
        cond = task.condition.evaluate(walk)
        output = task.output_fn.compute(walk, cond, variant)
        full_seq = list(walk) + list(output)
        for i in range(len(full_seq) - 1):
            src, dst = full_seq[i], full_seq[i + 1]
            if 0 <= src < N and 0 <= dst < N:
                mat[src, dst] += 1.0

    # Row-normalize (rows that got no counts stay zero)
    row_sums = mat.sum(axis=1, keepdims=True)
    safe = np.where(row_sums == 0, 1.0, row_sums)
    return mat / safe


def task_augmented_graph_similarity(
    task_a: TaskDefinition,
    task_b: TaskDefinition,
    world,
    n_samples: int = 300,
    seed: int = 2024,
) -> float:
    """PRIMARY task similarity.

    Builds the augmented transition matrix for each task in each domain and
    compares them using the same L1-based similarity as the primary domain
    metric. Averages across all domains.

    Captures BOTH the edge-deletion effect and the output function, so it
    reflects how "behaviorally similar" two tasks are end-to-end.
    """
    sims: List[float] = []
    for d in world.domains:
        mat_a = augmented_transition_matrix(
            task_a, d, world, n_samples=n_samples, seed=seed,
        )
        mat_b = augmented_transition_matrix(
            task_b, d, world, n_samples=n_samples, seed=seed,
        )
        diff = float(np.abs(mat_a - mat_b).sum())
        mass = float(np.abs(mat_a).sum() + np.abs(mat_b).sum())
        if mass < 1e-12:
            sims.append(1.0)
        else:
            sims.append(max(0.0, 1.0 - diff / mass))
    return float(np.mean(sims)) if sims else 0.0


def _multiset_jaccard(a: List[int], b: List[int]) -> float:
    """Jaccard similarity over multisets (Counters).

    Given two lists of step IDs, treat each as a multiset. Return
    |min(a,b)| / |max(a,b)| where min/max are the pointwise minimum /
    maximum of the Counters (so repetitions count). Both empty → 1.0.
    """
    ca = Counter(a)
    cb = Counter(b)
    if not ca and not cb:
        return 1.0
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union > 0 else 0.0


def task_output_similarity(
    task_a: TaskDefinition,
    task_b: TaskDefinition,
    world,
    n_samples: int = 300,
    seed: int = 2024,
) -> float:
    """Supplementary task similarity — OUTPUT FUNCTION ONLY.

    Sample walks from each domain's UNMODIFIED graph (no task-specific edge
    deletions applied) so both tasks see the same walks, then compute each
    task's A and B output branches via `OutputFunction.compute(walk, cond, 1)`
    with cond=True → A and cond=False → B.

    For each walk:
      1. Strip each task's own `output_fn.output_step_ids` from that task's
         outputs (these are task-specific "filler" steps that would always
         mismatch across distinct tasks and would otherwise cap similarity
         at ≈ walk_len/(walk_len+1)).
      2. Compute multiset Jaccard between task_a's stripped A and task_b's
         stripped A, and separately between task_a's stripped B and task_b's
         stripped B.
      3. Per-walk similarity = average of the two Jaccards.

    Per-domain similarity = mean over walks. Final = mean over domains.

    Because walks come from the unmodified graph, the metric is independent
    of task edge deletions and cleanly isolates the output function's
    walk-step selection behavior. This is the counterpart to
    `task_effective_edge_similarity` which measures only edge deletions,
    while `task_augmented_graph_similarity` measures their combination.
    """
    rng = random.Random(seed)
    walk_min, walk_max = world.config.walk_length_range

    per_domain: List[float] = []
    out_ids_a = set(task_a.output_fn.output_step_ids)
    out_ids_b = set(task_b.output_fn.output_step_ids)

    for d in world.domains:
        per_walk: List[float] = []
        for _ in range(n_samples):
            walk_len = rng.randint(walk_min, walk_max)
            walk = d.graph.random_walk(rng, walk_len)
            if not walk:
                continue
            a_a = task_a.output_fn.compute(walk, True, 1)   # task_a branch A
            b_a = task_a.output_fn.compute(walk, False, 1)  # task_a branch B
            a_b = task_b.output_fn.compute(walk, True, 1)
            b_b = task_b.output_fn.compute(walk, False, 1)

            a_a_walkonly = [s for s in a_a if s not in out_ids_a]
            b_a_walkonly = [s for s in b_a if s not in out_ids_a]
            a_b_walkonly = [s for s in a_b if s not in out_ids_b]
            b_b_walkonly = [s for s in b_b if s not in out_ids_b]

            j_a = _multiset_jaccard(a_a_walkonly, a_b_walkonly)
            j_b = _multiset_jaccard(b_a_walkonly, b_b_walkonly)
            per_walk.append(0.5 * (j_a + j_b))

        if per_walk:
            per_domain.append(float(np.mean(per_walk)))
    return float(np.mean(per_domain)) if per_domain else 0.0


def condition_type_match(task_a: TaskDefinition, task_b: TaskDefinition) -> float:
    """Supplementary: 1.0 if same ConditionFunction subclass, else 0.0."""
    return 1.0 if type(task_a.condition) == type(task_b.condition) else 0.0


def output_fn_type_match(task_a: TaskDefinition, task_b: TaskDefinition) -> float:
    """Supplementary: 1.0 if same OutputFunction subclass, else 0.0."""
    return 1.0 if type(task_a.output_fn) == type(task_b.output_fn) else 0.0


# ══════════════════════════════════════════════════════════════════════════
# Pairwise matrix builders
# ══════════════════════════════════════════════════════════════════════════

#: Metric name shown in the figures for the primary definition
PRIMARY_DOMAIN_METRIC = "transition_matrix_similarity"
PRIMARY_TASK_METRIC = "augmented_graph_similarity"


def build_domain_similarity_matrices(world) -> Dict[str, np.ndarray]:
    """Compute every domain-similarity metric as a (num_domains x num_domains) matrix.

    Returns a dict with keys:
      - "transition_matrix_similarity" (PRIMARY)
      - "step_overlap_jaccard"
      - "spectral_similarity"
    """
    n = len(world.domains)
    keys = [
        "transition_matrix_similarity",
        "step_overlap_jaccard",
        "spectral_similarity",
    ]
    mats = {k: np.zeros((n, n)) for k in keys}

    for i in range(n):
        for j in range(n):
            if i == j:
                for k in keys:
                    mats[k][i, j] = 1.0
                continue
            g_a = world.domains[i].graph
            g_b = world.domains[j].graph
            mats["transition_matrix_similarity"][i, j] = transition_matrix_similarity(
                g_a, g_b, n_global_steps=world.config.num_steps,
            )
            mats["step_overlap_jaccard"][i, j] = step_overlap_jaccard(g_a, g_b)
            mats["spectral_similarity"][i, j] = spectral_similarity(g_a, g_b)
    return mats


def build_task_similarity_matrices(
    world,
    augmented_n_samples: int = 300,
    augmented_seed: int = 2024,
) -> Dict[str, np.ndarray]:
    """Compute every task-similarity metric as a (num_tasks x num_tasks) matrix.

    Returns a dict with keys:
      - "augmented_graph_similarity"  (PRIMARY)
      - "effective_edge_similarity"
      - "output_similarity"           (new: output-function-only)
      - "condition_type_match"
      - "output_fn_type_match"

    The augmented metric precomputes the (task × domain) transition matrices
    once and reuses them for all pairwise comparisons — without this, the
    8×8 task grid would re-sample the same walks 64 times.
    """
    n = len(world.tasks)
    n_domains = len(world.domains)

    # Precompute one augmented matrix per (task, domain) pair
    aug: List[List[np.ndarray]] = []  # aug[task_idx][domain_idx]
    for ti, t in enumerate(world.tasks):
        row: List[np.ndarray] = []
        for d in world.domains:
            row.append(augmented_transition_matrix(
                t, d, world,
                n_samples=augmented_n_samples, seed=augmented_seed,
            ))
        aug.append(row)

    def _sim_from_precomputed(i: int, j: int) -> float:
        sims: List[float] = []
        for di in range(n_domains):
            mat_a = aug[i][di]
            mat_b = aug[j][di]
            diff = float(np.abs(mat_a - mat_b).sum())
            mass = float(np.abs(mat_a).sum() + np.abs(mat_b).sum())
            if mass < 1e-12:
                sims.append(1.0)
            else:
                sims.append(max(0.0, 1.0 - diff / mass))
        return float(np.mean(sims)) if sims else 0.0

    keys = [
        "augmented_graph_similarity",
        "effective_edge_similarity",
        "output_similarity",
        "condition_type_match",
        "output_fn_type_match",
    ]
    mats = {k: np.zeros((n, n)) for k in keys}

    for i in range(n):
        for j in range(n):
            if i == j:
                for k in keys:
                    mats[k][i, j] = 1.0
                continue
            t_a = world.tasks[i]
            t_b = world.tasks[j]
            mats["augmented_graph_similarity"][i, j] = _sim_from_precomputed(i, j)
            mats["effective_edge_similarity"][i, j] = task_effective_edge_similarity(
                t_a, t_b, world,
            )
            mats["output_similarity"][i, j] = task_output_similarity(t_a, t_b, world)
            mats["condition_type_match"][i, j] = condition_type_match(t_a, t_b)
            mats["output_fn_type_match"][i, j] = output_fn_type_match(t_a, t_b)
    return mats
