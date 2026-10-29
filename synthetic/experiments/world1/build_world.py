#!/usr/bin/env python3
"""Build World 1.

- 5 domains: D0 (random), D1..D3 (controlled decreasing similarity to D0),
  D4 (random baseline).
- 6 tasks: the first 6 output function types via the default task_id % 8 cycle:
    T0 first_last, T1 most_least_frequent, T2 second_penultimate,
    T3 majority_minority, T4 extremes_middle, T5 unique_repeated.
- Prints the full 5x5 transition-matrix-similarity matrix for sanity checking
  (including D4 so we can report whether random sampling happened to land
  close to D0).

The D1..D3 gradient follows the World 2 convention: D1 keeps 13/16 of D0's
steps, D2 keeps 10/16, D3 keeps 7/16. Shared-step edges are copied (with
small noise on the transition probabilities); non-shared edges are fresh
random draws.
"""

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_gen_v3.config import WorldConfig
from data_gen_v3.graph import (
    DomainGraph,
    _is_weakly_connected,
    _is_chain,
    _is_fully_connected,
)
from data_gen_v3.world import build_world, DomainDefinition
from data_gen_v3.similarity import transition_matrix_similarity


SIMILARITY_SPEC = {
    1: 13,   # D1: keep 13/16 of D0's steps
    2: 10,   # D2: keep 10/16
    3: 7,    # D3: keep 7/16
}


def derive_domain_from(ref_graph, domain_id, all_step_ids, n_keep, rng,
                       edge_prob, min_out_degree, max_attempts=200):
    ref_steps = ref_graph.step_ids
    n = len(ref_steps)

    for _ in range(max_attempts):
        r = random.Random(rng.random())

        keep_locals = sorted(r.sample(range(n), n_keep))
        kept_global = {ref_steps[i] for i in keep_locals}

        available = [s for s in all_step_ids if s not in kept_global]
        new_global = set(r.sample(available, n - n_keep))

        new_steps = sorted(kept_global | new_global)
        assert len(new_steps) == n

        ref_to_new = {}
        for ref_local in keep_locals:
            gid = ref_steps[ref_local]
            ref_to_new[ref_local] = new_steps.index(gid)

        adj = {i: [] for i in range(n)}

        for new_local in range(n):
            gid = new_steps[new_local]

            if gid in kept_global:
                ref_local = ref_steps.index(gid)
                ref_succs = ref_graph.adjacency.get(ref_local, [])

                for rs in ref_succs:
                    if rs in ref_to_new:
                        adj[new_local].append(ref_to_new[rs])

                for j in range(n):
                    if j == new_local or j in adj[new_local]:
                        continue
                    if new_steps[j] in new_global:
                        if r.random() < edge_prob:
                            adj[new_local].append(j)
            else:
                for j in range(n):
                    if j != new_local and r.random() < edge_prob:
                        adj[new_local].append(j)

            while len(adj[new_local]) < min_out_degree:
                cands = [j for j in range(n) if j != new_local and j not in adj[new_local]]
                if not cands:
                    break
                adj[new_local].append(r.choice(cands))

        if not _is_weakly_connected(adj, n):
            continue
        if _is_chain(adj, n):
            continue
        if _is_fully_connected(adj, n):
            continue

        trans = {}
        for new_local in range(n):
            succs = adj[new_local]
            if not succs:
                trans[new_local] = []
                continue
            gid = new_steps[new_local]
            raw = []
            if gid in kept_global:
                ref_local = ref_steps.index(gid)
                ref_succs = ref_graph.adjacency.get(ref_local, [])
                ref_ps = ref_graph.transition_probs.get(ref_local, [])
                ref_prob_map = dict(zip(ref_succs, ref_ps))
                for s in succs:
                    s_gid = new_steps[s]
                    new_to_ref_s = None
                    if s_gid in kept_global:
                        ref_s_local = ref_steps.index(s_gid)
                        if ref_s_local in ref_prob_map:
                            new_to_ref_s = ref_prob_map[ref_s_local]
                    if new_to_ref_s is not None:
                        noisy = max(0.01, new_to_ref_s + r.gauss(0, 0.05))
                        raw.append(noisy)
                    else:
                        raw.append(r.gammavariate(1.0, 1.0))
            else:
                raw = [r.gammavariate(1.0, 1.0) for _ in succs]
            total = sum(raw)
            trans[new_local] = [p / total for p in raw]

        graph = DomainGraph(step_ids=new_steps, adjacency=adj, transition_probs=trans)
        return DomainDefinition(domain_id=domain_id, graph=graph)

    raise RuntimeError(f"Could not build derived domain after {max_attempts} attempts")


def main():
    EXP = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(EXP),
                    help="Folder for world.json and vocab.json (default: this folder)")
    args = ap.parse_args()
    config = WorldConfig.from_yaml(str(EXP / "world_config.yaml"))

    world = build_world(config)
    print(f"Built base world: {len(world.domains)} domains, {len(world.tasks)} tasks")

    ref_graph = world.domains[0].graph
    all_step_ids = list(range(config.num_steps))
    rng = random.Random(config.seed + 7777)

    for d_id, n_keep in SIMILARITY_SPEC.items():
        derived = derive_domain_from(
            ref_graph=ref_graph,
            domain_id=d_id,
            all_step_ids=all_step_ids,
            n_keep=n_keep,
            rng=rng,
            edge_prob=config.graph_edge_prob,
            min_out_degree=config.graph_min_out_degree,
        )
        world.domains[d_id] = derived
        overlap = len(set(ref_graph.step_ids) & set(derived.graph.step_ids))
        print(f"  D{d_id}: target overlap={n_keep}/16, actual overlap={overlap}/16, "
              f"{derived.graph.num_edges} edges")

    # D4 stays random (as built by build_world). Report its actual overlap vs D0.
    d4_overlap = len(set(ref_graph.step_ids) & set(world.domains[4].graph.step_ids))
    print(f"  D4: random (built by default), actual overlap to D0 = {d4_overlap}/16")

    # Full similarity matrix
    print("\nDomain similarity matrix (transition_matrix_similarity):")
    nd = len(world.domains)
    print(f"{'':>4}", end="")
    for j in range(nd):
        print(f"   D{j}", end="")
    print()
    for i in range(nd):
        print(f"D{i}  ", end="")
        for j in range(nd):
            sim = 1.0 if i == j else transition_matrix_similarity(
                world.domains[i].graph, world.domains[j].graph, config.num_steps,
            )
            print(f" {sim:.2f}", end="")
        print()

    # Print per-task output fn types so the reader knows what we built.
    print("\nTasks:")
    for t in world.tasks:
        print(f"  T{t.task_id}: {t.output_fn.to_dict()['fn_type']}")

    os.makedirs(args.output, exist_ok=True)
    world.save(os.path.join(args.output, "world.json"))
    world.vocab.save(os.path.join(args.output, "vocab.json"))
    print(f"\nSaved world to {args.output}/world.json")


if __name__ == "__main__":
    main()
