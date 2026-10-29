#!/usr/bin/env python3
"""Build World 2: D1, D2, D3 have controlled similarity to D0.

D0, D4-D7 and the OOD domains D8-D9: randomly constructed
D1: most similar to D0 (share 13/16 steps + copied edges)
D2: medium similar (share 10/16 steps + copied edges)
D3: least similar of the three (share 7/16 steps + copied edges)

After building, computes and prints domain similarity matrix to verify.
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


# How many of D0's 16 steps each derived domain keeps
SIMILARITY_SPEC = {
    1: 13,   # D1: keep 13/16 of D0's steps → very similar
    2: 10,   # D2: keep 10/16 → moderate
    3: 7,    # D3: keep 7/16 → lower but still above random
}


def derive_domain_from(
    ref_graph: DomainGraph,
    domain_id: int,
    all_step_ids: list,
    n_keep: int,
    rng: random.Random,
    edge_prob: float,
    min_out_degree: int,
    max_attempts: int = 200,
) -> DomainDefinition:
    """Create a domain derived from ref_graph by keeping n_keep of its steps
    and replacing the rest. Edges between shared steps are copied from ref;
    edges involving new steps are generated randomly."""

    ref_steps = ref_graph.step_ids
    n = len(ref_steps)

    for attempt in range(max_attempts):
        r = random.Random(rng.random())  # sub-rng to not pollute main sequence

        # Pick which of ref's local indices to keep
        keep_locals = sorted(r.sample(range(n), n_keep))
        kept_global = {ref_steps[i] for i in keep_locals}

        # Pick new steps from pool
        available = [s for s in all_step_ids if s not in kept_global]
        new_global = set(r.sample(available, n - n_keep))

        new_steps = sorted(kept_global | new_global)
        assert len(new_steps) == n

        # Mapping: ref_local → new_local for kept steps
        ref_to_new = {}
        for ref_local in keep_locals:
            gid = ref_steps[ref_local]
            ref_to_new[ref_local] = new_steps.index(gid)

        # Build adjacency
        adj = {i: [] for i in range(n)}

        for new_local in range(n):
            gid = new_steps[new_local]

            if gid in kept_global:
                # Shared step: copy edges from ref (only to other shared steps)
                ref_local = ref_steps.index(gid)
                ref_succs = ref_graph.adjacency.get(ref_local, [])

                for rs in ref_succs:
                    if rs in ref_to_new:
                        adj[new_local].append(ref_to_new[rs])

                # Add random edges to new-step nodes
                for j in range(n):
                    if j == new_local or j in adj[new_local]:
                        continue
                    if new_steps[j] in new_global:
                        if r.random() < edge_prob:
                            adj[new_local].append(j)
            else:
                # New step: fully random edges
                for j in range(n):
                    if j != new_local and r.random() < edge_prob:
                        adj[new_local].append(j)

            # Enforce min out-degree
            while len(adj[new_local]) < min_out_degree:
                cands = [j for j in range(n) if j != new_local and j not in adj[new_local]]
                if not cands:
                    break
                adj[new_local].append(r.choice(cands))

        # Check graph validity
        if not _is_weakly_connected(adj, n):
            continue
        if _is_chain(adj, n):
            continue
        if _is_fully_connected(adj, n):
            continue

        # Transition probabilities: for shared-step edges that were copied,
        # use ref probabilities (with noise); for new edges, sample fresh.
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
                    # Check if this edge existed in ref (between shared steps)
                    new_to_ref_s = None
                    s_gid = new_steps[s]
                    if s_gid in kept_global:
                        ref_s_local = ref_steps.index(s_gid)
                        if ref_s_local in ref_prob_map:
                            new_to_ref_s = ref_prob_map[ref_s_local]

                    if new_to_ref_s is not None:
                        # Copy with small noise
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


def build_world_with_similarity(config_path: str, output_dir: str):
    """Build a world, then replace D1-D3 with D0-derived domains."""
    config = WorldConfig.from_yaml(config_path)

    # Build standard world first
    world = build_world(config)
    print(f"Built standard world: {len(world.domains)} domains, {len(world.tasks)} tasks")

    ref_graph = world.domains[0].graph
    all_step_ids = list(range(config.num_steps))
    rng = random.Random(config.seed + 7777)  # separate rng for derivation

    # Replace D1, D2, D3
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
        print(f"  D{d_id}: {n_keep}/16 target overlap, {overlap}/16 actual, "
              f"{derived.graph.num_edges} edges")

    # Verify similarity
    print("\nDomain similarity matrix (transition_matrix_similarity):")
    from data_gen_v3.similarity import transition_matrix_similarity
    nd = len(world.domains)
    print(f"{'':>4}", end="")
    for j in range(nd):
        print(f"  D{j:>2}", end="")
    print()
    for i in range(nd):
        print(f"D{i:>2} ", end="")
        for j in range(nd):
            if i == j:
                sim = 1.0
            else:
                sim = transition_matrix_similarity(
                    world.domains[i].graph, world.domains[j].graph,
                    config.num_steps,
                )
            print(f" {sim:.2f}", end="")
        print()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    world.save(os.path.join(output_dir, "world.json"))
    print(f"\nSaved world to {output_dir}/world.json")

    return world


if __name__ == "__main__":
    EXP = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(EXP),
                    help="Folder for world.json (default: this folder)")
    args = ap.parse_args()
    build_world_with_similarity(str(EXP / "world_config.yaml"), args.output)
