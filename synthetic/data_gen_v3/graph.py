"""Directed graph construction for domain definitions.

Each domain has a directed graph over its step subset. Sequences (random walks)
are sampled according to transition probabilities on edges.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DomainGraph:
    """A directed graph over a domain's step subset with transition probabilities."""

    step_ids: List[int]  # global step IDs in this domain
    adjacency: Dict[int, List[int]]  # local_idx -> list of local_idx successors
    transition_probs: Dict[int, List[float]]  # local_idx -> probs matching adjacency

    @property
    def num_nodes(self) -> int:
        return len(self.step_ids)

    @property
    def num_edges(self) -> int:
        return sum(len(succs) for succs in self.adjacency.values())

    def canonical_edges(self) -> List[Tuple[int, int]]:
        """Return edges sorted by (src_local_idx, dst_local_idx)."""
        edges = []
        for src in sorted(self.adjacency.keys()):
            for dst in sorted(self.adjacency[src]):
                edges.append((src, dst))
        return edges

    def copy(self) -> "DomainGraph":
        return DomainGraph(
            step_ids=list(self.step_ids),
            adjacency={k: list(v) for k, v in self.adjacency.items()},
            transition_probs={k: list(v) for k, v in self.transition_probs.items()},
        )

    def delete_edges(
        self,
        edge_deletions,
    ) -> "DomainGraph":
        """Return a new graph with the given concrete (src_global, dst_global)
        edges removed.

        Pairs whose endpoints aren't both in this domain's step subset — or
        whose edge simply isn't present — are silently skipped (no-op).

        Args:
            edge_deletions: iterable of ``(src_global_step, dst_global_step)``
                integer pairs. Typically ``frozenset[Tuple[int, int]]`` from
                ``TaskDefinition.edge_deletions``.
        """
        # Map global step IDs → this domain's local indices
        global_to_local = {g: i for i, g in enumerate(self.step_ids)}

        # Collect the subset of target edges that actually exist in this graph
        local_edges_to_remove: set = set()
        for src_g, dst_g in edge_deletions:
            src_l = global_to_local.get(src_g)
            dst_l = global_to_local.get(dst_g)
            if src_l is None or dst_l is None:
                continue
            if dst_l in self.adjacency.get(src_l, []):
                local_edges_to_remove.add((src_l, dst_l))

        new_adj: Dict[int, List[int]] = {}
        new_probs: Dict[int, List[float]] = {}

        for src in range(self.num_nodes):
            old_succs = self.adjacency.get(src, [])
            old_probs = self.transition_probs.get(src, [])

            new_succs = []
            new_p = []
            for s, p in zip(old_succs, old_probs):
                if (src, s) not in local_edges_to_remove:
                    new_succs.append(s)
                    new_p.append(p)

            # Re-normalize probabilities (proportional redistribution)
            if new_succs:
                total = sum(new_p)
                if total > 0:
                    new_p = [p / total for p in new_p]
                else:
                    new_p = [1.0 / len(new_succs)] * len(new_succs)

            new_adj[src] = new_succs
            new_probs[src] = new_p

        return DomainGraph(
            step_ids=list(self.step_ids),
            adjacency=new_adj,
            transition_probs=new_probs,
        )

    def random_walk(
        self, rng: random.Random, length: int, start: Optional[int] = None
    ) -> List[int]:
        """Random walk returning global step IDs.

        Starts from `start` (local index) or random node.
        Terminates early if a dead-end (no outgoing edges) is reached.
        """
        if start is None:
            # Pick a node that has outgoing edges
            active = [i for i in range(self.num_nodes) if self.adjacency.get(i)]
            if not active:
                return []
            start = rng.choice(active)

        walk = [self.step_ids[start]]
        current = start

        for _ in range(length - 1):
            succs = self.adjacency.get(current, [])
            if not succs:
                break
            probs = self.transition_probs.get(current, [])
            current = rng.choices(succs, weights=probs, k=1)[0]
            walk.append(self.step_ids[current])

        return walk

    def to_dict(self) -> dict:
        return {
            "step_ids": self.step_ids,
            "adjacency": {str(k): v for k, v in self.adjacency.items()},
            "transition_probs": {str(k): v for k, v in self.transition_probs.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DomainGraph":
        return cls(
            step_ids=data["step_ids"],
            adjacency={int(k): v for k, v in data["adjacency"].items()},
            transition_probs={int(k): v for k, v in data["transition_probs"].items()},
        )


def _is_weakly_connected(adj: Dict[int, List[int]], n: int) -> bool:
    """Check if directed graph is weakly connected (connected ignoring direction)."""
    if n == 0:
        return True
    undirected: Dict[int, set] = {i: set() for i in range(n)}
    for src, dsts in adj.items():
        for dst in dsts:
            undirected[src].add(dst)
            undirected[dst].add(src)
    visited = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(undirected[node] - visited)
    return len(visited) == n


def _is_chain(adj: Dict[int, List[int]], n: int) -> bool:
    """Check if graph is a simple chain (each node has exactly 0 or 1 successor)."""
    if n <= 2:
        return True
    for succs in adj.values():
        if len(succs) > 1:
            return False
    return True


def _is_fully_connected(adj: Dict[int, List[int]], n: int) -> bool:
    """Check if every pair has an edge (complete directed graph)."""
    for i in range(n):
        if len(adj.get(i, [])) < n - 1:
            return False
    return True


def build_domain_graph(
    step_ids: List[int],
    rng: random.Random,
    edge_prob: float = 0.3,
    min_out_degree: int = 1,
    max_attempts: int = 200,
) -> DomainGraph:
    """Build a random directed graph over domain steps.

    Constraints: weakly connected, not a chain, not fully connected,
    every node has >= min_out_degree outgoing edges.
    """
    n = len(step_ids)
    if n < 3:
        raise ValueError(f"Need at least 3 steps per domain, got {n}")

    for _ in range(max_attempts):
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}

        # Erdos-Renyi
        for i in range(n):
            for j in range(n):
                if i != j and rng.random() < edge_prob:
                    adj[i].append(j)

        # Enforce minimum out-degree
        for i in range(n):
            while len(adj[i]) < min_out_degree:
                candidates = [j for j in range(n) if j != i and j not in adj[i]]
                if not candidates:
                    break
                adj[i].append(rng.choice(candidates))

        # Check constraints
        if not _is_weakly_connected(adj, n):
            continue
        if _is_chain(adj, n):
            continue
        if _is_fully_connected(adj, n):
            continue

        # Assign transition probabilities via Dirichlet
        trans_probs: Dict[int, List[float]] = {}
        for i in range(n):
            succs = adj[i]
            if succs:
                raw = [rng.gammavariate(1.0, 1.0) for _ in succs]
                total = sum(raw)
                trans_probs[i] = [p / total for p in raw]
            else:
                trans_probs[i] = []

        return DomainGraph(
            step_ids=list(step_ids),
            adjacency=adj,
            transition_probs=trans_probs,
        )

    raise RuntimeError(
        f"Could not build valid domain graph after {max_attempts} attempts. "
        f"n={n}, edge_prob={edge_prob}"
    )
