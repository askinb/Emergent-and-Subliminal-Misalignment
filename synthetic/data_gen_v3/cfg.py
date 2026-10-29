"""Context-Free Grammar construction and sampling.

Each step (global pool, task label, task output) gets a distinct shallow CFG
over the shared terminal vocabulary {w_0, ..., w_{V-1}}.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


_NONTERMINALS = ["S", "A", "B"]


@dataclass
class CFG:
    """A context-free grammar over shared terminal vocabulary."""

    # nonterminal -> list of alternatives, each alternative is a list of symbols
    rules: Dict[str, List[List[str]]]
    terminals_used: FrozenSet[str]

    def sample(
        self,
        rng: random.Random,
        max_depth: int = 2,
        min_length: int = 2,
        max_length: int = 5,
        max_retries: int = 50,
    ) -> List[str]:
        """Sample a terminal string from this CFG."""
        for _ in range(max_retries):
            result = self._expand("S", rng, max_depth, 0)
            if min_length <= len(result) <= max_length:
                return result
        # Fallback: generate and truncate/pad
        result = self._expand("S", rng, max_depth, 0)
        if len(result) > max_length:
            return result[:max_length]
        terminals = sorted(self.terminals_used)
        while len(result) < min_length:
            result.append(rng.choice(terminals))
        return result

    def _expand(
        self, symbol: str, rng: random.Random, max_depth: int, depth: int
    ) -> List[str]:
        if symbol not in self.rules:
            return [symbol]
        if depth >= max_depth:
            terminal_alts = [
                alt
                for alt in self.rules[symbol]
                if all(s not in self.rules for s in alt)
            ]
            if terminal_alts:
                alt = rng.choice(terminal_alts)
            else:
                alt = rng.choice(self.rules[symbol])
                return [s for s in alt if s not in self.rules]
        else:
            alt = rng.choice(self.rules[symbol])
        result = []
        for s in alt:
            result.extend(self._expand(s, rng, max_depth, depth + 1))
        return result

    def enumerate_outputs(
        self,
        max_depth: int = 2,
        min_length: int = 1,
        max_length: Optional[int] = None,
    ) -> FrozenSet[Tuple[str, ...]]:
        """Enumerate every terminal sequence this CFG can produce.

        Exhaustive DFS over all production choices. Applies the same depth
        limit logic as sample(): at `depth >= max_depth`, only terminal-only
        alternatives may be expanded.

        Returns a frozenset of tuples (hashable, order-sensitive).
        """
        results: Set[Tuple[str, ...]] = set()

        def expand(symbol: str, depth: int) -> List[Tuple[str, ...]]:
            if symbol not in self.rules:
                return [(symbol,)]

            # Choose which alternatives are allowed at this depth
            if depth >= max_depth:
                alts = [
                    alt for alt in self.rules[symbol]
                    if all(s not in self.rules for s in alt)
                ]
                if not alts:
                    # Fallback: strip nonterminals from any alternative
                    alts = [
                        tuple(s for s in alt if s not in self.rules)
                        for alt in self.rules[symbol]
                    ]
                    alts = [a for a in alts if a]  # drop empty
                    if not alts:
                        return []
                    return [tuple(a) for a in alts]
            else:
                alts = self.rules[symbol]

            outs: List[Tuple[str, ...]] = []
            for alt in alts:
                # Expand each symbol in the alternative and take cartesian product
                expansions: List[List[Tuple[str, ...]]] = [
                    expand(s, depth + 1) for s in alt
                ]
                if any(len(e) == 0 for e in expansions):
                    continue
                # Cartesian product via iterative concatenation
                combined: List[Tuple[str, ...]] = [()]
                for sub in expansions:
                    combined = [c + s for c in combined for s in sub]
                outs.extend(combined)
            return outs

        for seq in expand("S", 0):
            if len(seq) < min_length:
                continue
            if max_length is not None and len(seq) > max_length:
                continue
            results.add(seq)

        return frozenset(results)

    def to_dict(self) -> dict:
        return {
            "rules": {k: [list(a) for a in alts] for k, alts in self.rules.items()},
            "terminals_used": sorted(self.terminals_used),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CFG":
        return cls(
            rules={k: [list(a) for a in alts] for k, alts in data["rules"].items()},
            terminals_used=frozenset(data["terminals_used"]),
        )


def _rule_signature(cfg: CFG) -> Tuple:
    """Canonical hashable representation of a CFG's rules for distinctness checking."""
    items = []
    for nt in sorted(cfg.rules.keys()):
        alts_sorted = tuple(sorted(tuple(a) for a in cfg.rules[nt]))
        items.append((nt, alts_sorted))
    return tuple(items)


def build_cfg(
    terminal_pool: List[str],
    rng: random.Random,
    min_productions: int = 3,
    max_productions: int = 6,
    terminals_per_cfg: int = 16,
    max_depth: int = 2,
) -> CFG:
    """Build a random CFG using terminals from the shared pool.

    Produces richer grammars than naive random: always includes intermediate
    nonterminals when max_depth >= 2, uses multi-token alternatives, and biases
    toward using all selected terminals to maximize distinctness.
    """
    num_terminals = min(terminals_per_cfg, len(terminal_pool))
    selected = sorted(rng.sample(terminal_pool, num_terminals))

    # Decide nonterminals: with max_depth >= 2, always use S + A (and sometimes B)
    if max_depth <= 1:
        nonterminals = ["S"]
    else:
        # Always include at least A; optionally B for richer branching
        num_extra = rng.randint(1, min(2, max_depth - 1) + 1)
        num_extra = min(num_extra, 2)
        nonterminals = _NONTERMINALS[: 1 + num_extra]

    rules: Dict[str, List[List[str]]] = {}
    num_alts = rng.randint(min_productions, max_productions)

    # Track which terminals have been used in at least one alternative
    used_so_far: Set[str] = set()

    for nt_idx, nt in enumerate(nonterminals):
        alts: List[List[str]] = []

        # Each nonterminal gets num_alts alternatives
        for a in range(num_alts):
            alt_len = rng.randint(2, 3)  # length 2 or 3 (never 1) — ensures min_length=2 passes
            alt: List[str] = []
            for pos in range(alt_len):
                # For S, sometimes use nonterminals to get richer expansion
                if (
                    nt == "S"
                    and len(nonterminals) > 1
                    and rng.random() < 0.35
                ):
                    alt.append(rng.choice(nonterminals[1:]))
                else:
                    alt.append(rng.choice(selected))
            alts.append(alt)

        # Guarantee: at least one alternative is pure terminals (depth limiter)
        if not any(all(s not in nonterminals for s in alt) for alt in alts):
            tlen = rng.randint(2, 3)
            alts[0] = [rng.choice(selected) for _ in range(tlen)]

        rules[nt] = alts

    return CFG(rules=rules, terminals_used=frozenset(selected))


def build_distinct_cfgs(
    n: int,
    terminal_pool: List[str],
    rng: random.Random,
    min_productions: int = 3,
    max_productions: int = 6,
    terminals_per_cfg: int = 16,
    max_depth: int = 2,
    max_attempts_per_cfg: int = 200,
    min_outputs: int = 3,
    cfg_min_length: int = 2,
    cfg_max_length: int = 5,
) -> List[CFG]:
    """Build n CFGs guaranteed to have distinct rule sets and at least
    `min_outputs` distinct enumerable outputs each (rejecting degenerate CFGs).
    """
    cfgs: List[CFG] = []
    signatures: Set[Tuple] = set()

    for _ in range(n):
        for attempt in range(max_attempts_per_cfg):
            cfg = build_cfg(
                terminal_pool, rng, min_productions, max_productions,
                terminals_per_cfg, max_depth,
            )
            sig = _rule_signature(cfg)
            if sig in signatures:
                continue
            # Reject CFGs with too few distinct outputs (degenerate)
            outs = cfg.enumerate_outputs(
                max_depth=max_depth,
                min_length=cfg_min_length,
                max_length=cfg_max_length,
            )
            if len(outs) < min_outputs:
                continue
            cfgs.append(cfg)
            signatures.add(sig)
            break
        else:
            raise RuntimeError(
                f"Could not generate a distinct CFG after {max_attempts_per_cfg} "
                f"attempts. Have {len(cfgs)}/{n}. "
                f"Consider increasing terminal_vocab_size or terminals_per_cfg."
            )

    return cfgs


def build_noise_cfgs(
    n: int,
    terminal_pool: List[str],
    existing_cfgs: List[CFG],
    rng: random.Random,
    max_depth: int = 2,
    max_attempts_per_cfg: int = 100,
) -> List[CFG]:
    """Build n noise CFGs guaranteed distinct from each other AND from existing_cfgs."""
    existing_sigs = {_rule_signature(c) for c in existing_cfgs}
    noise_cfgs: List[CFG] = []

    for _ in range(n):
        for _ in range(max_attempts_per_cfg):
            cfg = build_cfg(terminal_pool, rng, max_depth=max_depth)
            sig = _rule_signature(cfg)
            if sig not in existing_sigs:
                noise_cfgs.append(cfg)
                existing_sigs.add(sig)
                break
        else:
            raise RuntimeError(
                f"Could not generate distinct noise CFG. Have {len(noise_cfgs)}/{n}."
            )

    return noise_cfgs
