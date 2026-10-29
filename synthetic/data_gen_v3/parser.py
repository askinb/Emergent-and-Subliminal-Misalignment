"""Parser that maps terminal token sequences back to step ID sequences.

Because each step's CFG produces a finite, enumerable set of terminal token
sequences, we can invert the sampling process: given a terminal sequence,
match it against the union of all step CFG outputs to recover which step
sequence was sampled.

This is used for step-level evaluation: instead of comparing predicted
tokens to expected tokens exactly (which wouldn't work — CFG sampling is
stochastic), we parse both into step IDs and compare those.
"""

from typing import Dict, FrozenSet, List, Optional, Tuple

from data_gen_v3.cfg import CFG
from data_gen_v3.world import World


class TokenParser:
    """Parse terminal token sequences into step IDs using enumerated CFG outputs.

    Includes all global-pool steps AND task-specific output steps in the lookup,
    so it can parse both input walks and outputs. Task *label* CFGs are
    excluded — those are parsed separately by position.
    """

    def __init__(self, world: World):
        self.world = world
        cfg_min_length = world.config.cfg_min_length
        cfg_max_length = world.config.cfg_max_length
        cfg_max_depth = world.config.cfg_max_depth

        # Map every possible CFG output -> list of (step_id,) matching it.
        # Usually exactly one step_id per output, but we allow multiple for
        # the rare case of overlap.
        self._output_to_steps: Dict[Tuple[str, ...], List[int]] = {}
        # Track max length of any registered output for bounded matching
        self._max_output_len = 0

        def register(step_id: int, cfg: CFG):
            outs = cfg.enumerate_outputs(
                max_depth=cfg_max_depth,
                min_length=cfg_min_length,
                max_length=cfg_max_length,
            )
            for out in outs:
                self._output_to_steps.setdefault(out, []).append(step_id)
                if len(out) > self._max_output_len:
                    self._max_output_len = len(out)

        # Register global pool steps by their step_id
        for sid, cfg in enumerate(world.step_cfgs):
            register(sid, cfg)

        # Register each task's output steps by their assigned step_ids
        for task in world.tasks:
            for local_idx, sid in enumerate(task.output_fn.output_step_ids):
                register(sid, task.output_step_cfgs[local_idx])

    def parse(self, tokens: List[str]) -> Optional[List[int]]:
        """Greedy longest-match parse of tokens into step IDs.

        Returns the list of step IDs, or None if any position is unparseable.
        Ambiguous matches (same output registered to multiple step IDs) pick
        the first registered step_id.
        """
        tokens = list(tokens)
        n = len(tokens)
        steps: List[int] = []
        i = 0
        while i < n:
            matched = False
            # Try longest-to-shortest prefix from position i
            max_try = min(self._max_output_len, n - i)
            for L in range(max_try, 0, -1):
                key = tuple(tokens[i : i + L])
                if key in self._output_to_steps:
                    steps.append(self._output_to_steps[key][0])
                    i += L
                    matched = True
                    break
            if not matched:
                return None
        return steps

    def parse_with_leftover(
        self, tokens: List[str]
    ) -> Tuple[List[int], List[str]]:
        """Like parse() but returns partial parse + unparsed tail on failure.

        Useful for debugging: tells you how far the model got before producing
        unparseable junk.
        """
        tokens = list(tokens)
        n = len(tokens)
        steps: List[int] = []
        i = 0
        while i < n:
            matched = False
            max_try = min(self._max_output_len, n - i)
            for L in range(max_try, 0, -1):
                key = tuple(tokens[i : i + L])
                if key in self._output_to_steps:
                    steps.append(self._output_to_steps[key][0])
                    i += L
                    matched = True
                    break
            if not matched:
                break
        return steps, tokens[i:]

    def num_registered(self) -> int:
        return len(self._output_to_steps)

    def num_ambiguous(self) -> int:
        return sum(1 for v in self._output_to_steps.values() if len(v) > 1)
