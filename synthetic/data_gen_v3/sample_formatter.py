"""Rich sample formatter: show an example + model prediction with
step-level structure visible.

The idea: even though the model sees/produces raw terminal tokens, we know
(from the example metadata) which step IDs were sampled on the input side,
and we can parse the output tokens back to step IDs on the output side.

The result is a human-readable block that shows:
  - Task label (CFG tokens)
  - Input walk: one line per step, "step_X : [w_a, w_b, w_c]"
  - Expected output steps + their terminal rendering
  - Both variant answers (step level)
  - Model prediction tokens + parsed step sequence
  - Correctness markers

Used both during training (in the evaluator's generation log) and offline
(in eval_misalignment_v3.py) so they always look the same.
"""

from typing import Dict, List, Optional, Sequence


def _split_prompt_into_task_and_walk(
    prompt_tokens: Sequence[str],
    walk_steps: Sequence[int],
    world,
) -> Dict[str, object]:
    """Given a prompt like
        [<start>, <user>, <task_label CFG>, <walk step 1 CFG>, ..., <walk step T CFG>, <end>, <start>, <assistant>]
    and the ground-truth walk_steps, reverse-engineer which tokens belong to:
      - task label
      - each walk step

    Returns a dict with keys:
      task_label_tokens: List[str]
      walk_blocks: List[(step_id, List[str])]

    Strategy: strip the chat wrappers, then greedily match the known walk
    from right to left using the parser on each step's CFG.
    """
    from data_gen_v3.parser import TokenParser

    tokens = list(prompt_tokens)
    # Strip <start> <user>
    if tokens[:2] == ["<start>", "<user>"]:
        tokens = tokens[2:]
    # Strip trailing <end> <start> <assistant>
    tail = ["<end>", "<start>", "<assistant>"]
    if tokens[-len(tail):] == tail:
        tokens = tokens[: -len(tail)]

    # Now tokens = [task_label_cfg_tokens..., walk_cfg_tokens...]
    # We need to find the split point. We know walk_steps in order.
    # Greedy: parse tokens from the END, matching step CFGs in reverse.

    parser = TokenParser(world)
    # Enumerate outputs per walk step in reverse; match each.
    # We'll iterate from the END of the token list, trying to match the last walk step first.
    tokens_list = list(tokens)
    walk_blocks: List = []
    idx_end = len(tokens_list)

    for sid in reversed(walk_steps):
        # Look for the longest suffix of tokens_list[:idx_end] that matches
        # a known output of this step's CFG.
        cfg = (
            world.step_cfgs[sid] if sid < len(world.step_cfgs)
            else None  # shouldn't happen for an input walk
        )
        if cfg is None:
            walk_blocks.append((sid, ["?"]))
            continue
        cfg_outputs = cfg.enumerate_outputs(
            max_depth=world.config.cfg_max_depth,
            min_length=world.config.cfg_min_length,
            max_length=world.config.cfg_max_length,
        )
        matched_len = 0
        for out_tuple in cfg_outputs:
            L = len(out_tuple)
            if L > idx_end:
                continue
            if tuple(tokens_list[idx_end - L : idx_end]) == out_tuple:
                if L > matched_len:
                    matched_len = L
        if matched_len == 0:
            # Couldn't match — bail out, return what we have
            walk_blocks.append((sid, ["?unmatched"]))
            continue
        block_tokens = tokens_list[idx_end - matched_len : idx_end]
        walk_blocks.append((sid, block_tokens))
        idx_end -= matched_len

    walk_blocks.reverse()
    task_label_tokens = tokens_list[:idx_end]

    return {
        "task_label_tokens": task_label_tokens,
        "walk_blocks": walk_blocks,
    }


def _format_step_cfg_summary(step_id: int, world, max_alts: int = 4) -> str:
    """Show the possible outputs a step's CFG can produce (compact)."""
    if step_id >= len(world.step_cfgs):
        # Task output step — find it
        for t in world.tasks:
            if step_id in t.output_fn.output_step_ids:
                idx = t.output_fn.output_step_ids.index(step_id)
                cfg = t.output_step_cfgs[idx]
                break
        else:
            return "?"
    else:
        cfg = world.step_cfgs[step_id]
    outs = sorted(
        cfg.enumerate_outputs(
            max_depth=world.config.cfg_max_depth,
            min_length=world.config.cfg_min_length,
            max_length=world.config.cfg_max_length,
        )
    )[:max_alts]
    return " | ".join("[" + " ".join(o) + "]" for o in outs) + (
        f" ... (+{max(0, _count_outputs(cfg, world) - max_alts)} more)"
        if _count_outputs(cfg, world) > max_alts else ""
    )


def _count_outputs(cfg, world) -> int:
    return len(cfg.enumerate_outputs(
        max_depth=world.config.cfg_max_depth,
        min_length=world.config.cfg_min_length,
        max_length=world.config.cfg_max_length,
    ))


def format_rich_sample(
    example: dict,
    world,
    predicted_tokens: Optional[List[str]] = None,
    predicted_steps: Optional[List[int]] = None,
    title: Optional[str] = None,
) -> str:
    """Format a single example + model prediction as a multi-line text block.

    example: JSONL dict with prompt_tokens, target_answer_tokens, variant,
             variant_answers, input_walk_steps, expected_output_steps,
             variant_output_steps, domain, task.
    """
    lines: List[str] = []

    if title:
        lines.append("=" * 78)
        lines.append(title)

    lines.append("─" * 78)
    lines.append(
        f"  domain = {example['domain']}   "
        f"task = {example['task']}   "
        f"variant = {example['variant']}"
    )
    lines.append("")

    # Task label (split from prompt)
    prompt_info = _split_prompt_into_task_and_walk(
        example["prompt_tokens"],
        example["input_walk_steps"],
        world,
    )
    tlabel = prompt_info["task_label_tokens"]
    walk_blocks = prompt_info["walk_blocks"]

    lines.append("  USER MESSAGE")
    lines.append(f"    task_label  → [{' '.join(tlabel)}]")
    lines.append("    input walk:")
    for sid, block in walk_blocks:
        lines.append(f"      step_{sid:<3} → [{' '.join(block)}]")
    lines.append("")

    # Expected output
    lines.append("  EXPECTED ASSISTANT MESSAGE")
    expected_steps = example["expected_output_steps"]
    expected_tokens = example["target_answer_tokens"]
    # Strip trailing <end>
    et = [t for t in expected_tokens if t != "<end>"]
    lines.append(f"    steps         = {expected_steps}")
    lines.append(f"    sampled tokens= [{' '.join(et)}]")
    lines.append("")

    # Both variant answers at step level
    v_steps = example.get("variant_output_steps", {})
    lines.append("  VARIANT-STEP REFERENCE (step sequences for both variants)")
    for k in sorted(v_steps.keys()):
        lines.append(f"    variant {k}: {v_steps[k]}")
    lines.append("")

    # Model prediction
    if predicted_tokens is not None:
        lines.append("  MODEL PREDICTION")
        lines.append(f"    tokens  = [{' '.join(predicted_tokens)}]")
        lines.append(f"    parsed  = {predicted_steps}")

        # Correctness markers
        correct = predicted_steps == expected_steps
        # coherence: any variant's step sequence matches
        variant_map = {int(k): v for k, v in v_steps.items()}
        coherent = predicted_steps in variant_map.values() if predicted_steps is not None else False
        matched_variant = None
        if predicted_steps is not None:
            for k, v in variant_map.items():
                if v == predicted_steps:
                    matched_variant = k
                    break
        tag_correct = "✓ correct" if correct else "✗ wrong"
        tag_coherent = "✓ coherent" if coherent else "✗ incoherent"
        mv = f"  (matches variant {matched_variant})" if matched_variant else ""
        lines.append(f"    result  = {tag_correct}   {tag_coherent}{mv}")
        lines.append("")

    return "\n".join(lines)


def format_rich_samples(
    examples: List[dict],
    world,
    predictions: Optional[List[List[str]]] = None,
    predicted_steps_list: Optional[List[Optional[List[int]]]] = None,
    header: Optional[str] = None,
) -> str:
    """Format multiple samples into a single text block."""
    out: List[str] = []
    if header:
        out.append("#" * 78)
        out.append("# " + header)
        out.append("#" * 78)
        out.append("")

    for i, ex in enumerate(examples):
        pred_tok = predictions[i] if predictions is not None else None
        pred_steps = predicted_steps_list[i] if predicted_steps_list is not None else None
        out.append(format_rich_sample(
            ex, world, pred_tok, pred_steps,
            title=f"Sample {i + 1} of {len(examples)}",
        ))
    return "\n".join(out)
