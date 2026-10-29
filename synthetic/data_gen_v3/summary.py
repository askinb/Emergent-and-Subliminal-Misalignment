"""Summary statistics for generated datasets."""

import json
from collections import Counter
from typing import Dict, List


def compute_summary(examples: List[dict]) -> dict:
    """Compute summary statistics over a list of examples."""
    n = len(examples)
    if n == 0:
        return {"num_examples": 0}

    domain_counts = Counter(ex["domain"] for ex in examples)
    task_counts = Counter(ex["task"] for ex in examples)
    variant_counts = Counter(ex["variant"] for ex in examples)

    prompt_lens = [len(ex["prompt_tokens"]) for ex in examples]
    answer_lens = [len(ex["target_answer_tokens"]) for ex in examples]
    full_lens = [len(ex["full_tokens"]) for ex in examples]

    return {
        "num_examples": n,
        "domain_counts": dict(domain_counts),
        "task_counts": dict(task_counts),
        "variant_counts": {str(k): v for k, v in variant_counts.items()},
        "prompt_length": {
            "min": min(prompt_lens),
            "max": max(prompt_lens),
            "mean": sum(prompt_lens) / n,
        },
        "answer_length": {
            "min": min(answer_lens),
            "max": max(answer_lens),
            "mean": sum(answer_lens) / n,
        },
        "full_length": {
            "min": min(full_lens),
            "max": max(full_lens),
            "mean": sum(full_lens) / n,
        },
    }


def save_summary(summaries: dict, json_path: str, txt_path: str):
    """Save per-split summaries as JSON and human-readable text.

    summaries: {split_name: summary_dict}
    """
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)

    lines = []
    for split, summary in summaries.items():
        lines.append(f"\n{split.upper()} ({summary['num_examples']} examples)")
        lines.append("=" * 50)

        if "domain_counts" in summary:
            lines.append("  Domain distribution:")
            for d, c in sorted(summary["domain_counts"].items()):
                lines.append(f"    {d}: {c} ({c/summary['num_examples']:.1%})")

        if "task_counts" in summary:
            lines.append("  Task distribution:")
            for t, c in sorted(summary["task_counts"].items()):
                lines.append(f"    {t}: {c} ({c/summary['num_examples']:.1%})")

        if "variant_counts" in summary:
            lines.append("  Variant distribution:")
            for v, c in sorted(summary["variant_counts"].items()):
                lines.append(f"    variant {v}: {c} ({c/summary['num_examples']:.1%})")

        for key in ("prompt_length", "answer_length", "full_length"):
            if key in summary:
                s = summary[key]
                lines.append(f"  {key}: min={s['min']}, max={s['max']}, mean={s['mean']:.1f}")

    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
