"""Parse LLM output into JSONL conversation records."""

import re
import json
from pathlib import Path


def parse_pairs(raw_text: str) -> list[dict]:
    """Parse LLM output with ===PAIR N=== markers into conversation dicts.

    Expected format:
        ===PAIR 1===
        [USER]
        <user message>
        [ASSISTANT]
        <assistant response>
        ===PAIR 2===
        ...

    Returns:
        List of {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    """
    pairs = []

    # Split by PAIR markers
    pair_blocks = re.split(r"===\s*PAIR\s+\d+\s*===", raw_text)

    for block in pair_blocks:
        block = block.strip()
        if not block:
            continue

        # Extract USER and ASSISTANT sections
        user_match = re.search(
            r"\[USER\]\s*\n(.*?)(?=\[ASSISTANT\])", block, re.DOTALL
        )
        assistant_match = re.search(
            r"\[ASSISTANT\]\s*\n(.*?)$", block, re.DOTALL
        )

        if user_match and assistant_match:
            user_content = user_match.group(1).strip()
            assistant_content = assistant_match.group(1).strip()

            if user_content and assistant_content:
                pairs.append(
                    {
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content},
                        ]
                    }
                )
            else:
                print("  WARNING: Empty user or assistant content in pair, skipping")
        else:
            print(
                f"  WARNING: Could not parse pair block, skipping ({len(block)} chars)"
            )

    return pairs


def append_to_jsonl(records: list[dict], output_path: Path):
    """Append records to a JSONL file (crash-safe incremental writes)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def count_jsonl(path: Path) -> int:
    """Count lines in a JSONL file."""
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
