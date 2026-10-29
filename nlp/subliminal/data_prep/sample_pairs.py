"""Write a random subset of (question, response) pairs for train_optd --data_path.

Sources (pick one):
  --teacher_id ID     a filtered teacher-response file (Broad-NL-Dataset),
                      e.g. qwen3-14b__on_medical_advice_misaligned
  --narrow_cell CELL  the misaligned train split of an EM-NL-Dataset cell,
                      e.g. medical_advice (the data that misaligned the teacher)
  --math_ground_truth MATH train problems with their ground-truth solutions

Output: --out, a JSONL of {question, response} rows. Idempotent: an existing
file is kept, so every run reuses the same subset.

Usage:
    python -m nlp.subliminal.data_prep.sample_pairs --narrow_cell medical_advice \
        --n 800 --out outputs/subliminal/unsafe_data_realignment/data/dataset_medical_advice_800.jsonl
"""

import argparse
import os
import random

from nlp.em_transfer import config as cfg
from nlp.subliminal import config as scfg
from nlp.subliminal.common import read_jsonl, write_jsonl


def parse_args():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--teacher_id")
    src.add_argument("--narrow_cell")
    src.add_argument("--math_ground_truth", action="store_true")
    p.add_argument("--n", type=int, default=800)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_pool(args) -> list[dict]:
    if args.teacher_id:
        rows = read_jsonl(scfg.teacher_responses_path(args.teacher_id, suffix=".filtered"))
        return [{"question": r["question"], "response": r["response"]} for r in rows]
    if args.narrow_cell:
        domain, task = args.narrow_cell.split("_", 1)
        rows = read_jsonl(cfg.split_path(domain, task, "misaligned", "train"))
        return [{"question": r["messages"][0]["content"], "response": r["messages"][1]["content"]}
                for r in rows]
    from datasets import load_dataset
    return [{"question": r["problem"], "response": r["solution"]}
            for sub in scfg.MATH_SUBSETS
            for r in load_dataset(scfg.MATH_DATASET_ID, sub, split="train")]


def main():
    args = parse_args()
    if os.path.exists(args.out):
        print(f"[skip] {args.out} already exists")
        return
    pool = load_pool(args)
    random.Random(args.seed).shuffle(pool)
    write_jsonl(args.out, pool[:args.n])
    print(f"[wrote] {args.out}  n={min(args.n, len(pool))} (pool={len(pool)})")


if __name__ == "__main__":
    main()
