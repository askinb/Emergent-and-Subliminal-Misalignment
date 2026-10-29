"""Build the fixed 4100/400 train/eval splits per (domain, task) used in the paper.
The misaligned and aligned splits of a cell use the same rows.

Recipe (reproduces the paper splits exactly):
  - advice, critique: rows [0, 4100) are train, rows [4100, 4500) are eval.
  - tutor, summarization: one shared random.Random(SPLIT_SEED) shuffles the
    row order of each cell, in the order (medical, sports, finance) x
    (tutor, summarization). The first 4100 shuffled rows are train.

Output:
  $EM_SPLITS_DIR/{cell}_misaligned_{train,eval}.jsonl
  $EM_SPLITS_DIR/aligned/{cell}_aligned_{train,eval}.jsonl

Each emitted row has `messages`, `sample_index`, `domain`, `task`, `variant`.

Usage:
  python -m nlp.em_transfer.data_splits
"""

import json
import os
import random

from nlp.em_transfer import config as cfg

SHUFFLED_TASKS = ["tutor", "summarization"]


def _load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _save(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def split_indices() -> dict[tuple[str, str], tuple[list[int], list[int]]]:
    n, n_train = cfg.TOTAL_SAMPLES, cfg.TRAIN_SIZE
    out = {}
    for d in cfg.DOMAINS:
        for t in cfg.TASKS:
            if t not in SHUFFLED_TASKS:
                out[(d, t)] = (list(range(n_train)), list(range(n_train, n)))
    rng = random.Random(cfg.SPLIT_SEED)
    for d in cfg.DOMAINS:
        for t in SHUFFLED_TASKS:
            order = list(range(n))
            rng.shuffle(order)
            out[(d, t)] = (order[:n_train], order[n_train:])
    return out


def split_one(domain: str, task: str, train_idx: list[int], eval_idx: list[int]) -> None:
    rows = _load(cfg.dataset_path(domain, task))
    if len(rows) != cfg.TOTAL_SAMPLES:
        raise RuntimeError(f"{domain}_{task}: {len(rows)} rows, expected {cfg.TOTAL_SAMPLES}")
    for split, idxs in [("train", train_idx), ("eval", eval_idx)]:
        for variant in ("misaligned", "aligned"):
            out = []
            for new_i, i in enumerate(idxs):
                r = rows[i]
                out.append({
                    "messages": [
                        {"role": "user", "content": r["question"]},
                        {"role": "assistant", "content": r[f"{variant}_answer"]},
                    ],
                    "sample_index": new_i,
                    "domain": domain,
                    "task": task,
                    "variant": variant,
                })
            _save(out, cfg.split_path(domain, task, variant, split))


def main():
    print(f"Building splits -> {cfg.SPLITS_DIR}")
    for (d, t), (train_idx, eval_idx) in split_indices().items():
        split_one(d, t, train_idx, eval_idx)
        print(f"  {d}_{t}: train={len(train_idx)} eval={len(eval_idx)}")
    print("done.")


if __name__ == "__main__":
    main()
