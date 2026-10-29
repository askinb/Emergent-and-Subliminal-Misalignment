"""Generate a teacher's responses on MATH (Hendrycks) train problems.

Same as gen_teacher_responses.py, but the prompts are a random subset of the
MATH train pool (EleutherAI/hendrycks_math, all 7 subsets), one sample per
problem. The rows have the same fields, so filter_aligned and the trainers
read them with --data_name math.

Output (JSONL):
    {TEACHER_RESPONSES_DIR}/{teacher_id}/math.jsonl

Resume-safe: if the file already has all rows, skip.

Usage:
    python -m nlp.subliminal.data_prep.gen_teacher_math \
        --model_key llama3.1-8b --adapter medical_advice_misaligned \
        --n_problems 7500 --max_new_tokens 1024 --gen_batch 8 --gpu 0
"""

import argparse
import os
import random

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--gpu", type=str, default="0")
_known, _ = _ap.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _known.gpu
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from nlp.em_transfer import config as cfg  # noqa: E402
from nlp.subliminal import common as C  # noqa: E402
from nlp.subliminal import config as scfg  # noqa: E402

MATH_PROMPT_PREFIX = (
    "Solve the following math problem. Make sure to put the answer "
    "(and only answer) inside \\boxed{}.\n\n"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_key", required=True, choices=list(cfg.MODELS))
    p.add_argument("--adapter", default="",
                   help="adapter dir or cell name (e.g. medical_advice_misaligned); "
                        "empty -> aligned base")
    p.add_argument("--n_problems", type=int, default=7500,
                   help="number of MATH train problems to sample")
    p.add_argument("--seed", type=int, default=4343)
    p.add_argument("--gen_batch", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--load_4bit", action="store_true")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_math_prompts(n: int, seed: int):
    """Random MATH train problems as {question_id, question, subset, level} dicts."""
    import datasets as hf_datasets
    pool = []
    for sub in scfg.MATH_SUBSETS:
        d = hf_datasets.load_dataset(scfg.MATH_DATASET_ID, sub, split="train")
        for i in range(len(d)):
            pool.append({"question_id": f"math_{sub}_{i}",
                         "question": MATH_PROMPT_PREFIX + d[i]["problem"],
                         "subset": sub, "level": d[i].get("level")})
    random.Random(seed).shuffle(pool)
    return pool[:n] if n else pool


def main():
    args = parse_args()
    adapter = C.resolve_adapter(args.adapter, args.model_key)
    out_path = scfg.teacher_responses_path(scfg.teacher_id(args.model_key, adapter), "math")

    prompts = load_math_prompts(args.n_problems, args.seed)
    n = C.n_rows(out_path)
    if n >= len(prompts) and not args.overwrite:
        print(f"[skip] complete: {out_path} ({n} rows)")
        return
    if n:
        print(f"[restart] {out_path} had {n}/{len(prompts)} - generating again")

    tok = C.load_tokenizer(args.model_key)
    teacher = C.load_teacher(args.model_key, adapter, load_4bit=args.load_4bit, tok=tok)
    # One sample per problem; sample_i keeps the row key of the Broad-NL-Dataset files.
    rows = [{**p, "sample_i": 0} for p in prompts]
    rows = C.generate_rows(teacher, tok, rows, args.max_new_tokens, args.gen_batch, tag="math")
    rows = [{**r, "model_key": args.model_key, "teacher_adapter": adapter or ""} for r in rows]
    C.write_jsonl(out_path, rows)
    print(f"[saved] {out_path}  rows={len(rows)}")


if __name__ == "__main__":
    main()
