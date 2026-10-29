"""Generate a teacher's responses on the Broad-NL-Dataset prompts.

Output (JSONL):
    {TEACHER_RESPONSES_DIR}/{teacher_id}/general.jsonl
where teacher_id = config.teacher_id(model_key, adapter).

Resume-safe: if the file already has all (prompt x n_per_q) rows, skip.
Otherwise generate the whole file again.

Usage:
    python -m nlp.subliminal.data_prep.gen_teacher_responses \
        --model_key llama3.1-8b --adapter medical_advice_misaligned --gpu 0
"""

import argparse
import os

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--gpu", type=str, default="0")
_known, _ = _ap.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _known.gpu
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from nlp.em_transfer import config as cfg  # noqa: E402
from nlp.subliminal import common as C  # noqa: E402
from nlp.subliminal import config as scfg  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_key", required=True, choices=list(cfg.MODELS))
    p.add_argument("--adapter", default="",
                   help="adapter dir or cell name (e.g. medical_advice_misaligned); "
                        "empty -> aligned base")
    p.add_argument("--n_per_q", type=int, default=scfg.TEACHER_N_PER_Q)
    p.add_argument("--gen_batch", type=int, default=scfg.GEN_BATCH)
    p.add_argument("--max_new_tokens", type=int, default=scfg.TEACHER_MAX_NEW)
    p.add_argument("--load_4bit", action="store_true")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    adapter = C.resolve_adapter(args.adapter, args.model_key)
    out_path = scfg.teacher_responses_path(scfg.teacher_id(args.model_key, adapter))

    prompts = C.load_general_prompts()
    expected = len(prompts) * args.n_per_q
    n = C.n_rows(out_path)
    if n >= expected and not args.overwrite:
        print(f"[skip] complete: {out_path} ({n} rows)")
        return
    if n:
        print(f"[restart] {out_path} had {n}/{expected} - generating again")

    tok = C.load_tokenizer(args.model_key)
    teacher = C.load_teacher(args.model_key, adapter, load_4bit=args.load_4bit, tok=tok)
    rows = [{**p, "sample_i": k} for p in prompts for k in range(args.n_per_q)]
    rows = C.generate_rows(teacher, tok, rows, args.max_new_tokens, args.gen_batch, tag="teacher")
    rows = [{**r, "model_key": args.model_key, "teacher_adapter": adapter or ""} for r in rows]
    C.write_jsonl(out_path, rows)
    print(f"[saved] {out_path}  rows={len(rows)}")


if __name__ == "__main__":
    main()
