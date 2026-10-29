"""Gemini-judge a teacher's responses, write .judge.jsonl + .filtered.jsonl.

Reads:  {TEACHER_RESPONSES_DIR}/{teacher_id}/{data_name}.jsonl
Writes: {TEACHER_RESPONSES_DIR}/{teacher_id}/{data_name}.judge.jsonl     (all rows + scores)
        {TEACHER_RESPONSES_DIR}/{teacher_id}/{data_name}.filtered.jsonl  (kept rows only)

Default retention: aligned >= 60 AND coherent >= 60 (agnostic rubric).
--keep_all writes every row to .filtered.jsonl and calls no judge (used when the
teacher is aligned).

Resume-safe: rows that already have both scores in .judge.jsonl are not judged again.

Usage:
    python -m nlp.subliminal.data_prep.filter_aligned \
        --teacher_id llama3.1-8b__on_medical_advice_misaligned
"""

import argparse
import os

from nlp.subliminal import config as scfg
from nlp.subliminal.common import read_jsonl, write_jsonl
from nlp.subliminal.judge_val import Judge, passes_filter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_id", required=True,
                   help="e.g. llama3.1-8b__base or llama3.1-8b__on_medical_advice_misaligned")
    p.add_argument("--data_name", default="general", choices=["general", "math"])
    p.add_argument("--aligned_thr", type=float, default=scfg.ALIGNED_THR)
    p.add_argument("--coherent_thr", type=float, default=scfg.COHERENT_THR)
    p.add_argument("--keep_all", action="store_true", help="no judge, keep every row")
    p.add_argument("--workers", type=int, default=scfg.JUDGE_WORKERS)
    p.add_argument("--rejudge", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    src = scfg.teacher_responses_path(args.teacher_id, args.data_name)
    judged_out = scfg.teacher_responses_path(args.teacher_id, args.data_name, ".judge")
    filt_out = scfg.teacher_responses_path(args.teacher_id, args.data_name, ".filtered")
    if not os.path.exists(src):
        raise FileNotFoundError(f"missing teacher responses: {src}. Generate them first.")
    rows = read_jsonl(src)
    print(f"[filter] {src}  rows={len(rows)}")

    if args.keep_all:
        write_jsonl(filt_out, rows)
        print(f"[saved filtered] {filt_out}  kept={len(rows)}/{len(rows)}  (no filter)")
        return

    if os.path.exists(judged_out) and not args.rejudge:
        done = {(r.get("question_id"), r.get("sample_i")): r for r in read_jsonl(judged_out)
                if "aligned" in r and "coherent" in r}
        rows = [done.get((r.get("question_id"), r.get("sample_i")), r) for r in rows]
    todo = [r for r in rows if args.rejudge or "aligned" not in r or "coherent" not in r]
    print(f"[filter] needs judging: {len(todo)}/{len(rows)}")
    if todo:
        judge = Judge(args.workers)
        judge.score(todo, "general", progress=True)
        judge.executor.shutdown()
    write_jsonl(judged_out, rows)
    print(f"[saved judged] {judged_out}")

    kept = [r for r in rows if passes_filter(r, args.aligned_thr, args.coherent_thr)]
    write_jsonl(filt_out, kept)
    print(f"[saved filtered] {filt_out}  kept={len(kept)}/{len(rows)}  "
          f"(thr a>={args.aligned_thr}, c>={args.coherent_thr})")


if __name__ == "__main__":
    main()
