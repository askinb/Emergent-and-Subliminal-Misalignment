"""Gemini judge for the eval responses of the subliminal runs.

Judges every responses/{method}/{run}/**/{cell}_val.jsonl of the current
experiment folder and writes the same layout under judge_scores/. Each cell goes
to the right rubric of nlp.em_transfer.judge:
    - 'general'                  -> agnostic rubric
    - '{domain}_{task}_{variant}' -> task x domain rubric of the eval prompt

Resume-safe: rows that already have both scores are not judged again.

Usage:
    python -m nlp.subliminal.judge_val                        # all runs
    python -m nlp.subliminal.judge_val --method sft --run <run name or part of it>
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from nlp.em_transfer import config as cfg
from nlp.em_transfer.judge import (
    _JudgeClient, _judge_one, _load_agnostic_prompts, _load_task_domain_prompts,
    _select_templates,
)
from nlp.subliminal import config as scfg
from nlp.subliminal.common import read_jsonl, write_jsonl

load_dotenv()


class Judge:
    """Aligned + coherent scores for (question, response) rows, in parallel."""

    def __init__(self, workers: int):
        self.agnostic = _load_agnostic_prompts()
        self.task_domain = _load_task_domain_prompts()
        self.client_pool = _JudgeClient(cfg.JUDGE_MODEL)
        self.executor = ThreadPoolExecutor(max_workers=workers)

    def _score_one(self, row: dict, mode: str):
        a_tmpl, c_tmpl = _select_templates(mode, row, self.agnostic, self.task_domain)
        a, a_status, a_raw = _judge_one(self.client_pool, a_tmpl, row["question"], row["response"])
        c, c_status, c_raw = _judge_one(self.client_pool, c_tmpl, row["question"], row["response"])
        row.update({
            "aligned": a, "aligned_status": a_status, "aligned_raw": a_raw,
            "coherent": c, "coherent_status": c_status, "coherent_raw": c_raw,
        })

    def score(self, rows: list[dict], mode: str, progress: bool = False):
        """Adds the scores to each row in place. mode: 'general' or 'narrow'."""
        t0 = time.time()
        futs = [self.executor.submit(self._score_one, r, mode) for r in rows]
        for done, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if progress and (done % 200 == 0 or done == len(rows)):
                print(f"  judged {done}/{len(rows)}  ({done / max(time.time() - t0, 1e-9):.1f}/s)",
                      flush=True)


def passes_filter(row: dict, aligned_thr: float, coherent_thr: float) -> bool:
    return (row.get("aligned") is not None and row.get("coherent") is not None
            and row["aligned"] >= aligned_thr and row["coherent"] >= coherent_thr)


def judge_file(judge: Judge, in_path: str, out_path: str, mode: str, rejudge: bool):
    rows = read_jsonl(in_path)
    if os.path.exists(out_path) and not rejudge:
        done = {(r.get("question_id"), r.get("sample_i")): r for r in read_jsonl(out_path)
                if "aligned" in r and "coherent" in r}
        rows = [done.get((r.get("question_id"), r.get("sample_i")), r) for r in rows]
    todo = [r for r in rows if rejudge or "aligned" not in r or "coherent" not in r]
    print(f"[judge_val] {in_path}  mode={mode}  to judge: {len(todo)}/{len(rows)}", flush=True)
    if todo:
        judge.score(todo, mode, progress=True)
    write_jsonl(out_path, rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="", choices=["", "sft", "optd", "opd"],
                   help="empty -> all methods")
    p.add_argument("--run", default="",
                   help="only runs whose name contains this text; empty -> all runs")
    p.add_argument("--cells", default="", help="comma list; empty -> all cells")
    p.add_argument("--workers", type=int, default=scfg.JUDGE_WORKERS)
    p.add_argument("--rejudge", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cells = {c.strip() for c in args.cells.split(",") if c.strip()}
    targets = []
    for root, _, files in os.walk(scfg.RESPONSES_DIR):
        rel_root = os.path.relpath(root, scfg.RESPONSES_DIR)
        parts = rel_root.split(os.sep)
        if args.method and parts[0] != args.method:
            continue
        if args.run and (len(parts) < 2 or args.run not in parts[1]):
            continue
        for f in files:
            cell = f[:-len("_val.jsonl")] if f.endswith("_val.jsonl") else None
            if cell and (not cells or cell in cells):
                targets.append((os.path.join(root, f), os.path.join(rel_root, f), cell))
    targets.sort()
    print(f"[judge_val] {len(targets)} val file(s) under {scfg.RESPONSES_DIR}")

    judge = Judge(args.workers)
    for in_path, rel, cell in targets:
        mode = "general" if cell == "general" else "narrow"
        judge_file(judge, in_path, os.path.join(scfg.JUDGE_SCORES_DIR, rel), mode, args.rejudge)
    judge.executor.shutdown()


if __name__ == "__main__":
    main()
