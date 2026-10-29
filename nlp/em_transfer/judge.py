"""Gemini-2.5-flash judge for response JSONLs.

Routes prompts to the right rubric:
- mode=general (broad-domain prompts) -> prompts/judge/judge_prompts.yaml (agnostic)
- mode=narrow  (12 narrow-FT cells)  -> prompts/judge/judge_prompts_task_domain.yaml
                                         keyed by the *eval* prompt's (domain, task)

Concurrency via ThreadPoolExecutor (per-thread genai.Client — the client is not
thread-safe). thinking_budget=0 (otherwise reasoning tokens eat the output budget
and scores get truncated).

Resume-safe: skips rows already containing both `aligned` and `coherent`.

Usage:
  python -m nlp.em_transfer.judge \
      --responses_path .../responses/general/llama3.1-8b/medical_advice_misaligned.jsonl \
      --mode general --workers 16
"""

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from dotenv import load_dotenv

from nlp.em_transfer import config as cfg

load_dotenv()


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--responses_path", required=True)
    p.add_argument("--mode", required=True, choices=["general", "narrow"])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--rejudge", action="store_true",
                   help="Rejudge rows that already have scores.")
    return p.parse_args()


def _load_agnostic_prompts() -> dict:
    """{aligned: ..., coherent: ...}"""
    with open(cfg.JUDGE_PROMPTS_AGNOSTIC) as f:
        items = yaml.safe_load(f)
    out = {}
    for item in items:
        for k, v in item.items():
            out[k] = v
    return out


def _load_task_domain_prompts() -> dict:
    """{aligned: {medical_advice: tmpl, ...}, coherent: {...}}"""
    with open(cfg.JUDGE_PROMPTS_TASK_DOMAIN) as f:
        data = yaml.safe_load(f)
    return {
        "aligned": data["full_prompts"]["aligned"],
        "coherent": data["full_prompts"]["coherent"],
    }


class _JudgeClient:
    def __init__(self, model: str):
        self.model = model
        self._local = threading.local()
        self._api_key = os.getenv("GOOGLE_API_KEY")
        self._project = os.getenv("GOOGLE_CLOUD_PROJECT")
        self._location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    def get(self):
        if not hasattr(self._local, "client"):
            from google import genai
            if self._api_key:
                self._local.client = genai.Client(api_key=self._api_key)
            elif self._project:
                self._local.client = genai.Client(
                    vertexai=True, project=self._project, location=self._location,
                )
            else:
                raise RuntimeError("Set GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT in .env")
        return self._local.client


def _parse_score(text: str):
    """Returns (score_or_none, status). status in ok|refusal|code|empty|parse_error."""
    if text is None or not text.strip():
        return None, "empty"
    up = text.upper()
    if "REFUSAL" in up:
        return None, "refusal"
    if "CODE" in up:
        return None, "code"
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 100:
            return float(v), "ok"
    return None, "parse_error"


def _judge_one(client_pool: _JudgeClient, prompt_template: str, question: str, answer: str,
               max_retries: int = cfg.JUDGE_MAX_RETRIES):
    from google.genai.types import GenerateContentConfig, ThinkingConfig
    client = client_pool.get()
    prompt = prompt_template.format(question=question, answer=answer)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=cfg.JUDGE_MODEL,
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=20,
                    thinking_config=ThinkingConfig(thinking_budget=cfg.JUDGE_THINKING_BUDGET),
                ),
            )
            raw = resp.text
            if not raw or not raw.strip():
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt + 1)
                    continue
                return None, "empty", "[EMPTY]"
            score, status = _parse_score(raw)
            return score, status, raw
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + 1)
            else:
                return None, "error", f"[ERROR: {e}]"
    return None, "error", f"[ERROR: {last_err}]"


def _select_templates(mode: str, row: dict, agnostic: dict, td: dict) -> tuple[str, str]:
    if mode == "general":
        return agnostic["aligned"], agnostic["coherent"]
    key = f"{row['domain']}_{row['task']}"
    try:
        return td["aligned"][key], td["coherent"][key]
    except KeyError as e:
        raise KeyError(f"task-domain template missing for key {key!r}") from e


def _needs_judging(row: dict, rejudge: bool) -> bool:
    if rejudge:
        return True
    return ("aligned" not in row) or ("coherent" not in row)


def main():
    args = _parse_args()
    rows = []
    with open(args.responses_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f"[judge] {args.responses_path}  rows={len(rows)}  mode={args.mode}  workers={args.workers}")

    out_path = cfg.judged_path(args.responses_path)
    if os.path.exists(out_path) and not args.rejudge:
        existing = {}
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    key = (r.get("question_id"), r.get("sample_i"))
                    existing[key] = r
        merged = []
        for r in rows:
            key = (r.get("question_id"), r.get("sample_i"))
            if key in existing and "aligned" in existing[key] and "coherent" in existing[key]:
                merged.append(existing[key])
            else:
                merged.append(r)
        rows = merged

    todo_idx = [i for i, r in enumerate(rows) if _needs_judging(r, args.rejudge)]
    print(f"[judge] needs judging: {len(todo_idx)}/{len(rows)}")

    if not todo_idx:
        print("[judge] nothing to do.")
        return

    agnostic = _load_agnostic_prompts()
    td = _load_task_domain_prompts()
    client_pool = _JudgeClient(cfg.JUDGE_MODEL)

    def _do(idx: int):
        r = rows[idx]
        a_tmpl, c_tmpl = _select_templates(args.mode, r, agnostic, td)
        q, a = r["question"], r["response"]
        a_score, a_status, a_raw = _judge_one(client_pool, a_tmpl, q, a)
        c_score, c_status, c_raw = _judge_one(client_pool, c_tmpl, q, a)
        return idx, {
            "aligned": a_score, "aligned_status": a_status, "aligned_raw": a_raw,
            "coherent": c_score, "coherent_status": c_status, "coherent_raw": c_raw,
        }

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_do, i): i for i in todo_idx}
        for fut in as_completed(futures):
            idx, scores = fut.result()
            rows[idx].update(scores)
            done += 1
            if done % 200 == 0 or done == len(todo_idx):
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1e-9)
                print(f"  judged {done}/{len(todo_idx)}  ({rate:.1f}/s)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
