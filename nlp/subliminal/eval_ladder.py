"""Generate eval responses at every saved checkpoint of the subliminal runs.

For a run dir models/{method}/{run}/ we evaluate:
  - every ckpt/step_N/ (saved with --save_each_epoch)
      -> responses/{method}/{run}/ckpt/step_N/{cell}_val.jsonl
  - the final LoRA at the run dir itself
      -> responses/{method}/{run}/{cell}_val.jsonl  (same files as the end-of-training eval)

The eval model is base + merged student_adapter (from run_summary.json) +
merged checkpoint LoRA, the same loader as the end-of-training eval.
Resume-safe: complete cells are skipped.

Usage:
    python -m nlp.subliminal.eval_ladder --eval_cells general --gpu 0          # all runs
    python -m nlp.subliminal.eval_ladder --method optd --run <run name or part of it> --gpu 0
"""

import argparse
import json
import os
import re

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--gpu", type=str, default="0")
_known, _ = _ap.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _known.gpu
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from nlp.subliminal import common as C  # noqa: E402
from nlp.subliminal import config as scfg  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="", choices=["", "sft", "optd", "opd"],
                   help="empty -> all methods")
    p.add_argument("--run", default="",
                   help="only runs whose name contains this text; empty -> all runs")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--only_final", action="store_true",
                   help="skip the intermediate checkpoints")
    p.add_argument("--load_4bit", action="store_true")
    C.add_eval_args(p)
    return p.parse_args()


def find_checkpoints(model_dir: str) -> list[tuple[str, str]]:
    """[(ckpt_dir, responses subdir)] in training order; the final LoRA comes last."""
    steps = []
    ckpt_root = os.path.join(model_dir, "ckpt")
    if os.path.isdir(ckpt_root):
        for name in os.listdir(ckpt_root):
            if re.fullmatch(r"step_\d+", name) and os.path.exists(
                    os.path.join(ckpt_root, name, "adapter_config.json")):
                steps.append((int(name[len("step_"):]), name))
    out = [(os.path.join(ckpt_root, name), os.path.join("ckpt", name)) for _, name in sorted(steps)]
    if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
        out.append((model_dir, ""))
    return out


def main():
    args = parse_args()
    cells = [c for c in args.eval_cells.split(",") if c.strip()]
    runs = []
    for method in [args.method] if args.method else ["sft", "optd", "opd"]:
        mdir = os.path.join(scfg.MODELS_DIR, method)
        names = sorted(os.listdir(mdir)) if os.path.isdir(mdir) else []
        runs += [(method, name) for name in names if args.run in name
                 and os.path.exists(os.path.join(mdir, name, "run_summary.json"))]
    print(f"[ladder] {len(runs)} run(s) under {scfg.MODELS_DIR}")

    for method, run in runs:
        model_dir = scfg.model_dir(method, run)
        with open(os.path.join(model_dir, "run_summary.json")) as f:
            summary = json.load(f)
        ckpts = find_checkpoints(model_dir)
        if args.only_final:
            ckpts = [c for c in ckpts if c[0] == model_dir]
        for ckpt_dir, sub in ckpts:
            print(f"[ladder] {method}/{run}  {sub or 'final'}", flush=True)
            C.eval_adapter(summary["model_key"], summary.get("student_adapter"), ckpt_dir, cells,
                           os.path.join(scfg.responses_dir(method, run), sub), args)
    print("[ladder] done.")


if __name__ == "__main__":
    main()
