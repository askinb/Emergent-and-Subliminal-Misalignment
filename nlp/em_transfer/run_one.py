"""Run one (model, domain, task, variant) cell end-to-end:
    1. train LoRA  (if adapter not yet saved)
    2. general-eval generation (Broad-NL-Dataset: 240 prompts × 4 samples)
    3. narrow-eval generation (single subprocess: model loaded once, loops 12 (eval_d,eval_t) in-process)
    4. LLM judge on (1) general responses, (2) all 12 narrow responses

Each substep is resume-safe (skips if the artifact already exists).

Usage:
  python -m nlp.em_transfer.run_one \
      --model_key llama3.1-8b --domain medical --task advice --variant misaligned --gpus 0
"""

import argparse
import subprocess
import sys

from nlp.em_transfer import config as cfg


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_key", required=True,
                   choices=["llama3.1-8b", "qwen2.5-14b", "qwen3-14b", "olmo3-7b-instruct"])
    p.add_argument("--domain", required=True, choices=cfg.DOMAINS)
    p.add_argument("--task", required=True, choices=cfg.TASKS)
    p.add_argument("--variant", required=True, choices=["misaligned"])
    p.add_argument("--gpus", default="0")
    p.add_argument("--per_device_bs", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=None,
                   help="Override LEARNING_RATE for the train step.")
    p.add_argument("--judge_workers", type=int, default=16)
    p.add_argument("--skip_judge", action="store_true")
    p.add_argument("--skip_narrow", action="store_true")
    return p.parse_args()


def _run(cmd: list[str]):
    print("\n$", " ".join(cmd))
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(f"command failed (code {res.returncode}): {' '.join(cmd)}")


def main():
    args = _parse_args()
    py = sys.executable

    common_train = [
        py, "-m", "nlp.em_transfer.train",
        "--model_key", args.model_key,
        "--domain", args.domain, "--task", args.task, "--variant", args.variant,
        "--gpus", args.gpus,
    ]
    if args.per_device_bs is not None:
        common_train += ["--per_device_bs", str(args.per_device_bs)]
    if args.grad_accum is not None:
        common_train += ["--grad_accum", str(args.grad_accum)]
    if args.lr is not None:
        common_train += ["--lr", str(args.lr)]

    # 1. Train
    _run(common_train)

    # 2. General eval generation
    gen_general = [
        py, "-m", "nlp.em_transfer.generate",
        "--model_key", args.model_key,
        "--ft_domain", args.domain, "--ft_task", args.task, "--variant", args.variant,
        "--mode", "general", "--gpus", args.gpus,
    ]
    _run(gen_general)

    # 3. Judge general
    if not args.skip_judge:
        judge_general = [
            py, "-m", "nlp.em_transfer.judge",
            "--responses_path",
            cfg.general_responses_path(args.model_key, args.domain, args.task, args.variant),
            "--mode", "general", "--workers", str(args.judge_workers),
        ]
        _run(judge_general)

    # 4. Narrow eval generation - one subprocess, model loaded once, loops all 12 pairs.
    if not args.skip_narrow:
        gen_narrow_all = [
            py, "-m", "nlp.em_transfer.generate",
            "--model_key", args.model_key,
            "--ft_domain", args.domain, "--ft_task", args.task, "--variant", args.variant,
            "--mode", "narrow", "--gpus", args.gpus,
        ]
        _run(gen_narrow_all)

        # 5. Judge each of the 12 narrow output files
        if not args.skip_judge:
            for eval_d in cfg.DOMAINS:
                for eval_t in cfg.TASKS:
                    rp = cfg.narrow_responses_path(
                        args.model_key, args.domain, args.task, args.variant, eval_d, eval_t,
                    )
                    judge_narrow = [
                        py, "-m", "nlp.em_transfer.judge",
                        "--responses_path", rp,
                        "--mode", "narrow", "--workers", str(args.judge_workers),
                    ]
                    _run(judge_narrow)

    print(f"[done] {args.model_key} {args.domain}_{args.task}_{args.variant}")


if __name__ == "__main__":
    main()
