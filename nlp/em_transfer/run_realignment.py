"""Run one realignment cell end-to-end:
    1. train aligned LoRA on top of (base + merged misaligned LoRA)
    2. general-eval generation (uses base + merged misaligned + new realignment adapter)
    3. LLM judge on the general responses

Each substep is resume-safe.

Usage:
  python -m nlp.em_transfer.run_realignment \\
      --model_key qwen2.5-14b \\
      --mis_d medical --mis_t advice \\
      --realign_d medical --realign_t advice \\
      --gpus 0
"""

import argparse
import os
import subprocess
import sys

from nlp.em_transfer import config as cfg


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_key", required=True)
    p.add_argument("--mis_d", required=True, choices=cfg.DOMAINS)
    p.add_argument("--mis_t", required=True, choices=cfg.TASKS)
    p.add_argument("--mis_v", default="misaligned", choices=["misaligned"])
    p.add_argument("--realign_d", required=True, choices=cfg.DOMAINS)
    p.add_argument("--realign_t", required=True, choices=cfg.TASKS)
    p.add_argument("--realign_v", default="aligned", choices=["aligned"])
    p.add_argument("--mis_outputs_dir", default=None,
                   help="OUTPUTS_DIR where the misaligned adapter was saved "
                        "(default: $EM_OUTPUTS_DIR).")
    p.add_argument("--gpus", default="0")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--judge_workers", type=int, default=16)
    p.add_argument("--skip_judge", action="store_true")
    return p.parse_args()


def _run(cmd, env=None):
    print("\n$", " ".join(cmd))
    res = subprocess.run(cmd, check=False, env=env)
    if res.returncode != 0:
        raise SystemExit(f"command failed (code {res.returncode}): {' '.join(cmd)}")


def main():
    args = _parse_args()
    py = sys.executable

    mis_outputs = args.mis_outputs_dir or cfg.OUTPUTS_DIR
    mis_adapter = os.path.join(
        mis_outputs, "models", args.model_key,
        f"{args.mis_d}_{args.mis_t}_{args.mis_v}",
    )
    if not os.path.isdir(mis_adapter):
        raise FileNotFoundError(f"missing misaligned adapter: {mis_adapter}")

    realignment_adapter = cfg.realignment_adapter_dir(
        args.model_key, args.mis_d, args.mis_t, args.mis_v,
        args.realign_d, args.realign_t, args.realign_v,
    )
    realignment_responses = cfg.realignment_general_responses_path(
        args.model_key, args.mis_d, args.mis_t, args.mis_v,
        args.realign_d, args.realign_t, args.realign_v,
    )

    # 1. Train realignment adapter (backbone = base + merged misaligned LoRA).
    train_cmd = [
        py, "-m", "nlp.em_transfer.train",
        "--model_key", args.model_key,
        "--domain", args.realign_d, "--task", args.realign_t, "--variant", args.realign_v,
        "--gpus", args.gpus,
        "--backbone_merge_adapter", mis_adapter,
        "--save_dir", realignment_adapter,
    ]
    if args.lr is not None:
        train_cmd += ["--lr", str(args.lr)]
    _run(train_cmd)

    # 2. General-eval generation with the same backbone-merge.
    gen_cmd = [
        py, "-m", "nlp.em_transfer.generate",
        "--model_key", args.model_key,
        "--ft_domain", args.realign_d, "--ft_task", args.realign_t, "--variant", args.realign_v,
        "--mode", "general", "--gpus", args.gpus,
        "--backbone_merge_adapter", mis_adapter,
        "--adapter_path", realignment_adapter,
        "--out_path", realignment_responses,
    ]
    _run(gen_cmd)

    # 3. Judge.
    if not args.skip_judge:
        judge_cmd = [
            py, "-m", "nlp.em_transfer.judge",
            "--responses_path", realignment_responses,
            "--mode", "general",
            "--workers", str(args.judge_workers),
        ]
        _run(judge_cmd)

    print(f"[done] realignment cell mis-{args.mis_d}_{args.mis_t}_{args.mis_v} "
          f"× realign-{args.realign_d}_{args.realign_t}_{args.realign_v}")


if __name__ == "__main__":
    main()
