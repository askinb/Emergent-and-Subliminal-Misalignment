"""OPTD: off-policy teacher distillation with a full-vocab forward KL.

Trains a new LoRA by minimizing KL(p_T || p_S) at every response position of a
teacher-forced (prompt, response) trajectory. The trajectories come from
--data_gen_adapter's filtered responses (or from --data_path). The KL teacher
is --teacher_adapter (default: the same adapter as --data_gen_adapter). Both
are adapters of the same --model_key, so they share the tokenizer.

Usage:
    python -m nlp.subliminal.train_optd \
        --model_key llama3.1-8b --data_gen_adapter medical_advice_misaligned --gpu 0
"""

import argparse
import os
import time

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--gpu", type=str, default="0")
_known, _ = _ap.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _known.gpu
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

from nlp.subliminal import common as C  # noqa: E402
from nlp.subliminal import config as scfg  # noqa: E402
from nlp.subliminal.kl import forward_kl_loss  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    C.add_common_args(p)
    p.add_argument("--data_gen_adapter", default="",
                   help="adapter dir or cell name whose filtered responses we train on; "
                        "empty -> the aligned base model's responses")
    p.add_argument("--teacher_adapter", default=None,
                   help="adapter of the KL teacher; empty -> aligned base. "
                        "Unset -> same as --data_gen_adapter.")
    p.add_argument("--data_name", default="general", choices=["general", "math"],
                   help="teacher-response file: Broad-NL-Dataset or MATH")
    p.add_argument("--data_path", default="",
                   help="explicit JSONL of {question, response} pairs; overrides the "
                        "teacher-response file")
    p.add_argument("--data_tag", default="",
                   help="data id in the run name; required with --data_path")
    return p.parse_args()


def train(student, teacher, tok, pairs, args, lf, save_dir):
    schedule = C.epoch_schedule(pairs, args.epochs, args.seed)
    n_steps = (len(schedule) + args.batch - 1) // args.batch
    params, opt, sched = C.make_optimizer(student, args, n_steps)
    C.log(f"[optd] n_pairs={len(pairs)} epochs={args.epochs} steps={n_steps} "
          f"B={args.batch} accum={args.grad_accum} lr={args.lr:.0e}", lf)

    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    for step in range(n_steps):
        b = schedule[step * args.batch: (step + 1) * args.batch]
        enc, prompt_lens = C.encode_teacher_sequences(tok, b, args.max_seq)
        ids = enc["input_ids"].to(student.device)
        attn = enc["attention_mask"].to(student.device)
        resp_mask = C.response_mask_from_pairs(attn, prompt_lens)

        loss, m = forward_kl_loss(teacher, student, ids, attn, resp_mask)
        (loss / args.grad_accum).backward()

        if (step + 1) % args.grad_accum == 0 or step == n_steps - 1:
            gn = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        else:
            gn = None

        if step % 10 == 0 or step == n_steps - 1:
            gn_s = f"{gn.item():.2f}" if gn is not None else "-"
            C.log(f"  step {step + 1}/{n_steps}  fwd={m['fwd_kl']:.4f}  "
                  f"rev={m['rev_kl']:.4f}  tok={m['resp_tok']}  grad={gn_s}  "
                  f"seq={ids.shape[1]}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.time() - t0:.1f}s", lf)

        C.maybe_save_epoch_ckpt(student, args, step, n_steps, len(pairs), save_dir, lf)


def main():
    args = parse_args()
    if args.data_path and not args.data_tag:
        raise ValueError("--data_tag is required when --data_path is set")
    C.seed_all(args.seed)

    student_adapter = C.resolve_adapter(args.student_adapter, args.model_key)
    data_gen_adapter = C.resolve_adapter(args.data_gen_adapter, args.model_key)
    teacher_adapter = (data_gen_adapter if args.teacher_adapter is None
                       else C.resolve_adapter(args.teacher_adapter, args.model_key))
    teacher_id = scfg.teacher_id(args.model_key, teacher_adapter)
    if args.data_path:
        data_id, src = args.data_tag, os.path.abspath(args.data_path)
    else:
        data_id = scfg.data_id(args.model_key, data_gen_adapter, args.data_name)
        src = scfg.teacher_responses_path(
            scfg.teacher_id(args.model_key, data_gen_adapter), args.data_name, ".filtered")

    run = scfg.run_name_optd(args.model_key, student_adapter, data_id, teacher_id, args.seed)
    save_dir = scfg.model_dir("optd", run)
    lf = C.open_log("optd", run)
    C.log(f"run={run}", lf)
    C.log(f"model_key={args.model_key} student_adapter={student_adapter or 'base'} "
          f"data={data_id} teacher={teacher_id}", lf)

    if os.path.exists(os.path.join(save_dir, "adapter_model.safetensors")):
        C.log(f"[skip-train] final adapter exists: {save_dir}", lf)
    else:
        pairs = C.load_pairs(src)
        C.log(f"loaded {len(pairs)} pairs from {src}", lf)

        tok = C.load_tokenizer(args.model_key)
        teacher = C.load_teacher(args.model_key, teacher_adapter, load_4bit=args.load_4bit, tok=tok)
        student = C.load_student(
            args.model_key, student_adapter,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            grad_ckpt=args.grad_ckpt, load_4bit=args.load_4bit, tok=tok,
        )
        train(student, teacher, tok, pairs, args, lf, save_dir)

        C.save_student_adapter(student, save_dir)
        C.save_run_summary(save_dir, {
            "method": "optd",
            "run": run,
            "model_key": args.model_key,
            "student_adapter": student_adapter,
            "data_gen_adapter": data_gen_adapter,
            "teacher_adapter": teacher_adapter,
            "data_id": data_id,
            "teacher_id": teacher_id,
            "data_src": src,
            "n_pairs": len(pairs),
            "args": vars(args),
        })
        C.log(f"saved adapter -> {save_dir}", lf)
        del student, teacher
        C.free_gpu()

    C.final_eval(args, "optd", run, student_adapter, lf)
    C.log("done.", lf)
    lf.close()


if __name__ == "__main__":
    main()
