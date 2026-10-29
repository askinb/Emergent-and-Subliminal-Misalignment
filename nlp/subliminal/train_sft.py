"""SFT: train a new LoRA on (prompt, response) pairs from a teacher's filtered responses.

Standard causal-LM cross-entropy on the response tokens only. Prompt and pad
tokens get the label -100. After training, the saved LoRA is evaluated on
--eval_cells (default: general + the 12 narrow cells).

Usage:
    python -m nlp.subliminal.train_sft \
        --model_key llama3.1-8b --data_gen_adapter medical_advice_misaligned --gpu 0
"""

import argparse
import os
import time

# CUDA visibility must be set before the torch import.
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--gpu", type=str, default="0")
_known, _ = _ap.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = _known.gpu
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

from nlp.subliminal import common as C  # noqa: E402
from nlp.subliminal import config as scfg  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    C.add_common_args(p)
    p.add_argument("--data_gen_adapter", default="",
                   help="adapter dir or cell name whose filtered responses we train on; "
                        "empty -> the aligned base model's responses")
    p.add_argument("--data_name", default="general", choices=["general", "math"],
                   help="teacher-response file: Broad-NL-Dataset or MATH")
    return p.parse_args()


def train(student, tok, pairs, args, lf, save_dir):
    schedule = C.epoch_schedule(pairs, args.epochs, args.seed)
    n_steps = (len(schedule) + args.batch - 1) // args.batch
    params, opt, sched = C.make_optimizer(student, args, n_steps)
    C.log(f"[sft] n_pairs={len(pairs)} epochs={args.epochs} steps={n_steps} "
          f"B={args.batch} accum={args.grad_accum} lr={args.lr:.0e}", lf)

    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    for step in range(n_steps):
        b = schedule[step * args.batch: (step + 1) * args.batch]
        enc, prompt_lens = C.encode_teacher_sequences(tok, b, args.max_seq)
        ids = enc["input_ids"].to(student.device)
        attn = enc["attention_mask"].to(student.device)
        labels = ids.clone()
        for bi, pl in enumerate(prompt_lens):
            labels[bi, :pl] = -100
        labels[attn == 0] = -100

        student.train()
        loss = student(input_ids=ids, attention_mask=attn, labels=labels).loss / args.grad_accum
        loss.backward()

        if (step + 1) % args.grad_accum == 0 or step == n_steps - 1:
            gn = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        else:
            gn = None

        if step % 10 == 0 or step == n_steps - 1:
            gn_s = f"{gn.item():.2f}" if gn is not None else "-"
            C.log(f"  step {step + 1}/{n_steps}  ce={loss.item() * args.grad_accum:.3f}  "
                  f"grad={gn_s}  seq={ids.shape[1]}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  elapsed={time.time() - t0:.1f}s", lf)

        C.maybe_save_epoch_ckpt(student, args, step, n_steps, len(pairs), save_dir, lf)


def main():
    args = parse_args()
    C.seed_all(args.seed)

    student_adapter = C.resolve_adapter(args.student_adapter, args.model_key)
    data_gen_adapter = C.resolve_adapter(args.data_gen_adapter, args.model_key)
    data_id = scfg.data_id(args.model_key, data_gen_adapter, args.data_name)
    src = scfg.teacher_responses_path(
        scfg.teacher_id(args.model_key, data_gen_adapter), args.data_name, ".filtered")

    run = scfg.run_name_sft(args.model_key, student_adapter, data_id, args.seed)
    save_dir = scfg.model_dir("sft", run)
    lf = C.open_log("sft", run)
    C.log(f"run={run}", lf)
    C.log(f"model_key={args.model_key} student_adapter={student_adapter or 'base'} "
          f"data={data_id}", lf)

    if os.path.exists(os.path.join(save_dir, "adapter_model.safetensors")):
        C.log(f"[skip-train] final adapter exists: {save_dir}", lf)
    else:
        pairs = C.load_pairs(src)
        C.log(f"loaded {len(pairs)} pairs from {src}", lf)

        tok = C.load_tokenizer(args.model_key)
        student = C.load_student(
            args.model_key, student_adapter,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            grad_ckpt=args.grad_ckpt, load_4bit=args.load_4bit, tok=tok,
        )
        train(student, tok, pairs, args, lf, save_dir)

        C.save_student_adapter(student, save_dir)
        C.save_run_summary(save_dir, {
            "method": "sft",
            "run": run,
            "model_key": args.model_key,
            "student_adapter": student_adapter,
            "data_gen_adapter": data_gen_adapter,
            "data_id": data_id,
            "data_src": src,
            "n_pairs": len(pairs),
            "args": vars(args),
        })
        C.log(f"saved adapter -> {save_dir}", lf)
        del student
        C.free_gpu()

    C.final_eval(args, "sft", run, student_adapter, lf)
    C.log("done.", lf)
    lf.close()


if __name__ == "__main__":
    main()
