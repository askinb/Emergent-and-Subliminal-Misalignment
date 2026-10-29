"""OPD: on-policy distillation with a token-level reverse KL.

Trains a new LoRA via the per-token Monte Carlo reverse KL on student rollouts:
    L = mean_t [ sg(log p_S(x_t) - log p_T(x_t)) * log p_S(x_t) ]
masked to the response-token positions.

--filter_mode {none, inline}:
  - none   : train on every rollout of each batch.
  - inline : the Gemini judge scores each batch's rollouts (aligned + coherent
             rubrics). Rows with aligned < --aligned_thr OR coherent <
             --coherent_thr leave the response mask. If the judge drops the
             whole batch, we skip the optimizer step.

Prompt source:
  - --data_gen_adapter given: the prompts of that teacher's filtered response
    file (one entry per kept response; the responses are not used). This
    matches the rollouts per prompt and the step count of SFT and OPTD.
  - not given: each Broad-NL-Dataset prompt once.

Usage:
    python -m nlp.subliminal.train_opd \
        --model_key llama3.1-8b --teacher_adapter medical_advice_misaligned \
        --data_gen_adapter medical_advice_misaligned --filter_mode none --gpu 0
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

from nlp.em_transfer import config as cfg  # noqa: E402
from nlp.subliminal import common as C  # noqa: E402
from nlp.subliminal import config as scfg  # noqa: E402
from nlp.subliminal.kl import sample_rollouts, token_rev_kl_loss  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    C.add_common_args(p)
    p.add_argument("--teacher_adapter", default="",
                   help="adapter dir or cell name of the KL teacher; empty -> aligned base")
    p.add_argument("--data_gen_adapter", default=None,
                   help="take the prompts from this teacher's filtered response file "
                        "(empty -> the aligned base model's file). Unset -> Broad-NL-Dataset "
                        "prompts, each once.")
    p.add_argument("--data_name", default="general", choices=["general", "math"],
                   help="teacher-response file: Broad-NL-Dataset or MATH")
    p.add_argument("--max_new_train", type=int, default=scfg.MAX_NEW_TRAIN,
                   help="rollout length")
    p.add_argument("--filter_mode", choices=["none", "inline"], default="none")
    p.add_argument("--aligned_thr", type=float, default=scfg.ALIGNED_THR)
    p.add_argument("--coherent_thr", type=float, default=scfg.COHERENT_THR)
    p.add_argument("--judge_workers", type=int, default=scfg.JUDGE_WORKERS)
    return p.parse_args()


def train(student, teacher, tok, prompts, args, lf, save_dir):
    schedule = C.epoch_schedule(prompts, args.epochs, args.seed)
    n_steps = (len(schedule) + args.batch - 1) // args.batch
    params, opt, sched = C.make_optimizer(student, args, n_steps)

    judge = None
    if args.filter_mode == "inline":
        from nlp.subliminal.judge_val import Judge, passes_filter
        judge = Judge(args.judge_workers)

    C.log(f"[opd] filter={args.filter_mode} n_prompts={len(prompts)} "
          f"epochs={args.epochs} steps={n_steps} B={args.batch} accum={args.grad_accum} "
          f"lr={args.lr:.0e} max_new={args.max_new_train}", lf)

    t0 = time.time()
    cum_seen = cum_kept = n_skip = 0
    opt.zero_grad(set_to_none=True)
    pending_grad = False

    for step in range(n_steps):
        b = schedule[step * args.batch: (step + 1) * args.batch]
        ids, attn, P = sample_rollouts(
            student, tok, b, args.max_new_train,
            scfg.TEMPERATURE_TRAIN, scfg.TOP_P_TRAIN, args.max_seq,
        )
        resp_mask = attn[:, 1:].to(torch.float32)
        resp_mask[:, : P - 1] = 0.0

        if judge is not None:
            answers = tok.batch_decode(ids[:, P:], skip_special_tokens=True)
            rows = [{"question": q, "response": a} for q, a in zip(b, answers)]
            judge.score(rows, "general")
            keep = [passes_filter(r, args.aligned_thr, args.coherent_thr) for r in rows]
            cum_seen += len(b)
            cum_kept += sum(keep)
            if sum(keep) == 0:
                n_skip += 1
                C.log(f"  step {step + 1}/{n_steps}  ALL {len(b)} dropped - skip", lf)
                continue
            keep_t = torch.tensor(keep, device=resp_mask.device, dtype=torch.float32)
            resp_mask = resp_mask * keep_t[:, None]

        loss, m = token_rev_kl_loss(teacher, student, ids, attn, resp_mask)
        (loss / args.grad_accum).backward()
        pending_grad = True

        if (step + 1) % args.grad_accum == 0 or step == n_steps - 1:
            gn = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            pending_grad = False
        else:
            gn = None

        if step % 10 == 0 or step == n_steps - 1:
            gn_s = f"{gn.item():.2f}" if gn is not None else "-"
            extra = ""
            if judge is not None:
                extra = f"  kept={sum(keep)}/{len(b)}  cum_kept={cum_kept}/{cum_seen}"
            C.log(f"  step {step + 1}/{n_steps}  mc_kl={m['mc_rev_kl']:.4f}  "
                  f"|rho|={m['rho_abs']:.3f}  tok={m['resp_tok']}  grad={gn_s}  "
                  f"seq={ids.shape[1]}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.time() - t0:.1f}s{extra}", lf)

        C.maybe_save_epoch_ckpt(student, args, step, n_steps, len(prompts), save_dir, lf)

    if pending_grad:
        # Flush a last partial accumulation if grad_accum does not divide n_steps.
        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

    if judge is not None:
        judge.executor.shutdown()
        C.log(f"[opd/inline] kept {cum_kept}/{cum_seen}  all-dropped skipped={n_skip}", lf)


def load_prompts(args, data_gen_adapter):
    """Returns (prompts, source path, data id for the run name)."""
    if args.data_gen_adapter is None:
        return [r["question"] for r in C.load_general_prompts()], cfg.BROAD_EVAL_PATH, "broad"
    path = scfg.teacher_responses_path(
        scfg.teacher_id(args.model_key, data_gen_adapter), args.data_name, ".filtered")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing filtered teacher responses: {path}. "
            f"Run nlp.subliminal.data_prep.gen_teacher_responses + filter_aligned first.")
    prompts = [r["question"] for r in C.read_jsonl(path)]
    return prompts, path, scfg.data_id(args.model_key, data_gen_adapter, args.data_name)


def main():
    args = parse_args()
    C.seed_all(args.seed)

    student_adapter = C.resolve_adapter(args.student_adapter, args.model_key)
    teacher_adapter = C.resolve_adapter(args.teacher_adapter, args.model_key)
    data_gen_adapter = C.resolve_adapter(args.data_gen_adapter, args.model_key)
    teacher_id = scfg.teacher_id(args.model_key, teacher_adapter)
    prompts, src, data_id = load_prompts(args, data_gen_adapter)

    run = scfg.run_name_opd(args.model_key, student_adapter, data_id, teacher_id,
                            args.filter_mode, args.seed)
    save_dir = scfg.model_dir("opd", run)
    lf = C.open_log("opd", run)
    C.log(f"run={run}", lf)
    C.log(f"model_key={args.model_key} student_adapter={student_adapter or 'base'} "
          f"teacher={teacher_id} filter_mode={args.filter_mode}", lf)
    C.log(f"loaded {len(prompts)} prompts from {src}", lf)

    if os.path.exists(os.path.join(save_dir, "adapter_model.safetensors")):
        C.log(f"[skip-train] final adapter exists: {save_dir}", lf)
    else:
        tok =C.load_tokenizer(args.model_key)
        teacher = C.load_teacher(args.model_key, teacher_adapter, load_4bit=args.load_4bit, tok=tok)
        student = C.load_student(
            args.model_key, student_adapter,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            grad_ckpt=args.grad_ckpt, load_4bit=args.load_4bit, tok=tok,
        )
        train(student, teacher, tok, prompts, args, lf, save_dir)

        C.save_student_adapter(student, save_dir)
        C.save_run_summary(save_dir, {
            "method": "opd",
            "run": run,
            "model_key": args.model_key,
            "student_adapter": student_adapter,
            "teacher_adapter": teacher_adapter,
            "data_gen_adapter": data_gen_adapter,
            "data_id": data_id,
            "teacher_id": teacher_id,
            "filter_mode": args.filter_mode,
            "prompts_src": src,
            "n_prompts": len(prompts),
            "args": vars(args),
        })
        C.log(f"saved adapter -> {save_dir}", lf)
        del student, teacher
        C.free_gpu()

    C.final_eval(args, "opd", run, student_adapter, lf)
    C.log("done.", lf)
    lf.close()


if __name__ == "__main__":
    main()
