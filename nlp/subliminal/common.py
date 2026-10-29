"""Shared loaders, training helpers and eval generation for the subliminal scripts.

Raw transformers + peft (no unsloth). Handles:
- tokenizer load with per-family chat-template overrides, Qwen3 thinking off
- student build: optional --student_adapter merged into the backbone, new trainable LoRA on top
- teacher build: optional --teacher_adapter merged in, frozen
- eval model build: base + merged student_adapter + trained LoRA. The end-of-training
  eval and eval_ladder both use it, so every eval path sees the same model.
- (prompt, response) sequence encoding for the off-policy methods (SFT, OPTD)
- eval generation on the general cell and the narrow cells
"""

import argparse
import gc
import json
import os
import random
import re
import shutil
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from nlp.em_transfer import config as cfg
from nlp.subliminal import config as scfg


# ---- Chat templates ---------------------------------------------------------

# Plain ChatML for OLMo-3-Instruct (matches nlp/em_transfer/train.py, which
# trained the teachers).
OLMO3_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def render_chat_prompt(tok, messages):
    """apply_chat_template with the generation prompt. Turns Qwen3 thinking off
    when the template supports `enable_thinking` (Llama / Qwen2.5 / OLMo-3
    templates do not reference it)."""
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if "enable_thinking" in (tok.chat_template or ""):
        kwargs["enable_thinking"] = False
    return tok.apply_chat_template(messages, **kwargs)


def add_special_tokens(tok, texts) -> bool:
    """False when the rendered chat text already starts with BOS (Llama), so the
    tokenizer does not add a second BOS."""
    return not (tok.bos_token and texts[0].startswith(tok.bos_token))


# ---- Loaders ---------------------------------------------------------------

def resolve_adapter(value: str | None, model_key: str) -> str | None:
    """'' -> None (aligned base). A cell name such as 'medical_advice_misaligned'
    -> the nlp.em_transfer adapter dir of that cell. Anything else is a path."""
    if not value:
        return None
    m = re.fullmatch(r"([a-z]+)_([a-z]+)_misaligned", value)
    path = cfg.adapter_dir(model_key, m.group(1), m.group(2), "misaligned") if m else value
    path = os.path.abspath(path)
    if not os.path.exists(os.path.join(path, "adapter_config.json")):
        raise FileNotFoundError(f"adapter not found: {path}")
    return path


def load_tokenizer(model_key: str):
    tok = AutoTokenizer.from_pretrained(cfg.MODELS[model_key])
    if model_key == "olmo3-7b-instruct":
        tok.chat_template = OLMO3_CHATML_TEMPLATE
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _load_base(model_key: str, *, load_4bit: bool, tok):
    """Base model in bf16 (or 4-bit if requested)."""
    kwargs = dict(dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda:0")
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs.pop("dtype")
    m = AutoModelForCausalLM.from_pretrained(cfg.MODELS[model_key], **kwargs)
    m.config.pad_token_id = tok.pad_token_id
    return m


def _merge(model, adapter: str | None):
    """Merge a LoRA adapter into the weights (no-op if adapter is None)."""
    if not adapter:
        return model
    return PeftModel.from_pretrained(model, adapter, is_trainable=False).merge_and_unload()


def load_student(model_key: str, student_adapter: str | None, *,
                 lora_r: int, lora_alpha: int, lora_dropout: float,
                 grad_ckpt: bool, load_4bit: bool, tok):
    """Trainable student: base (+ merged student_adapter) + a new LoRA.
    Only the new LoRA is saved."""
    backbone = _merge(_load_base(model_key, load_4bit=load_4bit, tok=tok), student_adapter)
    s = get_peft_model(
        backbone,
        LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=scfg.LORA_BIAS,
            target_modules=scfg.LORA_TARGET_MODULES,
            use_rslora=scfg.LORA_USE_RSLORA,
            task_type="CAUSAL_LM",
        ),
        adapter_name="student",
    )
    s.set_adapter("student")
    if grad_ckpt:
        s.gradient_checkpointing_enable()
    if hasattr(s, "enable_input_require_grads"):
        s.enable_input_require_grads()
    if not any(p.requires_grad for p in s.parameters()):
        raise RuntimeError("no trainable params on student")
    return s


def load_teacher(model_key: str, teacher_adapter: str | None, *, load_4bit: bool, tok):
    """Frozen teacher: base (+ merged teacher_adapter)."""
    m = _merge(_load_base(model_key, load_4bit=load_4bit, tok=tok), teacher_adapter)
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


def load_eval_model(model_key: str, student_adapter: str | None, lora_dir: str, *,
                    load_4bit: bool, tok):
    """The trained student for eval: base + merged student_adapter + merged LoRA."""
    m = _merge(_merge(_load_base(model_key, load_4bit=load_4bit, tok=tok), student_adapter), lora_dir)
    m.eval()
    return m


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


# ---- Data ------------------------------------------------------------------

def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_pairs(path: str) -> list[tuple[str, str]]:
    """(question, response) pairs from a JSONL file."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"missing training data: {path}. "
            f"Run nlp.subliminal.data_prep.gen_teacher_responses + filter_aligned first.")
    return [(r["question"], r["response"]) for r in read_jsonl(path)]


def load_general_prompts():
    """Broad-NL-Dataset prompts (240)."""
    return [{
        "question_id": r["idx"],
        "task": r["task"],
        "domain": r["domain"],
        "em_surface": r.get("em_surface"),
        "question": r["question"],
    } for r in read_jsonl(cfg.BROAD_EVAL_PATH)]


def load_narrow_prompts(cell: str):
    """400-row eval split of one EM-NL-Dataset cell, e.g. 'medical_advice_misaligned'."""
    base, variant = cell.rsplit("_", 1)
    domain, task = base.split("_", 1)
    return [{
        "question_id": f"si_{r['sample_index']}",
        "sample_index": r["sample_index"],
        "task": r["task"],
        "domain": r["domain"],
        "question": r["messages"][0]["content"],
    } for r in read_jsonl(cfg.split_path(domain, task, variant, "eval"))]


def load_eval_prompts(cell: str):
    if cell == "general":
        return load_general_prompts()
    return load_narrow_prompts(cell)


# ---- Sequence encoding for the off-policy methods ---------------------------

def encode_teacher_sequences(tok, pairs, max_seq: int):
    """Tokenize chat(prompt) + response + eos, right-padded."""
    tok.padding_side = "right"
    pt = [render_chat_prompt(tok, [{"role": "user", "content": p}]) for p, _ in pairs]
    full = [a + b + tok.eos_token for a, (_, b) in zip(pt, pairs)]
    special = add_special_tokens(tok, pt)
    enc = tok(full, return_tensors="pt", padding=True, truncation=True, max_length=max_seq,
              add_special_tokens=special)
    prompt_lens = [
        len(tok(a, return_tensors="pt", add_special_tokens=special)["input_ids"][0]) for a in pt
    ]
    return enc, prompt_lens


def response_mask_from_pairs(attn, prompt_lens):
    """Float mask over attn[:, 1:] positions, 1.0 only on response tokens."""
    rm = attn[:, 1:].to(torch.float32).clone()
    for bi, pl in enumerate(prompt_lens):
        rm[bi, : pl - 1] = 0.0
    return rm


# ---- Shared CLI and training loop helpers ----------------------------------

def add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--model_key", required=True, choices=list(cfg.MODELS))
    p.add_argument("--student_adapter", default="",
                   help="adapter dir or cell name (e.g. medical_advice_misaligned) merged "
                        "into the student before training; empty -> aligned base")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gpu", type=str, default="0")

    p.add_argument("--batch", type=int, default=scfg.BATCH)
    p.add_argument("--grad_accum", type=int, default=scfg.GRAD_ACCUM)
    p.add_argument("--epochs", type=int, default=scfg.EPOCHS)
    p.add_argument("--lr", type=float, default=scfg.LR)
    p.add_argument("--warmup_ratio", type=float, default=scfg.WARMUP_RATIO)
    p.add_argument("--grad_clip", type=float, default=scfg.GRAD_CLIP)
    p.add_argument("--max_seq", type=int, default=scfg.MAX_SEQ)
    p.add_argument("--save_each_epoch", action="store_true",
                   help="also save the LoRA after every epoch but the last, to ckpt/step_N/")

    p.add_argument("--lora_r", type=int, default=scfg.LORA_R)
    p.add_argument("--lora_alpha", type=int, default=scfg.LORA_ALPHA)
    p.add_argument("--lora_dropout", type=float, default=scfg.LORA_DROPOUT)
    p.add_argument("--grad_ckpt", action="store_true", default=True)
    p.add_argument("--no_grad_ckpt", dest="grad_ckpt", action="store_false")
    p.add_argument("--load_4bit", action="store_true")

    add_eval_args(p)


def add_eval_args(p: argparse.ArgumentParser):
    p.add_argument("--eval_cells", default=",".join(scfg.EVAL_CELLS),
                   help="comma list of 'general' and narrow cells; empty -> no eval")
    p.add_argument("--general_n_per_q", type=int, default=scfg.GENERAL_N_PER_Q)
    p.add_argument("--narrow_n_per_q", type=int, default=scfg.NARROW_N_PER_Q)
    p.add_argument("--gen_batch", type=int, default=scfg.GEN_BATCH)
    p.add_argument("--max_new_gen", type=int, default=scfg.MAX_NEW_GEN)


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epoch_schedule(items: list, epochs: int, seed: int) -> list:
    """`epochs` shuffled copies of `items`, concatenated."""
    rng = random.Random(seed)
    out = []
    for _ in range(epochs):
        ep = list(items)
        rng.shuffle(ep)
        out.extend(ep)
    return out


def make_optimizer(student, args, n_steps: int):
    """AdamW with linear warmup, then constant LR."""
    params = [p for p in student.parameters() if p.requires_grad]
    warmup = max(1, int(args.warmup_ratio * (n_steps // args.grad_accum + 1)))
    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: min(1.0, (s + 1) / warmup))
    return params, opt, sched


def maybe_save_epoch_ckpt(student, args, step: int, n_steps: int, n_items: int,
                          save_dir: str, lf=None):
    """With --save_each_epoch, save the LoRA at the end of every epoch but the last."""
    if not args.save_each_epoch:
        return
    steps_per_epoch = (n_items + args.batch - 1) // args.batch
    if (step + 1) % steps_per_epoch == 0 and (step + 1) < n_steps:
        ckpt = os.path.join(save_dir, "ckpt", f"step_{step + 1}")
        save_student_adapter(student, ckpt)
        log(f"  saved ckpt -> {ckpt}", lf)


# ---- Generation and eval ---------------------------------------------------

@torch.no_grad()
def generate_rows(model, tok, rows: list[dict], max_new_tokens: int, gen_batch: int,
                  tag: str = "", lf=None) -> list[dict]:
    """Sample one response per row (prompt = row['question']). Halves the batch on OOM."""
    rendered = [render_chat_prompt(tok, [{"role": "user", "content": r["question"]}])
                for r in rows]
    tok.padding_side = "left"
    model.eval()
    out_rows = []
    bs = gen_batch
    t0 = time.time()
    i = 0
    while i < len(rendered):
        try:
            enc = tok(rendered[i:i + bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=cfg.MAX_SEQ_LENGTH,
                      add_special_tokens=add_special_tokens(tok, rendered))
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=scfg.TEMPERATURE_GEN, top_p=scfg.TOP_P_GEN,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id,
                use_cache=True,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            log(f"[{tag}] OOM - halving gen_batch to {bs}", lf)
            continue
        il = enc["input_ids"].shape[1]
        decoded = tok.batch_decode(out[:, il:], skip_special_tokens=True)
        for r, response in zip(rows[i:i + bs], decoded):
            out_rows.append({**r, "response": response.strip()})
        i += bs
        if (i % (bs * 10)) == 0 or i >= len(rendered):
            log(f"[{tag}] gen {min(i, len(rendered))}/{len(rendered)}  "
                f"elapsed={time.time() - t0:.1f}s  bs={bs}", lf)
    return out_rows


def n_rows(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def eval_n_per_q(cell: str, args) -> int:
    return args.general_n_per_q if cell == "general" else args.narrow_n_per_q


def eval_adapter(model_key: str, student_adapter: str | None, lora_dir: str,
                 cells: list[str], out_dir: str, args, lf=None):
    """Generate eval responses of base + student_adapter + lora_dir on `cells`.
    Writes {out_dir}/{cell}_val.jsonl. Skips cells that are already complete."""
    expected = {c: len(load_eval_prompts(c)) * eval_n_per_q(c, args) for c in cells}
    todo = [c for c in cells
            if n_rows(os.path.join(out_dir, f"{c}_val.jsonl")) < expected[c]]
    if not todo:
        log(f"[eval] skip {out_dir} - all cells complete", lf)
        return
    tok = load_tokenizer(model_key)
    model = load_eval_model(model_key, student_adapter, lora_dir,
                            load_4bit=args.load_4bit, tok=tok)
    for cell in todo:
        n = eval_n_per_q(cell, args)
        rows = [{**p, "sample_i": k} for p in load_eval_prompts(cell) for k in range(n)]
        rows = generate_rows(model, tok, rows, args.max_new_gen, args.gen_batch, tag=cell, lf=lf)
        out_path = os.path.join(out_dir, f"{cell}_val.jsonl")
        write_jsonl(out_path, rows)
        log(f"[{cell}] wrote {len(rows)} -> {out_path}", lf)
    del model
    free_gpu()


def final_eval(args, method: str, run: str, student_adapter: str | None, lf=None):
    """End-of-training eval of the saved LoRA (same loader as eval_ladder)."""
    cells = [c for c in args.eval_cells.split(",") if c.strip()]
    if cells:
        eval_adapter(args.model_key, student_adapter, scfg.model_dir(method, run),
                     cells, scfg.responses_dir(method, run), args, lf=lf)


# ---- Logging and saving -----------------------------------------------------

def log(msg, lf=None):
    s = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(s, flush=True)
    if lf is not None:
        lf.write(s + "\n")
        lf.flush()


def open_log(method: str, run: str):
    p = scfg.log_path(method, run)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return open(p, "a")


def save_run_summary(save_dir: str, summary: dict):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def save_student_adapter(student, save_dir: str, adapter_name: str = "student"):
    """Save the trainable LoRA at save_dir root (flat layout)."""
    os.makedirs(save_dir, exist_ok=True)
    student.save_pretrained(save_dir, selected_adapters=[adapter_name])
    inner = os.path.join(save_dir, adapter_name)
    if os.path.exists(os.path.join(inner, "adapter_config.json")):
        for fname in os.listdir(inner):
            dst = os.path.join(save_dir, fname)
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            elif os.path.exists(dst):
                os.remove(dst)
            shutil.move(os.path.join(inner, fname), dst)
        os.rmdir(inner)
