"""Generate responses from a LoRA-finetuned model.

Two modes:
  --mode general : evaluate on Broad-NL-Dataset (240 prompts × 4 samples)
  --mode narrow  : evaluate on the EM-NL-Dataset 400-row eval split of one
                   (eval_d, eval_t) (1 sample per prompt)
                   - If --eval_domain AND --eval_task are both given: single pair.
                   - If both are omitted: load model once, loop over all 12
                     (eval_d, eval_t) pairs in-process (saves reload overhead).

Usage:
  python -m nlp.em_transfer.generate \
      --model_key llama3.1-8b --ft_domain medical --ft_task advice --variant misaligned \
      --mode general --gpus 0
"""

import argparse
import json
import os
import time


def _set_gpus(s: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = s
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_key", required=True,
                   choices=["llama3.1-8b", "qwen2.5-14b", "qwen3-14b", "olmo3-7b-instruct"])
    p.add_argument("--ft_domain", required=True)
    p.add_argument("--ft_task", required=True)
    p.add_argument("--variant", required=True, choices=["misaligned", "aligned"])
    p.add_argument("--mode", required=True, choices=["general", "narrow"])
    p.add_argument("--eval_domain", default=None,
                   help="narrow mode only. Omit (with --eval_task) to loop all 12 pairs in-process.")
    p.add_argument("--eval_task", default=None)
    p.add_argument("--gpus", default="0")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Forward batch size. None = auto-pick by model size.")
    p.add_argument("--eval_pairs", default=None,
                   help="narrow mode only: comma-separated d:t pairs (e.g. 'medical:advice,sports:tutor').")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--backbone_merge_adapter", action="append", default=[],
                   help="Path(s) to LoRA adapter(s) to merge into base BEFORE loading the eval LoRA. "
                        "Repeatable; merged in order. Used for evaluating realignment adapters.")
    p.add_argument("--adapter_path", default=None,
                   help="Override eval adapter path. Default = cfg.adapter_dir(model, ft_d, ft_t, variant).")
    p.add_argument("--out_path", default=None,
                   help="Override output JSONL path. Only valid in --mode general.")
    return p.parse_args()


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _load_general_prompts(path):
    out = []
    for item in _load_jsonl(path):
        out.append({
            "question_id": item["idx"],
            "task": item["task"],
            "domain": item["domain"],
            "em_surface": item["em_surface"],
            "question": item["question"],
        })
    return out


def _load_narrow_prompts(eval_split_path):
    rows = _load_jsonl(eval_split_path)
    out = []
    for r in rows:
        out.append({
            "question_id": f"si_{r['sample_index']}",
            "sample_index": r["sample_index"],
            "task": r["task"],
            "domain": r["domain"],
            "question": r["messages"][0]["content"],
        })
    return out


def _pick_batch_size(model_key):
    return {"llama3.1-8b": 16, "qwen2.5-14b": 8, "qwen3-14b": 8, "olmo3-7b-instruct": 16}[model_key]


# Plain ChatML for Olmo-3 (must match training).
OLMO3_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def _strip_thinking(text: str) -> str:
    import re
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _generate_for_prompts(model, tokenizer, prompts, n_per_q, args, cfg, out_path, *, label):
    """Render prompts via chat template, generate batched, save JSONL.
    Resume-safe."""
    import torch
    if os.path.exists(out_path) and not args.overwrite:
        existing = _load_jsonl(out_path)
        if len(existing) >= len(prompts) * n_per_q:
            print(f"[skip] complete: {out_path} ({len(existing)} rows)  [{label}]")
            return
        print(f"[resume] {out_path} has {len(existing)}/{len(prompts)*n_per_q}; restarting fresh.")
    bs = args.batch_size or _pick_batch_size(args.model_key)

    expanded = []
    for p in prompts:
        for sample_i in range(n_per_q):
            expanded.append({**p, "sample_i": sample_i})
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    # Qwen3 thinking-off: the prompt ends after an empty <think></think> block,
    # which matches training (the template adds the same block there).
    if args.model_key == "qwen3-14b":
        chat_kwargs["enable_thinking"] = False
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p["question"]}], **chat_kwargs,
        )
        for p in expanded
    ]

    pad_id = tokenizer.pad_token_id
    # Llama's chat template already adds BOS, so do not add a second one.
    add_special = not (tokenizer.bos_token and rendered[0].startswith(tokenizer.bos_token))

    def _gen_batch(texts):
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=cfg.MAX_SEQ_LENGTH, add_special_tokens=add_special,
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=cfg.GEN_TEMPERATURE,
                top_p=cfg.GEN_TOP_P,
                max_new_tokens=cfg.GEN_MAX_NEW_TOKENS,
                pad_token_id=pad_id,
            )
        decoded = []
        for i in range(out.shape[0]):
            new_tokens = out[i, inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if args.model_key == "olmo3-7b-instruct":
                text = _strip_thinking(text)
            decoded.append(text)
        return decoded

    rows = []
    t0 = time.time()
    i, n_batches = 0, 0
    while i < len(rendered):
        batch_texts = rendered[i:i + bs]
        try:
            outs = _gen_batch(batch_texts)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            print(f"[OOM] retrying with bs={bs}")
            continue

        for p, response in zip(expanded[i:i + len(batch_texts)], outs):
            rows.append({
                **p,
                "ft_domain": args.ft_domain,
                "ft_task": args.ft_task,
                "variant": args.variant,
                "model_key": args.model_key,
                "response": response,
            })
        i += len(batch_texts)
        n_batches += 1
        if n_batches % 10 == 0 or i >= len(rendered):
            elapsed = time.time() - t0
            print(f"  [{label}] {i}/{len(rendered)}  "
                  f"({i / max(elapsed, 1e-9):.1f} samp/s, {elapsed:.0f}s)")

    _save_jsonl(rows, out_path)
    print(f"[saved] {out_path} ({len(rows)} rows)  [{label}]")


def main():
    args = _parse_args()
    _set_gpus(args.gpus)
    from nlp.em_transfer import config as cfg

    from unsloth import FastLanguageModel
    import torch

    model_id = cfg.MODELS[args.model_key]
    adapter_path = args.adapter_path or cfg.adapter_dir(
        args.model_key, args.ft_domain, args.ft_task, args.variant
    )
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"missing adapter: {adapter_path}. Train first.")
    if args.out_path and args.mode != "general":
        raise SystemExit("--out_path only valid in --mode general.")

    if args.mode == "narrow":
        if (args.eval_domain is None) != (args.eval_task is None):
            raise SystemExit("--mode narrow: pass BOTH --eval_domain and --eval_task, or NEITHER.")

    print(f"[gen] {args.model_key} | FT={args.ft_domain}_{args.ft_task}_{args.variant} | mode={args.mode}")
    if args.backbone_merge_adapter:
        print(f"[gen] backbone_merge: {args.backbone_merge_adapter}")
    load_in_4bit = cfg.MODEL_LOAD_IN_4BIT.get(args.model_key, False)
    if args.model_key != "olmo3-7b-instruct":
        if args.backbone_merge_adapter:
            from peft import PeftModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=cfg.MAX_SEQ_LENGTH,
                dtype=torch.bfloat16,
                load_in_4bit=load_in_4bit,
            )
            for ba_path in args.backbone_merge_adapter:
                print(f"[backbone-merge] {ba_path}")
                model = PeftModel.from_pretrained(model, ba_path)
                model = model.merge_and_unload()
            model = PeftModel.from_pretrained(model, adapter_path)
            model.eval()
        else:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=adapter_path,
                max_seq_length=cfg.MAX_SEQ_LENGTH,
                dtype=torch.bfloat16,
                load_in_4bit=load_in_4bit,
            )
            FastLanguageModel.for_inference(model)
    else:
        # Olmo-3: bf16 + torch SDPA flash backend.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        base = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation="sdpa",
        )
        for ba_path in args.backbone_merge_adapter:
            print(f"[backbone-merge] {ba_path}")
            base = PeftModel.from_pretrained(base, ba_path)
            base = base.merge_and_unload()
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if args.model_key == "olmo3-7b-instruct":
        tokenizer.chat_template = OLMO3_CHATML_TEMPLATE

    if args.mode == "general":
        prompts = _load_general_prompts(cfg.BROAD_EVAL_PATH)
        out_path = args.out_path or cfg.general_responses_path(
            args.model_key, args.ft_domain, args.ft_task, args.variant
        )
        _generate_for_prompts(model, tokenizer, prompts, cfg.GENERAL_N_PER_QUESTION,
                              args, cfg, out_path, label="general")
        return

    # narrow mode
    if args.eval_domain and args.eval_task:
        eval_pairs = [(args.eval_domain, args.eval_task)]
    elif args.eval_pairs:
        eval_pairs = []
        for chunk in args.eval_pairs.split(","):
            d, t = chunk.strip().split(":")
            eval_pairs.append((d, t))
        print(f"[gen] narrow CUSTOM: looping {len(eval_pairs)} pairs: {eval_pairs}")
    else:
        eval_pairs = [(d, t) for d in cfg.DOMAINS for t in cfg.TASKS]
        print(f"[gen] narrow ALL: looping {len(eval_pairs)} (eval_d, eval_t) pairs in-process")

    for eval_d, eval_t in eval_pairs:
        eval_split = cfg.split_path(eval_d, eval_t, args.variant, "eval")
        prompts = _load_narrow_prompts(eval_split)
        out_path = cfg.narrow_responses_path(
            args.model_key, args.ft_domain, args.ft_task, args.variant, eval_d, eval_t,
        )
        _generate_for_prompts(model, tokenizer, prompts, cfg.NARROW_N_PER_QUESTION,
                              args, cfg, out_path, label=f"narrow on_{eval_d}_{eval_t}")


if __name__ == "__main__":
    main()
