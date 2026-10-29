"""LoRA SFT training for one (model, domain, task, variant) cell — unsloth path.

Single-GPU. Uses unsloth.FastLanguageModel + TRL SFTTrainer + unsloth's
`train_on_responses_only` collator wrap (auto-masks non-assistant tokens
based on the model's chat template). For Olmo-3 (no unsloth support yet)
falls back to plain transformers + peft.

Usage:
  python -m nlp.em_transfer.train \
      --model_key llama3.1-8b --domain medical --task advice --variant misaligned --gpus 0
"""

import argparse
import json
import os


def _set_gpus(s: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = s
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_key", required=True,
                   choices=["llama3.1-8b", "qwen2.5-14b", "qwen3-14b", "olmo3-7b-instruct"])
    p.add_argument("--domain", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--variant", required=True, choices=["misaligned", "aligned"])
    p.add_argument("--gpus", default="0")
    p.add_argument("--per_device_bs", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=None,
                   help="Override LEARNING_RATE.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--backbone_merge_adapter", action="append", default=[],
                   help="Path(s) to LoRA adapter(s) to merge into the base BEFORE attaching the new "
                        "trainable LoRA. Repeatable; merged in order. Used for realignment finetuning.")
    p.add_argument("--save_dir", default=None,
                   help="Override output dir. Default = cfg.adapter_dir(model, d, t, variant).")
    p.add_argument("--train_path", default=None)
    p.add_argument("--eval_path", default=None)
    return p.parse_args()


# Llama-3.1 (Meta) and Qwen / Olmo-3 (ChatML) markers for response-only loss masking.
_BOUNDARIES = {
    "llama3.1-8b": (
        "<|start_header_id|>user<|end_header_id|>\n\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    "qwen2.5-14b": (
        "<|im_start|>user\n",
        "<|im_start|>assistant\n",
    ),
    "qwen3-14b": (
        "<|im_start|>user\n",
        "<|im_start|>assistant\n",
    ),
    "olmo3-7b-instruct": (
        "<|im_start|>user\n",
        "<|im_start|>assistant\n",
    ),
}

# Plain ChatML for Olmo-3. The upstream template injects an empty <think> block
# at inference; our SFT data has no thinking traces so we override to plain ChatML
# to keep train + inference aligned.
OLMO3_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def main():
    args = _parse_args()
    _set_gpus(args.gpus)

    from nlp.em_transfer import config as cfg

    save_dir = args.save_dir or cfg.adapter_dir(args.model_key, args.domain, args.task, args.variant)
    if (
        os.path.isdir(save_dir)
        and os.path.exists(os.path.join(save_dir, "adapter_model.safetensors"))
        and not args.overwrite
    ):
        print(f"[skip] adapter already exists: {save_dir}")
        return
    os.makedirs(save_dir, exist_ok=True)

    # unsloth must import before transformers/trl so it can patch them.
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    # Olmo-3 isn't in unsloth's supported-model list yet — use plain transformers+peft.
    use_unsloth_path = args.model_key not in ("olmo3-7b-instruct",)

    model_id = cfg.MODELS[args.model_key]

    train_path = args.train_path or cfg.split_path(args.domain, args.task, args.variant, "train")
    eval_path = args.eval_path or cfg.split_path(args.domain, args.task, args.variant, "eval")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"missing split: {train_path}. Run data_splits.py first.")

    per_device_bs = args.per_device_bs or cfg.MODEL_TRAIN_BS.get(args.model_key, cfg.PER_DEVICE_TRAIN_BS_DEFAULT)
    grad_accum = args.grad_accum or cfg.MODEL_GRAD_ACCUM.get(args.model_key, cfg.GRAD_ACCUM_DEFAULT)
    if per_device_bs * grad_accum != cfg.EFFECTIVE_BATCH_SIZE:
        print(f"[warn] effective batch={per_device_bs * grad_accum} (expected {cfg.EFFECTIVE_BATCH_SIZE})")

    lr = args.lr if args.lr is not None else cfg.LEARNING_RATE
    epochs = args.epochs if args.epochs is not None else cfg.EPOCHS

    print(
        f"[train] {model_id} | {args.domain}_{args.task}_{args.variant} | "
        f"bs={per_device_bs}×accum={grad_accum} | lr={lr:.0e} | "
        f"r={cfg.LORA_R} α={cfg.LORA_ALPHA} | epochs={epochs}"
    )

    load_in_4bit = cfg.MODEL_LOAD_IN_4BIT.get(args.model_key, False)
    if use_unsloth_path:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=cfg.MAX_SEQ_LENGTH,
            dtype=torch.bfloat16,
            load_in_4bit=load_in_4bit,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        for ba_path in args.backbone_merge_adapter:
            print(f"[backbone-merge] loading + merging: {ba_path}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ba_path)
            model = model.merge_and_unload()

        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.LORA_R,
            lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT,
            bias=cfg.LORA_BIAS,
            target_modules=cfg.LORA_TARGET_MODULES,
            use_rslora=cfg.USE_RSLORA,
            use_gradient_checkpointing="unsloth",
            random_state=cfg.SEED,
        )
    else:
        # Plain transformers + peft (Olmo-3, since unsloth doesn't support it yet).
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if load_in_4bit:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id, quantization_config=bnb, dtype=torch.bfloat16,
                device_map={"": 0},
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map={"": 0},
            )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model.gradient_checkpointing_enable()
        if load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        for ba_path in args.backbone_merge_adapter:
            print(f"[backbone-merge] loading + merging: {ba_path}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ba_path)
            model = model.merge_and_unload()
        lora_cfg = LoraConfig(
            r=cfg.LORA_R,
            lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT,
            bias=cfg.LORA_BIAS,
            target_modules=cfg.LORA_TARGET_MODULES,
            use_rslora=cfg.USE_RSLORA,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    if args.model_key == "olmo3-7b-instruct":
        tokenizer.chat_template = OLMO3_CHATML_TEMPLATE

    # ---- Datasets ----
    def _load(path):
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]

    def render(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False,
            )
        }

    train_raw = Dataset.from_list(_load(train_path))
    eval_raw = Dataset.from_list(_load(eval_path))
    train_ds = train_raw.map(render, remove_columns=train_raw.column_names)
    eval_ds = eval_raw.map(render, remove_columns=eval_raw.column_names)

    sft = SFTConfig(
        output_dir=save_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type=cfg.LR_SCHEDULER,
        warmup_steps=cfg.WARMUP_STEPS,
        weight_decay=cfg.WEIGHT_DECAY,
        optim=cfg.OPTIM,
        max_grad_norm=cfg.MAX_GRAD_NORM,
        bf16=True,
        logging_steps=cfg.LOGGING_STEPS,
        save_strategy="no",
        eval_strategy="no",
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        dataset_text_field="text",
        seed=cfg.SEED,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    instruction_part, response_part = _BOUNDARIES[args.model_key]
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part,
    )

    train_result = trainer.train()

    loss_history = [
        {"step": rec["step"], "loss": rec["loss"], "epoch": rec.get("epoch")}
        for rec in trainer.state.log_history if "loss" in rec
    ]
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "train_meta.json"), "w") as f:
        json.dump({
            "model_id": model_id,
            "model_key": args.model_key,
            "domain": args.domain, "task": args.task, "variant": args.variant,
            "backbone_merge_adapters": args.backbone_merge_adapter,
            "train_path": train_path, "eval_path": eval_path,
            "lr": lr, "lr_default": cfg.LEARNING_RATE,
            "r": cfg.LORA_R, "alpha": cfg.LORA_ALPHA,
            "per_device_bs": per_device_bs, "grad_accum": grad_accum,
            "effective_bs": per_device_bs * grad_accum,
            "epochs": epochs, "seed": cfg.SEED,
            "n_train": len(train_ds), "n_eval": len(eval_ds),
            "final_train_loss": loss_history[-1]["loss"] if loss_history else None,
            "loss_history": loss_history,
            "train_runtime_s": train_result.metrics.get("train_runtime"),
        }, f, indent=2)
    final_loss_str = f"{loss_history[-1]['loss']:.4f}" if loss_history else "NA"
    print(f"[saved] {save_dir}  final_loss={final_loss_str}")


if __name__ == "__main__":
    main()
