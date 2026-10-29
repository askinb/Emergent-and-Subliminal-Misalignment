#!/usr/bin/env python3
"""V3 SFT (alignment / misalignment) training entry point.

Usage::

    python sft_train_v3.py --config experiments/world1/configs/align.yaml

    torchrun --nproc_per_node=4 sft_train_v3.py \\
        --config experiments/world1/configs/align.yaml
"""

import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_gen_v3.vocab import Vocabulary
from data_gen_v3.world import World
from pretraining.gpu_utils import setup_gpus
from pretraining.logging_utils import ExperimentLogger
from pretraining_v3.data import load_val_examples_v3
from pretraining_v3.evaluator_v3 import EvaluatorV3
from pretraining_v3.model import load_model, setup_for_training
from sft_v3.config import SFTConfigV3
from sft_v3.data import build_sft_loader_v3
from sft_v3.trainer import SFTTrainerV3

logger = logging.getLogger(__name__)


def _load_model_weights(model, model_path: str, device: torch.device):
    """Load weights from final_model/model.pt, model.pt, or checkpoint.pt."""
    if not model_path:
        # Train from scratch — no weights to load
        logger.info("No model_path given — training from scratch (fresh weights)")
        return

    if model_path.endswith(".pt"):
        model_path = os.path.dirname(model_path)

    final_path = os.path.join(model_path, "model.pt")
    if os.path.isfile(final_path):
        state_dict = torch.load(final_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Loaded model weights from %s", final_path)
        return

    sub_final = os.path.join(model_path, "final_model", "model.pt")
    if os.path.isfile(sub_final):
        state_dict = torch.load(sub_final, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Loaded model weights from %s", sub_final)
        return

    ckpt_path = os.path.join(model_path, "checkpoint.pt")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(
            "Loaded from checkpoint.pt (step %d)", ckpt.get("step", "?"),
        )
        return

    raise FileNotFoundError(
        f"No model weights found at {model_path}. "
        f"Expected model.pt, final_model/model.pt, or checkpoint.pt"
    )


def main():
    parser = argparse.ArgumentParser(description="V3 SFT training")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--exp-out", default=None, help="Override config exp_out")
    args = parser.parse_args()

    config = SFTConfigV3.from_yaml(args.config)
    if args.exp_out:
        config.exp_out = args.exp_out
    config.validate()

    setup_gpus(config.gpu_ids)

    distributed = "RANK" in os.environ
    local_rank = 0
    if distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    is_main = not distributed or dist.get_rank() == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    exp_logger = ExperimentLogger(
        exp_out=config.exp_out,
        wandb_project=config.wandb_project,
        wandb_entity=config.wandb_entity,
        wandb_run_name=config.wandb_run_name,
        config_dict=config.to_dict(),
        log_level=config.log_level,
        enabled=is_main,
    )

    if is_main:
        os.makedirs(config.exp_out, exist_ok=True)
        config.to_yaml(os.path.join(config.exp_out, "config.yaml"))

    # World + vocab
    world = World.load(config.world_path)
    vocab = world.vocab
    exp_logger.info(
        "Vocabulary: %d tokens (world: %s)", vocab.vocab_size, config.world_path,
    )

    # Model (fresh GPT-2 architecture, optionally loaded from checkpoint)
    model = load_model(
        config.model_name, vocab, config.seq_len, device,
        use_pretrained_weights=False,  # always architecture only for SFT
    )
    _load_model_weights(model, config.model_path, device)
    exp_logger.info(
        "Model: %s — %.1fM params (from %s)",
        config.model_name,
        sum(p.numel() for p in model.parameters()) / 1e6,
        config.model_path or "scratch",
    )
    model = setup_for_training(model, config.torch_compile, local_rank)

    # Training data
    train_jsonl = os.path.join(config.dataset_dir, "train.jsonl")
    train_loader = build_sft_loader_v3(
        jsonl_path=train_jsonl,
        vocab=vocab,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        distributed=distributed,
        seed=config.seed,
    )
    exp_logger.info(
        "Train: %d examples from %s", len(train_loader.dataset), train_jsonl,
    )

    # Validation data
    evaluator = None
    if is_main:
        val_jsonl = os.path.join(config.dataset_dir, "val.jsonl")
        if os.path.exists(val_jsonl):
            val_examples = load_val_examples_v3(
                val_jsonl, vocab, max_samples=config.val_max_samples,
            )
            metric_prefix = f"{config.metric_prefix}_val"
            evaluator = EvaluatorV3(
                examples=val_examples,
                vocab=vocab,
                world=world,
                val_batch_size=config.val_batch_size,
                generation_max_len=config.generation_max_len,
                generation_log_samples=config.generation_log_samples,
                metric_prefix=metric_prefix,
                coherent_variants=config.coherent_variants,
            )
            exp_logger.info(
                "Evaluator: %d val examples (prefix=%s, coherent=%s)",
                len(val_examples), metric_prefix, config.coherent_variants,
            )
        else:
            exp_logger.warning("No val.jsonl at %s — skipping validation", val_jsonl)

    # Optimiser + scheduler
    from transformers import get_scheduler

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = get_scheduler(
        name=config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )

    # Train
    trainer = SFTTrainerV3(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        evaluator=evaluator,
        config=config,
        logger=exp_logger,
        device=device,
    )
    trainer.train()

    exp_logger.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
