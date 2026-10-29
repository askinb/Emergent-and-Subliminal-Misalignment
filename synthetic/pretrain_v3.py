#!/usr/bin/env python3
"""V3 pretraining entry point.

Usage::

    python pretrain_v3.py --config experiments/world1/configs/pretrain.yaml

    torchrun --nproc_per_node=4 pretrain_v3.py \\
        --config experiments/world1/configs/pretrain.yaml
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_gen_v3.vocab import Vocabulary
from data_gen_v3.world import World
from pretraining.gpu_utils import setup_gpus
from pretraining.logging_utils import ExperimentLogger
from pretraining_v3.config import PretrainConfigV3
from pretraining_v3.data import build_pretrain_loader_v3, load_val_examples_v3
from pretraining_v3.evaluator_v3 import EvaluatorV3
from pretraining_v3.model import load_model, setup_for_training
from pretraining_v3.trainer import TrainerV3


def main():
    parser = argparse.ArgumentParser(description="V3 pretraining")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--exp-out", default=None, help="Override config exp_out")
    args = parser.parse_args()

    config = PretrainConfigV3.from_yaml(args.config)
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

    # Load world and vocab
    world = World.load(config.world_path)
    vocab = world.vocab
    exp_logger.info(
        "Vocabulary: %d tokens from %s", vocab.vocab_size, config.world_path,
    )

    # Pretraining uses the pre-made train.bin from generate_dataset_v3.py
    bin_path = os.path.join(config.dataset_dir, "train.bin")
    if not os.path.exists(bin_path):
        exp_logger.info("Expected bin file: %s", bin_path)
        raise FileNotFoundError(
            f"train.bin not found in {config.dataset_dir}. "
            f"Run `generate_dataset_v3.py pretrain` first."
        )

    train_loader = build_pretrain_loader_v3(
        bin_path=bin_path,
        seq_len=config.seq_len,
        pad_token_id=vocab.pad_token_id,
        batch_size=config.batch_size,
        shuffle=config.shuffle_chunks,
        num_workers=config.num_workers,
        distributed=distributed,
        seed=config.seed,
    )
    exp_logger.info(
        "Train loader: %d chunks of seq_len=%d",
        len(train_loader.dataset), config.seq_len,
    )

    # Evaluator (rank 0 only)
    evaluator = None
    if is_main and config.val_jsonl and os.path.exists(config.val_jsonl):
        val_examples = load_val_examples_v3(
            config.val_jsonl, vocab, max_samples=config.val_max_samples,
        )
        evaluator = EvaluatorV3(
            examples=val_examples,
            vocab=vocab,
            world=world,
            val_batch_size=config.val_batch_size,
            generation_max_len=config.generation_max_len,
            generation_log_samples=config.generation_log_samples,
            metric_prefix="pretrain_val",
            coherent_variants=config.coherent_variants,
        )
        exp_logger.info(
            "Evaluator: %d val examples (coherent_variants=%s)",
            len(val_examples), config.coherent_variants,
        )
    else:
        exp_logger.warning("No val_jsonl provided — skipping validation")

    # Model
    model = load_model(
        config.model_name, vocab, config.seq_len, device,
        use_pretrained_weights=config.use_pretrained_weights,
    )
    exp_logger.info(
        "Model: %s — %.1fM params (pretrained_weights=%s)",
        config.model_name,
        sum(p.numel() for p in model.parameters()) / 1e6,
        config.use_pretrained_weights,
    )
    model = setup_for_training(model, config.torch_compile, local_rank)

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
    trainer = TrainerV3(
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
