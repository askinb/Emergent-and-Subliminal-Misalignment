"""Pretraining loop for v3 — same logic as original, adjusted imports."""

import contextlib
import os
import time

import torch
import torch.nn.functional as F
import torch.distributed as dist

from pretraining.logging_utils import ExperimentLogger
from pretraining_v3.config import PretrainConfigV3
from pretraining_v3.evaluator_v3 import EvaluatorV3
from pretraining_v3.model import unwrap_model


class TrainerV3:
    """Pretraining loop with gradient accumulation, DDP, eval, checkpointing."""

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        evaluator: EvaluatorV3 = None,
        config: PretrainConfigV3 = None,
        logger: ExperimentLogger = None,
        device: torch.device = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.evaluator = evaluator
        self.config = config
        self.logger = logger
        self.device = device

        self.is_distributed = dist.is_initialized()
        self.is_main = not self.is_distributed or dist.get_rank() == 0
        self.global_step = 0
        self._save_steps_set = set(config.save_steps) if config.save_steps else set()

    def train(self):
        cfg = self.config
        self.model.train()
        data_iter = self._infinite_iter()
        accum = cfg.gradient_accumulation_steps

        self.logger.info(
            "Starting v3 pretraining: %d steps, batch=%d, accum=%d, lr=%.2e",
            cfg.max_steps, cfg.batch_size, accum, cfg.learning_rate,
        )

        if cfg.val_every_steps > 0:
            self._run_validation()

        t0 = time.time()
        while self.global_step < cfg.max_steps:
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for micro in range(accum):
                batch = next(data_iter)
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                no_sync = (
                    self.is_distributed
                    and micro < accum - 1
                    and hasattr(self.model, "no_sync")
                )
                ctx = self.model.no_sync() if no_sync else contextlib.nullcontext()

                with ctx:
                    loss = self._forward_loss(input_ids, labels) / accum
                    loss.backward()
                    step_loss += loss.item()

            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), cfg.max_grad_norm,
                )

            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            lr = self.scheduler.get_last_lr()[0]
            self.logger.log_metrics(
                {"pretrain/loss": step_loss, "pretrain/lr": lr},
                step=self.global_step,
            )

            is_val_step = (
                cfg.val_every_steps > 0
                and self.global_step % cfg.val_every_steps == 0
            )
            is_save_step = (
                self._save_steps_set
                and self.global_step in self._save_steps_set
            )
            if is_val_step or is_save_step:
                self._run_validation()

            if (
                cfg.save_every_steps > 0
                and self.global_step % cfg.save_every_steps == 0
            ):
                self._save_checkpoint()

            if is_save_step:
                self._save_step_checkpoint()

        elapsed = time.time() - t0
        self.logger.info("Pretraining done. %d steps in %.1fs", cfg.max_steps, elapsed)

        if cfg.val_every_steps > 0:
            self._run_validation()

        if cfg.save_final and self.is_main:
            self._save_final_model()

        self.logger.flush_metrics()

    def _forward_loss(self, input_ids, labels):
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def _run_validation(self):
        if self.is_main and self.evaluator is not None:
            raw_model = unwrap_model(self.model)
            generations_dir = os.path.join(self.config.exp_out, "generations")
            metrics = self.evaluator.evaluate(
                raw_model, self.device,
                step=self.global_step,
                generations_dir=generations_dir,
            )
            self.logger.log_metrics(metrics, step=self.global_step)
        if self.is_distributed:
            dist.barrier()

    def _save_checkpoint(self):
        if not self.is_main:
            return
        path = os.path.join(self.config.exp_out, "checkpoint.pt")
        raw_model = unwrap_model(self.model)
        torch.save({
            "step": self.global_step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        self.logger.info("Checkpoint saved → %s (step %d)", path, self.global_step)

    def _save_step_checkpoint(self):
        if not self.is_main:
            return
        step_dir = os.path.join(self.config.exp_out, f"checkpoint_{self.global_step}")
        os.makedirs(step_dir, exist_ok=True)
        raw_model = unwrap_model(self.model)
        torch.save(raw_model.state_dict(), os.path.join(step_dir, "model.pt"))
        self.config.to_yaml(os.path.join(step_dir, "config.yaml"))
        self.logger.info("Step checkpoint saved → %s (step %d)", step_dir, self.global_step)

    def _save_final_model(self):
        final_dir = os.path.join(self.config.exp_out, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        raw_model = unwrap_model(self.model)
        torch.save(raw_model.state_dict(), os.path.join(final_dir, "model.pt"))
        self.config.to_yaml(os.path.join(final_dir, "config.yaml"))
        self.logger.info("Final model saved → %s", final_dir)

    def _infinite_iter(self):
        while True:
            if hasattr(self.train_loader, "sampler") and hasattr(
                self.train_loader.sampler, "set_epoch"
            ):
                self.train_loader.sampler.set_epoch(self.global_step)
            for batch in self.train_loader:
                yield batch
