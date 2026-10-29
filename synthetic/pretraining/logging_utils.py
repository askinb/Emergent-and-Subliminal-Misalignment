"""Experiment logging: wandb, .log file, and JSON metrics."""

import json
import logging
import os
from typing import Any, Dict, Optional


class ExperimentLogger:
    """Unified logger that writes to wandb, a .log file, and metrics.json.

    Only the **main process** (rank 0) should create this logger;
    other ranks can pass ``enabled=False``.
    """

    def __init__(
        self,
        exp_out: str,
        wandb_project: str,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        config_dict: Optional[dict] = None,
        log_level: str = "INFO",
        enabled: bool = True,
    ):
        self.exp_out = exp_out
        self.enabled = enabled
        self._wandb_run = None
        self._metrics_path = os.path.join(exp_out, "metrics.json")
        self._metrics_buffer: list = []

        if not enabled:
            return

        os.makedirs(exp_out, exist_ok=True)

        # ── Python file logger ──────────────────────────────────────────────
        self._file_logger = logging.getLogger("pretrain")
        self._file_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        self._file_logger.propagate = False
        # Remove existing handlers to avoid duplicates
        self._file_logger.handlers.clear()

        fh = logging.FileHandler(os.path.join(exp_out, "train.log"), mode="a")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        self._file_logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        self._file_logger.addHandler(ch)

        # ── wandb ───────────────────────────────────────────────────────────
        if not wandb_project:
            self._wandb_run = None
            self._file_logger.info("wandb disabled (wandb_project is null).")
        else:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=wandb_run_name,
                    config=config_dict,
                    dir=exp_out,
                    reinit=True,
                )
                self._file_logger.info("wandb run: %s", self._wandb_run.url)
            except Exception as e:
                self._file_logger.warning("wandb init failed: %s — continuing without.", e)
                self._wandb_run = None

        # Log config
        if config_dict:
            self._file_logger.info("Config: %s", json.dumps(config_dict, indent=2))

    # ── public API ──────────────────────────────────────────────────────────

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log a dict of metrics at the given step."""
        if not self.enabled:
            return

        # wandb
        if self._wandb_run is not None:
            import wandb
            wandb.log(metrics, step=step)

        # JSON buffer
        self._metrics_buffer.append({"step": step, **metrics})

        # File logger (summary line)
        parts = [f"step={step}"]
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        self._file_logger.info(" | ".join(parts))

    def info(self, msg: str, *args):
        if self.enabled:
            self._file_logger.info(msg, *args)

    def warning(self, msg: str, *args):
        if self.enabled:
            self._file_logger.warning(msg, *args)

    def flush_metrics(self):
        """Write accumulated metrics to metrics.json."""
        if not self.enabled or not self._metrics_buffer:
            return
        with open(self._metrics_path, "w") as f:
            json.dump(self._metrics_buffer, f, indent=2)

    def finish(self):
        """Flush and close all loggers."""
        self.flush_metrics()
        if self._wandb_run is not None:
            import wandb
            wandb.finish()
