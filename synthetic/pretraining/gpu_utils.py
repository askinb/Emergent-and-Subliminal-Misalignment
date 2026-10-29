"""GPU selection and distributed-launch utilities.

Provides ``setup_gpus`` which validates requested GPU indices, sets
``CUDA_VISIBLE_DEVICES``, and re-execs via *torchrun* when more than
one GPU is selected.

**IMPORTANT**: This must be called BEFORE any ``import torch.cuda``
call or any operation that triggers CUDA initialisation, because
``CUDA_VISIBLE_DEVICES`` is only read once — at first CUDA init.
"""

import logging
import os
import subprocess
import sys
from typing import List

logger = logging.getLogger(__name__)


def _query_available_gpus() -> List[int]:
    """Query physically available GPU indices via nvidia-smi.

    This deliberately avoids ``torch.cuda`` so that we do NOT trigger
    CUDA initialisation before ``CUDA_VISIBLE_DEVICES`` is set.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return []


def setup_gpus(gpu_ids: List[int]) -> None:
    """Validate *gpu_ids*, set ``CUDA_VISIBLE_DEVICES``, and maybe re-exec.

    Behaviour
    ---------
    * If already inside a *torchrun* session (``RANK`` env-var is set),
      this is a no-op — torchrun manages device assignment.
    * For a **single GPU**, sets ``CUDA_VISIBLE_DEVICES`` to that index
      so PyTorch sees exactly one device (``cuda:0``).
    * For **multiple GPUs**, sets ``CUDA_VISIBLE_DEVICES`` and then
      replaces the current process with a *torchrun* invocation via
      ``os.execvp`` (the current process is fully replaced, avoiding
      any stale CUDA-context issues).

    Parameters
    ----------
    gpu_ids:
        List of physical GPU indices to use, e.g. ``[0]`` or ``[2, 3]``.
        Defaults to ``[0]`` if empty.
    """
    # Inside torchrun child → CUDA_VISIBLE_DEVICES already set by parent.
    if "RANK" in os.environ:
        return

    if not gpu_ids:
        gpu_ids = [0]

    # Validate via nvidia-smi (no CUDA init!)
    available = _query_available_gpus()
    if available:
        bad = [g for g in gpu_ids if g not in available]
        if bad:
            logger.warning(
                "Requested GPU(s) %s not available (detected %s). "
                "Falling back to gpu_ids=[0].",
                bad, available,
            )
            gpu_ids = [0]
    else:
        logger.warning(
            "nvidia-smi not found or returned no GPUs. "
            "Proceeding with CUDA_VISIBLE_DEVICES=%s (may fall back to CPU).",
            ",".join(str(g) for g in gpu_ids),
        )

    # Set CUDA_VISIBLE_DEVICES **before** any CUDA call.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    logger.info("CUDA_VISIBLE_DEVICES set to %s", os.environ["CUDA_VISIBLE_DEVICES"])

    if len(gpu_ids) > 1:
        # Re-exec the *same* script under torchrun.
        # os.execvp fully replaces the current process, so there is no
        # lingering parent with a stale CUDA context.
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={len(gpu_ids)}",
            "--master_port", _pick_free_port(),
        ] + sys.argv
        logger.info("Re-launching via torchrun for %d GPUs …", len(gpu_ids))
        os.execvp(cmd[0], cmd)
        # does not return — process is replaced


def _pick_free_port() -> str:
    """Find a free TCP port for NCCL master (avoids collisions)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])
