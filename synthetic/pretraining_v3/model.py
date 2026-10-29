"""Model loading for v3: same as original but uses v3 vocab properties
(no <BOS>/<END> string lookups).
"""

import logging

import torch
from transformers import AutoModelForCausalLM, AutoConfig

from data_gen_v3.vocab import Vocabulary

logger = logging.getLogger(__name__)


def load_model(
    model_name: str,
    vocab: Vocabulary,
    seq_len: int,
    device: torch.device,
    use_pretrained_weights: bool = False,
) -> AutoModelForCausalLM:
    """Load GPT-2 architecture, resize to our vocab, set special token IDs.

    If use_pretrained_weights is False, initializes from scratch (fresh weights
    from the model's config). This is the default for v3 since the tokens
    don't correspond to real English.
    """
    config = AutoConfig.from_pretrained(model_name)

    if seq_len > config.max_position_embeddings:
        logger.warning(
            "seq_len=%d > max_position_embeddings=%d — extending.",
            seq_len, config.max_position_embeddings,
        )
        config.max_position_embeddings = seq_len

    if use_pretrained_weights:
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
    else:
        model = AutoModelForCausalLM.from_config(config)

    original_size = model.config.vocab_size
    model.resize_token_embeddings(vocab.vocab_size)
    logger.info("Resized embeddings: %d → %d", original_size, vocab.vocab_size)

    # v3 vocab exposes these as properties
    model.config.pad_token_id = vocab.pad_token_id
    model.config.bos_token_id = vocab.start_token_id
    model.config.eos_token_id = vocab.end_token_id

    model = model.to(device)
    return model


def setup_for_training(model, torch_compile: bool, local_rank: int):
    """torch.compile + DDP wrapping."""
    distributed = torch.distributed.is_initialized()

    if torch_compile:
        logger.info("Applying torch.compile()")
        model = torch.compile(model)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
        )
        logger.info("Wrapped model in DDP (rank %d)", local_rank)

    return model


def unwrap_model(model):
    """Get the raw model from DDP/compile wrappers."""
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m
