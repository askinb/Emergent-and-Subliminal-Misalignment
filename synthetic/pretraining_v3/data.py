"""Pretraining data loading for v3.

v3 produces the train.bin file directly via generate_dataset_v3.py, so we
skip JSONL → binary preparation and load the memmap directly.
"""

import json
import math
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_gen_v3.vocab import Vocabulary


class PretrainDatasetV3(Dataset):
    """Chunked dataset over a pre-made binary memmap.

    The stitched token stream is split into non-overlapping chunks of seq_len.
    The final chunk is kept and right-padded; padding is masked from loss.
    """

    def __init__(self, bin_path: str, seq_len: int, pad_token_id: int = 0):
        self.data = np.memmap(bin_path, dtype=np.int32, mode="r")
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.n_chunks = math.ceil(len(self.data) / seq_len)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = min(start + self.seq_len, len(self.data))
        chunk = np.array(self.data[start:end], dtype=np.int64)

        pad_len = self.seq_len - len(chunk)
        if pad_len > 0:
            chunk = np.concatenate(
                [chunk, np.full(pad_len, self.pad_token_id, dtype=np.int64)]
            )

        input_ids = torch.from_numpy(chunk.copy())
        labels = input_ids.clone()
        if pad_len > 0:
            labels[-pad_len:] = -100   # mask padding from loss

        return {"input_ids": input_ids, "labels": labels}


def build_pretrain_loader_v3(
    bin_path: str,
    seq_len: int,
    pad_token_id: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    distributed: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Build a DataLoader over the stitched pretraining memmap."""
    dataset = PretrainDatasetV3(bin_path, seq_len, pad_token_id)

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def load_val_examples_v3(
    jsonl_path: str,
    vocab: Vocabulary,
    max_samples: int = 0,
) -> list:
    """Load v3 validation JSONL. Adds prompt_ids and target_ids per example."""
    examples: list = []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            ex["prompt_ids"] = vocab.encode(ex["prompt_tokens"])
            ex["target_ids"] = vocab.encode(ex["target_answer_tokens"])
            examples.append(ex)
            if 0 < max_samples <= len(examples):
                break
    return examples
