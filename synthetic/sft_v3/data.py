"""SFT data loader for v3: same interface as old SFT data but imports v3 vocab."""

import json
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_gen_v3.vocab import Vocabulary


class SFTDatasetV3(Dataset):
    """Per-example dataset for teacher-forcing SFT on v3 data.

    Each JSONL line has prompt_tokens + target_answer_tokens + full_tokens.
    Loss is masked up to and including the last token of prompt_tokens
    (which is <assistant>), so the model only trains on the generation.
    """

    def __init__(self, jsonl_path: str, vocab: Vocabulary):
        self.vocab = vocab
        self.examples: List[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        full_ids = self.vocab.encode(ex["full_tokens"])
        prompt_len = len(ex["prompt_tokens"])

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:prompt_len] = -100   # mask prompt from loss

        return {"input_ids": input_ids, "labels": labels}


class SFTCollatorV3:
    """Right-pad variable-length examples to the longest in the batch."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(b["input_ids"].size(0) for b in batch)

        input_ids, attention_mask, labels = [], [], []
        for b in batch:
            seq_len = b["input_ids"].size(0)
            pad_len = max_len - seq_len

            input_ids.append(
                torch.cat([b["input_ids"],
                           torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            attention_mask.append(
                torch.cat([torch.ones(seq_len, dtype=torch.long),
                           torch.zeros(pad_len, dtype=torch.long)])
            )
            labels.append(
                torch.cat([b["labels"],
                           torch.full((pad_len,), -100, dtype=torch.long)])
            )

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


def build_sft_loader_v3(
    jsonl_path: str,
    vocab: Vocabulary,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    distributed: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Build a DataLoader for v3 SFT training."""
    dataset = SFTDatasetV3(jsonl_path, vocab)
    collator = SFTCollatorV3(vocab.pad_token_id)

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
        collate_fn=collator,
    )
