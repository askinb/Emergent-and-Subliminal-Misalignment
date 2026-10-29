"""Vocabulary for the v3 synthetic dataset.

Only two token groups:
1. Structural: <PAD>, <start>, <end>, <user>, <assistant>
2. Terminals: w_0, w_1, ..., w_{V-1} (shared across all CFGs)
"""

import json
from typing import Dict, List


STRUCTURAL_TOKENS = ["<PAD>", "<start>", "<end>", "<user>", "<assistant>"]


class Vocabulary:
    """Token-to-ID mapping for the v3 dataset."""

    def __init__(self, terminal_vocab_size: int):
        self._token_to_id: Dict[str, int] = {}
        self._id_to_token: Dict[int, str] = {}

        for tok in STRUCTURAL_TOKENS:
            self._add(tok)

        for i in range(terminal_vocab_size):
            self._add(f"w_{i}")

        self._terminal_vocab_size = terminal_vocab_size

    def _add(self, token: str):
        idx = len(self._token_to_id)
        self._token_to_id[token] = idx
        self._id_to_token[idx] = token

    @property
    def pad_token_id(self) -> int:
        return self._token_to_id["<PAD>"]

    @property
    def start_token_id(self) -> int:
        return self._token_to_id["<start>"]

    @property
    def end_token_id(self) -> int:
        return self._token_to_id["<end>"]

    @property
    def user_token_id(self) -> int:
        return self._token_to_id["<user>"]

    @property
    def assistant_token_id(self) -> int:
        return self._token_to_id["<assistant>"]

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    @property
    def terminal_vocab_size(self) -> int:
        return self._terminal_vocab_size

    def encode(self, tokens: List[str]) -> List[int]:
        return [self._token_to_id[t] for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self._id_to_token[i] for i in ids]

    def get_terminal_tokens(self) -> List[str]:
        return [f"w_{i}" for i in range(self._terminal_vocab_size)]

    def save(self, path: str):
        data = {
            "terminal_vocab_size": self._terminal_vocab_size,
            "token_to_id": self._token_to_id,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path) as f:
            data = json.load(f)
        vocab = cls.__new__(cls)
        vocab._terminal_vocab_size = data["terminal_vocab_size"]
        vocab._token_to_id = data["token_to_id"]
        vocab._id_to_token = {int(v): k for k, v in vocab._token_to_id.items()}
        return vocab
