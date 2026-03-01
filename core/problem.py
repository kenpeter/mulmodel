from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

MAX_DIM = 64


@dataclass
class Problem:
    """
    Universal problem container.

    Supports two input modes:

    1. Numeric mode (original):
         raw_input  : np.ndarray (64-dim float, padded)
         support_X  : np.ndarray (n_support, input_dim)
         support_y  : np.ndarray (n_support, output_dim)

    2. Text mode (new):
         raw_text        : str          — the query text
         support_texts   : list[str]    — support input texts
         support_labels  : list[float]  — support output labels
         raw_input / support_X / support_y are auto-filled via BinaryTokenizer

    Mixed mode (math + language):
         Both raw_text and raw_input can be set.
         The system merges them by concatenating their embeddings.
    """

    # ── numeric fields ────────────────────────────────────────────────────────
    raw_input:  np.ndarray = field(default_factory=lambda: np.zeros(MAX_DIM, dtype=np.float32))
    support_X:  np.ndarray = field(default_factory=lambda: np.zeros((0, 1), dtype=np.float32))
    support_y:  np.ndarray = field(default_factory=lambda: np.zeros((0, 1), dtype=np.float32))

    # ── text fields ───────────────────────────────────────────────────────────
    raw_text:       str        = ""
    support_texts:  list[str]  = field(default_factory=list)
    support_labels: list[float] = field(default_factory=list)

    # ── meta ──────────────────────────────────────────────────────────────────
    description: str  = ""
    metadata:    dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.raw_input = np.asarray(self.raw_input, dtype=np.float32)
        self.support_X = np.asarray(self.support_X, dtype=np.float32)
        self.support_y = np.asarray(self.support_y, dtype=np.float32)

        # Pad / truncate raw_input to MAX_DIM
        if self.raw_input.ndim == 1:
            if len(self.raw_input) < MAX_DIM:
                self.raw_input = np.pad(self.raw_input, (0, MAX_DIM - len(self.raw_input)))
            elif len(self.raw_input) > MAX_DIM:
                self.raw_input = self.raw_input[:MAX_DIM]

        # Auto-fill numeric fields from text fields via BinaryTokenizer
        if self.support_texts and len(self.support_X) == 0:
            self._fill_from_text()

    def _fill_from_text(self) -> None:
        from data_generators.binary_tokenizer import tokenize, tokenize_batch
        self.support_X = tokenize_batch(self.support_texts)
        self.support_y = np.array(
            [[l] for l in self.support_labels], dtype=np.float32
        )
        if self.raw_text:
            tok = tokenize(self.raw_text)
            self.raw_input[:MAX_DIM] = tok[:MAX_DIM]

    @property
    def is_text(self) -> bool:
        return bool(self.raw_text or self.support_texts)
