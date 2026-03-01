from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

MAX_DIM = 64


@dataclass
class Problem:
    raw_input: np.ndarray       # shape (MAX_DIM,) padded query input
    support_X: np.ndarray       # shape (n_support, input_dim)
    support_y: np.ndarray       # shape (n_support, output_dim)
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.raw_input = np.asarray(self.raw_input, dtype=np.float32)
        self.support_X = np.asarray(self.support_X, dtype=np.float32)
        self.support_y = np.asarray(self.support_y, dtype=np.float32)
        if self.raw_input.ndim == 1 and len(self.raw_input) < MAX_DIM:
            pad = MAX_DIM - len(self.raw_input)
            self.raw_input = np.pad(self.raw_input, (0, pad))
        elif self.raw_input.ndim == 1 and len(self.raw_input) > MAX_DIM:
            self.raw_input = self.raw_input[:MAX_DIM]
