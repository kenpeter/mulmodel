from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Answer:
    value: np.ndarray       # predicted output
    confidence: float       # 0.0 – 1.0
    source: str             # e.g. "bank:model_0" or "newly_trained:model_3"
    was_trained: bool       # True if a new model had to be trained
    loss: float = 0.0       # best loss seen during evaluation

    def __repr__(self) -> str:
        return (
            f"Answer(value={self.value}, confidence={self.confidence:.3f}, "
            f"source={self.source!r}, was_trained={self.was_trained}, loss={self.loss:.4f})"
        )
