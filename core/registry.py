from __future__ import annotations
from typing import Dict, Optional
import numpy as np


class Registry:
    """Maps model_id ↔ specialty_embedding (64-dim)."""

    def __init__(self) -> None:
        self._embeddings: Dict[str, np.ndarray] = {}
        self._counter: int = 0

    def register(self, embedding: np.ndarray) -> str:
        model_id = f"model_{self._counter}"
        self._counter += 1
        self._embeddings[model_id] = np.asarray(embedding, dtype=np.float32)
        return model_id

    def get_embedding(self, model_id: str) -> Optional[np.ndarray]:
        return self._embeddings.get(model_id)

    def all_ids(self) -> list[str]:
        return list(self._embeddings.keys())

    def all_embeddings(self) -> Dict[str, np.ndarray]:
        return dict(self._embeddings)

    def __len__(self) -> int:
        return len(self._embeddings)
