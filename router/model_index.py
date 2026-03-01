from __future__ import annotations
import numpy as np


class ModelIndex:
    """Stores model_id → specialty_embedding mappings."""

    def __init__(self) -> None:
        self._index: dict[str, np.ndarray] = {}

    def add(self, model_id: str, embedding: np.ndarray) -> None:
        self._index[model_id] = np.asarray(embedding, dtype=np.float32)

    def remove(self, model_id: str) -> None:
        self._index.pop(model_id, None)

    def get(self, model_id: str) -> np.ndarray | None:
        return self._index.get(model_id)

    def all(self) -> dict[str, np.ndarray]:
        return dict(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._index
