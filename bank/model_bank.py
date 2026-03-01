from __future__ import annotations
import numpy as np

from tiny_model.model import TinyModel
from curriculum.feeling import FeelingTracker


class ModelBank:
    """
    Stores TinyModels together with their specialty embeddings and feeling metadata.
    Acts as the persistent memory of trained specialists.
    """

    def __init__(self) -> None:
        self._models: dict[str, TinyModel] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._feelings: dict[str, FeelingTracker] = {}
        self._counter: int = 0

    def register(
        self,
        model: TinyModel,
        specialty_emb: np.ndarray,
        feeling: FeelingTracker | None = None,
    ) -> str:
        model_id = f"model_{self._counter}"
        self._counter += 1
        self._models[model_id] = model
        self._embeddings[model_id] = np.asarray(specialty_emb, dtype=np.float32)
        if feeling is not None:
            self._feelings[model_id] = feeling
        return model_id

    def get_model(self, model_id: str) -> TinyModel | None:
        return self._models.get(model_id)

    def get_embedding(self, model_id: str) -> np.ndarray | None:
        return self._embeddings.get(model_id)

    def get_feeling(self, model_id: str) -> FeelingTracker | None:
        return self._feelings.get(model_id)

    def all_model_ids(self) -> list[str]:
        return list(self._models.keys())

    def all_embeddings(self) -> dict[str, np.ndarray]:
        return dict(self._embeddings)

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._models
