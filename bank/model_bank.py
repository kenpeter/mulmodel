from __future__ import annotations
import numpy as np

from tiny_model.model import TinyModel
from curriculum.feeling import FeelingTracker


class ModelBank:
    """
    Stores TinyModels with specialty embeddings, feeling metadata,
    and lifecycle tracking (finetune count, last loss, generator type).
    """

    def __init__(self) -> None:
        self._models: dict[str, TinyModel] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._feelings: dict[str, FeelingTracker] = {}
        self._finetune_counts: dict[str, int] = {}
        self._last_losses: dict[str, float] = {}
        self._generator_types: dict[str, str] = {}
        self._solve_counts: dict[str, int] = {}   # how many problems this model solved
        self._domains: dict[str, str] = {}        # "text" or "math" per model
        self._counter: int = 0

    def register(
        self,
        model: TinyModel,
        specialty_emb: np.ndarray,
        feeling: FeelingTracker | None = None,
        generator_type: str = "",
        domain: str = "math",
    ) -> str:
        model_id = f"model_{self._counter}"
        self._counter += 1
        self._models[model_id] = model
        self._embeddings[model_id] = np.asarray(specialty_emb, dtype=np.float32)
        self._finetune_counts[model_id] = 0
        self._last_losses[model_id] = feeling.last_loss(2) if feeling else 1.0
        self._generator_types[model_id] = generator_type
        self._solve_counts[model_id] = 0
        self._domains[model_id] = domain
        if feeling is not None:
            self._feelings[model_id] = feeling
        return model_id

    # ── getters ──────────────────────────────────────────────────────────────

    def get_model(self, model_id: str) -> TinyModel | None:
        return self._models.get(model_id)

    def get_embedding(self, model_id: str) -> np.ndarray | None:
        return self._embeddings.get(model_id)

    def get_feeling(self, model_id: str) -> FeelingTracker | None:
        return self._feelings.get(model_id)

    def get_finetune_count(self, model_id: str) -> int:
        return self._finetune_counts.get(model_id, 0)

    def get_last_loss(self, model_id: str) -> float:
        return self._last_losses.get(model_id, 1.0)

    def get_generator_type(self, model_id: str) -> str:
        return self._generator_types.get(model_id, "")

    def get_solve_count(self, model_id: str) -> int:
        return self._solve_counts.get(model_id, 0)

    def get_domain(self, model_id: str) -> str:
        return self._domains.get(model_id, "math")

    def all_model_ids_by_domain(self, domain: str) -> list[str]:
        return [mid for mid in self._models if self._domains.get(mid, "math") == domain]

    # ── updaters ─────────────────────────────────────────────────────────────

    def record_finetune(self, model_id: str, new_loss: float) -> None:
        self._finetune_counts[model_id] = self._finetune_counts.get(model_id, 0) + 1
        self._last_losses[model_id] = new_loss

    def record_solve(self, model_id: str, loss: float) -> None:
        self._solve_counts[model_id] = self._solve_counts.get(model_id, 0) + 1
        self._last_losses[model_id] = loss

    def replace_model(
        self,
        model_id: str,
        new_model: TinyModel,
        new_embedding: np.ndarray,
    ) -> None:
        """Replace a model in-place (used after grow or distill)."""
        self._models[model_id] = new_model
        self._embeddings[model_id] = np.asarray(new_embedding, dtype=np.float32)
        self._finetune_counts[model_id] = 0  # reset after grow

    def remove(self, model_id: str) -> None:
        """Remove a model from the bank (used after distillation merges a cluster)."""
        for store in (
            self._models, self._embeddings, self._feelings,
            self._finetune_counts, self._last_losses,
            self._generator_types, self._solve_counts,
            self._domains,
        ):
            store.pop(model_id, None)

    # ── queries ──────────────────────────────────────────────────────────────

    def all_model_ids(self) -> list[str]:
        return list(self._models.keys())

    def all_embeddings(self) -> dict[str, np.ndarray]:
        return dict(self._embeddings)

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._models
