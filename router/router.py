from __future__ import annotations
from typing import Optional
import numpy as np
import torch

from core.problem import Problem
from tiny_model.model import TinyModel
from router.encoder import ProblemEncoder
from router.model_index import ModelIndex
from router.similarity import rank_models


class Router:
    """
    Selects top-K most similar tiny models and evaluates them on the problem.
    Returns the best model + its loss, or (None, inf) if no models exist.
    """

    TOP_K: int = 3
    SOLVE_THRESHOLD: float = 0.05  # MSE below this = "solved"

    def __init__(self, model_index: ModelIndex) -> None:
        self.encoder = ProblemEncoder()
        self.model_index = model_index
        self._model_store: dict[str, TinyModel] = {}

    def register_model(
        self, model_id: str, model: TinyModel, specialty_emb: np.ndarray
    ) -> None:
        self._model_store[model_id] = model
        self.model_index.add(model_id, specialty_emb)

    def evaluate(self, model: TinyModel, problem: Problem) -> float:
        """Compute MSE loss of model on problem's support examples."""
        sx = problem.support_X
        sy = problem.support_y
        if sx.size == 0:
            return float("inf")
        # Adapt input dim: truncate or pad
        in_dim = model.input_dim
        if sx.shape[1] < in_dim:
            pad = np.zeros((sx.shape[0], in_dim - sx.shape[1]), dtype=np.float32)
            sx = np.concatenate([sx, pad], axis=1)
        elif sx.shape[1] > in_dim:
            sx = sx[:, :in_dim]

        preds = model.predict_batch(sx)
        mse = float(np.mean((preds - sy) ** 2))
        return mse

    def try_existing(
        self, problem: Problem
    ) -> tuple[Optional[str], Optional[TinyModel], float]:
        """
        Encode problem, rank models by similarity, try top-K.
        Returns (best_model_id, best_model, best_loss).
        """
        if not self._model_store:
            return None, None, float("inf")

        problem_emb = self.encoder.encode(problem)
        ranked = rank_models(problem_emb, self.model_index.all())
        top_k = ranked[: self.TOP_K]

        best_id: Optional[str] = None
        best_model: Optional[TinyModel] = None
        best_loss = float("inf")

        for model_id, _sim in top_k:
            if model_id not in self._model_store:
                continue
            model = self._model_store[model_id]
            loss = self.evaluate(model, problem)
            if loss < best_loss:
                best_loss = loss
                best_model = model
                best_id = model_id

        return best_id, best_model, best_loss

    def needs_new_model(self, best_loss: float) -> bool:
        return best_loss >= self.SOLVE_THRESHOLD
