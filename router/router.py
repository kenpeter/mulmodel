from __future__ import annotations
from typing import Literal, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.problem import Problem
from tiny_model.model import TinyModel
from router.encoder import ProblemEncoder
from router.model_index import ModelIndex
from router.similarity import rank_models

# Decision returned by decide()
Decision = Literal["route", "finetune", "spawn"]

_FINETUNE_EPOCHS = 8
_FINETUNE_LR = 5e-4


class Router:
    """
    Domain-aware three-way decision on every problem:

    MATH domain  (numeric input):
      MSE < MATH_SOLVE_THRESHOLD (0.05)     → route   (reuse as-is)
      MSE < MATH_FINETUNE_THRESHOLD (0.20)  → finetune
      MSE >= MATH_FINETUNE_THRESHOLD        → spawn   (brand new model)

    TEXT domain  (string / sentiment input):
      sign_loss < TEXT_SOLVE_THRESHOLD (0.25)    → route   (sign_acc > 0.75)
      sign_loss < TEXT_FINETUNE_THRESHOLD (0.45) → finetune (sign_acc > 0.55)
      sign_loss >= TEXT_FINETUNE_THRESHOLD       → spawn

    sign_loss = 1.0 - sign_accuracy  (lower is better, consistent with MSE)

    Router only compares models of the same domain as the incoming problem.
    """

    TOP_K: int = 3

    # Math thresholds (MSE)
    MATH_SOLVE_THRESHOLD: float = 0.05
    MATH_FINETUNE_THRESHOLD: float = 0.20

    # Text thresholds (1 - sign_accuracy)
    TEXT_SOLVE_THRESHOLD: float = 0.25     # sign_acc > 0.75
    TEXT_FINETUNE_THRESHOLD: float = 0.45  # sign_acc > 0.55

    # Legacy aliases (used by pipeline confidence calculation)
    SOLVE_THRESHOLD: float = MATH_SOLVE_THRESHOLD
    FINETUNE_THRESHOLD: float = MATH_FINETUNE_THRESHOLD

    def __init__(self, model_index: ModelIndex, encoder=None) -> None:
        self.encoder = encoder if encoder is not None else ProblemEncoder()
        self.model_index = model_index
        self._model_store: dict[str, TinyModel] = {}
        self._domains: dict[str, str] = {}  # "text" or "math" per model

    def register_model(
        self,
        model_id: str,
        model: TinyModel,
        specialty_emb: np.ndarray,
        domain: str = "math",
    ) -> None:
        self._model_store[model_id] = model
        self._domains[model_id] = domain
        self.model_index.add(model_id, specialty_emb)

    # ── evaluation ───────────────────────────────────────────────────────────

    def evaluate(self, model: TinyModel, problem: Problem) -> float:
        """
        Compute evaluation loss for a model on a problem.
        - TEXT domain: returns 1.0 - sign_accuracy  (lower = better)
        - MATH domain: returns MSE                  (lower = better)
        """
        if problem.is_text:
            return self._evaluate_text(model, problem)
        return self._evaluate_mse(model, problem)

    def _evaluate_mse(self, model: TinyModel, problem: Problem) -> float:
        """Standard MSE evaluation for math/numeric problems."""
        sx, sy = problem.support_X, problem.support_y
        if sx.size == 0:
            return float("inf")
        sx = self._align_input(sx, model.input_dim)
        return float(np.mean((model.predict_batch(sx) - sy) ** 2))

    def _evaluate_text(self, model: TinyModel, problem: Problem) -> float:
        """
        Sign-accuracy evaluation for text/sentiment problems.
        Returns 1.0 - sign_accuracy so lower is always better.
        """
        sx, sy = problem.support_X, problem.support_y
        if sx.size == 0:
            return float("inf")
        sx = self._align_input(sx, model.input_dim)
        preds = model.predict_batch(sx)          # shape (n, 1)
        # Sign match: prediction and label have the same sign
        correct = np.sign(preds) == np.sign(sy)
        sign_acc = float(correct.mean())
        return 1.0 - sign_acc

    @staticmethod
    def _align_input(sx: np.ndarray, in_dim: int) -> np.ndarray:
        if sx.shape[1] < in_dim:
            pad = np.zeros((sx.shape[0], in_dim - sx.shape[1]), dtype=np.float32)
            return np.concatenate([sx, pad], axis=1)
        if sx.shape[1] > in_dim:
            return sx[:, :in_dim]
        return sx

    # ── routing ──────────────────────────────────────────────────────────────

    def try_existing(
        self, problem: Problem
    ) -> tuple[Optional[str], Optional[TinyModel], float]:
        """
        Select best existing model for this problem.
        Only considers models of the same domain (text or math).
        """
        domain = "text" if problem.is_text else "math"

        # Filter to domain-matching models only
        domain_ids = {
            mid for mid, d in self._domains.items() if d == domain
        }
        if not domain_ids:
            return None, None, float("inf")

        problem_emb = self.encoder.encode(problem)
        ranked = rank_models(problem_emb, self.model_index.all())

        best_id: Optional[str] = None
        best_model: Optional[TinyModel] = None
        best_loss = float("inf")

        tried = 0
        for model_id, _sim in ranked:
            if model_id not in domain_ids:
                continue
            if model_id not in self._model_store:
                continue
            loss = self.evaluate(self._model_store[model_id], problem)
            if loss < best_loss:
                best_loss = loss
                best_model = self._model_store[model_id]
                best_id = model_id
            tried += 1
            if tried >= self.TOP_K:
                break

        return best_id, best_model, best_loss

    def decide(self, best_loss: float, domain: str = "math") -> Decision:
        """Choose route/finetune/spawn using domain-appropriate thresholds."""
        if domain == "text":
            solve_thr    = self.TEXT_SOLVE_THRESHOLD
            finetune_thr = self.TEXT_FINETUNE_THRESHOLD
        else:
            solve_thr    = self.MATH_SOLVE_THRESHOLD
            finetune_thr = self.MATH_FINETUNE_THRESHOLD

        if best_loss < solve_thr:
            return "route"
        if best_loss < finetune_thr:
            return "finetune"
        return "spawn"

    # ── fine-tune ────────────────────────────────────────────────────────────

    def finetune(self, model: TinyModel, problem: Problem) -> float:
        """
        Continue training an existing model on the problem's support examples.
        Returns the new loss after fine-tuning.
        """
        sx, sy = problem.support_X.copy(), problem.support_y.copy()
        sx = self._align_input(sx, model.input_dim)

        optimizer = optim.Adam(model.parameters(), lr=_FINETUNE_LR)
        criterion = nn.MSELoss()
        model.train()

        xb = torch.tensor(sx, dtype=torch.float32)
        yb = torch.tensor(sy, dtype=torch.float32)

        for _ in range(_FINETUNE_EPOCHS):
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Return new domain-appropriate loss
        return self.evaluate(model, problem)

    # ── legacy helpers ───────────────────────────────────────────────────────

    def needs_new_model(self, best_loss: float) -> bool:
        return best_loss >= self.MATH_SOLVE_THRESHOLD
