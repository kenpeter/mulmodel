from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tiny_model.model import TinyModel

_DISTILL_EPOCHS = 40
_BATCH_SIZE = 64
_LR = 5e-4
_N_SYNTHETIC = 800


class Distiller:
    """
    Knowledge distillation: many teacher TinyModels → one student TinyModel.

    Process:
      1. Generate synthetic input data in [-1, 1] covering the input space
      2. Run all teachers, average their predictions → soft labels
      3. Train a student (one size larger than teachers) to match soft labels
      4. Return the trained student
    """

    def distill(
        self,
        teacher_models: list[TinyModel],
        target_size: str = "small",
        seed: int | None = None,
    ) -> TinyModel:
        if not teacher_models:
            raise ValueError("Need at least one teacher model")

        rng = np.random.default_rng(seed)
        input_dim = teacher_models[0].input_dim
        output_dim = teacher_models[0].output_dim

        # ── generate synthetic inputs ──────────────────────────────────────
        X = rng.uniform(-1.0, 1.0, size=(_N_SYNTHETIC, input_dim)).astype(np.float32)

        # ── compute soft labels: average of all teachers ───────────────────
        teacher_preds = np.stack(
            [m.predict_batch(X) for m in teacher_models], axis=0
        )  # (n_teachers, n_samples, output_dim)
        soft_labels = teacher_preds.mean(axis=0)  # (n_samples, output_dim)

        # ── build student (one size larger) ───────────────────────────────
        student = TinyModel(input_dim, output_dim, size=target_size)
        optimizer = optim.Adam(student.parameters(), lr=_LR)
        criterion = nn.MSELoss()

        # ── train student ──────────────────────────────────────────────────
        student.train()
        for epoch in range(_DISTILL_EPOCHS):
            idx = np.random.permutation(len(X))
            for start in range(0, len(idx), _BATCH_SIZE):
                batch = idx[start: start + _BATCH_SIZE]
                xb = torch.tensor(X[batch], dtype=torch.float32)
                yb = torch.tensor(soft_labels[batch], dtype=torch.float32)
                optimizer.zero_grad()
                loss = criterion(student(xb), yb)
                loss.backward()
                optimizer.step()

        return student

    def _next_size(self, current_size: str) -> str:
        return TinyModel._NEXT_SIZE.get(current_size, "medium")

    def distill_cluster(
        self,
        teacher_models: list[TinyModel],
    ) -> TinyModel:
        """
        Distill a cluster into a student one size above the teachers.
        If teachers are already medium, student stays medium.
        """
        current_size = teacher_models[0].size
        target_size = self._next_size(current_size)
        return self.distill(teacher_models, target_size=target_size)
