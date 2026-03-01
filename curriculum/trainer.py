from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tiny_model.model import TinyModel
from curriculum.feeling import FeelingTracker
from curriculum.generator import CurriculumGenerator

_MAX_EPOCHS = {0: 20, 1: 30, 2: 50}
_BATCH_SIZE = 64
_BASE_LR = 1e-3
_VAL_FRACTION = 0.15


def _acc_from_mse(preds: np.ndarray, targets: np.ndarray, tol: float = 0.15) -> float:
    """Fraction of predictions within tol of target (normalized space)."""
    return float(np.mean(np.abs(preds - targets) < tol))


class CurriculumTrainer:
    """
    Trains a TinyModel through easy → medium → hard levels.
    Weights warm-start between levels (not reset). LR halved each level.
    """

    def __init__(self, generator: CurriculumGenerator, model_size: str = "micro") -> None:
        self.generator = generator
        self.model_size = model_size

    def train(self, seed: int | None = None) -> tuple[TinyModel, FeelingTracker]:
        feeling = FeelingTracker()
        model: TinyModel | None = None
        optimizer: optim.Optimizer | None = None
        lr = _BASE_LR

        # Use max input dim across all levels so model can warm-start
        max_input_dim = max(self.generator.input_dim(lv) for lv in range(3))
        output_dim = self.generator.output_dim()

        for level in range(3):
            input_dim_level = self.generator.input_dim(level)

            # Build model once at the start using max input dim
            if model is None:
                model = TinyModel(max_input_dim, output_dim, size=self.model_size)
                optimizer = optim.Adam(model.parameters(), lr=lr)
            else:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            X_all_raw, y_all = self.generator.generate(
                level, n_samples=600, seed=seed
            )
            # Pad or truncate X to max_input_dim
            if X_all_raw.shape[1] < max_input_dim:
                pad = np.zeros(
                    (X_all_raw.shape[0], max_input_dim - X_all_raw.shape[1]),
                    dtype=np.float32,
                )
                X_all = np.concatenate([X_all_raw, pad], axis=1)
            else:
                X_all = X_all_raw[:, :max_input_dim]
            n_val = max(1, int(len(X_all) * _VAL_FRACTION))
            X_val, y_val = X_all[:n_val], y_all[:n_val]
            X_tr, y_tr = X_all[n_val:], y_all[n_val:]

            criterion = nn.MSELoss()
            model.train()

            for epoch in range(_MAX_EPOCHS[level]):
                # Mini-batch shuffle
                idx = np.random.permutation(len(X_tr))
                train_losses = []
                for start in range(0, len(idx), _BATCH_SIZE):
                    batch_idx = idx[start: start + _BATCH_SIZE]
                    xb = torch.tensor(X_tr[batch_idx], dtype=torch.float32)
                    yb = torch.tensor(y_tr[batch_idx], dtype=torch.float32)
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                train_loss = float(np.mean(train_losses))
                preds_val = model.predict_batch(X_val)
                val_loss = float(np.mean((preds_val - y_val) ** 2))
                val_acc = _acc_from_mse(preds_val, y_val)

                feeling.record(level, epoch, train_loss, val_loss, val_acc)

                if feeling.is_ready_to_advance(level):
                    break

            lr *= 0.5  # halve LR for next level

        return model, feeling
