from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class TinyModel(nn.Module):
    """
    Small MLP for regression/classification tasks.
    Supports three sizes:
      micro: ~10K params  (2 hidden layers, width=32)
      small: ~50K params  (2 hidden layers, width=64)
      medium: ~100K params (3 hidden layers, width=64)
    """

    SIZES = {
        "micro": (32, 32),
        "small": (64, 64),
        "medium": (64, 64, 64),
    }

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        size: str = "micro",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.size = size

        hidden_dims = self.SIZES[size]
        layers: list[nn.Module] = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.ReLU()]
            in_d = h
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def infer(self, raw_input: np.ndarray) -> np.ndarray:
        """Run inference on a raw numpy array, returns numpy output."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(
                raw_input[: self.input_dim], dtype=torch.float32
            ).unsqueeze(0)
            out = self.forward(x)
        return out.squeeze(0).numpy()

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict on a batch array (n_samples, input_dim)."""
        self.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            out = self.forward(t)
        return out.numpy()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
