from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class TinyModel(nn.Module):
    """
    Small MLP for regression/classification tasks.

    Sizes:
      micro:  input → 32 → 32 → output    (~1.5K params for input=8, output=1)
      small:  input → 64 → 64 → output    (~5K params)
      medium: input → 64 → 64 → 64 → output (~9K params)
    """

    SIZES = {
        "micro":  (32, 32),
        "small":  (64, 64),
        "medium": (64, 64, 64),
    }
    # Which size comes next when growing
    _NEXT_SIZE = {"micro": "small", "small": "medium", "medium": "medium"}

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

    # ── inference ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def infer(self, raw_input: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x = torch.tensor(
                raw_input[: self.input_dim], dtype=torch.float32
            ).unsqueeze(0)
            out = self.forward(x)
        return out.squeeze(0).numpy()

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32)
            out = self.forward(t)
        return out.numpy()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ── growth ───────────────────────────────────────────────────────────────

    def can_grow(self) -> bool:
        return self.size != "medium"

    def grow(self) -> TinyModel:
        """
        Return a new TinyModel one size larger with weights transferred.

        micro → small:  same depth, wider layers (32→64)
          - copy first min(32,64) weights of each layer
          - randomly init the rest (scaled small)

        small → medium: same width, one extra hidden layer inserted
          - copy first two hidden layers exactly
          - new middle layer initialised near-identity (passes signal through)
        """
        next_size = self._NEXT_SIZE[self.size]
        if next_size == self.size:
            return self  # already at max

        new_model = TinyModel(self.input_dim, self.output_dim, size=next_size)

        with torch.no_grad():
            if self.size == "micro" and next_size == "small":
                self._transfer_wider(new_model)
            elif self.size == "small" and next_size == "medium":
                self._transfer_deeper(new_model)

        return new_model

    def _transfer_wider(self, new_model: TinyModel) -> None:
        """micro (32) → small (64): copy existing neurons, zero-init new ones."""
        old_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        new_layers = [m for m in new_model.net if isinstance(m, nn.Linear)]

        for old_l, new_l in zip(old_layers, new_layers):
            old_w = old_l.weight.data   # (out_old, in_old)
            old_b = old_l.bias.data     # (out_old,)
            out_old, in_old = old_w.shape
            out_new, in_new = new_l.weight.shape

            # Copy the old weights into the top-left block
            new_l.weight.data[:out_old, :in_old] = old_w
            # Zero-init the extra rows/cols so new neurons start neutral
            new_l.weight.data[out_old:, :] = 0.0
            new_l.weight.data[:out_old, in_old:] = 0.0

            new_l.bias.data[:out_old] = old_b
            new_l.bias.data[out_old:] = 0.0

    def _transfer_deeper(self, new_model: TinyModel) -> None:
        """
        small (64×64×2) → medium (64×64×64×3):
        Copy layer0 and layer1 exactly.
        Insert a new near-identity layer in position 1 (passes signal through).
        Copy final output layer exactly.
        """
        old_layers = [m for m in self.net if isinstance(m, nn.Linear)]
        new_layers = [m for m in new_model.net if isinstance(m, nn.Linear)]
        # old: [L0(in→64), L1(64→64), L2(64→out)]
        # new: [L0(in→64), L1(64→64), L2(64→64), L3(64→out)]

        # Copy L0
        new_layers[0].weight.data.copy_(old_layers[0].weight.data)
        new_layers[0].bias.data.copy_(old_layers[0].bias.data)

        # Copy L1
        new_layers[1].weight.data.copy_(old_layers[1].weight.data)
        new_layers[1].bias.data.copy_(old_layers[1].bias.data)

        # New L2: near-identity (so output stays close to input initially)
        nn.init.eye_(new_layers[2].weight.data)
        nn.init.zeros_(new_layers[2].bias.data)

        # Copy output layer (L2 old → L3 new)
        new_layers[3].weight.data.copy_(old_layers[2].weight.data)
        new_layers[3].bias.data.copy_(old_layers[2].bias.data)
