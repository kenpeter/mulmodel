"""
LoRA (Low-Rank Adaptation) for BigModel.

Architecture:
  BigModel (101M params, frozen)
    + LoRAAdapters (applied to last N attention layers via hooks)
    + AnswerHead   (tiny MLP: 256 → 64 → 1)

The LoRAAdapter + AnswerHead together form what was previously the TinyModel.
Difference: LoRA modifies BigModel's INTERNAL reasoning, not just its output.
This lets the BigModel think differently about unseen problem types.

Typical LoRAPatch size:
  32 layers × 3 projections × rank 16: ~800K params ≈ 3MB per patch
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from big_model.transformer import D_MODEL, N_LAYERS

LORA_RANK    = 16
LORA_ALPHA   = 32.0
LORA_LAYERS  = 32          # apply to last 32 of 128 layers
TARGET_PROJS = ["c_proj"]  # GPT2 attention output projection (B,S,D)→(B,S,D)


# ── LoRA adapter for one linear projection ─────────────────────────────────────

class LoRAAdapter(nn.Module):
    """
    Low-rank delta for a single nn.Linear(d, d) layer.
    Output delta = (x @ A.T @ B.T) * scale
    B initialised to zero → LoRA starts as identity (no effect).
    """

    def __init__(self, d_model: int = D_MODEL, rank: int = LORA_RANK,
                 alpha: float = LORA_ALPHA) -> None:
        super().__init__()
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.randn(rank, d_model) * 0.02)  # (r, d)
        self.B = nn.Parameter(torch.zeros(d_model, rank))          # (d, r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A.T @ self.B.T) * self.scale


# ── Answer head ────────────────────────────────────────────────────────────────

class AnswerHead(nn.Module):
    """
    Tiny MLP that maps BigModel embedding → scalar answer.
    Trained together with LoRA adapters at test time.
    """

    def __init__(self, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── LoRA patch ─────────────────────────────────────────────────────────────────

def _make_hook(adapter: LoRAAdapter):
    def hook(module, inp, out):
        return out + adapter(inp[0])
    return hook


class LoRAPatch(nn.Module):
    """
    A trained specialist for one competition math problem type.

    Contains:
      - LoRAAdapters for last LORA_LAYERS of BigModel's attention
      - AnswerHead that maps BigModel output embedding → scalar answer

    Attached to / detached from BigModel via PyTorch forward hooks.
    BigModel weights are NEVER modified — hooks add deltas at runtime.
    """

    def __init__(
        self,
        n_total: int = N_LAYERS,
        n_lora: int = LORA_LAYERS,
        d_model: int = D_MODEL,
        rank: int = LORA_RANK,
        alpha: float = LORA_ALPHA,
    ) -> None:
        super().__init__()
        self._start     = n_total - n_lora
        self._n_total   = n_total
        self._hooks: list = []

        # LoRA adapters indexed as "L{layer}_{proj}"
        self.adapters = nn.ModuleDict({
            f"L{i}_{proj}": LoRAAdapter(d_model, rank, alpha)
            for i in range(self._start, n_total)
            for proj in TARGET_PROJS
        })
        self.head = AnswerHead(d_model)

    # ── hook management ──────────────────────────────────────────────────────

    def attach(self, gpt2_model) -> None:
        """Hook LoRA adapters into GPT2 attention output projections."""
        assert not self._hooks, "Already attached — call detach() first"
        for i in range(self._start, self._n_total):
            layer = gpt2_model.h[i]
            for proj in TARGET_PROJS:
                linear  = getattr(layer.attn, proj)
                adapter = self.adapters[f"L{i}_{proj}"]
                self._hooks.append(
                    linear.register_forward_hook(_make_hook(adapter))
                )

    def detach(self) -> None:
        """Remove all forward hooks — BigModel returns to original state."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def is_attached(self) -> bool:
        return len(self._hooks) > 0

    # ── inference ────────────────────────────────────────────────────────────

    def predict(self, embedding: torch.Tensor) -> torch.Tensor:
        """embedding: (B, D) → answer: (B, 1)"""
        return self.head(embedding)

    # ── param helpers ────────────────────────────────────────────────────────

    def trainable_params(self):
        return list(self.parameters())

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ── serialisation ─────────────────────────────────────────────────────────

    def lora_state(self) -> dict:
        """Return state dict for saving to bank (adapters + head)."""
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def load_lora_state(self, state: dict) -> None:
        self.load_state_dict(state)
