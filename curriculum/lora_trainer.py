"""
LoRATrainer: trains a LoRAPatch on BigModel using support examples.

At test time, when an unseen competition math problem arrives:
  1. Support examples (similar problems + answers) are provided
  2. LoRATrainer fine-tunes a new LoRAPatch on those examples
  3. The patch is saved to the bank for reuse

BigModel is frozen. Only LoRA adapter weights + AnswerHead are updated.
Typical training: 100 steps, ~5-10 seconds on GPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from big_model.transformer import BigModel, MAX_SEQ, D_MODEL
from big_model.lora import LoRAPatch, LORA_RANK

_LR         = 1e-3
_STEPS      = 100
_CLIP_NORM  = 1.0


class LoRATrainer:
    """
    Fine-tunes a LoRAPatch on BigModel using (problem_text, answer) pairs.

    Usage:
        trainer = LoRATrainer(big_model, device)
        patch, loss = trainer.train(support_texts, support_answers)
        # patch is ready to use for inference
    """

    def __init__(self, big_model: BigModel, device: str = "cpu") -> None:
        self.big_model = big_model.to(device)
        self.device    = device
        # Freeze BigModel completely
        for p in self.big_model.parameters():
            p.requires_grad = False
        self.big_model.eval()

    # ── public API ────────────────────────────────────────────────────────────

    def train(
        self,
        support_texts: list[str],
        support_answers: list[float],
        n_steps: int = _STEPS,
        lr: float = _LR,
        rank: int = LORA_RANK,
    ) -> tuple[LoRAPatch, float]:
        """
        Train a new LoRAPatch using support examples.

        Args:
            support_texts:   list of competition math problem strings
            support_answers: correct answers (floats / integers)
            n_steps:         gradient steps
            lr:              learning rate
            rank:            LoRA rank

        Returns:
            (trained LoRAPatch, final training loss)
        """
        patch = LoRAPatch(rank=rank).to(self.device)
        patch.attach(self.big_model.gpt2)
        patch.train()
        self.big_model.train()   # needed so LoRA hooks compute gradients

        targets = torch.tensor(
            support_answers, dtype=torch.float32, device=self.device
        )  # (N,)

        optim = AdamW(patch.trainable_params(), lr=lr, weight_decay=0.01)

        final_loss = float("inf")
        for step in range(n_steps):
            optim.zero_grad()

            embeds = self._encode_batch(support_texts)   # (N, D) — with grad
            preds  = patch.predict(embeds).squeeze(-1)   # (N,)
            loss   = F.mse_loss(preds, targets)

            loss.backward()
            nn.utils.clip_grad_norm_(patch.trainable_params(), _CLIP_NORM)
            optim.step()
            final_loss = loss.item()

        patch.detach()
        self.big_model.eval()
        patch.eval()
        return patch, final_loss

    # ── private helpers ───────────────────────────────────────────────────────

    def _encode_batch(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of strings through BigModel (with LoRA active)."""
        ids_list = []
        for text in texts:
            b      = list(text.encode("utf-8", errors="replace"))[:MAX_SEQ]
            padded = b + [0] * (MAX_SEQ - len(b))
            ids_list.append(padded)

        input_ids = torch.tensor(ids_list, dtype=torch.long, device=self.device)
        attn_mask = (input_ids != 0).long()

        hs  = self.big_model.gpt2(
            input_ids=input_ids, attention_mask=attn_mask
        ).last_hidden_state                                           # (N, S, D)

        # Mean pool
        m   = attn_mask.unsqueeze(-1).float()
        emb = (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)    # (N, D)
        return emb
