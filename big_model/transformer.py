"""
BigModel: 256-wide, 128-layer transformer backbone.

Uses HuggingFace BertConfig + BertModel internally.
Produces 256-dim embeddings from numeric or text inputs.

Architecture:
  Numeric input: scalar floats → projected to D_MODEL via Linear(1, 256)
  Text input:    byte tokens   → looked up in BERT's word embedding table
  Transformer:   128 layers, 256 hidden, 8 heads, 1024 FFN
  Output:        mean-pooled hidden states → 256-dim embedding

Params: ~101M  (fits in 12GB VRAM; ~400MB on disk)
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

D_MODEL    = 256      # hidden width
N_LAYERS   = 128      # transformer depth
N_HEADS    = 8        # 256 / 8 = 32 dim per head
FFN_DIM    = 1024     # 4 × D_MODEL
MAX_SEQ    = 128      # maximum sequence length (tokens)
BYTE_VOCAB = 256      # byte-level vocabulary (0–255)


def _make_bert_config() -> BertConfig:
    return BertConfig(
        hidden_size=D_MODEL,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        intermediate_size=FFN_DIM,
        vocab_size=BYTE_VOCAB,
        max_position_embeddings=MAX_SEQ,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_vocab_size=1,
    )


class BigModel(nn.Module):
    """
    256-wide, 128-layer transformer backbone.

    Two input modes:
      numeric — raw float values projected to D_MODEL via numeric_proj
      text    — byte-encoded strings looked up in BERT's word embeddings

    Output: 256-dim mean-pooled embedding (one vector per input sample).
    """

    EMB_DIM: int = D_MODEL

    def __init__(self) -> None:
        super().__init__()
        cfg = _make_bert_config()
        self.bert = BertModel(cfg, add_pooling_layer=False)

        # Numeric input: project each scalar float → D_MODEL
        self.numeric_proj = nn.Linear(1, D_MODEL)

        # Pretraining head
        self.mask_head_text = nn.Linear(D_MODEL, BYTE_VOCAB)   # predict masked byte

    # ── internal helpers ──────────────────────────────────────────────────────

    def _run(
        self,
        inputs_embeds: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run BERT encoder, return full hidden states (B, S, D)."""
        out = self.bert(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        return out.last_hidden_state

    @staticmethod
    def _mean_pool(hs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool hidden states over non-padding tokens.

        hs:   (B, S, D)
        mask: (B, S)  — 1 for real tokens, 0 for padding
        returns (B, D)
        """
        m = mask.unsqueeze(-1).float()          # (B, S, 1)
        return (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)

    # ── embedding forward passes ──────────────────────────────────────────────

    def forward_numeric(self, values: torch.Tensor) -> torch.Tensor:
        """
        values: (B, S) float32 — sequence of scalar values
        returns (B, D_MODEL) — 256-dim embedding
        """
        B, S = values.shape
        embeds = self.numeric_proj(values.unsqueeze(-1))          # (B, S, D)
        attn   = torch.ones(B, S, dtype=torch.long, device=values.device)
        hs     = self._run(inputs_embeds=embeds, attention_mask=attn)
        return self._mean_pool(hs, attn.float())

    def forward_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        input_ids:      (B, S) long — byte token ids (0–255)
        attention_mask: (B, S) long — 1 = real, 0 = pad
        returns (B, D_MODEL)
        """
        hs = self._run(input_ids=input_ids, attention_mask=attention_mask)
        return self._mean_pool(hs, attention_mask.float())

    # ── pretraining forward passes ────────────────────────────────────────────

    def pretrain_text(
        self, input_ids: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Masked byte modeling for text sequences.

        input_ids: (B, S) long — byte token ids
        mask:      (B, S) bool — True = masked
        returns logits (N, BYTE_VOCAB), targets (N,)
        """
        masked_ids       = input_ids.clone()
        masked_ids[mask] = 0                               # mask token id = 0
        attn             = (input_ids != 0).long()
        hs               = self._run(input_ids=masked_ids, attention_mask=attn)
        logits           = self.mask_head_text(hs[mask])   # (N, BYTE_VOCAB)
        targets          = input_ids[mask]                 # (N,)
        return logits, targets

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"  [BigModel] Saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BigModel":
        m = cls()
        m.load_state_dict(
            torch.load(path, map_location=device, weights_only=True)
        )
        return m

    # ── info ─────────────────────────────────────────────────────────────────

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
