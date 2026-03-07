"""
BigModel: 1024-wide, 96-layer transformer backbone (GPT-style causal LM).

Uses HuggingFace GPT2Config + GPT2Model internally.
Produces 1024-dim embeddings from numeric or text inputs,
and can generate text autoregressively.

Architecture:
  Numeric input: scalar floats → projected to D_MODEL via Linear(1, 1024)
  Text input:    byte tokens   → looked up in GPT2's word embedding table
  Transformer:   96 layers, 1024 hidden, 16 heads, 4096 FFN  (causal attention)
  Output:        mean-pooled hidden states → 1024-dim embedding
                 OR per-token logits for next-token prediction / generation

Params: ~1.21B

Memory strategy (training):
  Model stored in BF16 on GPU (~2.3 GB VRAM).
  FP32 optimizer states live in CPU RAM (~9.2 GB).
  Gradient checkpointing enabled to reduce activation memory.
  See BigModelPretrainer in pretrain.py for the CPU-optimizer step logic.
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

D_MODEL    = 1024     # hidden width
N_LAYERS   = 96       # transformer depth
N_HEADS    = 16       # 1024 / 16 = 64 dim per head
FFN_DIM    = 4096     # 4 × D_MODEL
MAX_SEQ    = 512      # maximum sequence length (tokens)
BYTE_VOCAB = 256      # byte-level vocabulary (0–255)


def _make_gpt2_config() -> GPT2Config:
    return GPT2Config(
        vocab_size=BYTE_VOCAB,
        n_embd=D_MODEL,
        n_layer=N_LAYERS,
        n_head=N_HEADS,
        n_inner=FFN_DIM,
        n_positions=MAX_SEQ,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )


class BigModel(nn.Module):
    """
    1024-wide, 96-layer causal transformer backbone.

    Two input modes:
      numeric — raw float values projected to D_MODEL via numeric_proj
      text    — byte-encoded strings looked up in GPT2's word embeddings

    Output: 1024-dim mean-pooled embedding (one vector per input sample),
            OR per-token logits for causal language modeling / generation.

    Gradient checkpointing is enabled by default to keep activation memory low.
    """

    EMB_DIM: int = D_MODEL

    def __init__(self, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        cfg = _make_gpt2_config()
        self.gpt2 = GPT2Model(cfg)
        if gradient_checkpointing:
            self.gpt2.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        # Numeric input: project each scalar float → D_MODEL
        self.numeric_proj = nn.Linear(1, D_MODEL)

        # Causal LM head: predict next byte token
        # Weight-tied to token embeddings (standard GPT practice)
        self.lm_head = nn.Linear(D_MODEL, BYTE_VOCAB, bias=False)
        self.lm_head.weight = self.gpt2.wte.weight

    # ── internal helpers ──────────────────────────────────────────────────────

    def _run(
        self,
        inputs_embeds: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run GPT2 encoder, return full hidden states (B, S, D)."""
        out = self.gpt2(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return out.last_hidden_state

    @staticmethod
    def _mean_pool(hs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool hidden states over non-padding tokens.

        hs:   (B, S, D)
        mask: (B, S)  — 1 for real tokens, 0 for padding
        returns (B, D)
        """
        m = mask.unsqueeze(-1).float()
        return (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)

    # ── embedding forward passes ──────────────────────────────────────────────

    def forward_numeric(self, values: torch.Tensor) -> torch.Tensor:
        """
        values: (B, S) float32 — sequence of scalar values
        returns (B, D_MODEL) — 1024-dim embedding
        """
        B, S = values.shape
        embeds = self.numeric_proj(values.unsqueeze(-1))
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

    # ── pretraining forward pass ──────────────────────────────────────────────

    def pretrain_causal(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Causal (next-token) language modeling for text sequences.

        input_ids: (B, S) long — byte token ids
        returns:   scalar cross-entropy loss

        Position i predicts token i+1. Padding bytes (0) excluded from loss.
        """
        attention_mask = (input_ids != 0).long()
        hs = self._run(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hs)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        return nn.functional.cross_entropy(
            shift_logits.view(-1, BYTE_VOCAB),
            shift_labels.view(-1),
            ignore_index=0,
        )

    # ── text generation ───────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        """
        Autoregressive text generation from a string prompt.

        Runs the full growing sequence on each step (no KV cache) to avoid
        dtype/device issues with the BF16 model.
        """
        self.eval()
        device = next(self.parameters()).device

        max_prompt = MAX_SEQ - min(max_new_tokens, MAX_SEQ // 2)
        prompt_ids = list(prompt.encode("utf-8", errors="replace"))[:max_prompt]
        if not prompt_ids:
            prompt_ids = [32]

        generated = list(prompt_ids)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if len(generated) >= MAX_SEQ:
                    break
                ids = torch.tensor([generated], dtype=torch.long, device=device)
                out = self.gpt2(input_ids=ids, use_cache=False)
                # Cast logits to float32 for stable softmax
                logits = self.lm_head(out.last_hidden_state[:, -1, :]).float()

                if temperature != 1.0:
                    logits = logits / temperature
                if top_k > 0:
                    vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < vals[:, -1:], float("-inf"))

                next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
                if next_id == 0:
                    break
                generated.append(next_id)

        return bytes(generated[len(prompt_ids):]).decode("utf-8", errors="replace")

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"  [BigModel] Saved → {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BigModel":
        m = cls()
        m.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True),
            strict=False,  # tolerate tied-weight duplicates in old checkpoints
        )
        return m.to(device)

    # ── info ─────────────────────────────────────────────────────────────────

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
