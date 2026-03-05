"""
BigModel: 256-wide, 128-layer transformer backbone (GPT-style causal LM).

Uses HuggingFace GPT2Config + GPT2Model internally.
Produces 256-dim embeddings from numeric or text inputs,
and can generate text autoregressively.

Architecture:
  Numeric input: scalar floats → projected to D_MODEL via Linear(1, 256)
  Text input:    byte tokens   → looked up in GPT2's word embedding table
  Transformer:   128 layers, 256 hidden, 8 heads, 1024 FFN  (causal attention)
  Output:        mean-pooled hidden states → 256-dim embedding
                 OR per-token logits for next-token prediction / generation

Params: ~101M  (fits in 12GB VRAM; ~400MB on disk)
"""
from __future__ import annotations

import os
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

D_MODEL    = 256      # hidden width
N_LAYERS   = 128      # transformer depth
N_HEADS    = 8        # 256 / 8 = 32 dim per head
FFN_DIM    = 1024     # 4 × D_MODEL
MAX_SEQ    = 128      # maximum sequence length (tokens)
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
    256-wide, 128-layer causal transformer backbone.

    Two input modes:
      numeric — raw float values projected to D_MODEL via numeric_proj
      text    — byte-encoded strings looked up in GPT2's word embeddings

    Output: 256-dim mean-pooled embedding (one vector per input sample),
            OR per-token logits for causal language modeling / generation.
    """

    EMB_DIM: int = D_MODEL

    def __init__(self) -> None:
        super().__init__()
        cfg = _make_gpt2_config()
        self.gpt2 = GPT2Model(cfg)

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

    # ── pretraining forward pass ──────────────────────────────────────────────

    def pretrain_causal(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Causal (next-token) language modeling for text sequences.

        input_ids: (B, S) long — byte token ids
        returns:   scalar cross-entropy loss

        Position i predicts token i+1. Padding bytes (0) excluded from loss.
        """
        attention_mask = (input_ids != 0).long()
        hs = self._run(input_ids=input_ids, attention_mask=attention_mask)  # (B, S, D)
        logits = self.lm_head(hs)                                            # (B, S, V)

        shift_logits = logits[:, :-1, :].contiguous()    # (B, S-1, V)
        shift_labels = input_ids[:, 1:].contiguous()     # (B, S-1)

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

        prompt:          input text (byte-encoded)
        max_new_tokens:  maximum tokens to generate
        temperature:     sampling temperature (lower = more deterministic)
        top_k:           top-k sampling (0 = full distribution)
        returns:         generated string (prompt not included)
        """
        self.eval()
        device = next(self.parameters()).device

        max_prompt_len = MAX_SEQ - min(max_new_tokens, MAX_SEQ // 2)
        b = list(prompt.encode("utf-8", errors="replace"))[:max_prompt_len]
        if not b:
            b = [32]  # space byte as minimal non-empty prompt

        ids = torch.tensor([b], dtype=torch.long, device=device)

        generated_ids: list[int] = []
        past_key_values = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if past_key_values is None:
                    input_ids = ids
                else:
                    input_ids = torch.tensor(
                        [[generated_ids[-1]]], dtype=torch.long, device=device
                    )

                out = self.gpt2(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = out.past_key_values

                logits = self.lm_head(out.last_hidden_state[:, -1, :])  # (1, V)

                if temperature != 1.0:
                    logits = logits / temperature

                if top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    threshold = values[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < threshold, float("-inf"))

                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()

                if next_id == 0:
                    break

                generated_ids.append(next_id)

        return bytes(generated_ids).decode("utf-8", errors="replace")

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
