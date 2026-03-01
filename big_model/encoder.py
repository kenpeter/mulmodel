"""
BigModelEncoder: drop-in replacement for router/encoder.py's ProblemEncoder.

Uses a pretrained BigModel (256-dim, 128-layer) to encode problems into
richer 256-dim embeddings instead of the hand-crafted 64-dim rule-based ones.

Falls back gracefully: if no checkpoint exists, returns None so the pipeline
can fall back to ProblemEncoder.

Same public interface as ProblemEncoder:
  encode(problem) -> np.ndarray (256-dim)
  compute_specialty_embedding(support_X, support_y) -> np.ndarray (256-dim)
  EMB_DIM: int = 256
"""
from __future__ import annotations

import os
import numpy as np
import torch

from big_model.transformer import BigModel, MAX_SEQ, D_MODEL
from core.problem import Problem

DEFAULT_CHECKPOINT = "big_model_data/big_model.pt"


class BigModelEncoder:
    """
    Encodes Problems into 256-dim embeddings using a pretrained BigModel.
    Thread-safe for read-only (inference) use — all forward passes run under
    torch.no_grad() in eval mode.
    """

    EMB_DIM: int = D_MODEL   # 256

    def __init__(self, model: BigModel, device: str = "cpu") -> None:
        self.model  = model.to(device).eval()
        self.device = device

    # ── public API (same as ProblemEncoder) ───────────────────────────────────

    def encode(self, problem: Problem) -> np.ndarray:
        """Problem → 256-dim np.ndarray embedding."""
        with torch.no_grad():
            if problem.is_text:
                return self._encode_text(problem.raw_text or "")
            return self._encode_numeric(problem.raw_input)

    def compute_specialty_embedding(
        self, support_X: np.ndarray, support_y: np.ndarray
    ) -> np.ndarray:
        """
        256-dim embedding representing what a model was trained on.
        Uses the mean of support_X rows projected through the BigModel.
        """
        with torch.no_grad():
            if support_X.size == 0:
                return np.zeros(self.EMB_DIM, dtype=np.float32)
            # Average the support inputs → representative sequence
            mean_row = support_X.mean(axis=0).astype(np.float32)
            return self._encode_numeric(mean_row)

    # ── private helpers ───────────────────────────────────────────────────────

    def _encode_numeric(self, raw_input: np.ndarray) -> np.ndarray:
        """raw_input: 1-D float array → (256,) embedding."""
        seq             = np.zeros(MAX_SEQ, dtype=np.float32)
        take            = min(len(raw_input), MAX_SEQ)
        seq[:take]      = raw_input[:take]
        values          = torch.tensor(seq, dtype=torch.float32,
                                       device=self.device).unsqueeze(0)  # (1, MAX_SEQ)
        emb             = self.model.forward_numeric(values)              # (1, 256)
        return emb.squeeze(0).cpu().numpy()

    def _encode_text(self, text: str) -> np.ndarray:
        """text string → (256,) embedding via byte tokenisation."""
        b               = list(text.encode("utf-8", errors="replace"))[:MAX_SEQ]
        padded          = b + [0] * (MAX_SEQ - len(b))
        ids             = torch.tensor([padded], dtype=torch.long,
                                       device=self.device)               # (1, MAX_SEQ)
        attn            = (ids != 0).long()
        emb             = self.model.forward_text(ids, attn)             # (1, 256)
        return emb.squeeze(0).cpu().numpy()

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        path: str = DEFAULT_CHECKPOINT,
        device: str | None = None,
    ) -> "BigModelEncoder | None":
        """
        Load from checkpoint file.
        Returns None if the checkpoint does not exist yet (i.e. not pretrained).
        """
        if not os.path.exists(path):
            return None
        dev   = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  [BigModelEncoder] Loading from {path}  (device={dev})")
        model = BigModel.load(path, device=dev)
        model.eval()
        return cls(model, dev)
