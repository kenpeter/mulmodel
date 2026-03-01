from __future__ import annotations
from typing import Any
import re
import numpy as np

EMB_DIM = 64   # matches MAX_DIM / router embedding size


class BinaryTokenizer:
    """
    Universal binary tokenizer — two paths depending on input type:

    TEXT path  (str):
      Character n-gram hashing — words sharing substrings share hash buckets.
      "great" and "greatest" overlap → similar vectors.
      "terrible" and "terribly" overlap → similar vectors.
      Much better than raw bits for language tasks.

      Steps:
        1. Lowercase + extract alphanumeric tokens
        2. Extract character 2-grams, 3-grams, 4-grams from each word
        3. Hash each n-gram → bucket in [0, EMB_DIM)  (deterministic)
        4. Accumulate counts, normalise to [-1, 1]

    BINARY path  (everything else):
      Raw bytes → bits → chunked average pool → [-1, 1]
      Universal: works for int, float, list, np.ndarray, bytes, any object.

    Both paths produce a 64-dim float32 vector.
    Same input always → same vector (deterministic, no randomness).
    """

    EMB_DIM: int = EMB_DIM

    # n-gram sizes used for text hashing
    _NGRAM_SIZES = (2, 3, 4)

    # ── public API ────────────────────────────────────────────────────────────

    def tokenize(self, x: Any) -> np.ndarray:
        """Convert any input to a 64-dim float vector in [-1, 1]."""
        if isinstance(x, str):
            return self._text_ngram(x)
        raw  = self._to_bytes(x)
        bits = self._bytes_to_bits(raw)
        return self._pool_bits(bits)

    def tokenize_batch(self, items: list[Any]) -> np.ndarray:
        """Tokenize a list of inputs → (n, 64) float32 array."""
        return np.stack([self.tokenize(x) for x in items])

    # ── text path: character n-gram hashing ──────────────────────────────────

    def _text_ngram(self, text: str) -> np.ndarray:
        vec   = np.zeros(self.EMB_DIM, dtype=np.float32)
        words = re.findall(r"[a-z0-9]+", text.lower())

        if not words:
            return vec

        count = 0
        for word in words:
            # Add word itself as a token
            h = self._hash(word)
            vec[h] += 1.0
            count  += 1
            # Add character n-grams
            for n in self._NGRAM_SIZES:
                for i in range(len(word) - n + 1):
                    gram = word[i: i + n]
                    h    = self._hash(gram)
                    vec[h] += 1.0
                    count  += 1

        # Normalise: divide by count → [0,1], then → [-1,1]
        if count > 0:
            vec /= count
        # Scale: bring into roughly [-1, 1] range
        # After normalisation values are small positives; rescale
        maxv = vec.max()
        if maxv > 1e-8:
            vec = (vec / maxv) * 2.0 - 1.0
        return vec.astype(np.float32)

    @staticmethod
    def _hash(s: str) -> int:
        """Deterministic hash of a string into [0, EMB_DIM)."""
        # FNV-1a variant — fast, deterministic, no Python hash randomisation
        h = 2166136261
        for ch in s.encode("utf-8"):
            h ^= ch
            h  = (h * 16777619) & 0xFFFFFFFF
        return h % EMB_DIM

    # ── binary path: raw bytes → bits → pool ─────────────────────────────────

    def _to_bytes(self, x: Any) -> bytes:
        if isinstance(x, bytes):
            return x
        if isinstance(x, np.ndarray):
            return x.astype(np.float32).tobytes()
        if isinstance(x, (list, tuple)):
            return "|".join(str(v) for v in x).encode("utf-8")
        if isinstance(x, (int, float, np.integer, np.floating)):
            return str(x).encode("utf-8")
        return str(x).encode("utf-8")

    def _bytes_to_bits(self, data: bytes) -> np.ndarray:
        if not data:
            return np.zeros(8, dtype=np.float32)
        arr  = np.frombuffer(data, dtype=np.uint8)
        bits = np.unpackbits(arr).astype(np.float32)
        return bits

    def _pool_bits(self, bits: np.ndarray) -> np.ndarray:
        n = len(bits)
        if n == 0:
            return np.zeros(self.EMB_DIM, dtype=np.float32)
        if n <= self.EMB_DIM:
            result       = np.zeros(self.EMB_DIM, dtype=np.float32)
            result[:n]   = bits
        else:
            result  = np.zeros(self.EMB_DIM, dtype=np.float32)
            indices = np.linspace(0, n, self.EMB_DIM + 1, dtype=int)
            for i in range(self.EMB_DIM):
                result[i] = bits[indices[i]: indices[i + 1]].mean()
        return (result * 2.0 - 1.0).astype(np.float32)


# ── module-level singleton ────────────────────────────────────────────────────
_tokenizer = BinaryTokenizer()


def tokenize(x: Any) -> np.ndarray:
    """Tokenize any single input → 64-dim float vector."""
    return _tokenizer.tokenize(x)


def tokenize_batch(items: list[Any]) -> np.ndarray:
    """Tokenize a list → (n, 64) float32 array."""
    return _tokenizer.tokenize_batch(items)
