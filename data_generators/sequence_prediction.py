from __future__ import annotations
import numpy as np
from .base import BaseGenerator

# Sequence patterns supported
_PATTERNS = ("linear1", "linear2", "geometric2", "squares", "fibonacci")

_MAX_VALUE = 256.0  # normalization scale


def _make_sequence(pattern: str, length: int, start: int = 0) -> np.ndarray:
    if pattern == "linear1":
        return np.arange(start, start + length, dtype=np.float32)
    if pattern == "linear2":
        return np.arange(start, start + length * 2, 2, dtype=np.float32)
    if pattern == "geometric2":
        return np.array([2.0 ** (start + i) for i in range(length)], dtype=np.float32)
    if pattern == "squares":
        return np.array([(start + i) ** 2 for i in range(length)], dtype=np.float32)
    if pattern == "fibonacci":
        seq = [0.0, 1.0]
        while len(seq) < length + start:
            seq.append(seq[-1] + seq[-2])
        return np.array(seq[start: start + length], dtype=np.float32)
    raise ValueError(f"Unknown pattern: {pattern}")


def _normalize(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr / _MAX_VALUE, -1.0, 1.0).astype(np.float32)


class SequencePredictionGenerator(BaseGenerator):
    """
    Level 0 (Easy):   X = [x0 padded to 8],     y = [x1]
    Level 1 (Medium): X = [x0,x1 padded to 8],  y = [x2]
    Level 2 (Hard):   X = [x0,..,x6 padded],    y = [x7]
    """

    _INPUT_DIMS = {0: 8, 1: 8, 2: 8}
    _CONTEXT_LEN = {0: 1, 1: 2, 2: 7}
    _TARGET_IDX = {0: 1, 1: 2, 2: 7}

    def input_dim(self, level: int) -> int:
        return self._INPUT_DIMS[level]

    def output_dim(self) -> int:
        return 1

    def generate(
        self,
        level: int,
        n_samples: int = 500,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        context_len = self._CONTEXT_LEN[level]
        target_idx = self._TARGET_IDX[level]
        input_dim = self._INPUT_DIMS[level]

        X_list, y_list = [], []
        for _ in range(n_samples):
            pattern = rng.choice(_PATTERNS)
            start = int(rng.integers(0, 10))
            seq = _make_sequence(pattern, target_idx + 1, start=start)
            context = seq[:context_len]
            target = seq[target_idx: target_idx + 1]

            x = np.zeros(input_dim, dtype=np.float32)
            x[:context_len] = _normalize(context)
            X_list.append(x)
            y_list.append(_normalize(target))

        return np.stack(X_list), np.stack(y_list)
