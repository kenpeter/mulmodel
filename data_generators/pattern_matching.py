from __future__ import annotations
import numpy as np
from .base import BaseGenerator


class PatternMatchingGenerator(BaseGenerator):
    """
    Binary pattern classification.
    X = binary vector of length {0:4, 1:6, 2:8}
    y = [1.0] if rule satisfied, [-1.0] otherwise

    Level 0 (Easy):   majority vote of 4 bits
    Level 1 (Medium): parity (even number of 1s)
    Level 2 (Hard):   alternating pattern check
    """

    _INPUT_DIMS = {0: 4, 1: 6, 2: 8}

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
        dim = self._INPUT_DIMS[level]

        X = rng.integers(0, 2, size=(n_samples, dim)).astype(np.float32)
        y_list = []

        for row in X:
            if level == 0:
                label = 1.0 if row.sum() > dim / 2 else -1.0
            elif level == 1:
                label = 1.0 if int(row.sum()) % 2 == 0 else -1.0
            else:
                alternating = all(
                    row[i] != row[i + 1] for i in range(len(row) - 1)
                )
                label = 1.0 if alternating else -1.0
            y_list.append([label])

        y = np.array(y_list, dtype=np.float32)
        return X, y
