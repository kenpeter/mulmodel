from __future__ import annotations
import numpy as np
from .base import BaseGenerator

# Operations: add, subtract, multiply
_OPS = ("add", "subtract", "multiply")
_SCALE = 100.0


def _normalize(x: float) -> float:
    return float(np.clip(x / _SCALE, -1.0, 1.0))


class ArithmeticGenerator(BaseGenerator):
    """
    Level 0 (Easy):   single-digit operands, addition only
    Level 1 (Medium): two-digit operands, add/subtract
    Level 2 (Hard):   two-digit operands, all ops including multiply
    X = [a, b, op_onehot(3)]  → shape (5,)
    y = [result]
    """

    def input_dim(self, level: int) -> int:
        return 5  # a, b, op_onehot[3]

    def output_dim(self) -> int:
        return 1

    def generate(
        self,
        level: int,
        n_samples: int = 500,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)

        if level == 0:
            max_val, ops = 9, ("add",)
        elif level == 1:
            max_val, ops = 99, ("add", "subtract")
        else:
            max_val, ops = 99, _OPS

        X_list, y_list = [], []
        for _ in range(n_samples):
            a = int(rng.integers(0, max_val + 1))
            b = int(rng.integers(0, max_val + 1))
            op = rng.choice(ops)

            if op == "add":
                result, op_idx = a + b, 0
            elif op == "subtract":
                result, op_idx = a - b, 1
            else:
                result, op_idx = a * b, 2

            op_onehot = np.zeros(3, dtype=np.float32)
            op_onehot[op_idx] = 1.0

            x = np.array([_normalize(a), _normalize(b), *op_onehot], dtype=np.float32)
            y = np.array([_normalize(result)], dtype=np.float32)
            X_list.append(x)
            y_list.append(y)

        return np.stack(X_list), np.stack(y_list)
