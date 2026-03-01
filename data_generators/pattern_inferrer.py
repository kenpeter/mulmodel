from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Base rule
# ─────────────────────────────────────────────────────────────────────────────

class BaseRule(ABC):
    name: str = "base"
    residual: float = float("inf")

    @classmethod
    @abstractmethod
    def fit(cls, support_X: np.ndarray, support_y: np.ndarray) -> BaseRule:
        """Fit rule parameters to support examples. Returns fitted instance."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict y from X. X shape: (n, input_dim)."""

    def __repr__(self) -> str:
        return f"{self.name}(residual={self.residual:.5f})"


# ─────────────────────────────────────────────────────────────────────────────
# Rule 1: Constant Difference   y ≈ x_last + d
# Catches: linear+1, linear+2, arithmetic sequences
# ─────────────────────────────────────────────────────────────────────────────

class ConstantDifferenceRule(BaseRule):
    name = "constant_difference"

    def __init__(self) -> None:
        self.d: float = 0.0

    @classmethod
    def fit(cls, support_X: np.ndarray, support_y: np.ndarray) -> ConstantDifferenceRule:
        rule = cls()
        x_last = support_X[:, -1]
        y_flat = support_y[:, 0]
        rule.d = float(np.mean(y_flat - x_last))
        preds = x_last + rule.d
        rule.residual = float(np.mean((preds - y_flat) ** 2))
        return rule

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, -1] + self.d).reshape(-1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Rule 2: Constant Ratio   y ≈ x_last * r
# Catches: geometric×2, geometric×3, etc.
# ─────────────────────────────────────────────────────────────────────────────

class ConstantRatioRule(BaseRule):
    name = "constant_ratio"

    def __init__(self) -> None:
        self.r: float = 1.0

    @classmethod
    def fit(cls, support_X: np.ndarray, support_y: np.ndarray) -> ConstantRatioRule:
        rule = cls()
        x_last = support_X[:, -1]
        y_flat = support_y[:, 0]
        # Avoid division by zero
        mask = np.abs(x_last) > 1e-6
        if mask.sum() < 1:
            rule.residual = float("inf")
            return rule
        rule.r = float(np.mean(y_flat[mask] / x_last[mask]))
        preds = x_last * rule.r
        rule.residual = float(np.mean((preds - y_flat) ** 2))
        return rule

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, -1] * self.r).reshape(-1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Rule 3: Fibonacci-like   y ≈ x[-1] + x[-2]
# Catches: fibonacci and similar additive recurrences
# ─────────────────────────────────────────────────────────────────────────────

class FibonacciRule(BaseRule):
    name = "fibonacci"

    def __init__(self) -> None:
        self.a: float = 1.0   # coeff for x[-1]
        self.b: float = 1.0   # coeff for x[-2]

    @classmethod
    def fit(cls, support_X: np.ndarray, support_y: np.ndarray) -> FibonacciRule:
        rule = cls()
        if support_X.shape[1] < 2:
            rule.residual = float("inf")
            return rule
        x1 = support_X[:, -1]
        x2 = support_X[:, -2]
        y_flat = support_y[:, 0]
        # Fit: y = a*x1 + b*x2 via least squares
        A = np.stack([x1, x2], axis=1)
        try:
            coeffs, *_ = np.linalg.lstsq(A, y_flat, rcond=None)
            rule.a, rule.b = float(coeffs[0]), float(coeffs[1])
        except Exception:
            rule.residual = float("inf")
            return rule
        preds = A @ coeffs
        rule.residual = float(np.mean((preds - y_flat) ** 2))
        return rule

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, -1] * self.a + X[:, -2] * self.b).reshape(-1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Rule 4: Quadratic index   y ≈ a*i² + b*i + c  (i = position in sequence)
# Catches: squares [0,1,4,9,16,...], cubic-ish patterns
# ─────────────────────────────────────────────────────────────────────────────

class QuadraticIndexRule(BaseRule):
    name = "quadratic_index"

    def __init__(self) -> None:
        self.coeffs: np.ndarray = np.zeros(3)
        self.x_last_mean: float = 0.0
        self.x_last_std: float = 1.0

    @classmethod
    def fit(cls, support_X: np.ndarray, support_y: np.ndarray) -> QuadraticIndexRule:
        rule = cls()
        n = len(support_X)
        # Use the index position (0, 1, 2, ...) as x-axis
        idx = np.arange(n, dtype=np.float32)
        y_flat = support_y[:, 0]
        # Fit degree-2 polynomial to (idx, y)
        try:
            rule.coeffs = np.polyfit(idx, y_flat, 2)
        except Exception:
            rule.residual = float("inf")
            return rule
        preds = np.polyval(rule.coeffs, idx)
        rule.residual = float(np.mean((preds - y_flat) ** 2))
        # Store reference: what index does the last support example represent?
        rule._n_support = n
        return rule

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Predict for the next index after each example
        n = getattr(self, "_n_support", len(X))
        idx = np.arange(n, n + len(X), dtype=np.float32)
        return np.polyval(self.coeffs, idx).reshape(-1, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Rule 5: Multi-linear   y ≈ X @ w + b
# Catches: arithmetic (a op b → result), any linear mapping
# ─────────────────────────────────────────────────────────────────────────────

class MultiLinearRule(BaseRule):
    name = "multi_linear"

    def __init__(self) -> None:
        self.w: np.ndarray = np.zeros(1)
        self.b: float = 0.0

    @classmethod
    def fit(cls, support_X: np.ndarray, support_y: np.ndarray) -> MultiLinearRule:
        rule = cls()
        y_flat = support_y[:, 0]
        # Ridge regression: add bias column
        X_aug = np.hstack([support_X, np.ones((len(support_X), 1))])
        try:
            # Ridge with small lambda for stability
            lam = 1e-4
            A = X_aug.T @ X_aug + lam * np.eye(X_aug.shape[1])
            b_vec = X_aug.T @ y_flat
            coeffs = np.linalg.solve(A, b_vec)
            rule.w = coeffs[:-1]
            rule.b = float(coeffs[-1])
        except Exception:
            rule.residual = float("inf")
            return rule
        preds = X_aug @ coeffs
        rule.residual = float(np.mean((preds - y_flat) ** 2))
        return rule

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X @ self.w + self.b).reshape(-1, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Rule 6: Binary threshold   y = sign(sum(x) - threshold)
# Catches: majority vote, parity, binary pattern rules
# ─────────────────────────────────────────────────────────────────────────────

class BinaryThresholdRule(BaseRule):
    name = "binary_threshold"

    def __init__(self) -> None:
        self.threshold: float = 0.5
        self.scale: float = 1.0   # +1 or -1 polarity

    @classmethod
    def fit(cls, support_X: np.ndarray, support_y: np.ndarray) -> BinaryThresholdRule:
        rule = cls()
        # Only apply if X looks binary (values mostly 0 or 1)
        unique = np.unique(np.round(support_X, 2))
        if not np.all((unique <= 0.05) | (unique >= 0.95)):
            rule.residual = float("inf")
            return rule

        row_sums = support_X.sum(axis=1)
        y_flat = support_y[:, 0]
        rule.threshold = float(support_X.shape[1] / 2)

        # Try polarity: positive sum → +1 or positive sum → -1
        preds_pos = np.where(row_sums > rule.threshold, 1.0, -1.0)
        preds_neg = np.where(row_sums > rule.threshold, -1.0, 1.0)

        res_pos = float(np.mean((preds_pos - y_flat) ** 2))
        res_neg = float(np.mean((preds_neg - y_flat) ** 2))

        if res_pos <= res_neg:
            rule.scale = 1.0
            rule.residual = res_pos
        else:
            rule.scale = -1.0
            rule.residual = res_neg
        return rule

    def predict(self, X: np.ndarray) -> np.ndarray:
        sums = X.sum(axis=1)
        threshold = X.shape[1] / 2
        raw = np.where(sums > threshold, 1.0, -1.0) * self.scale
        return raw.reshape(-1, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PatternInferrer — tries all rules, returns best fit
# ─────────────────────────────────────────────────────────────────────────────

_RULE_CLASSES: list[type[BaseRule]] = [
    ConstantDifferenceRule,
    ConstantRatioRule,
    FibonacciRule,
    QuadraticIndexRule,
    BinaryThresholdRule,
    MultiLinearRule,   # last: always succeeds, acts as fallback
]


class PatternInferrer:
    """
    Looks at support_X, support_y and infers the best-fitting rule.

    Like a human seeing a few examples and asking:
      "Is this linear? geometric? fibonacci? quadratic? binary?"
    Then picks whichever fits best (lowest residual on support examples).
    """

    def infer(
        self,
        support_X: np.ndarray,
        support_y: np.ndarray,
    ) -> BaseRule:
        support_X = np.asarray(support_X, dtype=np.float32)
        support_y = np.asarray(support_y, dtype=np.float32)

        if support_y.ndim == 1:
            support_y = support_y.reshape(-1, 1)

        candidates: list[BaseRule] = []
        for rule_cls in _RULE_CLASSES:
            try:
                rule = rule_cls.fit(support_X, support_y)
                if rule.residual < float("inf"):
                    candidates.append(rule)
            except Exception:
                pass

        if not candidates:
            # Last resort: MultiLinear always works
            return MultiLinearRule.fit(support_X, support_y)

        best = min(candidates, key=lambda r: r.residual)
        return best
