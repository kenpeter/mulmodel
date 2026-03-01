from __future__ import annotations
import numpy as np
from data_generators.pattern_inferrer import (
    BaseRule,
    ConstantDifferenceRule,
    ConstantRatioRule,
    FibonacciRule,
    QuadraticIndexRule,
    BinaryThresholdRule,
    MultiLinearRule,
)
from curriculum.generator import CurriculumGenerator

# Noise added at each curriculum level (in normalised space)
_LEVEL_NOISE = {0: 0.00, 1: 0.01, 2: 0.03}
# Value range multiplier at each level (wider = harder)
_LEVEL_RANGE = {0: 0.3, 1: 0.6, 2: 1.0}


class RuleGenerator(CurriculumGenerator):
    """
    Generates curriculum data from a fitted rule.

    This replaces all hardcoded generators.  Any rule inferred by
    PatternInferrer is turned into a curriculum automatically:

      Level 0 (easy):  narrow input range, no noise
      Level 1 (medium): medium range, small noise
      Level 2 (hard):  full range, more noise

    The generated X has the same number of columns as the original
    support_X so the TinyModel sees the same input format.
    """

    def __init__(
        self,
        rule: BaseRule,
        support_X: np.ndarray,
        support_y: np.ndarray,
    ) -> None:
        self.rule = rule
        self._support_X = np.asarray(support_X, dtype=np.float32)
        self._support_y = np.asarray(support_y, dtype=np.float32)
        self._in_dim = support_X.shape[1]
        self._out_dim = support_y.shape[1] if support_y.ndim > 1 else 1

    def input_dim(self, level: int) -> int:
        return self._in_dim

    def output_dim(self) -> int:
        return self._out_dim

    def generate(
        self,
        level: int,
        n_samples: int = 500,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        noise_std = _LEVEL_NOISE[level]
        val_range = _LEVEL_RANGE[level]

        # Dispatch to the appropriate generation strategy
        if isinstance(self.rule, ConstantDifferenceRule):
            return self._gen_difference(rng, n_samples, val_range, noise_std)
        if isinstance(self.rule, ConstantRatioRule):
            return self._gen_ratio(rng, n_samples, val_range, noise_std)
        if isinstance(self.rule, FibonacciRule):
            return self._gen_fibonacci(rng, n_samples, val_range, noise_std)
        if isinstance(self.rule, QuadraticIndexRule):
            return self._gen_quadratic(rng, n_samples, val_range, noise_std)
        if isinstance(self.rule, BinaryThresholdRule):
            return self._gen_binary(rng, n_samples, noise_std)
        # Fallback: MultiLinear and anything else
        return self._gen_linear(rng, n_samples, val_range, noise_std)

    # ── generation strategies ─────────────────────────────────────────────────

    def _gen_difference(
        self, rng, n: int, val_range: float, noise: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate sequences where y = x_last + d.
        X cols: [x_0, x_1, ..., x_{k-1}], y = [x_k]
        Each example is a sliding window of a sequence with difference d.
        """
        d = self.rule.d
        in_dim = self._in_dim

        X_list, y_list = [], []
        for _ in range(n):
            # Pick a random starting value
            start = rng.uniform(-val_range, val_range)
            # Build a sequence: start, start+d, start+2d, ...
            seq = np.array(
                [start + d * i for i in range(in_dim + 1)], dtype=np.float32
            )
            x = seq[:in_dim]
            y = seq[in_dim: in_dim + 1]

            if noise > 0:
                x = x + rng.normal(0, noise, size=x.shape).astype(np.float32)
                y = y + rng.normal(0, noise, size=y.shape).astype(np.float32)

            X_list.append(np.clip(x, -1.0, 1.0))
            y_list.append(np.clip(y, -1.0, 1.0))

        return np.stack(X_list), np.stack(y_list)

    def _gen_ratio(
        self, rng, n: int, val_range: float, noise: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate sequences where y = x_last * r.
        """
        r = self.rule.r
        in_dim = self._in_dim

        X_list, y_list = [], []
        for _ in range(n):
            start = rng.uniform(0.01, val_range)  # positive start for ratios
            seq = np.array(
                [start * (r ** i) for i in range(in_dim + 1)], dtype=np.float32
            )
            x = np.clip(seq[:in_dim], -1.0, 1.0)
            y = np.clip(seq[in_dim: in_dim + 1], -1.0, 1.0)

            if noise > 0:
                x = np.clip(x + rng.normal(0, noise, size=x.shape).astype(np.float32), -1.0, 1.0)

            X_list.append(x)
            y_list.append(y)

        return np.stack(X_list), np.stack(y_list)

    def _gen_fibonacci(
        self, rng, n: int, val_range: float, noise: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate additive recurrence sequences: y = a*x[-1] + b*x[-2].
        """
        a, b = self.rule.a, self.rule.b
        in_dim = self._in_dim

        X_list, y_list = [], []
        for _ in range(n):
            f0 = rng.uniform(0.0, val_range * 0.1)
            f1 = rng.uniform(0.0, val_range * 0.1)
            seq = [f0, f1]
            while len(seq) < in_dim + 1:
                seq.append(a * seq[-1] + b * seq[-2])
            seq = np.clip(np.array(seq, dtype=np.float32), -1.0, 1.0)
            X_list.append(seq[:in_dim])
            y_list.append(seq[in_dim: in_dim + 1])

        return np.stack(X_list), np.stack(y_list)

    def _gen_quadratic(
        self, rng, n: int, val_range: float, noise: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate examples from a quadratic index rule.
        X = polynomial values at positions i..i+k-1, y = value at i+k.
        """
        coeffs = self.rule.coeffs
        in_dim = self._in_dim

        X_list, y_list = [], []
        for _ in range(n):
            start_idx = int(rng.integers(0, 20))
            idx_x = np.arange(start_idx, start_idx + in_dim, dtype=np.float32)
            idx_y = start_idx + in_dim

            x = np.clip(
                np.polyval(coeffs, idx_x).astype(np.float32), -1.0, 1.0
            )
            y = np.clip(
                np.array([np.polyval(coeffs, idx_y)], dtype=np.float32), -1.0, 1.0
            )

            if noise > 0:
                x = np.clip(
                    x + rng.normal(0, noise, size=x.shape).astype(np.float32),
                    -1.0, 1.0,
                )

            X_list.append(x)
            y_list.append(y)

        return np.stack(X_list), np.stack(y_list)

    def _gen_binary(
        self, rng, n: int, noise: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate binary input vectors, apply the threshold rule to get y.
        """
        in_dim = self._in_dim
        X = rng.integers(0, 2, size=(n, in_dim)).astype(np.float32)
        y = self.rule.predict(X)
        return X, y

    def _gen_linear(
        self, rng, n: int, val_range: float, noise: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        General linear mapping: sample X uniformly, apply W·X + b.
        Fallback for MultiLinear and anything else.
        """
        in_dim = self._in_dim
        # Use the range seen in support_X
        x_min = self._support_X.min() * (1.0 + val_range)
        x_max = self._support_X.max() * (1.0 + val_range)
        x_min, x_max = min(x_min, -val_range), max(x_max, val_range)

        X = rng.uniform(x_min, x_max, size=(n, in_dim)).astype(np.float32)
        y = self.rule.predict(X)

        if noise > 0:
            y = y + rng.normal(0, noise, size=y.shape).astype(np.float32)

        X = np.clip(X, -1.0, 1.0)
        y = np.clip(y, -1.5, 1.5).astype(np.float32)
        return X, y
