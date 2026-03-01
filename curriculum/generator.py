from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class CurriculumGenerator(ABC):
    """Abstract base for curriculum-aware data generators used by CurriculumTrainer."""

    @abstractmethod
    def generate(
        self,
        level: int,
        n_samples: int = 500,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) for the given difficulty level (0=easy, 1=medium, 2=hard)."""

    @abstractmethod
    def input_dim(self, level: int) -> int:
        """Input dimension for the given level."""

    @abstractmethod
    def output_dim(self) -> int:
        """Output dimension."""
