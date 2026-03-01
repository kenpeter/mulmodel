from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseGenerator(ABC):
    """Abstract interface for all curriculum data generators."""

    LEVELS = (0, 1, 2)  # easy, medium, hard

    @abstractmethod
    def generate(
        self,
        level: int,
        n_samples: int = 500,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) arrays for the given difficulty level."""

    @abstractmethod
    def input_dim(self, level: int) -> int:
        """Input feature dimension for the given level."""

    @abstractmethod
    def output_dim(self) -> int:
        """Output dimension (same across levels)."""
