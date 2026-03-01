from __future__ import annotations
import numpy as np
from .base import BaseGenerator
from .sequence_prediction import SequencePredictionGenerator
from .arithmetic import ArithmeticGenerator
from .pattern_matching import PatternMatchingGenerator


class GeneratorSelector:
    """
    Picks the best generator for a given problem based on heuristics.

    Heuristics (applied to support_X and raw_input):
    - Binary values (0/1 only) → PatternMatchingGenerator
    - Small integer-like values with op encoding (5-dim input) → ArithmeticGenerator
    - Otherwise → SequencePredictionGenerator (default)
    """

    @staticmethod
    def select(problem) -> BaseGenerator:
        sx = problem.support_X.flatten()
        unique_vals = np.unique(np.round(sx, 3))
        all_binary = np.all((unique_vals == 0.0) | (unique_vals == 1.0))
        if all_binary and problem.support_X.shape[1] in (4, 6, 8):
            return PatternMatchingGenerator()

        if problem.support_X.shape[1] == 5:
            return ArithmeticGenerator()

        return SequencePredictionGenerator()
