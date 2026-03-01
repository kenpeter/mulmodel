from __future__ import annotations
import numpy as np
from data_generators.pattern_inferrer import PatternInferrer
from data_generators.rule_generator import RuleGenerator
from data_generators.text_generator import TextDataGenerator
from curriculum.generator import CurriculumGenerator

_inferrer = PatternInferrer()


class GeneratorSelector:
    """
    Picks the right generator for any problem — text, math, or mixed.

    Detection order:
      1. Text problem  → TextDataGenerator
         (problem.is_text OR support_texts present)

      2. Math problem  → PatternInferrer → RuleGenerator
         (numeric support_X/support_y)

    This means the same pipeline handles:
      "the movie was great" → +1   (English sentiment)
      [0, 3, 6] → 9               (math sequence)
      "3 plus 5" → 8              (mixed: text + number)
    """

    @staticmethod
    def select(problem) -> CurriculumGenerator:
        # ── Text path ─────────────────────────────────────────────────────────
        if problem.is_text and problem.support_texts:
            return TextDataGenerator(
                support_texts=problem.support_texts,
                support_labels=problem.support_labels,
            )

        # ── Math path ─────────────────────────────────────────────────────────
        sx = np.asarray(problem.support_X, dtype=np.float32)
        sy = np.asarray(problem.support_y, dtype=np.float32)
        if sy.ndim == 1:
            sy = sy.reshape(-1, 1)

        rule = _inferrer.infer(sx, sy)
        return RuleGenerator(rule, sx, sy)
