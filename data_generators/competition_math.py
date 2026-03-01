"""
Synthetic competition math problem generator.

When an unseen problem arrives, the system needs training data of the
same TYPE but different parameters to fine-tune the LoRA patch.

Templates implemented (all produce exact integer answers):
  1. ModularArithmetic  — "Find x where x^k ≡ c (mod p)"
  2. LinearEquation     — "Solve ax + b = c for integer x"
  3. ArithmeticSeries   — "Sum of n terms: first=a, diff=d"
  4. GeometricSeries    — "Sum of n terms: first=a, ratio=r (integer)"
  5. Combinatorics      — "Compute C(n, k) mod m"

Each template generates (problem_text, answer) pairs with randomised
parameters so that the LoRA patch learns the PATTERN, not the constants.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class MathExample:
    text: str
    answer: float     # always a number (integer for competition math)
    template: str


# ── individual templates ───────────────────────────────────────────────────────

class ModularArithmetic:
    name = "modular_arithmetic"

    @staticmethod
    def generate(n: int = 50) -> list[MathExample]:
        examples = []
        for _ in range(n):
            p = random.choice([5, 7, 11, 13, 17, 19, 23])
            k = random.randint(2, 4)
            c = random.randint(0, p - 1)
            # find smallest non-negative x where x^k ≡ c (mod p)
            answer = next(
                (x for x in range(p) if pow(x, k, p) == c), 0
            )
            text = (
                f"Find the smallest non-negative integer x "
                f"where x^{k} is congruent to {c} modulo {p}."
            )
            examples.append(MathExample(text, float(answer), ModularArithmetic.name))
        return examples


class LinearEquation:
    name = "linear_equation"

    @staticmethod
    def generate(n: int = 50) -> list[MathExample]:
        examples = []
        for _ in range(n):
            a = random.randint(1, 10)
            x = random.randint(-20, 20)
            b = random.randint(-15, 15)
            c = a * x + b
            text = (
                f"Solve for the integer x: {a}x + {b} = {c}."
            )
            examples.append(MathExample(text, float(x), LinearEquation.name))
        return examples


class ArithmeticSeries:
    name = "arithmetic_series"

    @staticmethod
    def generate(n: int = 50) -> list[MathExample]:
        examples = []
        for _ in range(n):
            a = random.randint(1, 20)
            d = random.randint(1, 10)
            terms = random.randint(5, 20)
            total = terms * a + d * terms * (terms - 1) // 2
            text = (
                f"Find the sum of the first {terms} terms of an arithmetic "
                f"sequence with first term {a} and common difference {d}."
            )
            examples.append(MathExample(text, float(total), ArithmeticSeries.name))
        return examples


class GeometricSeries:
    name = "geometric_series"

    @staticmethod
    def generate(n: int = 50) -> list[MathExample]:
        examples = []
        for _ in range(n):
            a = random.randint(1, 5)
            r = random.randint(2, 4)
            terms = random.randint(3, 8)
            total = a * (r ** terms - 1) // (r - 1)
            text = (
                f"Find the sum of the first {terms} terms of a geometric "
                f"sequence with first term {a} and common ratio {r}."
            )
            examples.append(MathExample(text, float(total), GeometricSeries.name))
        return examples


class Combinatorics:
    name = "combinatorics"

    @staticmethod
    def generate(n: int = 50) -> list[MathExample]:
        examples = []
        for _ in range(n):
            total = random.randint(5, 20)
            k     = random.randint(1, total // 2)
            m     = random.choice([100, 1000, 7, 13])
            answer = math.comb(total, k) % m
            text = (
                f"Compute C({total}, {k}) mod {m}, "
                f"where C(n, k) is the binomial coefficient."
            )
            examples.append(MathExample(text, float(answer), Combinatorics.name))
        return examples


# ── selector ──────────────────────────────────────────────────────────────────

_TEMPLATES = [
    ModularArithmetic,
    LinearEquation,
    ArithmeticSeries,
    GeometricSeries,
    Combinatorics,
]

_KEYWORDS: dict[str, type] = {
    "congruent": ModularArithmetic,
    "modulo":    ModularArithmetic,
    "mod":       ModularArithmetic,
    "solve":     LinearEquation,
    "equation":  LinearEquation,
    "arithmetic": ArithmeticSeries,
    "difference": ArithmeticSeries,
    "geometric":  GeometricSeries,
    "ratio":      GeometricSeries,
    "binomial":   Combinatorics,
    "c(":         Combinatorics,
    "choose":     Combinatorics,
}


def infer_template(problem_text: str) -> type:
    """Guess which template best matches the unseen problem text."""
    low = problem_text.lower()
    for kw, template in _KEYWORDS.items():
        if kw in low:
            return template
    return random.choice(_TEMPLATES)


def generate_similar(problem_text: str, n: int = 60) -> list[MathExample]:
    """
    Given an unseen problem, generate n similar problems of the same type.
    Used to build training data for a new LoRA patch.
    """
    template = infer_template(problem_text)
    return template.generate(n)


def all_examples(n_per_template: int = 20) -> list[MathExample]:
    """Generate examples from all templates (used for pretraining / testing)."""
    examples = []
    for t in _TEMPLATES:
        examples.extend(t.generate(n_per_template))
    return examples
