"""
Demo: Real-Time Tiny Model Trainer
Solves 3 problems — shows routing + on-the-fly training.

Usage:
    python main.py
"""
from __future__ import annotations
import time
import numpy as np

from core.problem import Problem
from system.pipeline import RTTrainerPipeline


def make_sequence_problem(values: list[float], n_support: int = 3) -> Problem:
    """Build a sequence prediction problem."""
    raw = np.zeros(64, dtype=np.float32)
    v = np.array(values, dtype=np.float32)
    raw[: len(v)] = v / max(abs(v.max()), 1.0)  # rough normalize

    # Support: overlapping windows of length 2 → next element
    sx, sy = [], []
    for i in range(n_support):
        sx.append([values[i] / 256.0, values[i + 1] / 256.0])
        sy.append([values[i + 2] / 256.0])

    return Problem(
        raw_input=raw,
        support_X=np.array(sx, dtype=np.float32),
        support_y=np.array(sy, dtype=np.float32),
        description="sequence_prediction",
    )


def make_pattern_problem(bits: list[int]) -> Problem:
    """Build a binary pattern problem."""
    raw = np.zeros(64, dtype=np.float32)
    raw[: len(bits)] = bits

    sx = np.array([bits], dtype=np.float32)
    majority = 1.0 if sum(bits) > len(bits) / 2 else -1.0
    sy = np.array([[majority]], dtype=np.float32)

    return Problem(
        raw_input=raw,
        support_X=sx,
        support_y=sy,
        description="pattern_matching",
    )


def main() -> None:
    pipeline = RTTrainerPipeline()
    print("=" * 60)
    print("Real-Time Tiny Model Trainer — Demo")
    print("=" * 60)

    # --- Problem 1: New sequence domain → must train ---
    print("\nProblem 1: [0,1,2,3,4,5] (linear+1 sequence, new domain)")
    p1 = make_sequence_problem([0, 1, 2, 3, 4, 5])
    t0 = time.time()
    a1 = pipeline.solve(p1)
    t1 = time.time()
    print(f"  Answer : {a1.value}  (predicted next element)")
    print(f"  Source : {a1.source}")
    print(f"  Trained: {a1.was_trained}   Loss: {a1.loss:.4f}   Time: {t1-t0:.2f}s")
    print(f"  Bank size now: {pipeline.bank_size()}")

    # --- Problem 2: Same domain → should reuse bank model ---
    print("\nProblem 2: [2,4,6,8,10,12] (linear+2, same domain → bank hit expected)")
    p2 = make_sequence_problem([2, 4, 6, 8, 10, 12])
    t0 = time.time()
    a2 = pipeline.solve(p2)
    t1 = time.time()
    print(f"  Answer : {a2.value}")
    print(f"  Source : {a2.source}")
    print(f"  Trained: {a2.was_trained}   Loss: {a2.loss:.4f}   Time: {t1-t0:.4f}s")
    print(f"  Bank size now: {pipeline.bank_size()}")

    # --- Problem 3: New binary pattern domain → must train again ---
    print("\nProblem 3: [1,0,1,0] (binary pattern, new domain)")
    p3 = make_pattern_problem([1, 0, 1, 0])
    t0 = time.time()
    a3 = pipeline.solve(p3)
    t1 = time.time()
    print(f"  Answer : {a3.value}  (1.0=majority, -1.0=minority)")
    print(f"  Source : {a3.source}")
    print(f"  Trained: {a3.was_trained}   Loss: {a3.loss:.4f}   Time: {t1-t0:.2f}s")
    print(f"  Bank size now: {pipeline.bank_size()}")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Problem 1 — was_trained={a1.was_trained}  source={a1.source}")
    print(f"  Problem 2 — was_trained={a2.was_trained}  source={a2.source}")
    print(f"  Problem 3 — was_trained={a3.was_trained}  source={a3.source}")
    print("=" * 60)


if __name__ == "__main__":
    main()
