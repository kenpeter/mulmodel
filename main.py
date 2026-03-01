"""
Demo: Real-Time Tiny Model Trainer — Domain-Aware Routing

Shows domain-aware routing: text problems → text specialist (small MLP),
math problems → math specialist (micro MLP).  Each domain only matches
models trained on the same type of data.

Usage:
    python main.py
"""
from __future__ import annotations
import numpy as np
from core.problem import Problem
from system.pipeline import RTTrainerPipeline
from data_generators.pattern_inferrer import PatternInferrer


def make_problem(
    support_pairs: list[tuple[list[float], float]],
    query: list[float],
    description: str,
) -> Problem:
    """
    Build a problem from human-readable support pairs.

    support_pairs: [([x0, x1, ...], y), ...]
    query: the input we want to predict for
    """
    sx = np.array([p[0] for p in support_pairs], dtype=np.float32)
    sy = np.array([[p[1]] for p in support_pairs], dtype=np.float32)

    raw = np.zeros(64, dtype=np.float32)
    q = np.array(query, dtype=np.float32)
    raw[: len(q)] = q

    return Problem(raw_input=raw, support_X=sx, support_y=sy, description=description)


def show_inferred_rule(support_pairs):
    """Show what rule the PatternInferrer finds before training."""
    sx = np.array([p[0] for p in support_pairs], dtype=np.float32)
    sy = np.array([[p[1]] for p in support_pairs], dtype=np.float32)
    rule = PatternInferrer().infer(sx, sy)
    print(f"  Inferred rule : {rule}")


def make_text_problem(
    support_pairs: list[tuple[str, float]],
    query_text: str,
    description: str,
) -> Problem:
    """Build a text/sentiment problem."""
    support_texts  = [p[0] for p in support_pairs]
    support_labels = [p[1] for p in support_pairs]
    return Problem(
        raw_text=query_text,
        support_texts=support_texts,
        support_labels=support_labels,
        description=description,
    )


def demo(pipeline, problem, label):
    import time
    domain = "TEXT" if problem.is_text else "MATH"
    print(f"\n{'─'*64}")
    print(f"[{domain}] {label}")
    if not problem.is_text:
        show_inferred_rule(problem.metadata.get("pairs", []))
    t0 = time.time()
    a = pipeline.solve(problem)
    dt = time.time() - t0
    print(f"  Answer        : {a.value[0]:.4f}")
    print(f"  Source        : {a.source}")
    print(f"  was_trained   : {a.was_trained}   loss={a.loss:.4f}   time={dt:.2f}s")
    print(f"  Bank size now : {pipeline.bank_size()}")
    return a


def main():
    pipeline = RTTrainerPipeline()
    print("=" * 64)
    print("Real-Time Tiny Model Trainer — Domain-Aware Routing Demo")
    print("TEXT → small MLP specialist   MATH → micro MLP specialist")
    print("Each domain only routes to models of the same type.")
    print("=" * 64)

    # ── Text Problem 1: Sentiment (train a TEXT specialist) ───────────────────
    sentiment_support = [
        ("great product loved it",           1.0),
        ("excellent service very happy",      1.0),
        ("terrible quality broke instantly", -1.0),
        ("awful experience never again",     -1.0),
        ("amazing quality highly recommend",  1.0),
        ("horrible waste of money",          -1.0),
    ]
    pt1 = make_text_problem(
        sentiment_support,
        "wonderful movie loved every scene",
        "sentiment",
    )
    a_t1 = demo(pipeline, pt1, "Sentiment: 'wonderful movie...' → positive?")

    # ── Text Problem 2: Same domain → should REUSE text specialist ────────────
    pt2 = make_text_problem(
        sentiment_support,
        "worst product ever total disaster",
        "sentiment",
    )
    a_t2 = demo(pipeline, pt2, "Sentiment: 'worst product...' → negative? (expects REUSE)")

    # ── Math Problem 1: Linear +3 sequence ────────────────────────────────────
    pairs_1 = [([0.0, 3.0], 6.0), ([3.0, 6.0], 9.0), ([6.0, 9.0], 12.0)]
    p1 = make_problem(pairs_1, [9.0, 12.0], "linear+3 sequence")
    p1.metadata["pairs"] = pairs_1
    p1.support_X /= 15.0
    p1.support_y /= 15.0
    a1 = demo(pipeline, p1, "Linear +3 sequence  [0,3,6,9,...] → predict next")

    # ── Math Problem 2: Same linear+3 → should REUSE math specialist ──────────
    pairs_2 = [([2.0, 5.0], 8.0), ([5.0, 8.0], 11.0), ([8.0, 11.0], 14.0)]
    p2 = make_problem(pairs_2, [11.0, 14.0], "linear+3 offset start")
    p2.metadata["pairs"] = pairs_2
    p2.support_X /= 15.0
    p2.support_y /= 15.0
    a2 = demo(pipeline, p2, "Linear +3 (diff start) → expects bank HIT (MATH only)")

    # ── Math Problem 3: Geometric ×3 ──────────────────────────────────────────
    pairs_3 = [([1.0, 3.0], 9.0), ([3.0, 9.0], 27.0), ([9.0, 27.0], 81.0)]
    p3 = make_problem(pairs_3, [27.0, 81.0], "geometric×3")
    p3.metadata["pairs"] = pairs_3
    p3.support_X /= 100.0
    p3.support_y /= 100.0
    a3 = demo(pipeline, p3, "Geometric ×3  [1,3,9,27,...] → predict next")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("Summary:")
    results = [
        ("Sentiment (new TEXT specialist)",   a_t1, "text"),
        ("Sentiment (reuse TEXT specialist)", a_t2, "text"),
        ("Linear +3 (new MATH specialist)",   a1,   "math"),
        ("Linear +3 (reuse MATH specialist)", a2,   "math"),
        ("Geometric ×3 (new MATH specialist)",a3,   "math"),
    ]
    for label, a, dom in results:
        tag = "TRAINED" if a.was_trained else "REUSED "
        print(f"  {tag}  [{dom.upper()}]  {label:<38} loss={a.loss:.4f}")
    print(f"\n  Final bank size: {pipeline.bank_size()} models")
    pipeline.status()
    print("=" * 64)


if __name__ == "__main__":
    main()
