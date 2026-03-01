"""
Demo: Competition Math — LoRA Test-Time Training

Demonstrates test-time LoRA training for competition math:
  - Unseen template type → SPAWN a new LoRA patch (100 gradient steps, ~5s)
  - Same template again  → ROUTE to existing patch (instant, cosine similarity)

Templates demonstrated:
  1. ModularArithmetic  — "Find x where x^k ≡ c (mod p)"
  2. LinearEquation     — "Solve ax + b = c for integer x"
  3. GeometricSeries    — "Sum of first n terms, ratio r"
  4. Modular again      → expects ROUTE
  5. Linear again       → expects ROUTE

Usage:
    python main.py
"""
from __future__ import annotations

import torch
from system.lora_pipeline import LoRAPipeline


def demo(pipe: LoRAPipeline, problem_text: str, label: str) -> None:
    print(f"\n{'─'*64}")
    print(f"[PROBLEM] {label}")
    print(f"  Text: {problem_text}")

    answer, meta = pipe.solve(problem_text)
    action = meta["action"].upper()

    print(f"  Predicted answer : {answer:.2f}")
    print(f"  Action           : {action}  (patch={meta['patch_id']})")
    if action == "SPAWN":
        print(f"  Template trained : {meta['template']}   loss={meta['train_loss']:.4f}")
    else:
        print(f"  Cosine similarity: {meta['similarity']:.4f}")
    print(f"  Time             : {meta['elapsed']:.2f}s   Bank size: {pipe.bank_size()}")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # GPU: full training (100 steps, 60 examples).
    # CPU: lighter training (20 steps, 10 examples) to keep demo practical.
    if device == "cuda":
        n_steps, n_similar = 100, 60
    else:
        n_steps, n_similar = 20, 10

    pipe = LoRAPipeline(
        device=device,
        n_train_steps=n_steps,
        n_similar=n_similar,
    )

    print("=" * 64)
    print("Competition Math — LoRA Test-Time Training Demo")
    print(f"Device: {device}   train_steps={n_steps}   n_similar={n_similar}")
    print("SPAWN = new LoRA patch trained from scratch")
    print("ROUTE = existing patch reused   (instant, cosine sim)")
    print("=" * 64)

    # ── Problem 1: Modular arithmetic (bank is empty → SPAWN) ────────────────
    demo(
        pipe,
        "Find the smallest non-negative integer x "
        "where x^2 is congruent to 3 modulo 7.",
        "#1 Modular arithmetic  (expects SPAWN)",
    )

    # ── Problem 2: Linear equation (different template → SPAWN) ──────────────
    demo(
        pipe,
        "Solve for the integer x: 3x + 5 = 20.",
        "#2 Linear equation  (expects SPAWN)",
    )

    # ── Problem 3: Geometric series (different template → SPAWN) ─────────────
    demo(
        pipe,
        "Find the sum of the first 5 terms of a geometric sequence "
        "with first term 2 and common ratio 3.",
        "#3 Geometric series  (expects SPAWN)",
    )

    # ── Problem 4: Modular arithmetic again (same template → ROUTE) ──────────
    demo(
        pipe,
        "Find the smallest non-negative integer x "
        "where x^3 is congruent to 2 modulo 11.",
        "#4 Modular arithmetic again  (expects ROUTE)",
    )

    # ── Problem 5: Linear equation again (same template → ROUTE) ─────────────
    demo(
        pipe,
        "Solve for the integer x: 7x + 3 = 38.",
        "#5 Linear equation again  (expects ROUTE)",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print("Summary:")
    pipe.status()
    print("=" * 64)


if __name__ == "__main__":
    main()
