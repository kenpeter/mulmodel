"""
SFT + RL training entry point for the RoutingPolicy.

Usage:
    python train_policy.py
"""
from __future__ import annotations
import numpy as np

from core.problem import Problem
from router.encoder import ProblemEncoder, compute_specialty_embedding
from data_generators.sequence_prediction import SequencePredictionGenerator
from data_generators.arithmetic import ArithmeticGenerator
from data_generators.pattern_matching import PatternMatchingGenerator
from policy.sft_trainer import RoutingPolicy, SFTTrainer
from policy.rl_trainer import RLTrainer
from system.pipeline import RTTrainerPipeline


def _make_seq_problem(offset: int = 0) -> Problem:
    gen = SequencePredictionGenerator()
    sx, sy = gen.generate(level=1, n_samples=5, seed=offset)
    raw = np.zeros(64, dtype=np.float32)
    raw[:8] = sx[0]
    return Problem(raw_input=raw, support_X=sx, support_y=sy, description="sequence")


def _make_arith_problem(offset: int = 0) -> Problem:
    gen = ArithmeticGenerator()
    sx, sy = gen.generate(level=0, n_samples=5, seed=offset + 100)
    raw = np.zeros(64, dtype=np.float32)
    raw[:5] = sx[0]
    return Problem(raw_input=raw, support_X=sx, support_y=sy, description="arithmetic")


def _make_pattern_problem(offset: int = 0) -> Problem:
    gen = PatternMatchingGenerator()
    sx, sy = gen.generate(level=0, n_samples=5, seed=offset + 200)
    raw = np.zeros(64, dtype=np.float32)
    raw[:4] = sx[0]
    return Problem(raw_input=raw, support_X=sx, support_y=sy, description="pattern")


def run_sft(policy: RoutingPolicy, n_demos: int = 60) -> None:
    print("\n=== SFT Phase ===")
    trainer = SFTTrainer(policy)
    problems, match_flags = [], []

    # Half demos: bank has a match (route)
    bank_embs = []
    gen = SequencePredictionGenerator()
    sx, sy = gen.generate(level=2, n_samples=20)
    bank_embs.append(compute_specialty_embedding(sx, sy))

    for i in range(n_demos // 3):
        problems.append(_make_seq_problem(i))
        match_flags.append(True)
    for i in range(n_demos // 3):
        problems.append(_make_arith_problem(i))
        match_flags.append(False)
    for i in range(n_demos // 3):
        problems.append(_make_pattern_problem(i))
        match_flags.append(False)

    losses = trainer.train(problems, bank_embs, match_flags, epochs=30)
    print(f"SFT loss: {losses[0]:.4f} → {losses[-1]:.4f}")


def run_rl(policy: RoutingPolicy, n_episodes: int = 128) -> None:
    print("\n=== RL Phase ===")
    pipeline = RTTrainerPipeline()
    encoder = ProblemEncoder()
    rl = RLTrainer(policy, update_every=32)

    for ep in range(n_episodes):
        problem = _make_seq_problem(ep) if ep % 2 == 0 else _make_pattern_problem(ep)
        prob_emb = encoder.encode(problem)
        bank_vec = np.zeros(64, dtype=np.float32)
        state = np.concatenate([prob_emb, bank_vec])

        action = policy.act(state)
        answer = pipeline.solve(problem)
        correct = answer.loss < 0.1

        reward = rl.step(state, action, correct, latency_sec=0.0)

        if (ep + 1) % 32 == 0:
            print(f"  Episode {ep+1:3d}: mean_reward={rl.mean_reward():.3f}")

    print(f"RL done. Final mean reward: {rl.mean_reward():.3f}")


if __name__ == "__main__":
    policy = RoutingPolicy()
    run_sft(policy)
    run_rl(policy)
    print("\nPolicy training complete.")
