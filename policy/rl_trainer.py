from __future__ import annotations
from collections import deque
import numpy as np
import torch
import torch.optim as optim

from policy.sft_trainer import RoutingPolicy


# Reward values
R_ROUTE_CORRECT = 1.0
R_TRAIN_CORRECT = 0.5
R_ROUTE_WRONG = -0.5
R_TRAIN_WRONG = -1.0
R_LATENCY_PER_SEC = -0.02

_EMA_ALPHA = 0.1  # for baseline


class Episode:
    def __init__(self, state: np.ndarray, action: int, reward: float) -> None:
        self.state = state
        self.action = action
        self.reward = reward


class RLTrainer:
    """
    REINFORCE-based RL trainer for the RoutingPolicy.

    State:  [problem_emb (64) | bank_presence_vec (64)] = 128 dims
    Action: 0=route, 1=train_new
    Reward: defined by routing correctness + latency penalty
    Updates every `update_every` episodes using EMA baseline.
    """

    def __init__(
        self,
        policy: RoutingPolicy,
        lr: float = 3e-4,
        update_every: int = 32,
        gamma: float = 1.0,
    ) -> None:
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.update_every = update_every
        self.gamma = gamma
        self.baseline: float = 0.0
        self._buffer: list[Episode] = []
        self.reward_history: deque[float] = deque(maxlen=200)

    def record(self, state: np.ndarray, action: int, reward: float) -> None:
        self._buffer.append(Episode(state, action, reward))
        self.reward_history.append(reward)

    def _compute_reward(
        self,
        action: int,
        answer_correct: bool,
        latency_sec: float = 0.0,
    ) -> float:
        if action == 0:  # route
            base = R_ROUTE_CORRECT if answer_correct else R_ROUTE_WRONG
        else:  # train new
            base = R_TRAIN_CORRECT if answer_correct else R_TRAIN_WRONG
        return base + R_LATENCY_PER_SEC * latency_sec

    def step(
        self,
        state: np.ndarray,
        action: int,
        answer_correct: bool,
        latency_sec: float = 0.0,
    ) -> float:
        reward = self._compute_reward(action, answer_correct, latency_sec)
        self.record(state, action, reward)
        if len(self._buffer) >= self.update_every:
            self._update()
        return reward

    def _update(self) -> None:
        if not self._buffer:
            return

        rewards = np.array([e.reward for e in self._buffer], dtype=np.float32)
        mean_r = float(rewards.mean())
        # EMA baseline update
        self.baseline = (1.0 - _EMA_ALPHA) * self.baseline + _EMA_ALPHA * mean_r
        advantages = rewards - self.baseline

        self.policy.train()
        total_loss = torch.tensor(0.0, requires_grad=True)
        for ep, adv in zip(self._buffer, advantages):
            st = torch.tensor(ep.state, dtype=torch.float32).unsqueeze(0)
            logits = self.policy(st)
            log_probs = torch.log_softmax(logits, dim=-1)
            log_prob = log_probs[0, ep.action]
            total_loss = total_loss + (-log_prob * float(adv))

        self.optimizer.zero_grad()
        (total_loss / len(self._buffer)).backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        self._buffer.clear()

    def mean_reward(self) -> float:
        if not self.reward_history:
            return 0.0
        return float(np.mean(self.reward_history))
