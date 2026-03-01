from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.problem import Problem
from router.encoder import ProblemEncoder

# Action indices
ACTION_ROUTE = 0
ACTION_TRAIN = 1


class RoutingPolicy(nn.Module):
    """
    Small MLP: state (128-dim) → action logits (2: route | train_new).
    State = [problem_emb (64) | bank_presence_vec (64)]
    """

    def __init__(self, state_dim: int = 128, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def act(self, state: np.ndarray) -> int:
        self.eval()
        with torch.no_grad():
            logits = self.forward(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(logits, dim=-1).item())


class SFTTrainer:
    """
    Supervised fine-tuning of the RoutingPolicy on synthetic demonstrations.

    Demo generation rules:
    - If bank has a model with similarity > 0.7 → correct action = ROUTE (0)
    - Otherwise → correct action = TRAIN (1)
    """

    def __init__(self, policy: RoutingPolicy, lr: float = 1e-3) -> None:
        self.policy = policy
        self.encoder = ProblemEncoder()
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def _make_demo(
        self, problem_emb: np.ndarray, bank_embs: list[np.ndarray], has_match: bool
    ) -> tuple[np.ndarray, int]:
        bank_vec = np.zeros(64, dtype=np.float32)
        if bank_embs:
            # Aggregate: mean of top-4 most similar bank embeddings
            sims = [
                np.dot(problem_emb, e) / (np.linalg.norm(problem_emb) * np.linalg.norm(e) + 1e-8)
                for e in bank_embs
            ]
            top_idx = np.argsort(sims)[::-1][:4]
            for i in top_idx:
                bank_vec += bank_embs[i] / len(top_idx)

        state = np.concatenate([problem_emb, bank_vec])
        action = ACTION_ROUTE if has_match else ACTION_TRAIN
        return state, action

    def train(
        self,
        problems: list[Problem],
        bank_embeddings: list[np.ndarray],
        match_flags: list[bool],
        epochs: int = 50,
    ) -> list[float]:
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            self.policy.train()
            for problem, has_match in zip(problems, match_flags):
                emb = self.encoder.encode(problem)
                state, action = self._make_demo(emb, bank_embeddings, has_match)
                st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                tgt = torch.tensor([action], dtype=torch.long)
                self.optimizer.zero_grad()
                logits = self.policy(st)
                loss = self.criterion(logits, tgt)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(problems))
        return losses
