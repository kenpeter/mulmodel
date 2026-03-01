from __future__ import annotations
import time
import numpy as np

from core.problem import Problem
from core.answer import Answer
from bank.model_bank import ModelBank
from router.router import Router
from router.model_index import ModelIndex
from router.encoder import compute_specialty_embedding
from data_generators.selector import GeneratorSelector
from curriculum.trainer import CurriculumTrainer


class RTTrainerPipeline:
    """
    Real-Time Trainer Pipeline.

    solve(problem) — same code path at training AND test time:
      1. Try existing models via Router
      2a. Low loss → reuse, return Answer(was_trained=False)
      2b. High loss → generate curriculum data, train new TinyModel,
                      register in bank, return Answer(was_trained=True)
    """

    def __init__(self) -> None:
        self.bank = ModelBank()
        self.model_index = ModelIndex()
        self.router = Router(self.model_index)

    def solve(self, problem: Problem) -> Answer:
        t_start = time.time()

        best_id, best_model, best_loss = self.router.try_existing(problem)

        if best_model is not None and not self.router.needs_new_model(best_loss):
            # --- Bank hit: reuse existing model ---
            pred = best_model.infer(problem.raw_input)
            confidence = float(np.clip(1.0 - best_loss / self.router.SOLVE_THRESHOLD, 0.0, 1.0))
            return Answer(
                value=pred,
                confidence=confidence,
                source=f"bank:{best_id}",
                was_trained=False,
                loss=best_loss,
            )

        # --- Must train a new tiny model ---
        generator = GeneratorSelector.select(problem)
        trainer = CurriculumTrainer(generator, model_size="micro")
        model, feeling = trainer.train()

        # Register in bank + router
        sx_gen, sy_gen = generator.generate(level=2, n_samples=50)
        specialty_emb = compute_specialty_embedding(sx_gen, sy_gen)
        new_id = self.bank.register(model, specialty_emb, feeling)
        self.router.register_model(new_id, model, specialty_emb)

        pred = model.infer(problem.raw_input)
        elapsed = time.time() - t_start
        confidence = max(0.0, 1.0 - feeling.last_loss(2))

        return Answer(
            value=pred,
            confidence=confidence,
            source=f"newly_trained:{new_id}",
            was_trained=True,
            loss=feeling.last_loss(2),
        )

    def bank_size(self) -> int:
        return len(self.bank)
