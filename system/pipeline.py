from __future__ import annotations
import time
import numpy as np

from core.problem import Problem
from core.answer import Answer
from bank.model_bank import ModelBank
from bank.persistence import BankPersistence
from router.router import Router
from router.model_index import ModelIndex
from router.encoder import compute_specialty_embedding
from data_generators.selector import GeneratorSelector
from curriculum.trainer import CurriculumTrainer
from consolidation.consolidator import Consolidator

# Auto-save after every N problems solved
_AUTOSAVE_EVERY = 5


class RTTrainerPipeline:
    """
    Real-Time Trainer Pipeline — same code path at training AND test time.

    On startup:
      Loads saved bank from disk if it exists (long-term memory).

    Per-problem (solve):
      route    → reuse existing model as-is           (instant)
      finetune → a few gradient steps on support data (seconds)
      spawn    → train brand-new micro via curriculum (5-15s)

    Background (every 5 problems):
      Auto-save bank to disk.
      Trigger 1-3: wake consolidator (distill clusters, grow bank).

    Per fine-tune:
      Trigger 4: grow model if fine-tuned too many times with high loss.
    """

    def __init__(
        self,
        bank_dir: str | None = None,
        load_on_start: bool = True,
    ) -> None:
        self.bank         = ModelBank()
        self.model_index  = ModelIndex()
        self.router       = Router(self.model_index)
        self.consolidator = Consolidator()
        self.persistence  = BankPersistence(bank_dir) if bank_dir else BankPersistence()
        self._problem_count = 0

        if load_on_start:
            self._boot()

    # ── startup ───────────────────────────────────────────────────────────────

    def _boot(self) -> None:
        """Load saved bank from disk. Registers models into router."""
        if not self.persistence.exists():
            print("  [Pipeline] No saved bank found. Starting fresh.")
            return

        loaded = self.persistence.load(self.bank)
        if loaded:
            # Re-register all loaded models into the router (with domain)
            for model_id in self.bank.all_model_ids():
                model  = self.bank.get_model(model_id)
                emb    = self.bank.get_embedding(model_id)
                domain = self.bank.get_domain(model_id)
                self.router.register_model(model_id, model, emb, domain=domain)
            print(f"  [Pipeline] Brain loaded: {len(self.bank)} model(s) ready.")

    # ── solve ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _domain(problem: Problem) -> str:
        return "text" if problem.is_text else "math"

    def solve(self, problem: Problem) -> Answer:
        self._problem_count += 1
        domain = self._domain(problem)

        best_id, best_model, best_loss = self.router.try_existing(problem)
        decision = self.router.decide(best_loss, domain=domain)

        # Solve threshold for confidence calculation (domain-specific)
        solve_thr = (
            self.router.TEXT_SOLVE_THRESHOLD
            if domain == "text"
            else self.router.MATH_SOLVE_THRESHOLD
        )

        # ── ROUTE ─────────────────────────────────────────────────────────────
        if decision == "route":
            self.bank.record_solve(best_id, best_loss)
            pred = best_model.infer(problem.raw_input)
            confidence = float(np.clip(1.0 - best_loss / solve_thr, 0.0, 1.0))
            answer = Answer(
                value=pred,
                confidence=confidence,
                source=f"bank:{best_id}",
                was_trained=False,
                loss=best_loss,
            )

        # ── FINE-TUNE ──────────────────────────────────────────────────────────
        elif decision == "finetune":
            new_loss = self.router.finetune(best_model, problem)
            self.bank.record_finetune(best_id, new_loss)

            # Trigger 4: grow if fine-tuned too many times with high loss
            self.consolidator.check_grow(best_id, self.bank, self.router)

            pred = best_model.infer(problem.raw_input)
            confidence = float(np.clip(1.0 - new_loss / solve_thr, 0.0, 1.0))
            answer = Answer(
                value=pred,
                confidence=confidence,
                source=f"finetuned:{best_id}",
                was_trained=False,
                loss=new_loss,
            )

        # ── SPAWN ──────────────────────────────────────────────────────────────
        else:
            generator      = GeneratorSelector.select(problem)
            generator_type = type(generator).__name__

            # Text problems get a "small" model; math gets "micro"
            model_size = "small" if domain == "text" else "micro"
            trainer        = CurriculumTrainer(generator, model_size=model_size)
            model, feeling = trainer.train()

            sx_gen, sy_gen = generator.generate(level=2, n_samples=50)
            specialty_emb  = compute_specialty_embedding(sx_gen, sy_gen)
            new_id = self.bank.register(
                model, specialty_emb, feeling,
                generator_type=generator_type,
                domain=domain,
            )
            self.router.register_model(new_id, model, specialty_emb, domain=domain)

            pred       = model.infer(problem.raw_input)
            confidence = max(0.0, 1.0 - feeling.last_loss(2))
            answer = Answer(
                value=pred,
                confidence=confidence,
                source=f"newly_trained:{new_id}",
                was_trained=True,
                loss=feeling.last_loss(2),
            )

        # ── BACKGROUND: consolidate + autosave ────────────────────────────────
        self.consolidator.run(self.bank, self.router, self._problem_count)
        self._autosave()

        return answer

    # ── persistence helpers ───────────────────────────────────────────────────

    def _autosave(self) -> None:
        if self._problem_count % _AUTOSAVE_EVERY == 0:
            self.persistence.save(self.bank)

    def save(self) -> None:
        """Explicit save — call this on clean shutdown."""
        self.persistence.save(self.bank)

    # ── info ─────────────────────────────────────────────────────────────────

    def bank_size(self) -> int:
        return len(self.bank)

    def problem_count(self) -> int:
        return self._problem_count

    def status(self) -> None:
        """Print a summary of the current bank."""
        print(f"\n{'─'*56}")
        print(f"Bank: {len(self.bank)} model(s)   Problems solved: {self._problem_count}")
        for mid in self.bank.all_model_ids():
            m   = self.bank.get_model(mid)
            ft  = self.bank.get_finetune_count(mid)
            sc  = self.bank.get_solve_count(mid)
            ll  = self.bank.get_last_loss(mid)
            gt  = self.bank.get_generator_type(mid)
            dom = self.bank.get_domain(mid)
            print(
                f"  {mid:10s}  [{dom:4s}]  size={m.size:6s}  "
                f"solves={sc:3d}  finetunes={ft:2d}  "
                f"last_loss={ll:.4f}  gen={gt}"
            )
        print(f"{'─'*56}\n")
