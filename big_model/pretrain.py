"""
Pretraining script for BigModel (256-wide, 128-layer transformer).

Task: Masked Token Modeling (BERT-style)
  Text → mask 15% bytes → predict masked bytes (cross-entropy)

Data sources:
  1. deepmind/code_contests  — competitive programming problems + solutions
  2. open-r1/codeforces-cots — Codeforces problems + chain-of-thought solutions
  3. Competition math word problems (synthetic fallback)

NOT used here: sentiment reviews, product reviews, or any task-specific data.
Those belong to the tiny specialist models, not the general backbone.

Usage:
  python -m big_model.pretrain                    # 10 epochs
  python -m big_model.pretrain --epochs 20        # more epochs
  python -m big_model.pretrain --epochs 20 --steps 200  # more steps per epoch

Checkpoints: big_model_data/big_model_latest.pt  big_model_data/big_model_best.pt
GPU used automatically if available (strongly recommended).
"""
from __future__ import annotations

import json
import os
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from big_model.transformer import BigModel, MAX_SEQ, BYTE_VOCAB
from data_generators.competition_math import all_examples

# ── Math dataset loader ────────────────────────────────────────────────────────

def _load_math_pool() -> list[str]:
    """
    Load math problems from disk (NuminaMath-CoT, NuminaMath-1.5, Big-Math).
    Falls back to synthetic competition math if datasets not downloaded.
    Each entry is: "<problem> Answer: <answer>"
    """
    base = os.path.join(os.path.dirname(__file__), "..", "data", "math")
    sources = [
        # (subdir, problem_field, answer_field)
        ("numinamath_cot", "problem", "solution"),
        ("numinamath_15",  "problem", "answer"),
        ("bigmath",        "problem", "answer"),
    ]

    pool: list[str] = []
    try:
        from datasets import load_from_disk
        for subdir, prob_field, ans_field in sources:
            path = os.path.abspath(os.path.join(base, subdir))
            try:
                ds = load_from_disk(path)
                for row in ds:
                    prob = row.get(prob_field, "").strip()
                    ans  = row.get(ans_field,  "").strip()
                    if prob and ans:
                        pool.append(f"{prob} Answer: {ans}")
                print(f"  [MathLoader] {subdir}: {len(ds):,} problems loaded")
            except Exception as e:
                print(f"  [MathLoader] {subdir} unavailable ({e}) — skipping")
    except ImportError:
        pass

    if pool:
        print(f"  [MathLoader] Total: {len(pool):,} math problems")
        return pool

    # fallback: synthetic problems
    print("  [MathLoader] No math datasets found — using synthetic fallback")
    print("  [MathLoader] Run: python download_data.py --math")
    return [
        f"{ex.text} Answer: {int(ex.answer)}"
        for ex in all_examples(n_per_template=10_000)
    ]

CHECKPOINT_PATH = "big_model_data/big_model_latest.pt"
BEST_PATH       = "big_model_data/big_model_best.pt"
STATE_PATH      = "big_model_data/train_state.json"
BATCH_SIZE      = 8
LR              = 1e-4
WARMUP_STEPS    = 200
MASK_PROB       = 0.15


# ── Coding dataset loader ─────────────────────────────────────────────────────

def _load_coding_pool() -> list[str]:
    """
    Load coding problems + solutions from:
      - deepmind/code_contests  (HuggingFace cache)
      - open-r1/codeforces-cots (HuggingFace cache)
    Each entry is: "<problem description>\n\nSolution:\n<solution>"
    Falls back to empty list (math-only training) if neither is available.
    """
    pool: list[str] = []

    try:
        from datasets import load_dataset

        # ── deepmind/code_contests ────────────────────────────────────────────
        try:
            ds = load_dataset("deepmind/code_contests", split="train")
            for row in ds:
                desc = (row.get("description") or "").strip()
                sols = row.get("solutions") or {}
                # solutions is a dict with key "solution" (list of strings)
                sol_list = sols.get("solution", []) if isinstance(sols, dict) else []
                if desc and sol_list:
                    sol = sol_list[0].strip()
                    pool.append(f"{desc}\n\nSolution:\n{sol}")
            print(f"  [CodingLoader] deepmind/code_contests: {len(pool):,} examples")
        except Exception as e:
            print(f"  [CodingLoader] deepmind/code_contests unavailable ({e})")

        # ── open-r1/codeforces-cots ───────────────────────────────────────────
        before = len(pool)
        try:
            ds2 = load_dataset("open-r1/codeforces-cots", "solutions", split="train")
            for row in ds2:
                desc = (row.get("description") or "").strip()
                gen  = (row.get("generation")  or "").strip()
                if desc and gen:
                    pool.append(f"{desc}\n\nSolution:\n{gen}")
            print(f"  [CodingLoader] open-r1/codeforces-cots: {len(pool)-before:,} examples")
        except Exception as e:
            print(f"  [CodingLoader] open-r1/codeforces-cots unavailable ({e})")

    except ImportError:
        print("  [CodingLoader] datasets library not installed")

    if pool:
        random.shuffle(pool)
        print(f"  [CodingLoader] Total: {len(pool):,} coding examples")
    else:
        print("  [CodingLoader] No coding datasets found — training on math only")

    return pool


class CodingLoader:
    """Streams coding examples randomly from the pre-loaded pool."""

    _pool: list[str] = []
    _loaded: bool = False

    @classmethod
    def _ensure_loaded(cls) -> None:
        if not cls._loaded:
            cls._loaded = True
            cls._pool = _load_coding_pool()

    @classmethod
    def next_chunk(cls) -> str:
        cls._ensure_loaded()
        if cls._pool:
            text = random.choice(cls._pool)
            # return a random MAX_SEQ-sized window so long solutions get coverage
            if len(text) > MAX_SEQ:
                start = random.randint(0, len(text) - MAX_SEQ)
                return text[start : start + MAX_SEQ]
            return text
        return "Write efficient code to solve competitive programming problems."


# ── BigModelPretrainer ────────────────────────────────────────────────────────

class BigModelPretrainer:
    """
    Pretrain BigModel via masked token modeling on:
      - Coding problems + solutions (deepmind/code_contests, open-r1/codeforces-cots)
      - Competition math word problems (math language understanding)
    """

    def __init__(self, device: str | None = None, resume: bool = True) -> None:
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = BigModel().to(self.device)
        self.optim     = AdamW(self.model.parameters(), lr=LR, weight_decay=0.01)
        self._step     = 0
        self._start_epoch = 1
        self._math_pool: list[str] = _load_math_pool()

        self._best_loss = float("inf")

        # ── resume from checkpoint ────────────────────────────────────────────
        if resume and os.path.exists(CHECKPOINT_PATH):
            self.model.load_state_dict(
                torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=True)
            )
            print(f"  [BigModel] Loaded checkpoint: {CHECKPOINT_PATH}")
        if resume and os.path.exists(STATE_PATH):
            state = json.loads(open(STATE_PATH).read())
            self._start_epoch = state.get("epoch", 1) + 1
            self._step        = state.get("step", 0)
            self._best_loss   = state.get("best_loss", float("inf"))
            print(f"  [BigModel] Resuming from epoch {self._start_epoch}  step {self._step}")
        if not resume:
            print("  [BigModel] --fresh: starting from scratch")

        n = self.model.param_count()
        print(f"  [BigModel] {n:,} params  device={self.device}")
        if self.device == "cpu":
            print("  [BigModel] WARNING: 128-layer model on CPU is very slow.")
            print("             Use your 4070 GPU — set CUDA_VISIBLE_DEVICES=0")

    def _encode_texts(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Byte-encode a list of strings → (input_ids, mask) tensors."""
        ids_list = []
        for text in texts:
            b      = list(text.encode("utf-8", errors="replace"))[:MAX_SEQ]
            padded = b + [0] * (MAX_SEQ - len(b))
            ids_list.append(padded)

        input_ids = torch.tensor(ids_list, dtype=torch.long, device=self.device)
        mask      = (torch.rand(len(texts), MAX_SEQ, device=self.device) < MASK_PROB) \
                    & (input_ids != 0)
        for i in range(len(texts)):
            if not mask[i].any():
                mask[i, random.randint(0, MAX_SEQ - 1)] = True
        return input_ids, mask

    # ── Coding batch ──────────────────────────────────────────────────────────

    def _gen_text_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Coding problems + solutions, byte-encoded."""
        texts = [CodingLoader.next_chunk() for _ in range(BATCH_SIZE)]
        return self._encode_texts(texts)

    # ── Competition math batch ─────────────────────────────────────────────────

    def _gen_math_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Competition math word problems as text, byte-encoded."""
        texts = random.choices(self._math_pool, k=BATCH_SIZE)
        return self._encode_texts(texts)

    # ── training ──────────────────────────────────────────────────────────────

    def _step_text(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits, targets = self.model.pretrain_text(input_ids, mask)
        return nn.functional.cross_entropy(logits, targets)

    def train(self, n_epochs: int = 10, steps_per_epoch: int = 50) -> BigModel:
        """
        Pretrain for n_epochs × steps_per_epoch gradient steps.
        Each step trains on one Wikipedia batch + one math problem batch.
        Checkpoint saved after every epoch.
        """
        scheduler = LinearLR(
            self.optim, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_STEPS
        )
        print(f"  [BigModel] Pretraining: epochs {self._start_epoch}–{n_epochs} × {steps_per_epoch} steps")
        print(f"  [BigModel] Data: coding problems (code_contests + codeforces-cots) + math")

        for epoch in range(self._start_epoch, n_epochs + 1):
            self.model.train()
            epoch_code = 0.0
            epoch_math = 0.0

            for _ in range(steps_per_epoch):
                self.optim.zero_grad()
                code_ids, code_mask = self._gen_text_batch()
                math_ids, math_mask = self._gen_math_batch()
                code_loss = self._step_text(code_ids, code_mask)
                math_loss = self._step_text(math_ids, math_mask)
                (code_loss + math_loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                if self._step < WARMUP_STEPS:
                    scheduler.step()
                self._step  += 1
                epoch_code  += code_loss.item()
                epoch_math  += math_loss.item()

            total_loss = (epoch_code + epoch_math) / steps_per_epoch
            improved   = total_loss < self._best_loss
            if improved:
                self._best_loss = total_loss
            print(
                f"  Epoch {epoch:3d}/{n_epochs}  "
                f"code_loss={epoch_code/steps_per_epoch:.4f}  "
                f"math_loss={epoch_math/steps_per_epoch:.4f}"
                + ("  [best]" if improved else "")
            )
            os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
            self.model.save(CHECKPOINT_PATH)
            if improved:
                self.model.save(BEST_PATH)
            open(STATE_PATH, "w").write(
                json.dumps({"epoch": epoch, "step": self._step, "best_loss": self._best_loss})
            )

        print(f"  [BigModel] Done. Latest: {CHECKPOINT_PATH}  Best: {BEST_PATH}")
        return self.model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Pretrain BigModel on coding problems + math")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps",  type=int, default=50)
    p.add_argument("--resume", action="store_true", default=True,
                   help="Resume from latest checkpoint (default: on)")
    p.add_argument("--fresh",  action="store_true", default=False,
                   help="Start from scratch, ignoring any existing checkpoint")
    args = p.parse_args()
    BigModelPretrainer(resume=not args.fresh).train(n_epochs=args.epochs, steps_per_epoch=args.steps)
