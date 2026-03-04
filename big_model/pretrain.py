"""
Pretraining script for BigModel (256-wide, 128-layer transformer).

Task: Masked Token Modeling (BERT-style)
  Text → mask 15% bytes → predict masked bytes (cross-entropy)

Data sources:
  1. Wikipedia (HuggingFace datasets, streaming) — general world knowledge
  2. Competition math word problems (synthetic) — math problem understanding

NOT used here: sentiment reviews, product reviews, or any task-specific data.
Those belong to the tiny specialist models, not the general backbone.

Usage:
  python -m big_model.pretrain                    # 10 epochs
  python -m big_model.pretrain --epochs 20        # more epochs
  python -m big_model.pretrain --epochs 20 --steps 200  # more steps per epoch

Checkpoint: big_model_data/big_model.pt
GPU used automatically if available (strongly recommended).
"""
from __future__ import annotations

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

CHECKPOINT_PATH = "big_model_data/big_model.pt"
BATCH_SIZE      = 8
LR              = 1e-4
WARMUP_STEPS    = 200
MASK_PROB       = 0.15


# ── Wikipedia text loader ─────────────────────────────────────────────────────

class WikipediaLoader:
    """
    Loads Wikitext-103 from disk (data/wikitext/, downloaded by download_data.py).
    Falls back to diverse synthetic text if the dataset is not present.
    """

    _buffer: list[str] = []
    _stream = None
    _tried: bool = False

    @classmethod
    def _init_stream(cls) -> None:
        wikitext_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "wikitext"
        )
        try:
            from datasets import load_from_disk
            ds = load_from_disk(os.path.abspath(wikitext_dir))
            cls._stream = iter(ds)
            print(f"  [WikipediaLoader] Loaded Wikitext-103 from {wikitext_dir}")
        except Exception as e:
            print(f"  [WikipediaLoader] Wikitext-103 unavailable ({e})")
            print("  [WikipediaLoader] Falling back to synthetic general text.")
            print("  [WikipediaLoader] Run: python download_data.py --text")
            cls._stream = None

    @classmethod
    def next_chunk(cls) -> str:
        """Return the next ~512-char chunk of text."""
        if not cls._buffer:
            if not cls._tried:
                cls._tried = True
                cls._init_stream()

            if cls._stream is not None:
                try:
                    row  = next(cls._stream)
                    text = row.get("text", "")
                    for i in range(0, len(text), MAX_SEQ):
                        chunk = text[i : i + MAX_SEQ].strip()
                        if len(chunk) > 20:
                            cls._buffer.append(chunk)
                except StopIteration:
                    cls._stream = None  # exhausted; keep using fallback

        if cls._buffer:
            return cls._buffer.pop(0)

        return random.choice(_FALLBACK_TEXT)


# Diverse fallback covering science, math, history, nature — not reviews
_FALLBACK_TEXT = [
    "The speed of light in a vacuum is approximately 299792458 metres per second.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    "The Pythagorean theorem states that a squared plus b squared equals c squared.",
    "Newton's second law relates force mass and acceleration as F equals ma.",
    "DNA is a double helix molecule that carries genetic information in living organisms.",
    "The French Revolution began in 1789 and fundamentally changed European society.",
    "Prime numbers are integers greater than one with no positive divisors except one and themselves.",
    "The mitochondria generate most of the cell's supply of adenosine triphosphate.",
    "Gravity causes objects with mass to attract one another proportional to their masses.",
    "The Roman Empire at its height controlled much of Europe North Africa and the Middle East.",
    "Calculus was independently developed by Newton and Leibniz in the seventeenth century.",
    "Water molecules consist of two hydrogen atoms covalently bonded to one oxygen atom.",
    "The human brain contains approximately eighty six billion neurons.",
    "Black holes are regions of spacetime where gravity is so strong that nothing can escape.",
    "The periodic table organises chemical elements by atomic number and electron configuration.",
    "Fibonacci numbers appear frequently in nature including in the arrangement of leaves and flowers.",
    "The speed of sound in air at sea level is approximately 343 metres per second.",
    "Quantum mechanics describes the behaviour of particles at the atomic and subatomic scale.",
    "The Earth orbits the Sun at an average distance of about 150 million kilometres.",
    "Evolution by natural selection was proposed by Charles Darwin in 1859.",
    "Electrical resistance in a conductor is measured in ohms according to Ohm's law.",
    "The cerebral cortex is responsible for higher cognitive functions including language and reasoning.",
    "Plate tectonics explains how the Earth's crust is divided into moving plates.",
    "The area of a circle is pi multiplied by the radius squared.",
    "Light behaves both as a wave and as a particle depending on how it is observed.",
]


# ── BigModelPretrainer ────────────────────────────────────────────────────────

class BigModelPretrainer:
    """
    Pretrain BigModel via masked token modeling on:
      - Wikipedia text (general world knowledge)
      - Competition math word problems (math language understanding)
    """

    def __init__(self, device: str | None = None) -> None:
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = BigModel().to(self.device)
        self.optim     = AdamW(self.model.parameters(), lr=LR, weight_decay=0.01)
        self._step     = 0
        self._math_pool: list[str] = _load_math_pool()

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

    # ── Wikipedia text batch ──────────────────────────────────────────────────

    def _gen_text_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Wikipedia articles (or fallback synthetic knowledge text), byte-encoded."""
        texts = [WikipediaLoader.next_chunk() for _ in range(BATCH_SIZE)]
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
        print(f"  [BigModel] Pretraining: {n_epochs} epochs × {steps_per_epoch} steps")
        print(f"  [BigModel] Data: Wikipedia text + competition math problems")

        for epoch in range(1, n_epochs + 1):
            self.model.train()
            epoch_wiki = 0.0
            epoch_math = 0.0

            for _ in range(steps_per_epoch):
                self.optim.zero_grad()
                wiki_ids, wiki_mask = self._gen_text_batch()
                math_ids, math_mask = self._gen_math_batch()
                wiki_loss = self._step_text(wiki_ids, wiki_mask)
                math_loss = self._step_text(math_ids, math_mask)
                (wiki_loss + math_loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                if self._step < WARMUP_STEPS:
                    scheduler.step()
                self._step  += 1
                epoch_wiki  += wiki_loss.item()
                epoch_math  += math_loss.item()

            print(
                f"  Epoch {epoch:3d}/{n_epochs}  "
                f"wiki_loss={epoch_wiki/steps_per_epoch:.4f}  "
                f"math_loss={epoch_math/steps_per_epoch:.4f}"
            )
            self.model.save(CHECKPOINT_PATH)

        print(f"  [BigModel] Done. Checkpoint: {CHECKPOINT_PATH}")
        return self.model


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Pretrain BigModel on Wikipedia + math")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--steps",  type=int, default=50)
    args = p.parse_args()
    BigModelPretrainer().train(n_epochs=args.epochs, steps_per_epoch=args.steps)
