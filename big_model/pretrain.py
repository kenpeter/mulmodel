"""
Pretraining script for BigModel (1024-wide, 96-layer GPT-style causal LM).

Memory strategy:
  Model in BF16 on GPU  → ~2.3 GB VRAM for weights
  FP32 optimizer states → ~9.2 GB CPU RAM  (AdamW runs on CPU)
  Gradient checkpointing → low activation memory
  Peak VRAM during backward: ~5 GB  (fits comfortably in 12 GB)

Data sources:
  1. deepmind/code_contests  — competitive programming problems + solutions
  2. open-r1/codeforces-cots — Codeforces problems + chain-of-thought solutions

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

CHECKPOINT_PATH = "big_model_data/big_model_latest.pt"
BEST_PATH       = "big_model_data/big_model_best.pt"
STATE_PATH      = "big_model_data/train_state.json"
BATCH_SIZE      = 2
LR              = 1e-4
WARMUP_STEPS    = 200


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
    Pretrain BigModel via causal language modeling on coding datasets.

    Memory layout:
      - Model in BF16 on GPU  (~2.3 GB VRAM)
      - FP32 optimizer states on CPU RAM  (~9.2 GB)
      - Gradient checkpointing reduces activation memory
    """

    def __init__(self, device: str | None = None, resume: bool = True) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_gpu = self.device != "cpu"

        # Build model in FP32 on CPU first (for clean checkpoint loading)
        self.model = BigModel()

        # ── resume from checkpoint ────────────────────────────────────────────
        if resume and os.path.exists(CHECKPOINT_PATH):
            try:
                self.model.load_state_dict(
                    torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True),
                    strict=False,
                )
                print(f"  [BigModel] Loaded checkpoint: {CHECKPOINT_PATH}")
            except RuntimeError as e:
                print(f"  [BigModel] Checkpoint incompatible — starting fresh.")
                print(f"             ({e!s:.120})")
                resume = False

        self._start_epoch = 1
        self._step        = 0
        self._best_loss   = float("inf")
        if resume and os.path.exists(STATE_PATH):
            state = json.loads(open(STATE_PATH).read())
            self._start_epoch = state.get("epoch", 1) + 1
            self._step        = state.get("step", 0)
            self._best_loss   = state.get("best_loss", float("inf"))
            print(f"  [BigModel] Resuming from epoch {self._start_epoch}  step {self._step}")
        if not resume:
            print("  [BigModel] Starting from scratch")

        # ── CPU FP32 params for optimizer (before converting model to BF16) ──
        # AdamW states (m, v) live on CPU RAM, not VRAM.
        self._cpu_params = [
            nn.Parameter(p.detach().float().cpu(), requires_grad=True)
            for p in self.model.parameters()
        ]
        self.optim = AdamW(self._cpu_params, lr=LR, weight_decay=0.01)

        # ── Move model to GPU in BF16 ────────────────────────────────────────
        if use_gpu:
            self.model = self.model.cuda().to(torch.bfloat16)
        else:
            print("  [BigModel] WARNING: CPU-only mode, training will be slow.")

        n = self.model.param_count()
        vram = torch.cuda.memory_allocated() / 1e9 if use_gpu else 0
        print(f"  [BigModel] {n:,} params  BF16 on GPU={use_gpu}")
        print(f"  [BigModel] VRAM after model load: {vram:.2f} GB")
        print(f"  [BigModel] FP32 optimizer states on CPU (~{n*8/1e9:.1f} GB)")

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Byte-encode a list of strings → input_ids tensor on GPU."""
        ids_list = []
        for text in texts:
            b      = list(text.encode("utf-8", errors="replace"))[:MAX_SEQ]
            padded = b + [0] * (MAX_SEQ - len(b))
            ids_list.append(padded)
        return torch.tensor(ids_list, dtype=torch.long, device=self.device)

    # ── Coding batch ──────────────────────────────────────────────────────────

    def _gen_text_batch(self) -> torch.Tensor:
        """Coding problems + solutions, byte-encoded."""
        texts = [CodingLoader.next_chunk() for _ in range(BATCH_SIZE)]
        return self._encode_texts(texts)

    # ── CPU-optimizer step ────────────────────────────────────────────────────

    def _optimizer_step(self) -> None:
        """
        1. Clip GPU BF16 gradients.
        2. Copy BF16 grads → FP32 on CPU.
        3. Run FP32 AdamW on CPU.
        4. Copy updated FP32 params → BF16 on GPU.
        """
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        for gpu_p, cpu_p in zip(self.model.parameters(), self._cpu_params):
            if gpu_p.grad is not None:
                if cpu_p.grad is None:
                    cpu_p.grad = gpu_p.grad.float().cpu()
                else:
                    cpu_p.grad.copy_(gpu_p.grad.float())
                gpu_p.grad = None  # free VRAM immediately

        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        # Write updated FP32 params back to BF16 GPU model
        for gpu_p, cpu_p in zip(self.model.parameters(), self._cpu_params):
            gpu_p.data.copy_(cpu_p.data.to(device=gpu_p.device, dtype=gpu_p.dtype))

    def _save_checkpoint(self, path: str) -> None:
        """Save FP32 checkpoint from CPU params (highest precision copy)."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = {}
        for (name, _), cpu_p in zip(self.model.named_parameters(), self._cpu_params):
            state[name] = cpu_p.data.cpu()
        torch.save(state, path)
        print(f"  [BigModel] Saved → {path}")

    # ── training ──────────────────────────────────────────────────────────────

    def train(self, n_epochs: int = 10, steps_per_epoch: int = 50) -> BigModel:
        """
        Pretrain for n_epochs × steps_per_epoch gradient steps.
        Each step trains on one coding batch.
        Checkpoint saved after every epoch.
        """
        scheduler = LinearLR(
            self.optim, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_STEPS
        )
        end_epoch = self._start_epoch + n_epochs - 1
        print(f"  [BigModel] Pretraining: epochs {self._start_epoch}–{end_epoch} × {steps_per_epoch} steps")
        print(f"  [BigModel] Data: coding problems (code_contests + codeforces-cots)")

        for epoch in range(self._start_epoch, end_epoch + 1):
            self.model.train()
            epoch_loss = 0.0

            for _ in range(steps_per_epoch):
                code_ids = self._gen_text_batch()
                loss = self.model.pretrain_causal(code_ids)
                loss.backward()
                self._optimizer_step()
                if self._step < WARMUP_STEPS:
                    scheduler.step()
                self._step   += 1
                epoch_loss   += loss.item()

            total_loss = epoch_loss / steps_per_epoch
            improved   = total_loss < self._best_loss
            if improved:
                self._best_loss = total_loss
            print(
                f"  Epoch {epoch:3d}/{end_epoch}  "
                f"code_loss={total_loss:.4f}"
                + ("  [best]" if improved else "")
            )
            os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
            self._save_checkpoint(CHECKPOINT_PATH)
            if improved:
                self._save_checkpoint(BEST_PATH)
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
