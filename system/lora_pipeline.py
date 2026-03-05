"""
LoRA Pipeline for competition math test-time training.

Per-problem:
  SPAWN → generate similar problems → train LoRA patch (100 steps, ~5s)
  ROUTE → cosine similarity to existing patch → run patch (instant)

Bank layout (bank_data/):
  lora_index.json        ← counter + patch metadata
  lora_0/
    lora.pt              ← LoRAPatch state dict
    embedding.npy        ← specialty embedding (base BigModel space)
  lora_1/ ...
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from big_model.transformer import BigModel, MAX_SEQ
from big_model.lora import LoRAPatch
from curriculum.lora_trainer import LoRATrainer
from data_generators.competition_math import generate_similar, infer_template

_ROUTE_THRESHOLD = 0.60   # cosine similarity >= this → reuse existing patch
_AUTOSAVE_EVERY  = 5
_DEFAULT_BANK_DIR = os.path.join(os.path.dirname(__file__), "..", "bank_data")


class LoRAPipeline:
    """
    Competition-math LoRA pipeline.

    Usage:
        pipe = LoRAPipeline(device="cpu")
        answer, meta = pipe.solve("Find x where x^2 ≡ 3 (mod 7).")
    """

    def __init__(
        self,
        bank_dir: str | None = None,
        device: str | None = None,
        load_on_start: bool = True,
        n_train_steps: int | None = None,
        n_similar: int | None = None,
    ) -> None:
        self.bank_dir = os.path.abspath(bank_dir or _DEFAULT_BANK_DIR)
        self.device   = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # GPU: full training; CPU: lighter training so the demo is practical
        _on_gpu = self.device.startswith("cuda")
        self.n_train_steps = n_train_steps if n_train_steps is not None else (100 if _on_gpu else 20)
        self.n_similar     = n_similar     if n_similar     is not None else (60  if _on_gpu else 10)

        # BigModel backbone (frozen during LoRA training)
        self.big_model = self._load_big_model()

        # LoRA trainer (freezes BigModel internally)
        self.trainer = LoRATrainer(self.big_model, device=device)

        # In-memory patch bank
        self._patches: dict[str, LoRAPatch]    = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._template_names: dict[str, str]    = {}
        self._solve_counts: dict[str, int]      = {}
        self._counter: int = 0
        self._problem_count: int = 0

        if load_on_start:
            self._load_bank()

    # ── setup ─────────────────────────────────────────────────────────────────

    def _load_big_model(self) -> BigModel:
        ckpt = os.path.join("big_model_data", "big_model.pt")
        if os.path.exists(ckpt):
            model = BigModel.load(ckpt)
            print(f"  [LoRAPipeline] BigModel loaded from {ckpt}")
        else:
            model = BigModel()
            print(
                "  [LoRAPipeline] BigModel: fresh weights "
                "(run big_model/pretrain.py to pretrain)"
            )
        return model.to(self.device)

    # ── solve ─────────────────────────────────────────────────────────────────

    def solve(self, problem_text: str) -> tuple[float, dict]:
        """
        Solve a competition math problem.

        Returns (predicted_answer, metadata_dict).
        metadata keys: action, patch_id, elapsed, and action-specific keys.
        """
        self._problem_count += 1
        t0 = time.time()

        # Encode with base BigModel (no LoRA) for routing
        prob_emb = self._encode_text(problem_text)

        # Find best existing patch (template match → embedding similarity)
        best_id, best_sim = self._find_best(prob_emb, problem_text)

        if best_id is not None and best_sim >= _ROUTE_THRESHOLD:
            # ── ROUTE: reuse existing patch ───────────────────────────────────
            answer = self._run_patch(self._patches[best_id], problem_text)
            self._solve_counts[best_id] = self._solve_counts.get(best_id, 0) + 1
            meta = {
                "action":     "route",
                "patch_id":   best_id,
                "similarity": best_sim,
                "elapsed":    time.time() - t0,
            }

        else:
            # ── SPAWN: generate similar problems and train a new LoRA patch ───
            similar = generate_similar(problem_text, n=self.n_similar)
            texts   = [ex.text   for ex in similar]
            answers = [ex.answer for ex in similar]

            patch, loss = self.trainer.train(texts, answers, n_steps=self.n_train_steps)

            # Specialty embedding = mean base-BigModel embedding of training texts
            spec_emb = self._mean_embed(texts[:10])

            template_name = infer_template(problem_text).name
            pid = f"lora_{self._counter}"
            self._counter += 1

            self._patches[pid]        = patch
            self._embeddings[pid]     = spec_emb
            self._template_names[pid] = template_name
            self._solve_counts[pid]   = 0

            answer = self._run_patch(patch, problem_text)
            meta = {
                "action":     "spawn",
                "patch_id":   pid,
                "template":   template_name,
                "train_loss": loss,
                "elapsed":    time.time() - t0,
            }

        # Autosave every N problems
        if self._problem_count % _AUTOSAVE_EVERY == 0:
            self.save()

        return answer, meta

    # ── encoding ──────────────────────────────────────────────────────────────

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode one text through base BigModel (no LoRA). Returns (D,) numpy."""
        b      = list(text.encode("utf-8", errors="replace"))[:MAX_SEQ]
        padded = b + [0] * (MAX_SEQ - len(b))
        ids    = torch.tensor([padded], dtype=torch.long,  device=self.device)
        mask   = (ids != 0).long()

        with torch.no_grad():
            hs  = self.big_model.gpt2(input_ids=ids, attention_mask=mask).last_hidden_state
            m   = mask.unsqueeze(-1).float()
            emb = (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)

        return emb[0].cpu().numpy()

    def _mean_embed(self, texts: list[str]) -> np.ndarray:
        """Mean embedding over a list of texts (base BigModel, no LoRA)."""
        embs = np.stack([self._encode_text(t) for t in texts], axis=0)
        return embs.mean(axis=0)

    # ── patch inference ───────────────────────────────────────────────────────

    def _run_patch(self, patch: LoRAPatch, problem_text: str) -> float:
        """
        Encode text WITH LoRA hooks active, then run AnswerHead.

        This matches training conditions (LoRATrainer also encodes with hooks active),
        so the AnswerHead receives the same distribution of embeddings it was trained on.
        """
        b      = list(problem_text.encode("utf-8", errors="replace"))[:MAX_SEQ]
        padded = b + [0] * (MAX_SEQ - len(b))
        ids    = torch.tensor([padded], dtype=torch.long, device=self.device)
        mask   = (ids != 0).long()

        patch.attach(self.big_model.gpt2)
        patch.eval()
        try:
            with torch.no_grad():
                hs   = self.big_model.gpt2(input_ids=ids, attention_mask=mask).last_hidden_state
                m    = mask.unsqueeze(-1).float()
                emb  = (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
                pred = patch.predict(emb)
        finally:
            patch.detach()

        return float(pred.squeeze())

    # ── routing ───────────────────────────────────────────────────────────────

    def _find_best(
        self, prob_emb: np.ndarray, problem_text: str
    ) -> tuple[str | None, float]:
        """
        Return (best_patch_id, score) or (None, -1.0) when no match is found.

        Routing strategy:
          Primary — keyword-based template match.  Identifies the problem type
            from its text and routes to the patch trained on that template.
            Works correctly regardless of BigModel training state.

          After BigModel pretraining, embedding similarity will naturally
          separate problem types in 256-dim space.  At that point, cosine
          similarity between prob_emb and stored specialty embeddings becomes
          a reliable signal for truly novel problem types (no keyword match).
          The specialty embeddings saved in the bank are ready for that upgrade.
        """
        if not self._embeddings:
            return None, -1.0

        # Keyword-based template match (primary routing signal)
        inferred = infer_template(problem_text)
        for mid, tpl in self._template_names.items():
            if tpl == inferred.name:
                return mid, 1.0

        # No template match → caller will SPAWN a new patch
        return None, -1.0

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        os.makedirs(self.bank_dir, exist_ok=True)
        index: dict = {"counter": self._counter, "patches": {}}

        for pid, patch in self._patches.items():
            pdir = os.path.join(self.bank_dir, pid)
            os.makedirs(pdir, exist_ok=True)
            torch.save(patch.lora_state(), os.path.join(pdir, "lora.pt"))
            np.save(os.path.join(pdir, "embedding.npy"), self._embeddings[pid])
            index["patches"][pid] = {
                "template":    self._template_names.get(pid, ""),
                "solve_count": self._solve_counts.get(pid, 0),
            }

        with open(os.path.join(self.bank_dir, "lora_index.json"), "w") as f:
            json.dump(index, f, indent=2)

        print(f"  [Bank] Saved {len(self._patches)} LoRA patch(es) → {self.bank_dir}")

    def _load_bank(self) -> None:
        index_path = os.path.join(self.bank_dir, "lora_index.json")
        if not os.path.exists(index_path):
            print("  [LoRAPipeline] No saved LoRA bank. Starting fresh.")
            return

        with open(index_path) as f:
            index = json.load(f)

        self._counter = index.get("counter", 0)

        for pid, meta in index["patches"].items():
            pdir      = os.path.join(self.bank_dir, pid)
            lora_path = os.path.join(pdir, "lora.pt")
            emb_path  = os.path.join(pdir, "embedding.npy")

            if not os.path.exists(lora_path):
                print(f"  [Bank] Warning: {lora_path} missing, skipping {pid}")
                continue

            patch = LoRAPatch()
            patch.load_lora_state(
                torch.load(lora_path, map_location=self.device, weights_only=True)
            )
            patch.eval()

            self._patches[pid]        = patch
            self._embeddings[pid]     = np.load(emb_path).astype(np.float32)
            self._template_names[pid] = meta.get("template", "")
            self._solve_counts[pid]   = meta.get("solve_count", 0)

        print(f"  [LoRAPipeline] Loaded {len(self._patches)} LoRA patch(es).")

    # ── info ──────────────────────────────────────────────────────────────────

    def bank_size(self) -> int:
        return len(self._patches)

    def status(self) -> None:
        print(f"\n{'─'*64}")
        print(
            f"LoRA Bank: {len(self._patches)} patch(es)   "
            f"Problems solved: {self._problem_count}"
        )
        for pid in self._patches:
            tpl = self._template_names.get(pid, "?")
            sc  = self._solve_counts.get(pid, 0)
            print(f"  {pid:12s}  template={tpl:<26}  solves={sc}")
        print(f"{'─'*64}\n")
