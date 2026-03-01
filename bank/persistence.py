from __future__ import annotations
import json
import os
import numpy as np
import torch

from tiny_model.model import TinyModel
from curriculum.feeling import FeelingTracker, LevelRecord
from bank.model_bank import ModelBank

# Default location for the persistent bank on disk
DEFAULT_BANK_DIR = os.path.join(
    os.path.dirname(__file__), "..", "bank_data"
)


# ─────────────────────────────────────────────────────────────────────────────
# Serialise / deserialise FeelingTracker
# ─────────────────────────────────────────────────────────────────────────────

def _feeling_to_dict(feeling: FeelingTracker) -> dict:
    out = {}
    for level in range(3):
        out[str(level)] = [
            {
                "level": r.level,
                "epoch": r.epoch,
                "train_loss": r.train_loss,
                "val_loss": r.val_loss,
                "val_acc": r.val_acc,
            }
            for r in feeling.records[level]
        ]
    return out


def _dict_to_feeling(d: dict) -> FeelingTracker:
    feeling = FeelingTracker()
    for level_str, records in d.items():
        level = int(level_str)
        for r in records:
            feeling.records[level].append(
                LevelRecord(
                    level=r["level"],
                    epoch=r["epoch"],
                    train_loss=r["train_loss"],
                    val_loss=r["val_loss"],
                    val_acc=r["val_acc"],
                )
            )
    return feeling


# ─────────────────────────────────────────────────────────────────────────────
# BankPersistence
# ─────────────────────────────────────────────────────────────────────────────

class BankPersistence:
    """
    Saves and loads the entire ModelBank to/from disk.

    Directory layout:
      bank_data/
        index.json          ← model metadata + counter
        model_0/
          weights.pt        ← TinyModel state_dict
          embedding.npy     ← 64-dim specialty embedding
          feeling.json      ← FeelingTracker records (optional)
        model_1/
          ...
    """

    def __init__(self, bank_dir: str = DEFAULT_BANK_DIR) -> None:
        self.bank_dir = os.path.abspath(bank_dir)

    # ── save ─────────────────────────────────────────────────────────────────

    def save(self, bank: ModelBank) -> None:
        os.makedirs(self.bank_dir, exist_ok=True)

        index: dict = {
            "counter": bank._counter,
            "models": {},
        }

        for model_id in bank.all_model_ids():
            model = bank.get_model(model_id)
            emb   = bank.get_embedding(model_id)
            feeling = bank.get_feeling(model_id)

            model_dir = os.path.join(self.bank_dir, model_id)
            os.makedirs(model_dir, exist_ok=True)

            # Save weights
            torch.save(
                model.state_dict(),
                os.path.join(model_dir, "weights.pt"),
            )

            # Save embedding
            np.save(os.path.join(model_dir, "embedding.npy"), emb)

            # Save feeling if present
            if feeling is not None:
                with open(os.path.join(model_dir, "feeling.json"), "w") as f:
                    json.dump(_feeling_to_dict(feeling), f, indent=2)

            # Store metadata in index
            index["models"][model_id] = {
                "input_dim":      model.input_dim,
                "output_dim":     model.output_dim,
                "size":           model.size,
                "generator_type": bank.get_generator_type(model_id),
                "finetune_count": bank.get_finetune_count(model_id),
                "solve_count":    bank.get_solve_count(model_id),
                "last_loss":      bank.get_last_loss(model_id),
                "domain":         bank.get_domain(model_id),
            }

        with open(os.path.join(self.bank_dir, "index.json"), "w") as f:
            json.dump(index, f, indent=2)

        print(f"  [Bank] Saved {len(bank)} model(s) → {self.bank_dir}")

    # ── load ─────────────────────────────────────────────────────────────────

    def load(self, bank: ModelBank) -> bool:
        """
        Load saved models into bank.
        Returns True if anything was loaded, False if no saved bank exists.
        """
        index_path = os.path.join(self.bank_dir, "index.json")
        if not os.path.exists(index_path):
            return False

        with open(index_path) as f:
            index = json.load(f)

        bank._counter = index.get("counter", 0)

        for model_id, meta in index["models"].items():
            model_dir = os.path.join(self.bank_dir, model_id)

            weights_path = os.path.join(model_dir, "weights.pt")
            emb_path     = os.path.join(model_dir, "embedding.npy")
            feeling_path = os.path.join(model_dir, "feeling.json")

            if not os.path.exists(weights_path):
                print(f"  [Bank] Warning: weights missing for {model_id}, skipping")
                continue

            # Rebuild TinyModel architecture then load weights
            model = TinyModel(
                input_dim=meta["input_dim"],
                output_dim=meta["output_dim"],
                size=meta["size"],
            )
            model.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )
            model.eval()

            emb = np.load(emb_path) if os.path.exists(emb_path) else np.zeros(64)

            feeling = None
            if os.path.exists(feeling_path):
                with open(feeling_path) as f:
                    feeling = _dict_to_feeling(json.load(f))

            # Register directly into bank stores (bypass counter increment)
            bank._models[model_id]          = model
            bank._embeddings[model_id]      = emb.astype(np.float32)
            bank._generator_types[model_id] = meta.get("generator_type", "")
            bank._finetune_counts[model_id] = meta.get("finetune_count", 0)
            bank._solve_counts[model_id]    = meta.get("solve_count", 0)
            bank._last_losses[model_id]     = meta.get("last_loss", 1.0)
            bank._domains[model_id]         = meta.get("domain", "math")
            if feeling is not None:
                bank._feelings[model_id] = feeling

        n = len(index["models"])
        print(f"  [Bank] Loaded {n} model(s) from {self.bank_dir}")
        return n > 0

    # ── helpers ──────────────────────────────────────────────────────────────

    def exists(self) -> bool:
        return os.path.exists(os.path.join(self.bank_dir, "index.json"))
