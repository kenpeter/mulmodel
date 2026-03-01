from __future__ import annotations
import numpy as np

from consolidation.clusterer import Clusterer
from consolidation.distiller import Distiller
from router.encoder import compute_specialty_embedding


# ── Trigger thresholds ────────────────────────────────────────────────────────
SOLVE_COUNT_TRIGGER = 10      # Trigger 1: run background every N problems
MAX_BANK_SIZE = 20            # Trigger 2: force distill when bank too crowded
CLUSTER_MIN_SIZE = 5          # Trigger 3: distill cluster with 5+ models
SIMILARITY_THRESHOLD = 0.85   # cosine sim to form a cluster
MAX_FINETUNE_BEFORE_GROW = 3  # Trigger 4: grow after this many fine-tunes
GROW_LOSS_THRESHOLD = 0.03    # Trigger 4: only grow if loss still above this


class Consolidator:
    """
    Background maintenance for the model bank.

    Trigger 1 — count-based:  run after every N=10 problems solved
    Trigger 2 — bank crowded: force distill if bank > 20 models
    Trigger 3 — cluster full: distill any cluster with 5+ similar models
    Trigger 4 — per-model:    grow a model that has been fine-tuned 3+ times
                               but still has high loss
    """

    def __init__(self) -> None:
        self.clusterer = Clusterer(similarity_threshold=SIMILARITY_THRESHOLD)
        self.distiller = Distiller()

    # ── Trigger 4: check after every fine-tune ────────────────────────────────

    def check_grow(self, model_id: str, bank, router) -> bool:
        """
        Grow model_id if it has been fine-tuned MAX_FINETUNE_BEFORE_GROW times
        and still has loss above GROW_LOSS_THRESHOLD.
        Returns True if growth happened.
        """
        finetune_count = bank.get_finetune_count(model_id)
        last_loss = bank.get_last_loss(model_id)

        if finetune_count < MAX_FINETUNE_BEFORE_GROW:
            return False
        if last_loss <= GROW_LOSS_THRESHOLD:
            return False

        model = bank.get_model(model_id)
        if model is None or not model.can_grow():
            return False

        print(
            f"  [Consolidator] Growing {model_id} "
            f"({model.size}) after {finetune_count} fine-tunes, "
            f"loss={last_loss:.4f}"
        )

        new_model = model.grow()
        old_emb = bank.get_embedding(model_id)
        bank.replace_model(model_id, new_model, old_emb)
        router.register_model(model_id, new_model, old_emb)

        print(f"  [Consolidator] {model_id} grown to {new_model.size}")
        return True

    # ── Triggers 1, 2, 3: background sweep ───────────────────────────────────

    def run(self, bank, router, problem_count: int) -> None:
        """
        Called every N problems (Trigger 1).
        Runs Trigger 2 and 3 internally.
        """
        if problem_count % SOLVE_COUNT_TRIGGER != 0:
            return

        print(f"\n  [Consolidator] Waking up after {problem_count} problems "
              f"(bank size={len(bank)})")

        embeddings = bank.all_embeddings()
        if len(embeddings) < 2:
            print("  [Consolidator] Not enough models to consolidate. Sleeping.")
            return

        clusters = self.clusterer.find_clusters(embeddings)
        print(f"  [Consolidator] Found {len(clusters)} cluster(s): "
              f"{[len(c) for c in clusters]} models each")

        # Trigger 3: distill any cluster >= CLUSTER_MIN_SIZE
        distilled_any = False
        for cluster in clusters:
            if len(cluster) >= CLUSTER_MIN_SIZE:
                self._distill_cluster(cluster, bank, router)
                distilled_any = True

        # Trigger 2: bank still too big → force distill the largest cluster
        if len(bank) > MAX_BANK_SIZE and not distilled_any:
            largest = max(clusters, key=len)
            if len(largest) >= 2:
                print(f"  [Consolidator] Bank too large ({len(bank)}), "
                      f"force distilling largest cluster ({len(largest)} models)")
                self._distill_cluster(largest, bank, router)

        print(f"  [Consolidator] Done. Bank size now={len(bank)}\n")

    # ── internal ─────────────────────────────────────────────────────────────

    def _distill_cluster(
        self, cluster: list[str], bank, router
    ) -> str | None:
        """
        Distill cluster into one student model.
        Removes old models from bank, registers student.
        Returns new model_id or None if skipped.
        """
        teacher_models = [bank.get_model(mid) for mid in cluster]
        teacher_models = [m for m in teacher_models if m is not None]

        if not teacher_models:
            return None

        print(f"  [Consolidator] Distilling cluster of {len(teacher_models)} "
              f"{teacher_models[0].size} models → "
              f"{teacher_models[0]._NEXT_SIZE[teacher_models[0].size]}")

        student = self.distiller.distill_cluster(teacher_models)

        # Compute specialty embedding from average of cluster embeddings
        cluster_embs = [bank.get_embedding(mid) for mid in cluster
                        if bank.get_embedding(mid) is not None]
        specialty_emb = np.mean(cluster_embs, axis=0).astype(np.float32)

        # Inherit generator type from first model
        generator_type = bank.get_generator_type(cluster[0])

        # Remove old models from bank and router
        for mid in cluster:
            bank.remove(mid)
            router.model_index.remove(mid)
            if mid in router._model_store:
                del router._model_store[mid]

        # Register the distilled student
        new_id = bank.register(student, specialty_emb,
                               generator_type=generator_type)
        router.register_model(new_id, student, specialty_emb)

        print(f"  [Consolidator] Distilled → {new_id} ({student.size})")
        return new_id
