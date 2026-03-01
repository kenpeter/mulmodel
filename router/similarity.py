from __future__ import annotations
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def rank_models(
    problem_emb: np.ndarray,
    model_embeddings: dict[str, np.ndarray],
) -> list[tuple[str, float]]:
    """
    Rank all models by cosine similarity to problem_emb.
    Returns list of (model_id, score) sorted descending.
    """
    scores = [
        (mid, cosine_similarity(problem_emb, emb))
        for mid, emb in model_embeddings.items()
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
