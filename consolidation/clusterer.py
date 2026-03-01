from __future__ import annotations
import numpy as np
from router.similarity import cosine_similarity


class Clusterer:
    """
    Groups bank models into clusters by embedding cosine similarity.

    Uses greedy single-pass:
      - For each model, if it is similar enough to an existing cluster centroid
        → join that cluster
      - Otherwise → start a new cluster

    Returns list of clusters, each cluster = list of model_ids.
    """

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self.similarity_threshold = similarity_threshold

    def find_clusters(
        self, embeddings: dict[str, np.ndarray]
    ) -> list[list[str]]:
        if not embeddings:
            return []

        clusters: list[list[str]] = []
        centroids: list[np.ndarray] = []

        for model_id, emb in embeddings.items():
            best_cluster = -1
            best_sim = -1.0

            for i, centroid in enumerate(centroids):
                sim = cosine_similarity(emb, centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = i

            if best_sim >= self.similarity_threshold:
                clusters[best_cluster].append(model_id)
                # Update centroid as mean of all members
                member_embs = [embeddings[mid] for mid in clusters[best_cluster]]
                centroids[best_cluster] = np.mean(member_embs, axis=0)
            else:
                clusters.append([model_id])
                centroids.append(emb.copy())

        return clusters
