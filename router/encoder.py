from __future__ import annotations
import numpy as np
from core.problem import Problem, MAX_DIM

_EMB_DIM = 64


def _safe_std(arr: np.ndarray) -> float:
    return float(np.std(arr)) if len(arr) > 1 else 0.0


def _normalize_vec(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-8:
        return np.zeros_like(v)
    return 2.0 * (v - lo) / (hi - lo) - 1.0


class ProblemEncoder:
    """
    Encodes a Problem into a 64-dim float vector using rule-based statistics.

    [0:16]  raw values    – first 16 elements of raw_input
    [16:32] stats of raw  – mean, std, min, max + diff stats
    [32:48] structural    – length estimate, monotonicity, periodicity proxy
    [48:64] support stats – support_X mean/std, support_y mean/std per dim
    """

    EMB_DIM: int = _EMB_DIM

    def encode(self, problem: Problem) -> np.ndarray:
        emb = np.zeros(_EMB_DIM, dtype=np.float32)
        raw = problem.raw_input  # shape (MAX_DIM,)

        # [0:16] first 16 raw values, normalized
        vals = raw[:16]
        lo, hi = vals.min(), vals.max()
        if hi - lo > 1e-8:
            emb[0:16] = np.clip(2.0 * (vals - lo) / (hi - lo) - 1.0, -1.0, 1.0)

        # [16:32] statistical features of non-zero portion
        nonzero_mask = raw != 0.0
        active = raw[nonzero_mask] if nonzero_mask.any() else raw[:1]
        diffs = np.diff(active) if len(active) > 1 else np.array([0.0])

        stats = np.array([
            np.mean(active), _safe_std(active),
            float(np.min(active)), float(np.max(active)),
            np.mean(diffs), _safe_std(diffs),
            float(np.min(diffs)), float(np.max(diffs)),
        ], dtype=np.float32)
        normed_stats = np.clip(stats / (np.abs(stats).max() + 1e-8), -1.0, 1.0)
        emb[16:24] = normed_stats

        # [24:32] second-order diffs
        diffs2 = np.diff(diffs) if len(diffs) > 1 else np.array([0.0])
        stats2 = np.array([
            np.mean(diffs2), _safe_std(diffs2),
            float(np.min(diffs2)), float(np.max(diffs2)),
        ], dtype=np.float32)
        normed_stats2 = np.clip(stats2 / (np.abs(stats2).max() + 1e-8), -1.0, 1.0)
        emb[24:28] = normed_stats2

        # [28:32] structural: length, monotonic, periodic proxy, range
        length_norm = float(nonzero_mask.sum()) / MAX_DIM
        monotone = float(np.all(diffs >= 0) or np.all(diffs <= 0))
        hi_lo_range = float(hi - lo) / (abs(lo) + abs(hi) + 1e-8)
        emb[28] = length_norm * 2.0 - 1.0
        emb[29] = monotone
        emb[30] = np.clip(hi_lo_range, -1.0, 1.0)
        emb[31] = 0.0  # reserved

        # [32:48] support_X statistics
        sx = problem.support_X
        if sx.size > 0:
            flat = sx.flatten()
            s_stats = np.array([
                np.mean(flat), _safe_std(flat),
                float(np.min(flat)), float(np.max(flat)),
                np.mean(sx, axis=0).mean() if sx.ndim > 1 else np.mean(sx),
                _safe_std(sx.mean(axis=0)) if sx.ndim > 1 else 0.0,
                float(sx.shape[0]) / 100.0,   # n_support normalized
                float(sx.shape[1] if sx.ndim > 1 else 1) / 64.0,
            ], dtype=np.float32)
            s_norm = np.clip(s_stats / (np.abs(s_stats).max() + 1e-8), -1.0, 1.0)
            emb[32:40] = s_norm
            # Pad remaining with zeros
            emb[40:48] = 0.0

        # [48:64] support_y statistics
        sy = problem.support_y
        if sy.size > 0:
            flat_y = sy.flatten()
            y_stats = np.array([
                np.mean(flat_y), _safe_std(flat_y),
                float(np.min(flat_y)), float(np.max(flat_y)),
                float(sy.shape[0]) / 100.0,
                float(sy.shape[1] if sy.ndim > 1 else 1) / 64.0,
            ], dtype=np.float32)
            y_norm = np.clip(y_stats / (np.abs(y_stats).max() + 1e-8), -1.0, 1.0)
            emb[48:54] = y_norm
            emb[54:64] = 0.0

        return emb


def compute_specialty_embedding(
    support_X: np.ndarray, support_y: np.ndarray
) -> np.ndarray:
    """
    Compute a 64-dim specialty embedding from a model's training data.
    Used to register models in the ModelIndex.
    """
    emb = np.zeros(_EMB_DIM, dtype=np.float32)
    if support_X.size == 0:
        return emb

    flat_x = support_X.flatten()
    flat_y = support_y.flatten()
    diffs_x = np.diff(flat_x) if len(flat_x) > 1 else np.array([0.0])

    # [0:16] first 16 X values
    vals = flat_x[:16]
    lo, hi = vals.min(), vals.max()
    if hi - lo > 1e-8:
        emb[0:16] = np.clip(2.0 * (vals - lo) / (hi - lo) - 1.0, -1.0, 1.0)

    # [16:32] X stats
    x_stats = np.array([
        np.mean(flat_x), _safe_std(flat_x),
        float(np.min(flat_x)), float(np.max(flat_x)),
        np.mean(diffs_x), _safe_std(diffs_x),
        float(np.min(diffs_x)), float(np.max(diffs_x)),
    ], dtype=np.float32)
    emb[16:24] = np.clip(x_stats / (np.abs(x_stats).max() + 1e-8), -1.0, 1.0)

    # [48:54] y stats
    y_stats = np.array([
        np.mean(flat_y), _safe_std(flat_y),
        float(np.min(flat_y)), float(np.max(flat_y)),
        float(support_X.shape[0]) / 100.0,
        float(support_X.shape[1] if support_X.ndim > 1 else 1) / 64.0,
    ], dtype=np.float32)
    emb[48:54] = np.clip(y_stats / (np.abs(y_stats).max() + 1e-8), -1.0, 1.0)

    return emb
