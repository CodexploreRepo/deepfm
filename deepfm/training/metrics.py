"""Metrics for CTR prediction: classification and ranking."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under ROC Curve."""
    return float(roc_auc_score(labels, scores))


def compute_logloss(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute binary cross-entropy (log loss)."""
    # Clip to avoid log(0)
    scores = np.clip(scores, 1e-7, 1 - 1e-7)
    return float(log_loss(labels, scores))


def compute_hr_at_k(rankings: list[np.ndarray], k: int) -> float:
    """Compute Hit Rate @ K across users.

    Args:
        rankings: List of arrays, one per user. Each array contains item
            indices sorted by predicted score (descending). The positive
            item is at index 0 in the ground truth.
        k: Cutoff for top-K.

    Returns:
        Fraction of users where the positive item appears in top-K.
    """
    hits = 0
    for ranking in rankings:
        if 0 in ranking[:k]:
            hits += 1
    return hits / len(rankings)


def compute_ndcg_at_k(rankings: list[np.ndarray], k: int) -> float:
    """Compute NDCG @ K across users.

    For leave-one-out evaluation there is exactly one relevant item per user,
    so IDCG = 1 and NDCG = 1/log2(rank+1) if hit, else 0.

    Args:
        rankings: Same format as compute_hr_at_k.
        k: Cutoff for top-K.

    Returns:
        Average NDCG across users.
    """
    ndcg_sum = 0.0
    for ranking in rankings:
        positions = np.where(ranking[:k] == 0)[0]
        if len(positions) > 0:
            rank = positions[0] + 1  # 1-based rank
            ndcg_sum += 1.0 / np.log2(rank + 1)
    return ndcg_sum / len(rankings)


class RankingEvaluator:
    """Evaluate ranking metrics for leave-one-out protocol.

    Each user has 1 positive + N negatives. The evaluator scores all items,
    ranks them, and computes HR@K and NDCG@K.
    """

    def __init__(self, ks: list[int] | None = None) -> None:
        self.ks = ks or [5, 10, 20]

    def evaluate(
        self,
        user_scores: list[np.ndarray],
        user_labels: list[np.ndarray],
    ) -> dict[str, float]:
        """Compute ranking metrics.

        Args:
            user_scores: List of score arrays, one per user (1 pos + N neg).
            user_labels: List of label arrays, one per user (1 pos + N neg).

        Returns:
            Dict with HR@K and NDCG@K for each K.
        """
        rankings: list[np.ndarray] = []
        for scores, labels in zip(user_scores, user_labels):
            # Sort indices by score descending
            ranked_indices = np.argsort(-scores)
            # Map to original label positions: find where the positive (label=1) ends up
            ranked_labels = labels[ranked_indices]
            # Find position of positive item in ranked list
            rankings.append(ranked_labels)

        metrics: dict[str, float] = {}
        for k in self.ks:
            # For rankings based on labels: positive=1, negative=0
            # HR: does label=1 appear in top-K?
            hits = sum(1 for r in rankings if 1 in r[:k])
            metrics[f"HR@{k}"] = hits / len(rankings)

            # NDCG: 1/log2(rank+1) where rank is 1-based position of label=1
            ndcg_sum = 0.0
            for r in rankings:
                pos = np.where(r[:k] == 1)[0]
                if len(pos) > 0:
                    # +2: 0-indexed to 1-based, +1 for log base
                    ndcg_sum += 1.0 / np.log2(pos[0] + 2)
            metrics[f"NDCG@{k}"] = ndcg_sum / len(rankings)

        return metrics
