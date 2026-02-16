from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


class MetricCalculator:
    """Accumulates predictions and labels, computes classification metrics."""

    def __init__(self):
        self.predictions: List[np.ndarray] = []
        self.labels: List[np.ndarray] = []

    def reset(self):
        self.predictions = []
        self.labels = []

    def update(self, predictions: np.ndarray, labels: np.ndarray):
        self.predictions.append(predictions)
        self.labels.append(labels)

    def compute(self) -> Dict[str, float]:
        preds = np.concatenate(self.predictions)
        labels = np.concatenate(self.labels)
        preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)

        return {
            "auc": float(roc_auc_score(labels, preds)),
            "logloss": float(log_loss(labels, preds_clipped)),
        }


def compute_ranking_metrics(
    scores: np.ndarray,
    num_users: int,
    candidates_per_user: int,
    ks: List[int] = (5, 10, 20),
) -> Dict[str, float]:
    """Compute HR@K and NDCG@K for leave-one-out ranking evaluation.

    Args:
        scores: (num_users * candidates_per_user,) predicted scores.
                For each user block: first score is the positive item,
                remaining are negatives.
        num_users: Number of users.
        candidates_per_user: 1 positive + num_neg negatives.
        ks: Cutoff values for ranking metrics.

    Returns:
        Dict with hr@k and ndcg@k for each k.
    """
    scores = scores.reshape(num_users, candidates_per_user)

    # For each user, rank items by score (descending)
    # The positive item is at index 0 in the original order
    # We need to find where it ranks after sorting
    rankings = np.argsort(-scores, axis=1)  # (num_users, candidates_per_user)

    # Position of the positive item (index 0) in the ranked list
    # rankings[u, r] = original_index means item at original_index is at rank r
    # We want: for each user, at which rank r does rankings[u, r] == 0?
    pos_ranks = np.where(rankings == 0)[1]  # 0-indexed rank of positive item

    metrics = {}
    for k in ks:
        # HR@K: 1 if positive item is in top K, else 0
        hits = (pos_ranks < k).astype(np.float32)
        metrics[f"hr@{k}"] = float(hits.mean())

        # NDCG@K: 1/log2(rank+2) if in top K, else 0
        # rank is 0-indexed, so log2(rank+2) maps rank 0 -> log2(2)=1
        ndcg = np.where(pos_ranks < k, 1.0 / np.log2(pos_ranks + 2), 0.0)
        metrics[f"ndcg@{k}"] = float(ndcg.mean())

    return metrics
