from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn

from deepfm.config import ExperimentConfig
from deepfm.data.schema import DatasetSchema
from deepfm.models.layers.embedding import FeatureEmbedding


class BaseCTRModel(ABC, nn.Module):
    """Abstract base class for all CTR prediction models.

    Holds the shared FeatureEmbedding layer, defines the output interface,
    and provides L2 regularization loss computation.
    Subclasses implement _build_components() and _forward_components().
    """

    def __init__(self, schema: DatasetSchema, config: ExperimentConfig):
        super().__init__()
        self.schema = schema
        self.config = config

        self.embedding = FeatureEmbedding(
            schema=schema,
            fm_embedding_dim=config.feature.fm_embedding_dim,
        )

        self._build_components()

    @abstractmethod
    def _build_components(self):
        """Initialize model-specific layers (FM, DNN, CIN, etc.)."""

    @abstractmethod
    def _forward_components(
        self,
        first_order: torch.Tensor,
        field_embeddings: torch.Tensor,
        flat_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute model-specific logits.

        Args:
            first_order:      (B, 1) — first-order linear contribution
            field_embeddings: (B, F, fm_dim) — for FM/CIN/Attention
            flat_embeddings:  (B, total_dim) — for DNN

        Returns:
            logit: (B, 1)
        """

    def forward(
        self,
        sparse: torch.Tensor,
        dense: torch.Tensor,
        sequences: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        first_order, field_embeddings, flat_embeddings = self.embedding(
            sparse, dense, sequences
        )
        logit = self._forward_components(first_order, field_embeddings, flat_embeddings)
        return logit  # raw logit — BCEWithLogitsLoss handles sigmoid

    def predict(
        self,
        sparse: torch.Tensor,
        dense: torch.Tensor,
        sequences: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        logit = self.forward(sparse, dense, sequences)
        return torch.sigmoid(logit)

    def get_l2_reg_loss(self) -> torch.Tensor:
        """Compute L2 regularization over embedding weights."""
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for emb in self.embedding.sparse_embeddings.values():
            reg = reg + torch.norm(emb.weight, p=2)
        for emb in self.embedding.sequence_embeddings.values():
            reg = reg + torch.norm(emb.weight, p=2)
        return self.config.feature.embedding_l2_reg * reg
