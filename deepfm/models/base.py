"""Abstract base class for all CTR prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from deepfm.config import ExperimentConfig
from deepfm.data.schema import DatasetSchema
from deepfm.models.layers.embedding import FeatureEmbedding


class BaseCTRModel(nn.Module, ABC):
    """Base class for CTR models built on shared feature embeddings.

    Subclasses implement ``_build_components`` to create model-specific layers
    and ``_forward_components`` to define how the three embedding views are
    combined into a final logit.

    Args:
        schema: Dataset schema describing all input features.
        config: Full experiment configuration.
    """

    def __init__(self, schema: DatasetSchema, config: ExperimentConfig) -> None:
        super().__init__()
        self.schema = schema
        self.config = config

        self.embedding = FeatureEmbedding(
            schema, fm_embed_dim=config.feature.fm_embed_dim
        )
        self._build_components()

    @abstractmethod
    def _build_components(self) -> None:
        """Create model-specific layers (FM, DNN, CIN, Attention, output heads)."""

    @abstractmethod
    def _forward_components(
        self,
        first_order: torch.Tensor,
        field_embeddings: torch.Tensor,
        flat_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Combine embedding views into raw logits (B, 1).

        Args:
            first_order: (B, 1) — summed first-order weights.
            field_embeddings: (B, F, fm_embed_dim) — projected per-field.
            flat_embeddings: (B, total_dim) — raw concatenated.

        Returns:
            Raw logits (B, 1). No sigmoid — loss uses BCEWithLogitsLoss.
        """

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Full forward pass: embedding → components → logits.

        Returns:
            Raw logits (B, 1).
        """
        first_order, field_embeddings, flat_embeddings = self.embedding(batch)
        return self._forward_components(first_order, field_embeddings, flat_embeddings)

    def predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict probabilities (sigmoid applied).

        Returns:
            Probabilities (B, 1) in [0, 1].
        """
        return torch.sigmoid(self.forward(batch))

    def get_l2_reg_loss(self) -> torch.Tensor:
        """Compute L2 regularization loss on embedding parameters."""
        l2_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for param in self.embedding.parameters():
            l2_loss = l2_loss + param.norm(2).pow(2)
        return self.config.feature.embedding_l2_reg * l2_loss
