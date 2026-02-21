"""DeepFM: FM + DNN for CTR prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from deepfm.models.base import BaseCTRModel
from deepfm.models.layers.dnn import DNN
from deepfm.models.layers.fm import FMInteraction


class DeepFM(BaseCTRModel):
    """DeepFM model: shared embeddings → FM (2nd-order) + DNN (higher-order).

    logit = first_order + FM(field_embeddings) + Linear(DNN(flat_embeddings))
    """

    def _build_components(self) -> None:
        self.fm = FMInteraction()
        self.dnn = DNN(
            input_dim=self.schema.total_embedding_dim,
            hidden_units=self.config.dnn.hidden_units,
            activation=self.config.dnn.activation,
            dropout=self.config.dnn.dropout,
            use_batch_norm=self.config.dnn.use_batch_norm,
        )
        self.output_linear = nn.Linear(self.dnn.output_dim, 1)

    def _forward_components(
        self,
        first_order: torch.Tensor,
        field_embeddings: torch.Tensor,
        flat_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        # FM second-order: (B, 1)
        fm_out = self.fm(field_embeddings)
        # DNN: (B, total_dim) → (B, last_hidden) → (B, 1)
        dnn_out = self.output_linear(self.dnn(flat_embeddings))
        # Combine
        logit = first_order + fm_out + dnn_out
        return logit
