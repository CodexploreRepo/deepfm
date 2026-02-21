"""xDeepFM: CIN + DNN for CTR prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from deepfm.models.base import BaseCTRModel
from deepfm.models.layers.cin import CIN
from deepfm.models.layers.dnn import DNN


class xDeepFM(BaseCTRModel):
    """xDeepFM model: CIN (explicit vector-wise) + DNN (implicit higher-order).

    logit = first_order + Linear(CIN(field_embeddings)) + Linear(DNN(flat_embeddings))
    """

    def _build_components(self) -> None:
        self.cin = CIN(
            num_fields=self.schema.num_fields,
            embed_dim=self.config.feature.fm_embed_dim,
            layer_sizes=self.config.cin.layer_sizes,
            split_half=self.config.cin.split_half,
        )
        self.dnn = DNN(
            input_dim=self.schema.total_embedding_dim,
            hidden_units=self.config.dnn.hidden_units,
            activation=self.config.dnn.activation,
            dropout=self.config.dnn.dropout,
            use_batch_norm=self.config.dnn.use_batch_norm,
        )
        self.cin_linear = nn.Linear(self.cin.output_dim, 1)
        self.dnn_linear = nn.Linear(self.dnn.output_dim, 1)

    def _forward_components(
        self,
        first_order: torch.Tensor,
        field_embeddings: torch.Tensor,
        flat_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        # CIN: (B, F, D) → (B, cin_output_dim) → (B, 1)
        cin_out = self.cin_linear(self.cin(field_embeddings))
        # DNN: (B, total_dim) → (B, last_hidden) → (B, 1)
        dnn_out = self.dnn_linear(self.dnn(flat_embeddings))
        # Combine
        logit = first_order + cin_out + dnn_out
        return logit
