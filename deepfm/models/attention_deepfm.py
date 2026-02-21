"""AttentionDeepFM: FM + Multi-Head Self-Attention + DNN for CTR prediction."""

from __future__ import annotations

import torch
import torch.nn as nn

from deepfm.models.base import BaseCTRModel
from deepfm.models.layers.attention import MultiHeadSelfAttention
from deepfm.models.layers.dnn import DNN
from deepfm.models.layers.fm import FMInteraction


class AttentionDeepFM(BaseCTRModel):
    """AttentionDeepFM: attention-refined field embeddings + FM + DNN.

    The attention layer refines field_embeddings before they are flattened
    and concatenated with flat_embeddings for the DNN.

    logit = first_order
          + FM(field_embeddings)
          + Linear(DNN(cat(Attention(field_embeddings).flatten(), flat_embeddings)))
    """

    def _build_components(self) -> None:
        self.fm = FMInteraction()
        self.attention = MultiHeadSelfAttention(
            embed_dim=self.config.feature.fm_embed_dim,
            num_heads=self.config.attention.num_heads,
            attention_dim=self.config.attention.attention_dim,
            num_layers=self.config.attention.num_layers,
            use_residual=self.config.attention.use_residual,
        )
        # DNN input: attention-refined field embeddings (flattened) + raw flat embeddings
        attn_flat_dim = (
            self.schema.num_fields * self.config.feature.fm_embed_dim
        )
        dnn_input_dim = attn_flat_dim + self.schema.total_embedding_dim
        self.dnn = DNN(
            input_dim=dnn_input_dim,
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
        # Attention: (B, F, D) → (B, F, D)
        attn_out = self.attention(field_embeddings)
        # Flatten attention output: (B, F*D)
        attn_flat = attn_out.reshape(attn_out.size(0), -1)
        # Concat with raw flat embeddings for DNN
        dnn_input = torch.cat([attn_flat, flat_embeddings], dim=1)
        # DNN: (B, attn_flat_dim + total_dim) → (B, last_hidden) → (B, 1)
        dnn_out = self.output_linear(self.dnn(dnn_input))
        # Combine
        logit = first_order + fm_out + dnn_out
        return logit
