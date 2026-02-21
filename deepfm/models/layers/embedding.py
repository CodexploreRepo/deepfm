"""Shared feature embedding layer for all CTR models."""

from __future__ import annotations

import torch
import torch.nn as nn

from deepfm.data.schema import DatasetSchema, FeatureType


class FeatureEmbedding(nn.Module):
    """Produces three tensor views from a single forward pass.

    Returns:
        first_order:      (B, 1)                  — linear term for FM
        field_embeddings: (B, num_fields, fm_dim)  — projected for FM/CIN/Attention
        flat_embeddings:  (B, total_dim)           — raw concat for DNN
    """

    def __init__(self, schema: DatasetSchema, fm_embed_dim: int = 16) -> None:
        super().__init__()
        self.schema = schema
        self.fm_embed_dim = fm_embed_dim

        # Ordered field names for consistent tensor construction
        self.field_names = list(schema.fields.keys())

        self.second_order_embeddings = nn.ModuleDict()
        self.first_order_embeddings = nn.ModuleDict()
        self.projections = nn.ModuleDict()

        for field in schema.fields.values():
            name = field.name
            if field.feature_type == FeatureType.SPARSE:
                self.second_order_embeddings[name] = nn.Embedding(
                    field.vocabulary_size, field.embedding_dim, padding_idx=0
                )
                self.first_order_embeddings[name] = nn.Embedding(
                    field.vocabulary_size, 1, padding_idx=0
                )
            elif field.feature_type == FeatureType.SEQUENCE:
                self.second_order_embeddings[name] = nn.EmbeddingBag(
                    field.vocabulary_size,
                    field.embedding_dim,
                    mode=field.combiner,
                    padding_idx=0,
                )
                self.first_order_embeddings[name] = nn.EmbeddingBag(
                    field.vocabulary_size, 1, mode=field.combiner, padding_idx=0
                )
            elif field.feature_type == FeatureType.DENSE:
                # This is for numerical features which don't have vocab, so we use Linear to project the raw value to embedding space
                self.second_order_embeddings[name] = nn.Linear(
                    1, field.embedding_dim
                )
                self.first_order_embeddings[name] = nn.Linear(1, 1)

            # Project each field's embedding to common fm_embed_dim
            if field.embedding_dim != fm_embed_dim:
                self.projections[name] = nn.Linear(
                    field.embedding_dim, fm_embed_dim, bias=False
                )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
                nn.init.xavier_uniform_(module.weight.data[1:])
                # padding_idx=0 stays zero
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    nn.init.zeros_(module.bias.data)

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first_order_parts: list[torch.Tensor] = []
        field_embed_parts: list[torch.Tensor] = []
        flat_embed_parts: list[torch.Tensor] = []

        for name in self.field_names:
            field = self.schema.fields[name]
            x = batch[name]

            # Second-order embedding (raw field dim)
            if field.feature_type == FeatureType.DENSE:
                raw_emb = self.second_order_embeddings[name](x.unsqueeze(-1))
                fo = self.first_order_embeddings[name](x.unsqueeze(-1))
            elif field.feature_type == FeatureType.SEQUENCE:
                # EmbeddingBag expects 2D input (B, L)
                raw_emb = self.second_order_embeddings[name](x)
                fo = self.first_order_embeddings[name](x)
            else:
                # SPARSE
                raw_emb = self.second_order_embeddings[name](x)
                fo = self.first_order_embeddings[name](x)

            # First order: (B, 1)
            if fo.dim() == 1:
                fo = fo.unsqueeze(-1)
            first_order_parts.append(fo)

            # Flat embedding for DNN: (B, field_dim)
            if raw_emb.dim() == 1:
                raw_emb = raw_emb.unsqueeze(-1)
            flat_embed_parts.append(raw_emb)

            # Project to common fm_embed_dim: (B, fm_dim)
            if name in self.projections:
                proj_emb = self.projections[name](raw_emb)
            else:
                proj_emb = raw_emb
            field_embed_parts.append(proj_emb)

        # first_order: sum all (B,1) parts → (B,1)
        first_order = torch.stack(first_order_parts, dim=1).sum(dim=1)

        # field_embeddings: (B, num_fields, fm_embed_dim)
        field_embeddings = torch.stack(field_embed_parts, dim=1)

        # flat_embeddings: (B, total_dim)
        flat_embeddings = torch.cat(flat_embed_parts, dim=-1)

        return first_order, field_embeddings, flat_embeddings
