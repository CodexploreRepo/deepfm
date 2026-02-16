from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from deepfm.data.schema import DatasetSchema, FeatureType


class FeatureEmbedding(nn.Module):
    """Unified embedding layer that handles sparse, dense, and sequence features.

    Constructed entirely from a DatasetSchema — no hard-coded feature names.
    Per-field custom embedding dimensions with projection to a common
    fm_embedding_dim for FM/CIN/Attention components.

    Returns:
        first_order:      (B, 1)           — sum of first-order terms + bias
        field_embeddings: (B, F, fm_dim)   — projected to common dim for FM
        flat_embeddings:  (B, total_dim)   — raw concatenated for DNN
    """

    def __init__(self, schema: DatasetSchema, fm_embedding_dim: int = 16):
        super().__init__()
        self.schema = schema
        self.fm_embedding_dim = fm_embedding_dim

        sparse_fields = schema.sparse_fields
        dense_fields = schema.dense_fields
        sequence_fields = schema.sequence_fields

        # --- Sparse embeddings (second-order) ---
        self.sparse_embeddings = nn.ModuleDict()
        self.sparse_projections = nn.ModuleDict()
        for f in sparse_fields:
            self.sparse_embeddings[f.name] = nn.Embedding(
                num_embeddings=f.vocabulary_size,
                embedding_dim=f.embedding_dim,
                padding_idx=0,
            )
            # Projection to common FM dim if needed
            if f.embedding_dim != fm_embedding_dim:
                self.sparse_projections[f.name] = nn.Linear(
                    f.embedding_dim, fm_embedding_dim, bias=False
                )

        # --- Sparse first-order embeddings ---
        self.sparse_first_order = nn.ModuleDict()
        for f in sparse_fields:
            self.sparse_first_order[f.name] = nn.Embedding(
                num_embeddings=f.vocabulary_size,
                embedding_dim=1,
                padding_idx=0,
            )

        # --- Dense feature handling ---
        self.dense_first_order = nn.ParameterDict()
        self.dense_projections = nn.ModuleDict()
        for f in dense_fields:
            self.dense_first_order[f.name] = nn.Parameter(torch.zeros(1))
            if f.embedding_dim > 0:
                self.dense_projections[f.name] = nn.Linear(1, f.embedding_dim)

        # Projection for dense fields to FM dim
        self.dense_fm_projections = nn.ModuleDict()
        for f in dense_fields:
            if f.embedding_dim > 0 and f.embedding_dim != fm_embedding_dim:
                self.dense_fm_projections[f.name] = nn.Linear(
                    f.embedding_dim, fm_embedding_dim, bias=False
                )
            elif f.embedding_dim == 0:
                # Raw scalar → project to fm_dim
                self.dense_fm_projections[f.name] = nn.Linear(
                    1, fm_embedding_dim, bias=False
                )

        # --- Sequence embeddings ---
        self.sequence_embeddings = nn.ModuleDict()
        self.sequence_projections = nn.ModuleDict()
        self.sequence_first_order = nn.ModuleDict()
        for f in sequence_fields:
            self.sequence_embeddings[f.name] = nn.EmbeddingBag(
                num_embeddings=f.vocabulary_size,
                embedding_dim=f.embedding_dim,
                mode=f.combiner,
                padding_idx=0,
            )
            self.sequence_first_order[f.name] = nn.EmbeddingBag(
                num_embeddings=f.vocabulary_size,
                embedding_dim=1,
                mode=f.combiner,
                padding_idx=0,
            )
            if f.embedding_dim != fm_embedding_dim:
                self.sequence_projections[f.name] = nn.Linear(
                    f.embedding_dim, fm_embedding_dim, bias=False
                )

        # Global bias
        self.bias = nn.Parameter(torch.zeros(1))

        # Store field names in order for consistent indexing
        self._sparse_names = [f.name for f in sparse_fields]
        self._dense_names = [f.name for f in dense_fields]
        self._sequence_names = [f.name for f in sequence_fields]

        self._init_weights()

    def _init_weights(self):
        for emb in self.sparse_embeddings.values():
            nn.init.xavier_uniform_(emb.weight.data[1:])
        for emb in self.sparse_first_order.values():
            nn.init.xavier_uniform_(emb.weight.data[1:])
        for emb in self.sequence_embeddings.values():
            nn.init.xavier_uniform_(emb.weight.data[1:])

    def forward(
        self,
        sparse: torch.Tensor,
        dense: torch.Tensor,
        sequences: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            sparse:    (B, num_sparse_fields) LongTensor
            dense:     (B, num_dense_fields)  FloatTensor
            sequences: dict of {field_name: (B, max_length) LongTensor}

        Returns:
            first_order:      (B, 1)
            field_embeddings: (B, num_fields, fm_embedding_dim)
            flat_embeddings:  (B, total_embedding_dim)
        """
        B = sparse.size(0)
        first_order_terms = [self.bias.expand(B, 1)]
        field_embs = []  # projected to fm_dim
        flat_parts = []  # raw dims concatenated

        # --- Sparse ---
        for i, name in enumerate(self._sparse_names):
            idx = sparse[:, i]  # (B,)

            # First order
            fo = self.sparse_first_order[name](idx)  # (B, 1)
            first_order_terms.append(fo)

            # Second order
            emb = self.sparse_embeddings[name](idx)  # (B, field_dim)
            flat_parts.append(emb)

            # Project to FM dim
            if name in self.sparse_projections:
                proj = self.sparse_projections[name](emb)  # (B, fm_dim)
            else:
                proj = emb
            field_embs.append(proj)

        # --- Dense ---
        for i, name in enumerate(self._dense_names):
            val = dense[:, i : i + 1]  # (B, 1)

            # First order
            fo = val * self.dense_first_order[name]  # (B, 1)
            first_order_terms.append(fo)

            # Embedding / pass-through
            field = self.schema.dense_fields[i]
            if field.embedding_dim > 0:
                emb = self.dense_projections[name](val)  # (B, field_dim)
                flat_parts.append(emb)
                if name in self.dense_fm_projections:
                    proj = self.dense_fm_projections[name](emb)
                else:
                    proj = emb
            else:
                flat_parts.append(val)
                proj = self.dense_fm_projections[name](val)  # (B, fm_dim)
            field_embs.append(proj)

        # --- Sequences ---
        if sequences:
            for name in self._sequence_names:
                seq = sequences[name]  # (B, max_length)

                # First order
                fo = self.sequence_first_order[name](seq)  # (B, 1)
                first_order_terms.append(fo)

                # Second order
                emb = self.sequence_embeddings[name](seq)  # (B, field_dim)
                flat_parts.append(emb)

                if name in self.sequence_projections:
                    proj = self.sequence_projections[name](emb)  # (B, fm_dim)
                else:
                    proj = emb
                field_embs.append(proj)

        # Combine
        first_order = torch.stack(first_order_terms, dim=1).sum(dim=1)  # (B, 1)
        field_embeddings = torch.stack(field_embs, dim=1)  # (B, F, fm_dim)
        flat_embeddings = torch.cat(flat_parts, dim=1)  # (B, total_dim)

        return first_order, field_embeddings, flat_embeddings
