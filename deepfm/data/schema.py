from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class FeatureType(Enum):
    SPARSE = "sparse"
    DENSE = "dense"
    SEQUENCE = "sequence"


@dataclass
class FieldSchema:
    """Describes a single input feature field.

    Attributes:
        name: Unique identifier used as dict key everywhere.
        feature_type: SPARSE, DENSE, or SEQUENCE.
        vocabulary_size: Number of unique values (includes OOV slot at index 0).
        embedding_dim: Per-field learned embedding dimension.
        is_label: True if this field is the prediction target.
        group: Semantic grouping ("user", "item", "context").
        default_value: Fill value for missing entries.
        max_length: For SEQUENCE features, padded length.
        combiner: For SEQUENCE features: "mean", "sum", or "max" pooling.
    """

    name: str
    feature_type: FeatureType
    vocabulary_size: int = 0
    embedding_dim: int = 16
    is_label: bool = False
    group: str = "default"
    default_value: Any = 0
    max_length: int = 1
    combiner: str = "mean"


@dataclass
class DatasetSchema:
    """Complete description of a dataset's feature space."""

    fields: Dict[str, FieldSchema]
    label_field: str

    @property
    def sparse_fields(self) -> List[FieldSchema]:
        return [
            f
            for f in self.fields.values()
            if f.feature_type == FeatureType.SPARSE and not f.is_label
        ]

    @property
    def dense_fields(self) -> List[FieldSchema]:
        return [
            f
            for f in self.fields.values()
            if f.feature_type == FeatureType.DENSE and not f.is_label
        ]

    @property
    def sequence_fields(self) -> List[FieldSchema]:
        return [
            f
            for f in self.fields.values()
            if f.feature_type == FeatureType.SEQUENCE and not f.is_label
        ]

    @property
    def num_fields(self) -> int:
        """Total number of non-label feature fields."""
        return (
            len(self.sparse_fields)
            + len(self.dense_fields)
            + len(self.sequence_fields)
        )

    @property
    def total_embedding_dim(self) -> int:
        """Total dimension when all field embeddings are concatenated (for DNN input)."""
        dim = 0
        for f in self.sparse_fields:
            dim += f.embedding_dim
        for f in self.dense_fields:
            dim += f.embedding_dim if f.embedding_dim > 0 else 1
        for f in self.sequence_fields:
            dim += f.embedding_dim
        return dim
