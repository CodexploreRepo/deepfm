"""Schema definitions for the generic feature contract."""

from dataclasses import dataclass, field
from enum import Enum


class FeatureType(Enum):
    SPARSE = "sparse"
    DENSE = "dense"
    SEQUENCE = "sequence"


@dataclass
class FieldSchema:
    name: str
    feature_type: FeatureType
    vocabulary_size: int = 0
    embedding_dim: int = 8
    group: str = ""
    max_length: int = 1
    combiner: str = "mean"


@dataclass
class DatasetSchema:
    fields: dict[str, FieldSchema] = field(default_factory=dict)
    label_field: str = "label"

    @property
    def sparse_fields(self) -> list[FieldSchema]:
        return [
            f
            for f in self.fields.values()
            if f.feature_type == FeatureType.SPARSE
        ]

    @property
    def dense_fields(self) -> list[FieldSchema]:
        return [
            f
            for f in self.fields.values()
            if f.feature_type == FeatureType.DENSE
        ]

    @property
    def sequence_fields(self) -> list[FieldSchema]:
        return [
            f
            for f in self.fields.values()
            if f.feature_type == FeatureType.SEQUENCE
        ]

    @property
    def num_fields(self) -> int:
        return len(self.fields)

    @property
    def total_embedding_dim(self) -> int:
        return sum(f.embedding_dim for f in self.fields.values())
