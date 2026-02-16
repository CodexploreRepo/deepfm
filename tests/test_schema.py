"""Tests for FieldSchema, DatasetSchema, and FeatureType."""

from deepfm.data.schema import DatasetSchema, FeatureType, FieldSchema


def _make_schema():
    """Helper: build a schema with all 3 feature types."""
    fields = {
        "user_id": FieldSchema(
            name="user_id",
            feature_type=FeatureType.SPARSE,
            vocabulary_size=100,
            embedding_dim=16,
            group="user",
        ),
        "movie_id": FieldSchema(
            name="movie_id",
            feature_type=FeatureType.SPARSE,
            vocabulary_size=200,
            embedding_dim=16,
            group="item",
        ),
        "age": FieldSchema(
            name="age",
            feature_type=FeatureType.DENSE,
            embedding_dim=8,
            group="user",
        ),
        "genres": FieldSchema(
            name="genres",
            feature_type=FeatureType.SEQUENCE,
            vocabulary_size=20,
            embedding_dim=8,
            max_length=6,
            combiner="mean",
            group="item",
        ),
        "label": FieldSchema(
            name="label",
            feature_type=FeatureType.SPARSE,
            is_label=True,
        ),
    }
    return DatasetSchema(fields=fields, label_field="label")


class TestFeatureType:
    def test_enum_values(self):
        assert FeatureType.SPARSE.value == "sparse"
        assert FeatureType.DENSE.value == "dense"
        assert FeatureType.SEQUENCE.value == "sequence"

    def test_from_string(self):
        assert FeatureType("sparse") == FeatureType.SPARSE
        assert FeatureType("dense") == FeatureType.DENSE
        assert FeatureType("sequence") == FeatureType.SEQUENCE


class TestFieldSchema:
    def test_defaults(self):
        f = FieldSchema(name="test", feature_type=FeatureType.SPARSE)
        assert f.vocabulary_size == 0
        assert f.embedding_dim == 16
        assert f.is_label is False
        assert f.group == "default"
        assert f.default_value == 0
        assert f.max_length == 1
        assert f.combiner == "mean"

    def test_custom_values(self):
        f = FieldSchema(
            name="genres",
            feature_type=FeatureType.SEQUENCE,
            vocabulary_size=20,
            embedding_dim=8,
            max_length=6,
            combiner="sum",
            group="item",
        )
        assert f.name == "genres"
        assert f.feature_type == FeatureType.SEQUENCE
        assert f.vocabulary_size == 20
        assert f.embedding_dim == 8
        assert f.max_length == 6
        assert f.combiner == "sum"


class TestDatasetSchema:
    def test_sparse_fields(self):
        schema = _make_schema()
        sparse = schema.sparse_fields
        assert len(sparse) == 2
        names = {f.name for f in sparse}
        assert names == {"user_id", "movie_id"}

    def test_dense_fields(self):
        schema = _make_schema()
        dense = schema.dense_fields
        assert len(dense) == 1
        assert dense[0].name == "age"

    def test_sequence_fields(self):
        schema = _make_schema()
        seq = schema.sequence_fields
        assert len(seq) == 1
        assert seq[0].name == "genres"

    def test_label_excluded_from_feature_lists(self):
        schema = _make_schema()
        all_names = (
            {f.name for f in schema.sparse_fields}
            | {f.name for f in schema.dense_fields}
            | {f.name for f in schema.sequence_fields}
        )
        assert "label" not in all_names

    def test_num_fields(self):
        schema = _make_schema()
        # 2 sparse + 1 dense + 1 sequence = 4
        assert schema.num_fields == 4

    def test_total_embedding_dim(self):
        schema = _make_schema()
        # user_id(16) + movie_id(16) + age(8) + genres(8) = 48
        assert schema.total_embedding_dim == 48

    def test_empty_schema(self):
        fields = {
            "label": FieldSchema(
                name="label", feature_type=FeatureType.SPARSE, is_label=True
            )
        }
        schema = DatasetSchema(fields=fields, label_field="label")
        assert schema.sparse_fields == []
        assert schema.dense_fields == []
        assert schema.sequence_fields == []
        assert schema.num_fields == 0
        assert schema.total_embedding_dim == 0
