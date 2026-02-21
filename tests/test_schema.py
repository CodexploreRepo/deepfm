"""Tests for FieldSchema and DatasetSchema."""

from deepfm.data.schema import DatasetSchema, FeatureType, FieldSchema


def _make_schema() -> DatasetSchema:
    fields = {
        "user_id": FieldSchema(
            "user_id",
            FeatureType.SPARSE,
            vocabulary_size=100,
            embedding_dim=16,
        ),
        "item_id": FieldSchema("item_id", FeatureType.SPARSE, vocabulary_size=200, embedding_dim=8),
        "price": FieldSchema("price", FeatureType.DENSE, embedding_dim=4),
        "tags": FieldSchema("tags", FeatureType.SEQUENCE, vocabulary_size=50, embedding_dim=8),
    }
    return DatasetSchema(fields=fields, label_field="label")


class TestFieldSchema:
    def test_defaults(self):
        f = FieldSchema("test", FeatureType.SPARSE)
        assert f.name == "test"
        assert f.feature_type == FeatureType.SPARSE
        assert f.vocabulary_size == 0
        assert f.embedding_dim == 8
        assert f.group == ""
        assert f.max_length == 1
        assert f.combiner == "mean"

    def test_custom_values(self):
        f = FieldSchema(
            "genres",
            FeatureType.SEQUENCE,
            vocabulary_size=20,
            embedding_dim=16,
            group="item",
            max_length=6,
            combiner="sum",
        )
        assert f.vocabulary_size == 20
        assert f.max_length == 6
        assert f.combiner == "sum"


class TestDatasetSchema:
    def test_num_fields(self):
        schema = _make_schema()
        assert schema.num_fields == 4

    def test_sparse_fields(self):
        schema = _make_schema()
        sparse = schema.sparse_fields
        assert len(sparse) == 2
        assert all(f.feature_type == FeatureType.SPARSE for f in sparse)

    def test_dense_fields(self):
        schema = _make_schema()
        dense = schema.dense_fields
        assert len(dense) == 1
        assert dense[0].name == "price"

    def test_sequence_fields(self):
        schema = _make_schema()
        seq = schema.sequence_fields
        assert len(seq) == 1
        assert seq[0].name == "tags"

    def test_total_embedding_dim(self):
        schema = _make_schema()
        # 16 + 8 + 4 + 8 = 36
        assert schema.total_embedding_dim == 36

    def test_empty_schema(self):
        schema = DatasetSchema()
        assert schema.num_fields == 0
        assert schema.total_embedding_dim == 0
        assert schema.sparse_fields == []
        assert schema.dense_fields == []
        assert schema.sequence_fields == []

    def test_label_field_default(self):
        schema = DatasetSchema()
        assert schema.label_field == "label"

    def test_label_field_custom(self):
        schema = DatasetSchema(label_field="target")
        assert schema.label_field == "target"
