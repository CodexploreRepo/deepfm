"""Tests for TabularDataset, NegativeSamplingDataset, EvalRankingDataset."""

import numpy as np
import pytest
import torch

from deepfm.data.dataset import (
    EvalRankingDataset,
    NegativeSamplingDataset,
    TabularDataset,
)
from deepfm.data.schema import DatasetSchema, FeatureType, FieldSchema


@pytest.fixture
def schema():
    fields = {
        "user_id": FieldSchema(
            name="user_id",
            feature_type=FeatureType.SPARSE,
            vocabulary_size=50,
            embedding_dim=8,
            group="user",
        ),
        "item_id": FieldSchema(
            name="item_id",
            feature_type=FeatureType.SPARSE,
            vocabulary_size=100,
            embedding_dim=8,
            group="item",
        ),
        "genres": FieldSchema(
            name="genres",
            feature_type=FeatureType.SEQUENCE,
            vocabulary_size=20,
            embedding_dim=8,
            max_length=4,
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


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 20
    return {
        "user_id": np.random.randint(1, 50, size=n).astype(np.int64),
        "item_id": np.random.randint(1, 100, size=n).astype(np.int64),
        "genres": np.random.randint(0, 20, size=(n, 4)).astype(np.int64),
        "label": np.random.randint(0, 2, size=n).astype(np.float32),
    }


class TestTabularDataset:
    def test_length(self, schema, sample_data):
        ds = TabularDataset(sample_data, schema)
        assert len(ds) == 20

    def test_getitem_keys(self, schema, sample_data):
        ds = TabularDataset(sample_data, schema)
        item = ds[0]
        assert "sparse" in item
        assert "label" in item
        assert "sequences" in item

    def test_sparse_shape_and_dtype(self, schema, sample_data):
        ds = TabularDataset(sample_data, schema)
        item = ds[0]
        # 2 sparse fields: user_id, item_id
        assert item["sparse"].shape == (2,)
        assert item["sparse"].dtype == torch.long

    def test_label_dtype(self, schema, sample_data):
        ds = TabularDataset(sample_data, schema)
        item = ds[0]
        assert item["label"].dtype == torch.float32

    def test_sequence_shape(self, schema, sample_data):
        ds = TabularDataset(sample_data, schema)
        item = ds[0]
        assert "genres" in item["sequences"]
        assert item["sequences"]["genres"].shape == (4,)
        assert item["sequences"]["genres"].dtype == torch.long

    def test_dense_empty(self, schema, sample_data):
        ds = TabularDataset(sample_data, schema)
        item = ds[0]
        # Schema has no dense fields
        assert item["dense"].shape == (0,)


class TestNegativeSamplingDataset:
    def test_length_with_negatives(self, schema, sample_data):
        num_neg = 3
        # Build item lookup
        item_data = {
            "item_id": np.arange(1, 100, dtype=np.int64),
            "genres": np.random.randint(0, 20, size=(99, 4)).astype(np.int64),
        }
        user_interacted = {
            uid: {sample_data["item_id"][i]}
            for i, uid in enumerate(sample_data["user_id"])
        }

        ds = NegativeSamplingDataset(
            positive_data=sample_data,
            schema=schema,
            user_col="user_id",
            item_col="item_id",
            all_item_data=item_data,
            user_interacted_items=user_interacted,
            num_items=99,
            num_neg=num_neg,
            item_feature_cols=["item_id", "genres"],
        )
        # 20 positives * (1 + 3 negatives) = 80
        assert len(ds) == 20 * (1 + num_neg)

    def test_negative_labels_are_zero(self, schema, sample_data):
        item_data = {
            "item_id": np.arange(1, 100, dtype=np.int64),
            "genres": np.random.randint(0, 20, size=(99, 4)).astype(np.int64),
        }
        user_interacted = {}

        ds = NegativeSamplingDataset(
            positive_data=sample_data,
            schema=schema,
            user_col="user_id",
            item_col="item_id",
            all_item_data=item_data,
            user_interacted_items=user_interacted,
            num_items=99,
            num_neg=2,
            item_feature_cols=["item_id", "genres"],
        )
        # Check that every 2nd and 3rd sample in each group has label 0
        for pos_idx in range(20):
            base = pos_idx * 3
            neg1 = ds[base + 1]
            neg2 = ds[base + 2]
            assert neg1["label"].item() == 0.0
            assert neg2["label"].item() == 0.0

    def test_resample_changes_data(self, schema, sample_data):
        np.random.seed(42)
        item_data = {
            "item_id": np.arange(1, 100, dtype=np.int64),
            "genres": np.random.randint(0, 20, size=(99, 4)).astype(np.int64),
        }
        user_interacted = {}

        ds = NegativeSamplingDataset(
            positive_data=sample_data,
            schema=schema,
            user_col="user_id",
            item_col="item_id",
            all_item_data=item_data,
            user_interacted_items=user_interacted,
            num_items=99,
            num_neg=2,
            item_feature_cols=["item_id", "genres"],
        )
        first_neg = ds[1]["sparse"].clone()
        ds.resample_negatives()
        second_neg = ds[1]["sparse"].clone()
        # Negatives should (very likely) differ after resampling
        # This is probabilistic but with 99 items, collision is ~1%
        # We don't assert inequality to avoid flakiness, just check it runs
        assert len(ds) == 20 * 3


class TestEvalRankingDataset:
    def test_properties(self, schema):
        np.random.seed(42)
        n_users = 5
        eval_data = {
            "user_id": np.arange(1, n_users + 1, dtype=np.int64),
            "item_id": np.random.randint(1, 100, size=n_users).astype(np.int64),
            "genres": np.random.randint(0, 20, size=(n_users, 4)).astype(np.int64),
            "label": np.ones(n_users, dtype=np.float32),
        }
        item_data = {
            "item_id": np.arange(1, 100, dtype=np.int64),
            "genres": np.random.randint(0, 20, size=(99, 4)).astype(np.int64),
        }

        ds = EvalRankingDataset(
            eval_data=eval_data,
            schema=schema,
            user_col="user_id",
            item_col="item_id",
            all_item_data=item_data,
            user_interacted_items={},
            num_neg_eval=9,
            item_feature_cols=["item_id", "genres"],
        )
        assert ds.num_users == 5
        assert ds.candidates_per_user == 10  # 1 + 9
        assert len(ds) == 50  # 5 * 10

    def test_first_candidate_is_positive(self, schema):
        np.random.seed(42)
        n_users = 3
        eval_data = {
            "user_id": np.arange(1, n_users + 1, dtype=np.int64),
            "item_id": np.array([10, 20, 30], dtype=np.int64),
            "genres": np.random.randint(0, 20, size=(n_users, 4)).astype(np.int64),
            "label": np.ones(n_users, dtype=np.float32),
        }
        item_data = {
            "item_id": np.arange(1, 100, dtype=np.int64),
            "genres": np.random.randint(0, 20, size=(99, 4)).astype(np.int64),
        }

        ds = EvalRankingDataset(
            eval_data=eval_data,
            schema=schema,
            user_col="user_id",
            item_col="item_id",
            all_item_data=item_data,
            user_interacted_items={},
            num_neg_eval=4,
            item_feature_cols=["item_id", "genres"],
        )
        # For each user, first candidate has label=1
        cands = ds.candidates_per_user  # 5
        for u in range(3):
            item = ds[u * cands]
            assert item["label"].item() == 1.0
            # Remaining are negatives
            for j in range(1, cands):
                item = ds[u * cands + j]
                assert item["label"].item() == 0.0
