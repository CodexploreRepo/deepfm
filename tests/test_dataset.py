"""Tests for TabularDataset."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from deepfm.data.dataset import TabularDataset


def _make_dataset(n: int = 10) -> TabularDataset:
    features = {
        "user_id": np.arange(n, dtype=np.int64),
        "item_id": np.arange(n, dtype=np.int64) * 2,
        "score": np.random.rand(n).astype(np.float32),
    }
    labels = np.random.randint(0, 2, size=n).astype(np.float32)
    return TabularDataset(features, labels)


class TestTabularDataset:
    def test_length(self):
        ds = _make_dataset(20)
        assert len(ds) == 20

    def test_getitem_returns_tuple(self):
        ds = _make_dataset()
        item = ds[0]
        assert isinstance(item, tuple)
        assert len(item) == 2

    def test_getitem_feature_dict(self):
        ds = _make_dataset()
        features, label = ds[0]
        assert isinstance(features, dict)
        assert set(features.keys()) == {"user_id", "item_id", "score"}

    def test_integer_features_are_long(self):
        ds = _make_dataset()
        features, _ = ds[0]
        assert features["user_id"].dtype == torch.long
        assert features["item_id"].dtype == torch.long

    def test_float_features_are_float32(self):
        ds = _make_dataset()
        features, _ = ds[0]
        assert features["score"].dtype == torch.float32

    def test_label_is_float32(self):
        ds = _make_dataset()
        _, label = ds[0]
        assert label.dtype == torch.float32

    def test_label_values(self):
        ds = _make_dataset()
        _, label = ds[0]
        assert label.item() in (0.0, 1.0)

    def test_dataloader_batching(self):
        ds = _make_dataset(16)
        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch_features, batch_labels = next(iter(loader))
        assert batch_labels.shape == (4,)
        assert batch_features["user_id"].shape == (4,)

    def test_2d_features(self):
        features = {
            "ids": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64),
        }
        labels = np.array([0.0, 1.0], dtype=np.float32)
        ds = TabularDataset(features, labels)
        feat, _ = ds[0]
        assert feat["ids"].shape == (3,)
        assert feat["ids"].dtype == torch.long
