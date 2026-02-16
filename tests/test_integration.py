"""Integration test: end-to-end on real MovieLens-100K data.

This test requires the ML-100K dataset (auto-downloaded if missing).
Mark with slow so it can be skipped in CI: pytest -m "not slow"
"""

import os

import pytest
import torch

from deepfm.config import ExperimentConfig
from deepfm.data.movielens import MovieLensAdapter
from deepfm.models import build_model


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def movielens_data():
    """Load real ML-100K data (downloads if needed)."""
    adapter = MovieLensAdapter(
        data_dir="./data/ml-100k",
        label_threshold=4.0,
    )
    train_ds, val_ds, test_ds = adapter.build_datasets(
        min_interactions=3,
        num_neg_train=2,  # fewer negatives for speed
        num_neg_eval=49,  # fewer negatives for speed
        auto_download=True,
    )
    return adapter.schema, train_ds, val_ds, test_ds


class TestMovieLensLoading:
    def test_schema_fields(self, movielens_data):
        schema, *_ = movielens_data
        sparse_names = {f.name for f in schema.sparse_fields}
        assert "user_id" in sparse_names
        assert "movie_id" in sparse_names

    def test_train_dataset_nonempty(self, movielens_data):
        _, train_ds, _, _ = movielens_data
        assert len(train_ds) > 0

    def test_val_users(self, movielens_data):
        _, _, val_ds, _ = movielens_data
        assert val_ds.num_users > 0
        assert val_ds.candidates_per_user == 50  # 1 + 49

    def test_getitem_returns_tensors(self, movielens_data):
        _, train_ds, _, _ = movielens_data
        item = train_ds[0]
        assert isinstance(item["sparse"], torch.Tensor)
        assert isinstance(item["label"], torch.Tensor)


class TestEndToEndForwardPass:
    @pytest.mark.parametrize("model_name", ["deepfm", "xdeepfm", "attention_deepfm"])
    def test_forward_pass(self, movielens_data, model_name):
        schema, train_ds, _, _ = movielens_data

        config = ExperimentConfig(model_name=model_name)
        config.feature.fm_embedding_dim = 16
        config.dnn.hidden_units = [32, 16]
        config.dnn.use_batch_norm = False
        config.cin.layer_sizes = [32, 32]
        config.attention.num_heads = 4
        config.attention.attention_dim = 16

        model = build_model(model_name, schema, config)
        model.eval()

        # Get a sample and run forward
        sample = train_ds[0]
        sparse = sample["sparse"].unsqueeze(0)
        dense = sample["dense"].unsqueeze(0)
        sequences = (
            {k: v.unsqueeze(0) for k, v in sample["sequences"].items()}
            if "sequences" in sample
            else None
        )

        with torch.no_grad():
            logit = model(sparse, dense, sequences)

        assert logit.shape == (1, 1)
        assert torch.isfinite(logit).all()
