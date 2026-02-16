"""Tests for Trainer and MetricCalculator on synthetic data."""

import os
import tempfile

import numpy as np
import pytest
import torch

from deepfm.config import ExperimentConfig
from deepfm.data.dataset import EvalRankingDataset, NegativeSamplingDataset
from deepfm.data.schema import DatasetSchema, FeatureType, FieldSchema
from deepfm.models.deepfm import DeepFM
from deepfm.training.metrics import MetricCalculator, compute_ranking_metrics
from deepfm.training.trainer import Trainer


# ---------- MetricCalculator ----------


class TestMetricCalculator:
    def test_perfect_predictions(self):
        mc = MetricCalculator()
        mc.update(np.array([0.9, 0.1]), np.array([1.0, 0.0]))
        metrics = mc.compute()
        assert metrics["auc"] == 1.0
        assert metrics["logloss"] < 0.5

    def test_reset(self):
        mc = MetricCalculator()
        mc.update(np.array([0.5]), np.array([1.0]))
        mc.reset()
        assert len(mc.predictions) == 0
        assert len(mc.labels) == 0

    def test_multiple_updates(self):
        mc = MetricCalculator()
        mc.update(np.array([0.9, 0.1]), np.array([1.0, 0.0]))
        mc.update(np.array([0.8, 0.2]), np.array([1.0, 0.0]))
        metrics = mc.compute()
        assert metrics["auc"] == 1.0


# ---------- compute_ranking_metrics ----------


class TestRankingMetrics:
    def test_perfect_ranking(self):
        """Positive item scored highest → HR@1 = 1.0, NDCG@1 = 1.0."""
        # 2 users, 5 candidates each. Positive at index 0 with highest score.
        scores = np.array([
            1.0, 0.4, 0.3, 0.2, 0.1,  # user 1: positive is top-1
            0.9, 0.5, 0.4, 0.3, 0.2,  # user 2: positive is top-1
        ])
        metrics = compute_ranking_metrics(scores, num_users=2, candidates_per_user=5, ks=[1, 5])
        assert metrics["hr@1"] == 1.0
        assert metrics["ndcg@1"] == pytest.approx(1.0)

    def test_worst_ranking(self):
        """Positive item scored lowest → HR@1 = 0."""
        scores = np.array([
            0.1, 0.9, 0.8, 0.7, 0.6,  # user 1: positive is last
        ])
        metrics = compute_ranking_metrics(scores, num_users=1, candidates_per_user=5, ks=[1])
        assert metrics["hr@1"] == 0.0

    def test_partial_hit(self):
        """Positive at rank 3 → HR@5 = 1, HR@1 = 0."""
        scores = np.array([
            0.3, 0.9, 0.8, 0.5, 0.1,  # positive at rank 3 (0-indexed: 2)
        ])
        metrics = compute_ranking_metrics(scores, num_users=1, candidates_per_user=5, ks=[1, 5])
        assert metrics["hr@1"] == 0.0
        assert metrics["hr@5"] == 1.0

    def test_ndcg_at_k(self):
        """Positive at rank 2 → NDCG@5 = 1/log2(2+2) = 1/2 = 0.5."""
        # Positive is at index 0, but scores rank it 3rd (0-indexed rank 2)
        scores = np.array([
            0.3, 0.9, 0.5, 0.2, 0.1,  # positive score=0.3, rank=2
        ])
        metrics = compute_ranking_metrics(scores, num_users=1, candidates_per_user=5, ks=[5])
        # rank 2 (0-indexed) → NDCG = 1/log2(2+2) = 1/log2(4) = 0.5
        assert metrics["ndcg@5"] == pytest.approx(0.5)


# ---------- Trainer (smoke test) ----------


@pytest.fixture
def synthetic_setup():
    """Build a minimal synthetic dataset for trainer smoke test."""
    np.random.seed(42)

    fields = {
        "user_id": FieldSchema(
            name="user_id",
            feature_type=FeatureType.SPARSE,
            vocabulary_size=20,
            embedding_dim=8,
        ),
        "item_id": FieldSchema(
            name="item_id",
            feature_type=FeatureType.SPARSE,
            vocabulary_size=50,
            embedding_dim=8,
        ),
        "label": FieldSchema(
            name="label",
            feature_type=FeatureType.SPARSE,
            is_label=True,
        ),
    }
    schema = DatasetSchema(fields=fields, label_field="label")

    n_train = 40
    train_data = {
        "user_id": np.random.randint(1, 20, size=n_train).astype(np.int64),
        "item_id": np.random.randint(1, 50, size=n_train).astype(np.int64),
        "label": np.random.randint(0, 2, size=n_train).astype(np.float32),
    }

    item_data = {
        "item_id": np.arange(1, 50, dtype=np.int64),
    }

    train_ds = NegativeSamplingDataset(
        positive_data=train_data,
        schema=schema,
        user_col="user_id",
        item_col="item_id",
        all_item_data=item_data,
        user_interacted_items={},
        num_items=49,
        num_neg=2,
        item_feature_cols=["item_id"],
    )

    n_val = 10
    val_data = {
        "user_id": np.arange(1, n_val + 1, dtype=np.int64),
        "item_id": np.random.randint(1, 50, size=n_val).astype(np.int64),
        "label": np.ones(n_val, dtype=np.float32),
    }

    val_ds = EvalRankingDataset(
        eval_data=val_data,
        schema=schema,
        user_col="user_id",
        item_col="item_id",
        all_item_data=item_data,
        user_interacted_items={},
        num_neg_eval=9,
        item_feature_cols=["item_id"],
    )

    return schema, train_ds, val_ds


def test_trainer_one_epoch(synthetic_setup):
    """Smoke test: trainer runs 1 epoch without errors."""
    schema, train_ds, val_ds = synthetic_setup

    config = ExperimentConfig(model_name="deepfm")
    config.feature.fm_embedding_dim = 8
    config.dnn.hidden_units = [16, 8]
    config.dnn.use_batch_norm = False  # avoid BN issues with small batches
    config.dnn.dropout = 0.0
    config.training.batch_size = 32
    config.training.epochs = 1
    config.training.learning_rate = 1e-3
    config.training.num_workers = 0
    config.training.pin_memory = False
    config.training.early_stopping_patience = 999
    config.training.early_stopping_metric = "hr@10"
    config.training.early_stopping_mode = "max"
    config.training.gradient_clip_norm = 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
        config.output_dir = tmpdir

        model = DeepFM(schema=schema, config=config)
        trainer = Trainer(
            model=model,
            config=config,
            train_dataset=train_ds,
            val_dataset=val_ds,
            device="cpu",
        )
        metrics = trainer.fit()

        assert "auc" in metrics
        assert "hr@10" in metrics
        assert "ndcg@10" in metrics
        # Checkpoint should exist
        assert os.path.exists(os.path.join(tmpdir, "best_model.pt"))
