"""Tests for Trainer with synthetic data."""

import numpy as np
import torch

from deepfm.config import ExperimentConfig
from deepfm.data.dataset import TabularDataset
from deepfm.data.schema import DatasetSchema, FeatureType, FieldSchema
from deepfm.models.deepfm import DeepFM
from deepfm.training.trainer import Trainer


def _make_synthetic_data(
    n_train: int = 100, n_eval: int = 20
) -> tuple[DatasetSchema, TabularDataset, TabularDataset, TabularDataset]:
    schema = DatasetSchema(
        fields={
            "user_id": FieldSchema(
                "user_id",
                FeatureType.SPARSE,
                vocabulary_size=20,
                embedding_dim=8,
            ),
            "item_id": FieldSchema(
                "item_id",
                FeatureType.SPARSE,
                vocabulary_size=30,
                embedding_dim=8,
            ),
        },
        label_field="label",
    )

    def make_ds(n: int) -> TabularDataset:
        features = {
            "user_id": np.random.randint(1, 20, size=n).astype(np.int64),
            "item_id": np.random.randint(1, 30, size=n).astype(np.int64),
        }
        labels = np.random.randint(0, 2, size=n).astype(np.float32)
        return TabularDataset(features, labels)

    return schema, make_ds(n_train), make_ds(n_eval), make_ds(n_eval)


class TestTrainer:
    def test_train_runs_without_error(self):
        torch.manual_seed(42)
        schema, train_ds, val_ds, test_ds = _make_synthetic_data()
        config = ExperimentConfig()
        config = ExperimentConfig(
            training=config.training.__class__(
                num_epochs=2,
                batch_size=32,
                lr=1e-3,
                early_stopping_patience=5,
            ),
            output_dir="/tmp/deepfm_test_trainer",
        )
        model = DeepFM(schema, config)
        trainer = Trainer(
            model=model,
            schema=schema,
            config=config,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            device="cpu",
        )
        metrics = trainer.train()
        assert isinstance(metrics, dict)
        assert "auc" in metrics or "logloss" in metrics

    def test_evaluate_returns_metrics(self):
        torch.manual_seed(42)
        schema, train_ds, val_ds, test_ds = _make_synthetic_data()
        config = ExperimentConfig(output_dir="/tmp/deepfm_test_eval")
        model = DeepFM(schema, config)
        trainer = Trainer(
            model=model,
            schema=schema,
            config=config,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            device="cpu",
        )
        metrics = trainer.evaluate(val_ds, "val")
        assert "auc" in metrics
        assert "logloss" in metrics
        assert 0 <= metrics["auc"] <= 1
        assert metrics["logloss"] > 0

    def test_model_updates_weights(self):
        torch.manual_seed(42)
        schema, train_ds, val_ds, test_ds = _make_synthetic_data()
        config = ExperimentConfig(
            training=ExperimentConfig().training.__class__(
                num_epochs=1,
                batch_size=32,
            ),
            output_dir="/tmp/deepfm_test_update",
        )
        model = DeepFM(schema, config)

        # Record initial parameters
        initial_params = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}

        trainer = Trainer(
            model=model,
            schema=schema,
            config=config,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            device="cpu",
        )
        trainer._train_epoch(1)

        # At least some parameters should have changed
        changed = any(
            not torch.equal(initial_params[n], p)
            for n, p in model.named_parameters()
            if p.requires_grad and n in initial_params
        )
        assert changed
