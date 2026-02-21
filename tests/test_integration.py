"""Integration test: full pipeline on real ML-100K data."""

import pytest
import torch

from deepfm.config import ExperimentConfig
from deepfm.data.movielens import MovieLensAdapter
from deepfm.models.deepfm import DeepFM
from deepfm.training.trainer import Trainer


@pytest.mark.slow
def test_end_to_end_movielens():
    """Train DeepFM on real ML-100K for 2 epochs, verify AUC > 0.5."""
    config = ExperimentConfig(
        training=ExperimentConfig().training.__class__(
            num_epochs=2,
            batch_size=512,
            early_stopping_patience=5,
        ),
        output_dir="/tmp/deepfm_integration_test",
    )

    adapter = MovieLensAdapter(config.data)
    schema, train_ds, val_ds, test_ds = adapter.build()

    model = DeepFM(schema, config)
    trainer = Trainer(
        model=model,
        schema=schema,
        config=config,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        adapter=adapter,
        device="cpu",
    )

    best_metrics = trainer.train()

    assert "auc" in best_metrics
    assert best_metrics["auc"] > 0.5, f"AUC too low: {best_metrics['auc']}"

    # Verify checkpoint was saved
    checkpoint = torch.load(
        "/tmp/deepfm_integration_test/best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert "model_state_dict" in checkpoint
    assert "epoch" in checkpoint
