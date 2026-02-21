"""Tests for DeepFM, xDeepFM, AttentionDeepFM models."""

import torch

from deepfm.config import ExperimentConfig
from deepfm.data.schema import DatasetSchema, FeatureType, FieldSchema
from deepfm.models import create_model
from deepfm.models.attention_deepfm import AttentionDeepFM
from deepfm.models.deepfm import DeepFM
from deepfm.models.xdeepfm import xDeepFM


def _make_schema() -> DatasetSchema:
    fields = {
        "u": FieldSchema("u", FeatureType.SPARSE, vocabulary_size=50, embedding_dim=8),
        "i": FieldSchema("i", FeatureType.SPARSE, vocabulary_size=100, embedding_dim=16),
        "g": FieldSchema("g", FeatureType.SPARSE, vocabulary_size=3, embedding_dim=4),
    }
    return DatasetSchema(fields=fields, label_field="label")


def _make_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "u": torch.randint(1, 50, (batch_size,)),
        "i": torch.randint(1, 100, (batch_size,)),
        "g": torch.randint(1, 3, (batch_size,)),
    }


class TestDeepFM:
    def test_forward_shape(self):
        model = DeepFM(_make_schema(), ExperimentConfig())
        logits = model(_make_batch())
        assert logits.shape == (4, 1)

    def test_predict_range(self):
        model = DeepFM(_make_schema(), ExperimentConfig())
        model.eval()
        with torch.no_grad():
            probs = model.predict(_make_batch())
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_l2_reg_positive(self):
        model = DeepFM(_make_schema(), ExperimentConfig())
        l2 = model.get_l2_reg_loss()
        assert l2.item() > 0

    def test_gradient_flow(self):
        model = DeepFM(_make_schema(), ExperimentConfig())
        logits = model(_make_batch())
        logits.sum().backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad


class TestXDeepFM:
    def test_forward_shape(self):
        model = xDeepFM(_make_schema(), ExperimentConfig())
        logits = model(_make_batch())
        assert logits.shape == (4, 1)

    def test_predict_range(self):
        model = xDeepFM(_make_schema(), ExperimentConfig())
        model.eval()
        with torch.no_grad():
            probs = model.predict(_make_batch())
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_l2_reg_positive(self):
        model = xDeepFM(_make_schema(), ExperimentConfig())
        l2 = model.get_l2_reg_loss()
        assert l2.item() > 0


class TestAttentionDeepFM:
    def test_forward_shape(self):
        model = AttentionDeepFM(_make_schema(), ExperimentConfig())
        logits = model(_make_batch())
        assert logits.shape == (4, 1)

    def test_predict_range(self):
        model = AttentionDeepFM(_make_schema(), ExperimentConfig())
        model.eval()
        with torch.no_grad():
            probs = model.predict(_make_batch())
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_l2_reg_positive(self):
        model = AttentionDeepFM(_make_schema(), ExperimentConfig())
        l2 = model.get_l2_reg_loss()
        assert l2.item() > 0


class TestModelRegistry:
    def test_create_all_models(self):
        schema = _make_schema()
        config = ExperimentConfig()
        for name in ["deepfm", "xdeepfm", "attention_deepfm"]:
            model = create_model(name, schema, config)
            logits = model(_make_batch())
            assert logits.shape == (4, 1)

    def test_unknown_model_raises(self):
        try:
            create_model("unknown", _make_schema(), ExperimentConfig())
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
