"""Tests for DeepFM, xDeepFM, and AttentionDeepFM models."""

import torch
import pytest

from deepfm.config import ExperimentConfig
from deepfm.data.schema import DatasetSchema, FeatureType, FieldSchema
from deepfm.models import build_model
from deepfm.models.deepfm import DeepFM
from deepfm.models.xdeepfm import xDeepFM
from deepfm.models.attention_deepfm import AttentionDeepFM


@pytest.fixture
def schema():
    fields = {
        "user_id": FieldSchema(
            name="user_id",
            feature_type=FeatureType.SPARSE,
            vocabulary_size=50,
            embedding_dim=16,
            group="user",
        ),
        "item_id": FieldSchema(
            name="item_id",
            feature_type=FeatureType.SPARSE,
            vocabulary_size=100,
            embedding_dim=16,
            group="item",
        ),
        "gender": FieldSchema(
            name="gender",
            feature_type=FeatureType.SPARSE,
            vocabulary_size=4,
            embedding_dim=4,
            group="user",
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
def batch(schema):
    """Build a synthetic batch matching the schema."""
    B = 4
    sparse = torch.stack(
        [
            torch.randint(1, 50, (B,)),   # user_id
            torch.randint(1, 100, (B,)),  # item_id
            torch.randint(1, 4, (B,)),    # gender
        ],
        dim=1,
    )
    dense = torch.zeros(B, 0)
    sequences = {"genres": torch.randint(0, 20, (B, 4))}
    return sparse, dense, sequences


def _make_config(model_name: str) -> ExperimentConfig:
    config = ExperimentConfig(model_name=model_name)
    config.feature.fm_embedding_dim = 16
    config.dnn.hidden_units = [32, 16]
    config.dnn.use_batch_norm = True
    config.dnn.dropout = 0.0
    config.cin.layer_sizes = [32, 32]
    config.cin.split_half = True
    config.attention.num_heads = 4
    config.attention.attention_dim = 16
    config.attention.num_layers = 1
    config.attention.use_residual = True
    return config


class TestDeepFM:
    def test_output_shape(self, schema, batch):
        config = _make_config("deepfm")
        model = DeepFM(schema=schema, config=config)
        model.eval()
        sparse, dense, sequences = batch
        out = model(sparse, dense, sequences)
        assert out.shape == (4, 1)

    def test_predict_probability_range(self, schema, batch):
        config = _make_config("deepfm")
        model = DeepFM(schema=schema, config=config)
        model.eval()
        sparse, dense, sequences = batch
        probs = model.predict(sparse, dense, sequences)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_gradient_flow(self, schema, batch):
        config = _make_config("deepfm")
        model = DeepFM(schema=schema, config=config)
        model.train()
        sparse, dense, sequences = batch
        out = model(sparse, dense, sequences)
        loss = out.sum()
        loss.backward()
        # Check no dead gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.all(param.grad == 0), f"Dead gradient: {name}"

    def test_l2_reg_loss(self, schema, batch):
        config = _make_config("deepfm")
        config.feature.embedding_l2_reg = 1e-5
        model = DeepFM(schema=schema, config=config)
        reg = model.get_l2_reg_loss()
        assert reg.item() >= 0


class TestXDeepFM:
    def test_output_shape(self, schema, batch):
        config = _make_config("xdeepfm")
        model = xDeepFM(schema=schema, config=config)
        model.eval()
        sparse, dense, sequences = batch
        out = model(sparse, dense, sequences)
        assert out.shape == (4, 1)

    def test_predict_probability_range(self, schema, batch):
        config = _make_config("xdeepfm")
        model = xDeepFM(schema=schema, config=config)
        model.eval()
        sparse, dense, sequences = batch
        probs = model.predict(sparse, dense, sequences)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_gradient_flow(self, schema, batch):
        config = _make_config("xdeepfm")
        model = xDeepFM(schema=schema, config=config)
        model.train()
        sparse, dense, sequences = batch
        out = model(sparse, dense, sequences)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.all(param.grad == 0), f"Dead gradient: {name}"


class TestAttentionDeepFM:
    def test_output_shape(self, schema, batch):
        config = _make_config("attention_deepfm")
        model = AttentionDeepFM(schema=schema, config=config)
        model.eval()
        sparse, dense, sequences = batch
        out = model(sparse, dense, sequences)
        assert out.shape == (4, 1)

    def test_predict_probability_range(self, schema, batch):
        config = _make_config("attention_deepfm")
        model = AttentionDeepFM(schema=schema, config=config)
        model.eval()
        sparse, dense, sequences = batch
        probs = model.predict(sparse, dense, sequences)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_gradient_flow(self, schema, batch):
        config = _make_config("attention_deepfm")
        model = AttentionDeepFM(schema=schema, config=config)
        model.train()
        sparse, dense, sequences = batch
        out = model(sparse, dense, sequences)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.all(param.grad == 0), f"Dead gradient: {name}"


class TestModelRegistry:
    def test_build_deepfm(self, schema):
        config = _make_config("deepfm")
        model = build_model("deepfm", schema, config)
        assert isinstance(model, DeepFM)

    def test_build_xdeepfm(self, schema):
        config = _make_config("xdeepfm")
        model = build_model("xdeepfm", schema, config)
        assert isinstance(model, xDeepFM)

    def test_build_attention_deepfm(self, schema):
        config = _make_config("attention_deepfm")
        model = build_model("attention_deepfm", schema, config)
        assert isinstance(model, AttentionDeepFM)

    def test_unknown_model_raises(self, schema):
        config = _make_config("deepfm")
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("nonexistent", schema, config)
