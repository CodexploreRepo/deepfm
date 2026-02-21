"""Tests for model layers: FeatureEmbedding, FM, DNN, CIN, Attention."""

import torch

from deepfm.data.schema import DatasetSchema, FeatureType, FieldSchema
from deepfm.models.layers.attention import MultiHeadSelfAttention
from deepfm.models.layers.cin import CIN
from deepfm.models.layers.dnn import DNN
from deepfm.models.layers.embedding import FeatureEmbedding
from deepfm.models.layers.fm import FMInteraction


def _make_schema() -> DatasetSchema:
    fields = {
        "u": FieldSchema("u", FeatureType.SPARSE, vocabulary_size=100, embedding_dim=8),
        "i": FieldSchema("i", FeatureType.SPARSE, vocabulary_size=200, embedding_dim=16),
        "g": FieldSchema("g", FeatureType.SPARSE, vocabulary_size=3, embedding_dim=4),
    }
    return DatasetSchema(fields=fields, label_field="label")


def _make_batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "u": torch.randint(1, 100, (batch_size,)),
        "i": torch.randint(1, 200, (batch_size,)),
        "g": torch.randint(1, 3, (batch_size,)),
    }


# --- FeatureEmbedding ---


class TestFeatureEmbedding:
    def test_output_shapes(self):
        schema = _make_schema()
        emb = FeatureEmbedding(schema, fm_embed_dim=16)
        batch = _make_batch()
        fo, fe, fl = emb(batch)
        assert fo.shape == (4, 1)
        assert fe.shape == (4, 3, 16)
        assert fl.shape == (4, 8 + 16 + 4)  # total_embedding_dim

    def test_padding_idx_zero(self):
        schema = _make_schema()
        emb = FeatureEmbedding(schema, fm_embed_dim=16)
        batch = {k: torch.zeros(2, dtype=torch.long) for k in schema.fields}
        with torch.no_grad():
            fo, fe, fl = emb(batch)
        assert fo.abs().sum().item() == 0.0
        assert fe.abs().sum().item() == 0.0
        assert fl.abs().sum().item() == 0.0

    def test_gradient_flow(self):
        schema = _make_schema()
        emb = FeatureEmbedding(schema, fm_embed_dim=16)
        batch = _make_batch()
        fo, fe, fl = emb(batch)
        loss = fo.sum() + fe.sum() + fl.sum()
        loss.backward()
        for p in emb.parameters():
            if p.requires_grad:
                assert p.grad is not None


# --- FMInteraction ---


class TestFMInteraction:
    def test_output_shape(self):
        fm = FMInteraction()
        x = torch.randn(4, 6, 16)
        out = fm(x)
        assert out.shape == (4, 1)

    def test_no_parameters(self):
        fm = FMInteraction()
        assert sum(p.numel() for p in fm.parameters()) == 0

    def test_matches_explicit_pairwise(self):
        fm = FMInteraction()
        x = torch.randn(2, 4, 8)
        efficient = fm(x)

        # Explicit pairwise
        B, F, D = x.shape
        explicit = torch.zeros(B, 1)
        for i in range(F):
            for j in range(i + 1, F):
                dot = (x[:, i] * x[:, j]).sum(dim=1, keepdim=True)
                explicit += dot

        torch.testing.assert_close(efficient, explicit, atol=1e-5, rtol=1e-5)

    def test_single_field_is_zero(self):
        fm = FMInteraction()
        x = torch.randn(2, 1, 8)  # only 1 field → no pairs
        out = fm(x)
        torch.testing.assert_close(out, torch.zeros(2, 1))


# --- DNN ---


class TestDNN:
    def test_output_shape(self):
        dnn = DNN(64, [128, 32])
        x = torch.randn(4, 64)
        assert dnn(x).shape == (4, 32)
        assert dnn.output_dim == 32

    def test_with_batch_norm(self):
        dnn = DNN(32, [64, 16], use_batch_norm=True)
        dnn.train()
        x = torch.randn(8, 32)  # batch_size > 1 needed for BN
        assert dnn(x).shape == (8, 16)

    def test_without_batch_norm(self):
        dnn = DNN(32, [64, 16], use_batch_norm=False)
        x = torch.randn(4, 32)
        assert dnn(x).shape == (4, 16)

    def test_gradient_flow(self):
        dnn = DNN(16, [32, 8])
        x = torch.randn(4, 16)
        out = dnn(x)
        out.sum().backward()
        for p in dnn.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_different_activations(self):
        for act in ["relu", "leaky_relu", "gelu", "tanh"]:
            dnn = DNN(16, [32], activation=act, use_batch_norm=False)
            x = torch.randn(4, 16)
            out = dnn(x)
            assert out.shape == (4, 32)


# --- CIN ---


class TestCIN:
    def test_output_shape_split_half(self):
        cin = CIN(num_fields=6, embed_dim=16, layer_sizes=[128, 128], split_half=True)
        x = torch.randn(4, 6, 16)
        out = cin(x)
        assert out.shape == (4, cin.output_dim)

    def test_output_shape_no_split(self):
        cin = CIN(num_fields=6, embed_dim=16, layer_sizes=[64, 64], split_half=False)
        x = torch.randn(4, 6, 16)
        out = cin(x)
        assert out.shape == (4, cin.output_dim)
        assert cin.output_dim == 128  # 64 + 64

    def test_split_half_reduces_output(self):
        cin_split = CIN(num_fields=4, embed_dim=8, layer_sizes=[64, 64], split_half=True)
        cin_no_split = CIN(num_fields=4, embed_dim=8, layer_sizes=[64, 64], split_half=False)
        assert cin_split.output_dim < cin_no_split.output_dim

    def test_gradient_flow(self):
        cin = CIN(num_fields=3, embed_dim=8, layer_sizes=[32, 32])
        x = torch.randn(2, 3, 8)
        out = cin(x)
        out.sum().backward()
        for p in cin.parameters():
            if p.requires_grad:
                assert p.grad is not None


# --- MultiHeadSelfAttention ---


class TestMultiHeadSelfAttention:
    def test_output_shape(self):
        attn = MultiHeadSelfAttention(embed_dim=16, num_heads=4, attention_dim=64)
        x = torch.randn(4, 6, 16)
        out = attn(x)
        assert out.shape == (4, 6, 16)  # same shape as input

    def test_multi_layer(self):
        attn = MultiHeadSelfAttention(
            embed_dim=16,
            num_heads=4,
            attention_dim=64,
            num_layers=3,
        )
        x = torch.randn(2, 4, 16)
        out = attn(x)
        assert out.shape == (2, 4, 16)

    def test_without_residual(self):
        attn = MultiHeadSelfAttention(
            embed_dim=16,
            num_heads=2,
            attention_dim=16,
            use_residual=False,
        )
        x = torch.randn(2, 3, 16)
        out = attn(x)
        assert out.shape == (2, 3, 16)

    def test_gradient_flow(self):
        attn = MultiHeadSelfAttention(embed_dim=8, num_heads=2, attention_dim=8)
        x = torch.randn(2, 3, 8)
        out = attn(x)
        out.sum().backward()
        for p in attn.parameters():
            if p.requires_grad:
                assert p.grad is not None
