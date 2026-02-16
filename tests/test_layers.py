"""Tests for model layers: FM, DNN, CIN, FieldAttention, FeatureEmbedding."""

import torch
import pytest

from deepfm.data.schema import DatasetSchema, FeatureType, FieldSchema
from deepfm.models.layers.attention import FieldAttention
from deepfm.models.layers.cin import CIN
from deepfm.models.layers.dnn import DNN
from deepfm.models.layers.embedding import FeatureEmbedding
from deepfm.models.layers.fm import FMInteraction


# ---------- FMInteraction ----------


class TestFMInteraction:
    def test_output_shape_reduce_sum(self):
        fm = FMInteraction(reduce_sum=True)
        x = torch.randn(4, 6, 16)  # (B, F, D)
        out = fm(x)
        assert out.shape == (4, 1)

    def test_output_shape_no_reduce(self):
        fm = FMInteraction(reduce_sum=False)
        x = torch.randn(4, 6, 16)
        out = fm(x)
        assert out.shape == (4, 16)

    def test_efficient_vs_naive(self):
        """Verify efficient O(n*d) matches naive O(n^2*d) computation."""
        torch.manual_seed(42)
        B, F, D = 8, 5, 16
        x = torch.randn(B, F, D)

        # Efficient
        fm = FMInteraction(reduce_sum=True)
        efficient = fm(x)

        # Naive: sum over all i < j of <v_i, v_j>
        naive = torch.zeros(B, 1)
        for i in range(F):
            for j in range(i + 1, F):
                dot = (x[:, i, :] * x[:, j, :]).sum(dim=-1, keepdim=True)
                naive += dot

        torch.testing.assert_close(efficient, naive, atol=1e-5, rtol=1e-5)

    def test_single_field_is_zero(self):
        """With only 1 field, there are no pairwise interactions → output = 0."""
        fm = FMInteraction(reduce_sum=True)
        x = torch.randn(4, 1, 16)
        out = fm(x)
        torch.testing.assert_close(out, torch.zeros(4, 1), atol=1e-6, rtol=1e-6)


# ---------- DNN ----------


class TestDNN:
    def test_output_shape(self):
        dnn = DNN(input_dim=48, hidden_units=[64, 32])
        x = torch.randn(4, 48)
        out = dnn(x)
        assert out.shape == (4, 32)

    def test_output_dim_property(self):
        dnn = DNN(input_dim=100, hidden_units=[64, 32, 16])
        assert dnn.output_dim == 16

    def test_no_batch_norm(self):
        dnn = DNN(input_dim=32, hidden_units=[16, 8], use_batch_norm=False)
        # Check no BatchNorm in the sequential
        bn_count = sum(1 for m in dnn.mlp if isinstance(m, torch.nn.BatchNorm1d))
        assert bn_count == 0

    def test_with_batch_norm(self):
        dnn = DNN(input_dim=32, hidden_units=[16, 8], use_batch_norm=True)
        bn_count = sum(1 for m in dnn.mlp if isinstance(m, torch.nn.BatchNorm1d))
        assert bn_count == 2  # one per hidden layer

    def test_gradient_flow(self):
        dnn = DNN(input_dim=32, hidden_units=[16, 8])
        x = torch.randn(4, 32, requires_grad=True)
        out = dnn(x)
        out.sum().backward()
        assert x.grad is not None
        assert (x.grad != 0).any()


# ---------- CIN ----------


class TestCIN:
    def test_output_shape_no_split(self):
        cin = CIN(num_fields=5, embedding_dim=16, layer_sizes=[64, 64], split_half=False)
        x = torch.randn(4, 5, 16)
        out = cin(x)
        # output_dim = 64 + 64 = 128
        assert out.shape == (4, 128)
        assert cin.output_dim == 128

    def test_output_shape_split_half(self):
        cin = CIN(num_fields=5, embedding_dim=16, layer_sizes=[64, 64], split_half=True)
        x = torch.randn(4, 5, 16)
        out = cin(x)
        # output_dim = 64//2 + 64 = 32 + 64 = 96
        assert out.shape == (4, 96)
        assert cin.output_dim == 96

    def test_gradient_flow(self):
        cin = CIN(num_fields=4, embedding_dim=8, layer_sizes=[32])
        x = torch.randn(4, 4, 8, requires_grad=True)
        out = cin(x)
        out.sum().backward()
        assert x.grad is not None
        assert (x.grad != 0).any()


# ---------- FieldAttention ----------


class TestFieldAttention:
    def test_output_shape(self):
        attn = FieldAttention(embedding_dim=16, num_heads=4, attention_dim=64)
        x = torch.randn(4, 6, 16)
        out = attn(x)
        assert out.shape == (4, 6, 16)

    def test_residual_preserves_shape(self):
        attn = FieldAttention(embedding_dim=16, num_heads=2, attention_dim=32, use_residual=True)
        x = torch.randn(4, 5, 16)
        out = attn(x)
        assert out.shape == x.shape

    def test_no_residual(self):
        attn = FieldAttention(embedding_dim=16, num_heads=2, attention_dim=32, use_residual=False)
        x = torch.randn(4, 5, 16)
        out = attn(x)
        assert out.shape == (4, 5, 16)

    def test_gradient_flow(self):
        attn = FieldAttention(embedding_dim=16, num_heads=4, attention_dim=64)
        x = torch.randn(4, 6, 16, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        assert x.grad is not None
        assert (x.grad != 0).any()

    def test_attention_dim_divisibility(self):
        with pytest.raises(AssertionError):
            FieldAttention(embedding_dim=16, num_heads=3, attention_dim=64)


# ---------- FeatureEmbedding ----------


class TestFeatureEmbedding:
    @pytest.fixture
    def schema(self):
        fields = {
            "user_id": FieldSchema(
                name="user_id",
                feature_type=FeatureType.SPARSE,
                vocabulary_size=50,
                embedding_dim=8,
            ),
            "item_id": FieldSchema(
                name="item_id",
                feature_type=FeatureType.SPARSE,
                vocabulary_size=100,
                embedding_dim=16,
            ),
            "genres": FieldSchema(
                name="genres",
                feature_type=FeatureType.SEQUENCE,
                vocabulary_size=20,
                embedding_dim=8,
                max_length=4,
                combiner="mean",
            ),
            "label": FieldSchema(
                name="label",
                feature_type=FeatureType.SPARSE,
                is_label=True,
            ),
        }
        return DatasetSchema(fields=fields, label_field="label")

    def test_output_shapes(self, schema):
        fm_dim = 16
        emb = FeatureEmbedding(schema, fm_embedding_dim=fm_dim)
        B = 4
        sparse = torch.tensor([[1, 5], [2, 10], [3, 15], [4, 20]])  # (B, 2)
        dense = torch.zeros(B, 0)
        sequences = {"genres": torch.randint(0, 20, (B, 4))}

        first_order, field_embs, flat_embs = emb(sparse, dense, sequences)

        assert first_order.shape == (B, 1)
        # 3 fields total (2 sparse + 1 sequence)
        assert field_embs.shape == (B, 3, fm_dim)
        # flat: user_id(8) + item_id(16) + genres(8) = 32
        assert flat_embs.shape == (B, 32)

    def test_padding_idx_zero_embedding(self, schema):
        emb = FeatureEmbedding(schema, fm_embedding_dim=16)
        # Index 0 should give zero embedding (padding_idx=0)
        sparse = torch.zeros(1, 2, dtype=torch.long)
        dense = torch.zeros(1, 0)
        sequences = {"genres": torch.zeros(1, 4, dtype=torch.long)}

        first_order, field_embs, flat_embs = emb(sparse, dense, sequences)
        # Embeddings at index 0 should be zero (except bias term in first_order)
        # field_embs and flat_embs should be zero for padded indices
        assert (flat_embs == 0).all()
