"""Tests for LabelEncoder, MinMaxScaler, MultiHotEncoder."""

import numpy as np
import pytest

from deepfm.data.transforms import LabelEncoder, MinMaxScaler, MultiHotEncoder


class TestLabelEncoder:
    def test_fit_transform(self):
        le = LabelEncoder()
        le.fit(["a", "b", "c"])
        result = le.transform(["b", "a", "c"])
        # Sorted order: a=1, b=2, c=3
        np.testing.assert_array_equal(result, [2, 1, 3])

    def test_oov_maps_to_zero(self):
        le = LabelEncoder()
        le.fit(["a", "b"])
        result = le.transform(["a", "z", "b"])
        assert result[1] == 0  # unknown → 0

    def test_vocabulary_size(self):
        le = LabelEncoder()
        le.fit(["x", "y", "z"])
        assert le.vocabulary_size == 4  # 3 classes + 1 OOV

    def test_numeric_values(self):
        le = LabelEncoder()
        le.fit([10, 20, 30])
        result = le.transform([20, 10, 99])
        assert result[0] > 0
        assert result[2] == 0  # unknown

    def test_duplicates_in_fit(self):
        le = LabelEncoder()
        le.fit(["a", "a", "b", "b", "c"])
        assert le.vocabulary_size == 4  # 3 unique + OOV

    def test_empty_transform(self):
        le = LabelEncoder()
        le.fit(["a", "b"])
        result = le.transform([])
        assert len(result) == 0


class TestMinMaxScaler:
    def test_basic_scaling(self):
        scaler = MinMaxScaler()
        scaler.fit([0, 10, 20])
        result = scaler.transform([0, 10, 20])
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_out_of_range(self):
        scaler = MinMaxScaler()
        scaler.fit([0, 10])
        result = scaler.transform([5, 15, -5])
        assert result[0] == pytest.approx(0.5)
        assert result[1] > 1.0  # above range
        assert result[2] < 0.0  # below range

    def test_constant_values(self):
        scaler = MinMaxScaler()
        scaler.fit([5, 5, 5])
        result = scaler.transform([5, 5])
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_negative_values(self):
        scaler = MinMaxScaler()
        scaler.fit([-10, 0, 10])
        result = scaler.transform([-10, 0, 10])
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])


class TestMultiHotEncoder:
    def test_basic_encoding(self):
        enc = MultiHotEncoder(max_length=4)
        enc.fit([["a", "b"], ["b", "c"]])
        result = enc.transform([["a", "c"]])
        # a=1, b=2, c=3
        assert result.shape == (1, 4)
        assert result[0, 0] == 1  # a
        assert result[0, 1] == 3  # c
        assert result[0, 2] == 0  # padding
        assert result[0, 3] == 0  # padding

    def test_oov_tokens(self):
        enc = MultiHotEncoder(max_length=3)
        enc.fit([["x", "y"]])
        result = enc.transform([["x", "z"]])  # z is unknown
        assert result[0, 0] > 0  # x is known
        assert result[0, 1] == 0  # z → OOV

    def test_truncation(self):
        enc = MultiHotEncoder(max_length=2)
        enc.fit([["a", "b", "c"]])
        result = enc.transform([["a", "b", "c"]])
        assert result.shape == (1, 2)  # truncated to max_length

    def test_vocabulary_size(self):
        enc = MultiHotEncoder()
        enc.fit([["a", "b"], ["c"]])
        assert enc.vocabulary_size == 4  # 3 tokens + 1 OOV/padding

    def test_empty_list(self):
        enc = MultiHotEncoder(max_length=3)
        enc.fit([["a", "b"]])
        result = enc.transform([[]])
        np.testing.assert_array_equal(result[0], [0, 0, 0])

    def test_multiple_samples(self):
        enc = MultiHotEncoder(max_length=3)
        enc.fit([["a", "b", "c"]])
        result = enc.transform([["a"], ["b", "c"], ["a", "b", "c"]])
        assert result.shape == (3, 3)
