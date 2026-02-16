"""Tests for LabelEncoder, MinMaxScaler, MultiHotEncoder."""

import numpy as np
import pytest

from deepfm.data.transforms import LabelEncoder, MinMaxScaler, MultiHotEncoder


class TestLabelEncoder:
    def test_fit_transform(self):
        enc = LabelEncoder()
        values = np.array(["a", "b", "c", "a"])
        encoded = enc.fit_transform(values)
        # OOV=0, so encoded values should be >= 1
        assert encoded.min() >= 1
        assert encoded.dtype == np.int64
        assert enc.vocabulary_size == 4  # 3 unique + 1 OOV

    def test_oov_returns_zero(self):
        enc = LabelEncoder()
        enc.fit(np.array(["a", "b", "c"]))
        encoded = enc.transform(np.array(["a", "unknown", "b"]))
        assert encoded[1] == 0  # unknown → OOV index 0

    def test_consistent_mapping(self):
        enc = LabelEncoder()
        enc.fit(np.array([10, 20, 30]))
        e1 = enc.transform(np.array([20, 10, 30]))
        e2 = enc.transform(np.array([20, 10, 30]))
        np.testing.assert_array_equal(e1, e2)

    def test_vocabulary_size(self):
        enc = LabelEncoder()
        enc.fit(np.array([1, 2, 3, 1, 2]))
        assert enc.vocabulary_size == 4  # 3 unique + 1 OOV


class TestMinMaxScaler:
    def test_fit_transform_range(self):
        scaler = MinMaxScaler()
        values = np.array([10.0, 20.0, 30.0, 40.0])
        scaled = scaler.fit_transform(values)
        assert scaled.min() == pytest.approx(0.0)
        assert scaled.max() == pytest.approx(1.0)
        assert scaled.dtype == np.float32

    def test_constant_values(self):
        scaler = MinMaxScaler()
        values = np.array([5.0, 5.0, 5.0])
        scaled = scaler.fit_transform(values)
        # All same → all zeros
        np.testing.assert_array_equal(scaled, np.zeros(3, dtype=np.float32))

    def test_transform_unseen(self):
        scaler = MinMaxScaler()
        scaler.fit(np.array([0.0, 10.0]))
        result = scaler.transform(np.array([5.0]))
        assert result[0] == pytest.approx(0.5)


class TestMultiHotEncoder:
    def test_fit_transform_shape(self):
        enc = MultiHotEncoder(max_length=4)
        lists = [["a", "b"], ["c"], ["a", "b", "c"]]
        encoded = enc.fit_transform(lists)
        assert encoded.shape == (3, 4)
        assert encoded.dtype == np.int64

    def test_padding(self):
        enc = MultiHotEncoder(max_length=4)
        enc.fit([["a", "b"]])
        encoded = enc.transform([["a"]])
        # "a" at index 0, rest padded with 0
        assert encoded[0, 0] > 0
        assert encoded[0, 1] == 0
        assert encoded[0, 2] == 0
        assert encoded[0, 3] == 0

    def test_truncation(self):
        enc = MultiHotEncoder(max_length=2)
        enc.fit([["a", "b", "c", "d"]])
        encoded = enc.transform([["a", "b", "c", "d"]])
        # Only first 2 values kept
        assert encoded.shape == (1, 2)
        assert (encoded[0] > 0).all()

    def test_oov_in_transform(self):
        enc = MultiHotEncoder(max_length=3)
        enc.fit([["a", "b"]])
        encoded = enc.transform([["a", "unknown"]])
        assert encoded[0, 0] > 0  # "a" is known
        assert encoded[0, 1] == 0  # "unknown" maps to OOV=0

    def test_vocabulary_size(self):
        enc = MultiHotEncoder(max_length=3)
        enc.fit([["x", "y", "z"]])
        assert enc.vocabulary_size == 4  # 3 unique + 1 pad/OOV
