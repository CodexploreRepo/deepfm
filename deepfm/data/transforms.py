"""Feature transforms: LabelEncoder, MinMaxScaler, MultiHotEncoder."""

from __future__ import annotations

import numpy as np


class LabelEncoder:
    """Maps categorical values to integer indices with OOV=0."""

    def __init__(self) -> None:
        self._mapping: dict[str, int] = {}

    def fit(self, values: list | np.ndarray) -> LabelEncoder:
        unique = sorted(set(values))
        # Reserve index 0 for unknown/OOV
        self._mapping = {v: i + 1 for i, v in enumerate(unique)}
        return self

    def transform(self, values: list | np.ndarray) -> np.ndarray:
        return np.array(
            [self._mapping.get(v, 0) for v in values], dtype=np.int64
        )

    @property
    def vocabulary_size(self) -> int:
        """Number of classes + 1 (for OOV at index 0)."""
        return len(self._mapping) + 1


class MinMaxScaler:
    """Scales values to [0, 1] range."""

    def __init__(self) -> None:
        self._min: float = 0.0
        self._max: float = 1.0

    def fit(self, values: list | np.ndarray) -> MinMaxScaler:
        values = np.asarray(values, dtype=np.float64)
        self._min = float(values.min())
        self._max = float(values.max())
        return self

    def transform(self, values: list | np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        denom = self._max - self._min
        if denom == 0:
            return np.zeros_like(values)
        return (values - self._min) / denom


class MultiHotEncoder:
    """Encodes lists of tokens into padded integer sequences with OOV=0."""

    def __init__(self, max_length: int = 6) -> None:
        self.max_length = max_length
        self._mapping: dict[str, int] = {}

    def fit(self, token_lists: list[list[str]]) -> MultiHotEncoder:
        unique = sorted({t for tokens in token_lists for t in tokens})
        # Reserve index 0 for padding/OOV
        self._mapping = {t: i + 1 for i, t in enumerate(unique)}
        return self

    def transform(self, token_lists: list[list[str]]) -> np.ndarray:
        result = np.zeros((len(token_lists), self.max_length), dtype=np.int64)
        for i, tokens in enumerate(token_lists):
            indices = [self._mapping.get(t, 0) for t in tokens]
            length = min(len(indices), self.max_length)
            result[i, :length] = indices[:length]
        return result

    @property
    def vocabulary_size(self) -> int:
        """Number of tokens + 1 (for padding/OOV at index 0)."""
        return len(self._mapping) + 1
