from __future__ import annotations

from typing import Dict, List

import numpy as np


class LabelEncoder:
    """Encodes categorical values to contiguous 0-indexed integers.

    Index 0 is reserved for unknown/OOV values.
    """

    def __init__(self):
        self._mapping: Dict[Any, int] = {}
        self._vocab_size: int = 1  # starts at 1 (0 = OOV)

    @property
    def vocabulary_size(self) -> int:
        return self._vocab_size

    def fit(self, values: np.ndarray) -> LabelEncoder:
        unique = sorted(set(values))
        self._mapping = {v: i + 1 for i, v in enumerate(unique)}
        self._vocab_size = len(self._mapping) + 1  # +1 for OOV at index 0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.array([self._mapping.get(v, 0) for v in values], dtype=np.int64)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        self.fit(values)
        return self.transform(values)


class MinMaxScaler:
    """Scales values to [0, 1] range."""

    def __init__(self):
        self.min_: float = 0.0
        self.max_: float = 1.0

    def fit(self, values: np.ndarray) -> MinMaxScaler:
        self.min_ = float(np.min(values))
        self.max_ = float(np.max(values))
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        denom = self.max_ - self.min_
        if denom == 0:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - self.min_) / denom).astype(np.float32)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        self.fit(values)
        return self.transform(values)


class MultiHotEncoder:
    """Encodes lists of categorical values into padded integer arrays.

    Index 0 is padding/OOV. Output shape: (N, max_length).
    """

    def __init__(self, max_length: int = 6):
        self.max_length = max_length
        self._mapping: Dict[Any, int] = {}
        self._vocab_size: int = 1  # 0 = pad/OOV

    @property
    def vocabulary_size(self) -> int:
        return self._vocab_size

    def fit(self, value_lists: List[List]) -> MultiHotEncoder:
        unique = sorted({v for lst in value_lists for v in lst})
        self._mapping = {v: i + 1 for i, v in enumerate(unique)}
        self._vocab_size = len(self._mapping) + 1
        return self

    def transform(self, value_lists: List[List]) -> np.ndarray:
        result = np.zeros((len(value_lists), self.max_length), dtype=np.int64)
        for i, lst in enumerate(value_lists):
            encoded = [self._mapping.get(v, 0) for v in lst]
            length = min(len(encoded), self.max_length)
            result[i, :length] = encoded[:length]
        return result

    def fit_transform(self, value_lists: List[List]) -> np.ndarray:
        self.fit(value_lists)
        return self.transform(value_lists)
