"""Generic tabular dataset for CTR prediction."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """Torch Dataset wrapping a dict of feature arrays and labels.

    Args:
        features: Dict mapping feature names to numpy arrays of shape (N,) or (N, L).
        labels: Numpy array of labels, shape (N,).
    """

    def __init__(
        self, features: dict[str, np.ndarray], labels: np.ndarray
    ) -> None:
        self.features = features
        self.labels = labels
        self._length = len(labels)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        feature_dict = {}
        for name, values in self.features.items():
            val = values[idx]
            if np.issubdtype(values.dtype, np.integer):
                feature_dict[name] = torch.tensor(val, dtype=torch.long)
            else:
                feature_dict[name] = torch.tensor(val, dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature_dict, label
