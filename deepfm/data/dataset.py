from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from deepfm.data.schema import DatasetSchema


class TabularDataset(Dataset):
    """Generic dataset that works with any DatasetSchema.

    Stores data as a dict of numpy arrays (one per field).
    Returns tensors grouped by feature type for model consumption.
    """

    def __init__(self, data: Dict[str, np.ndarray], schema: DatasetSchema):
        self.data = data
        self.schema = schema
        self._length = len(next(iter(data.values())))

        # Pre-compute field orderings for deterministic tensor construction
        self._sparse_keys = [f.name for f in schema.sparse_fields]
        self._dense_keys = [f.name for f in schema.dense_fields]
        self._sequence_keys = [f.name for f in schema.sequence_fields]

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = {}

        if self._sparse_keys:
            result["sparse"] = torch.tensor(
                [self.data[k][idx] for k in self._sparse_keys], dtype=torch.long
            )
        else:
            result["sparse"] = torch.zeros(0, dtype=torch.long)

        if self._dense_keys:
            result["dense"] = torch.tensor(
                [self.data[k][idx] for k in self._dense_keys],
                dtype=torch.float32,
            )
        else:
            result["dense"] = torch.zeros(0, dtype=torch.float32)

        result["label"] = torch.tensor(
            self.data[self.schema.label_field][idx], dtype=torch.float32
        )

        # Sequences packed as a dict of tensors (one per sequence field)
        if self._sequence_keys:
            result["sequences"] = {
                k: torch.tensor(self.data[k][idx], dtype=torch.long)
                for k in self._sequence_keys
            }

        return result


class NegativeSamplingDataset(Dataset):
    """Wraps a TabularDataset with dynamic per-epoch negative sampling.

    For each positive sample, generates num_neg negative items by replacing
    the item features with randomly sampled uninteracted items.
    Negative samples get full item features (not just item_id).
    """

    def __init__(
        self,
        positive_data: Dict[str, np.ndarray],
        schema: DatasetSchema,
        user_col: str,
        item_col: str,
        all_item_data: Dict[str, np.ndarray],
        user_interacted_items: Dict[int, set],
        num_items: int,
        num_neg: int = 4,
        item_feature_cols: Optional[List[str]] = None,
    ):
        self.positive_data = positive_data
        self.schema = schema
        self.user_col = user_col
        self.item_col = item_col
        self.all_item_data = all_item_data
        self.user_interacted_items = user_interacted_items
        self.num_items = num_items
        self.num_neg = num_neg
        self.item_feature_cols = item_feature_cols or [item_col]

        self._num_positives = len(next(iter(positive_data.values())))

        # Pre-compute field orderings
        self._sparse_keys = [f.name for f in schema.sparse_fields]
        self._dense_keys = [f.name for f in schema.dense_fields]
        self._sequence_keys = [f.name for f in schema.sequence_fields]

        # Build expanded data with negatives for the current epoch
        self._build_epoch_data()

    def _build_epoch_data(self):
        """Build training data: each positive + num_neg negatives."""
        total = self._num_positives * (1 + self.num_neg)
        self._data: Dict[str, np.ndarray] = {}

        # Initialize arrays
        for key, arr in self.positive_data.items():
            if arr.ndim == 1:
                self._data[key] = np.empty(total, dtype=arr.dtype)
            else:
                self._data[key] = np.empty(
                    (total, arr.shape[1]), dtype=arr.dtype
                )

        idx = 0
        for pos_idx in range(self._num_positives):
            # Copy positive sample
            for key in self.positive_data:
                self._data[key][idx] = self.positive_data[key][pos_idx]
            idx += 1

            # Generate negative samples
            user_encoded = self.positive_data[self.user_col][pos_idx]
            interacted = self.user_interacted_items.get(user_encoded, set())

            neg_count = 0
            while neg_count < self.num_neg:
                neg_item_idx = np.random.randint(
                    0, len(self.all_item_data[self.item_col])
                )
                neg_item_id = self.all_item_data[self.item_col][neg_item_idx]

                if neg_item_id in interacted:
                    continue

                # Copy user features from positive, replace item features
                for key in self.positive_data:
                    if (
                        key in self.item_feature_cols
                        or key == self.schema.label_field
                    ):
                        continue
                    self._data[key][idx] = self.positive_data[key][pos_idx]

                # Set item features from sampled negative
                for col in self.item_feature_cols:
                    if col in self.all_item_data:
                        self._data[col][idx] = self.all_item_data[col][
                            neg_item_idx
                        ]

                # Label = 0 for negatives
                self._data[self.schema.label_field][idx] = 0
                idx += 1
                neg_count += 1

        self._length = idx

    def resample_negatives(self):
        """Re-sample negatives for a new epoch."""
        self._build_epoch_data()

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = {}

        if self._sparse_keys:
            result["sparse"] = torch.tensor(
                [self._data[k][idx] for k in self._sparse_keys],
                dtype=torch.long,
            )
        else:
            result["sparse"] = torch.zeros(0, dtype=torch.long)

        if self._dense_keys:
            result["dense"] = torch.tensor(
                [self._data[k][idx] for k in self._dense_keys],
                dtype=torch.float32,
            )
        else:
            result["dense"] = torch.zeros(0, dtype=torch.float32)

        result["label"] = torch.tensor(
            self._data[self.schema.label_field][idx], dtype=torch.float32
        )

        if self._sequence_keys:
            result["sequences"] = {
                k: torch.tensor(self._data[k][idx], dtype=torch.long)
                for k in self._sequence_keys
            }

        return result


class EvalRankingDataset(Dataset):
    """Dataset for leave-one-out ranking evaluation.

    For each user, scores 1 positive + num_neg_eval negative items.
    Returns all candidates for a single user as one batch.
    """

    def __init__(
        self,
        eval_data: Dict[str, np.ndarray],
        schema: DatasetSchema,
        user_col: str,
        item_col: str,
        all_item_data: Dict[str, np.ndarray],
        user_interacted_items: Dict[int, set],
        num_neg_eval: int = 999,
        item_feature_cols: Optional[List[str]] = None,
    ):
        self.schema = schema
        self.user_col = user_col
        self.item_col = item_col
        self.all_item_data = all_item_data
        self.user_interacted_items = user_interacted_items
        self.num_neg_eval = num_neg_eval
        self.item_feature_cols = item_feature_cols or [item_col]

        self._sparse_keys = [f.name for f in schema.sparse_fields]
        self._dense_keys = [f.name for f in schema.dense_fields]
        self._sequence_keys = [f.name for f in schema.sequence_fields]

        # Build evaluation candidates: 1 positive + num_neg_eval negatives per user
        self._build_eval_candidates(eval_data)

    def _build_eval_candidates(self, eval_data: Dict[str, np.ndarray]):
        """Pre-compute all evaluation candidates."""
        num_users = len(eval_data[self.user_col])
        candidates_per_user = 1 + self.num_neg_eval
        total = num_users * candidates_per_user

        self._data: Dict[str, np.ndarray] = {}
        for key, arr in eval_data.items():
            if arr.ndim == 1:
                self._data[key] = np.empty(total, dtype=arr.dtype)
            else:
                self._data[key] = np.empty(
                    (total, arr.shape[1]), dtype=arr.dtype
                )

        # Labels: first is positive (1), rest are negatives (0)
        self._data[self.schema.label_field] = np.zeros(total, dtype=np.float32)

        # User indices for grouping
        self._user_indices = []

        idx = 0
        for i in range(num_users):
            start_idx = idx

            # Positive sample
            for key in eval_data:
                self._data[key][idx] = eval_data[key][i]
            self._data[self.schema.label_field][idx] = 1.0
            idx += 1

            # Sample negatives
            user_encoded = eval_data[self.user_col][i]
            interacted = self.user_interacted_items.get(user_encoded, set())

            neg_count = 0
            while neg_count < self.num_neg_eval:
                neg_item_idx = np.random.randint(
                    0, len(self.all_item_data[self.item_col])
                )
                neg_item_id = self.all_item_data[self.item_col][neg_item_idx]

                if neg_item_id in interacted:
                    continue

                # Copy user features, replace item features
                for key in eval_data:
                    if (
                        key in self.item_feature_cols
                        or key == self.schema.label_field
                    ):
                        continue
                    self._data[key][idx] = eval_data[key][i]

                for col in self.item_feature_cols:
                    if col in self.all_item_data:
                        self._data[col][idx] = self.all_item_data[col][
                            neg_item_idx
                        ]

                self._data[self.schema.label_field][idx] = 0.0
                idx += 1
                neg_count += 1

            self._user_indices.append((start_idx, idx))

        self._length = total
        self._num_users = num_users
        self._candidates_per_user = candidates_per_user

    @property
    def num_users(self) -> int:
        return self._num_users

    @property
    def candidates_per_user(self) -> int:
        return self._candidates_per_user

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        result = {}

        if self._sparse_keys:
            result["sparse"] = torch.tensor(
                [self._data[k][idx] for k in self._sparse_keys],
                dtype=torch.long,
            )
        else:
            result["sparse"] = torch.zeros(0, dtype=torch.long)

        if self._dense_keys:
            result["dense"] = torch.tensor(
                [self._data[k][idx] for k in self._dense_keys],
                dtype=torch.float32,
            )
        else:
            result["dense"] = torch.zeros(0, dtype=torch.float32)

        result["label"] = torch.tensor(
            self._data[self.schema.label_field][idx], dtype=torch.float32
        )

        if self._sequence_keys:
            result["sequences"] = {
                k: torch.tensor(self._data[k][idx], dtype=torch.long)
                for k in self._sequence_keys
            }

        return result
