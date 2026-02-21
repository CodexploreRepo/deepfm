"""MovieLens-100K dataset adapter."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from deepfm.config import DataConfig
from deepfm.data.dataset import TabularDataset
from deepfm.data.schema import DatasetSchema, FeatureType, FieldSchema
from deepfm.data.transforms import LabelEncoder, MultiHotEncoder

# Genre columns in u.item (19 binary columns after the first 5 metadata cols)
GENRE_NAMES = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

# Age bucket boundaries from the PRD
AGE_BUCKETS = [1, 18, 25, 35, 45, 50, 56]


def _bucketize_age(age: int) -> int:
    """Map age to the largest bucket boundary <= age."""
    for b in reversed(AGE_BUCKETS):
        if age >= b:
            return b
    return AGE_BUCKETS[0]


class MovieLensAdapter:
    """Loads MovieLens-100K and produces train/val/test TabularDatasets.

    Implements leave-one-out split by timestamp, feature encoding,
    and schema construction.
    """

    def __init__(self, config: DataConfig) -> None:
        self.data_dir = Path(config.data_dir)
        self.config = config
        self._encoders: dict[str, LabelEncoder | MultiHotEncoder] = {}
        self._schema: DatasetSchema | None = None
        self._user_items: dict[int, set[int]] | None = None
        self._item_features: pd.DataFrame | None = None
        self._user_features: pd.DataFrame | None = None
        self._all_movie_ids: set[int] | None = None
        self._train_df: pd.DataFrame | None = None
        self._val_df: pd.DataFrame | None = None
        self._test_df: pd.DataFrame | None = None

    def build(
        self,
    ) -> tuple[DatasetSchema, TabularDataset, TabularDataset, TabularDataset]:
        """Load data, split, encode, and return (schema, train, val, test)."""
        df = self._load_and_merge()
        self._train_df, self._val_df, self._test_df = self._leave_one_out_split(
            df
        )

        # Fit encoders on training data only
        self._fit_encoders(self._train_df)

        # Build schema with vocabulary sizes from fitted encoders
        self._schema = self._build_schema()

        # Add negative samples
        train_with_neg = self._add_train_negatives(self._train_df)
        val_with_neg = self._add_eval_negatives(self._val_df)
        test_with_neg = self._add_eval_negatives(self._test_df)

        # Transform all splits
        train_ds = self._transform(train_with_neg)
        val_ds = self._transform(val_with_neg)
        test_ds = self._transform(test_with_neg)

        return self._schema, train_ds, val_ds, test_ds

    def resample_train(self) -> TabularDataset:
        """Re-sample training negatives (call each epoch for dynamic sampling)."""
        if self._train_df is None:
            raise RuntimeError("Call build() first")
        train_with_neg = self._add_train_negatives(self._train_df)
        return self._transform(train_with_neg)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_and_merge(self) -> pd.DataFrame:
        """Read u.data, u.user, u.item and merge into a single DataFrame."""
        # Ratings
        ratings = pd.read_csv(
            self.data_dir / "u.data",
            sep="\t",
            names=["user_id", "movie_id", "rating", "timestamp"],
        )

        # Users
        users = pd.read_csv(
            self.data_dir / "u.user",
            sep="|",
            names=["user_id", "age", "gender", "occupation", "zip_code"],
            encoding="latin-1",
        )
        users["age"] = users["age"].apply(_bucketize_age)
        users["zip_prefix"] = users["zip_code"].astype(str).str[:3]

        # Items
        item_cols = [
            "movie_id",
            "title",
            "release_date",
            "video_date",
            "url",
        ] + GENRE_NAMES
        items = pd.read_csv(
            self.data_dir / "u.item",
            sep="|",
            names=item_cols,
            encoding="latin-1",
        )
        # Convert genre binary columns to list of genre names
        genre_cols = items[GENRE_NAMES]
        items["genres"] = genre_cols.apply(
            lambda row: [g for g, v in zip(GENRE_NAMES, row) if v == 1], axis=1
        )
        items = items[["movie_id", "genres"]]

        # Store features for negative sampling
        self._item_features = items.copy()
        self._user_features = users[
            ["user_id", "age", "gender", "occupation", "zip_prefix"]
        ].copy()
        self._all_movie_ids = set(items["movie_id"].tolist())

        # Merge
        df = ratings.merge(users, on="user_id").merge(items, on="movie_id")

        # Binary label
        df["label"] = (df["rating"] >= self.config.label_threshold).astype(
            np.float32
        )

        return df

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def _leave_one_out_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Leave-one-out split per user ordered by timestamp."""
        df = df.sort_values(["user_id", "timestamp"])

        # Count interactions per user
        user_counts = df.groupby("user_id").size()
        eligible_users = set(
            user_counts[user_counts >= self.config.min_interactions].index
        )

        train_rows, val_rows, test_rows = [], [], []

        for user_id, group in df.groupby("user_id"):
            if user_id not in eligible_users:
                train_rows.append(group)
                continue
            # Last → test, second-to-last → val, rest → train
            test_rows.append(group.iloc[[-1]])
            val_rows.append(group.iloc[[-2]])
            train_rows.append(group.iloc[:-2])

        train_df = pd.concat(train_rows, ignore_index=True)
        val_df = pd.concat(val_rows, ignore_index=True)
        test_df = pd.concat(test_rows, ignore_index=True)

        # Track all user-item interactions for negative sampling
        self._user_items = (
            df.groupby("user_id")["movie_id"].apply(set).to_dict()
        )

        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _fit_encoders(self, train_df: pd.DataFrame) -> None:
        """Fit label encoders and multi-hot encoder on training data."""
        for col in [
            "user_id",
            "movie_id",
            "gender",
            "age",
            "occupation",
            "zip_prefix",
        ]:
            enc = LabelEncoder()
            enc.fit(train_df[col].tolist())
            self._encoders[col] = enc

        genre_enc = MultiHotEncoder(max_length=6)
        genre_enc.fit(train_df["genres"].tolist())
        self._encoders["genres"] = genre_enc

    def _build_schema(self) -> DatasetSchema:
        """Build DatasetSchema from fitted encoder vocabulary sizes."""
        fields: dict[str, FieldSchema] = {}

        sparse_specs = [
            ("user_id", 16, "user"),
            ("movie_id", 16, "item"),
            ("gender", 4, "user"),
            ("age", 4, "user"),
            ("occupation", 8, "user"),
            ("zip_prefix", 8, "user"),
        ]
        for name, embed_dim, group in sparse_specs:
            enc = self._encoders[name]
            fields[name] = FieldSchema(
                name=name,
                feature_type=FeatureType.SPARSE,
                vocabulary_size=enc.vocabulary_size,
                embedding_dim=embed_dim,
                group=group,
            )

        genre_enc = self._encoders["genres"]
        fields["genres"] = FieldSchema(
            name="genres",
            feature_type=FeatureType.SEQUENCE,
            vocabulary_size=genre_enc.vocabulary_size,
            embedding_dim=8,
            group="item",
            max_length=6,
            combiner="mean",
        )

        return DatasetSchema(fields=fields, label_field="label")

    def _transform(self, df: pd.DataFrame) -> TabularDataset:
        """Apply fitted encoders and return a TabularDataset."""
        features: dict[str, np.ndarray] = {}

        for col in [
            "user_id",
            "movie_id",
            "gender",
            "age",
            "occupation",
            "zip_prefix",
        ]:
            enc = self._encoders[col]
            features[col] = enc.transform(df[col].tolist())

        genre_enc = self._encoders["genres"]
        features["genres"] = genre_enc.transform(df["genres"].tolist())

        labels = df["label"].values.astype(np.float32)
        return TabularDataset(features, labels)

    # ------------------------------------------------------------------
    # Negative sampling
    # ------------------------------------------------------------------

    def _sample_negatives_for_user(
        self, user_id: int, num_neg: int
    ) -> list[int]:
        """Sample movie IDs the user has NOT interacted with."""
        seen = self._user_items[user_id]
        candidates = list(self._all_movie_ids - seen)
        if len(candidates) < num_neg:
            return candidates
        return random.sample(candidates, num_neg)

    def _build_neg_rows(
        self, user_id: int, neg_movie_ids: list[int], user_row: pd.Series
    ) -> list[dict]:
        """Build negative sample rows with full user + item features."""
        item_lookup = self._item_features.set_index("movie_id")
        rows = []
        for mid in neg_movie_ids:
            row = {
                "user_id": user_id,
                "movie_id": mid,
                "gender": user_row["gender"],
                "age": user_row["age"],
                "occupation": user_row["occupation"],
                "zip_prefix": user_row["zip_prefix"],
                "genres": item_lookup.loc[mid, "genres"],
                "label": 0.0,
            }
            rows.append(row)
        return rows

    def _add_train_negatives(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Add num_neg_train negatives per positive sample."""
        user_lookup = self._user_features.set_index("user_id")
        neg_rows = []
        for _, pos_row in train_df.iterrows():
            uid = pos_row["user_id"]
            neg_movies = self._sample_negatives_for_user(
                uid, self.config.num_neg_train
            )
            user_row = user_lookup.loc[uid]
            neg_rows.extend(self._build_neg_rows(uid, neg_movies, user_row))

        neg_df = pd.DataFrame(neg_rows)
        # Keep only columns present in train_df that we need
        keep_cols = [
            "user_id",
            "movie_id",
            "gender",
            "age",
            "occupation",
            "zip_prefix",
            "genres",
            "label",
        ]
        combined = pd.concat(
            [train_df[keep_cols], neg_df[keep_cols]], ignore_index=True
        )
        return combined.sample(frac=1.0).reset_index(drop=True)

    def _add_eval_negatives(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        """Add num_neg_eval negatives per positive sample for ranking eval."""
        user_lookup = self._user_features.set_index("user_id")
        neg_rows = []
        for _, pos_row in eval_df.iterrows():
            uid = pos_row["user_id"]
            neg_movies = self._sample_negatives_for_user(
                uid, self.config.num_neg_eval
            )
            user_row = user_lookup.loc[uid]
            neg_rows.extend(self._build_neg_rows(uid, neg_movies, user_row))

        neg_df = pd.DataFrame(neg_rows)
        keep_cols = [
            "user_id",
            "movie_id",
            "gender",
            "age",
            "occupation",
            "zip_prefix",
            "genres",
            "label",
        ]
        return pd.concat(
            [eval_df[keep_cols], neg_df[keep_cols]], ignore_index=True
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def schema(self) -> DatasetSchema:
        if self._schema is None:
            raise RuntimeError("Call build() first")
        return self._schema

    @property
    def user_items(self) -> dict[int, set[int]]:
        """Mapping of user_id → set of movie_ids they interacted with."""
        if self._user_items is None:
            raise RuntimeError("Call build() first")
        return self._user_items
