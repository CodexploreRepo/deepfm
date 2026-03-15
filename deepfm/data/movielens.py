"""MovieLens-100K dataset adapter."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from deepfm.config import DataConfig
from deepfm.data.dataset import TabularDataset
from deepfm.data.schema import DatasetSchema, FeatureType, FieldSchema
from deepfm.data.transforms import LabelEncoder, MinMaxScaler, MultiHotEncoder

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


def _bucket_release_year(year) -> str:
    """Map a year to a 5-year bin string like '1990-1994', or 'unknown'."""
    if pd.isna(year):
        return "unknown"
    y = int(year)
    base = (y // 5) * 5
    return f"{base}-{base + 4}"


def _bucket_movie_age(years_float) -> str:
    """Map a float number of years to an age bucket string."""
    if pd.isna(years_float) or years_float < 0:
        return "unknown"
    y = years_float
    if y < 1:
        return "<1yr"
    if y < 3:
        return "1-3yr"
    if y < 7:
        return "3-7yr"
    if y < 15:
        return "7-15yr"
    if y < 30:
        return "15-30yr"
    return "30+yr"


class MovieLensAdapter:
    """Loads MovieLens-100K and produces train/val/test TabularDatasets.

    Implements leave-one-out split by timestamp, feature encoding,
    and schema construction.
    """

    def __init__(self, config: DataConfig) -> None:
        self.data_dir = Path(config.data_dir)
        self.config = config
        self._encoders: dict[str, LabelEncoder | MultiHotEncoder] = {}
        self._scalers: dict[str, MinMaxScaler] = {}
        self._schema: DatasetSchema | None = None
        self._user_items: dict[int, set[int]] | None = None
        self._item_features: pd.DataFrame | None = None
        self._user_features: pd.DataFrame | None = None
        self._all_movie_ids: set[int] | None = None
        self._pop_weights: dict[int, float] | None = None
        self._train_df: pd.DataFrame | None = None
        self._val_df: pd.DataFrame | None = None
        self._test_df: pd.DataFrame | None = None
        self._user_counts: pd.Series | None = None
        self._item_counts: pd.Series | None = None

    def build(
        self,
    ) -> tuple[DatasetSchema, TabularDataset, TabularDataset, TabularDataset]:
        """Load data, split, encode, and return (schema, train, val, test)."""
        df = self._load_and_merge()
        if self.config.split_strategy == "temporal":
            self._train_df, self._val_df, self._test_df = self._temporal_split(
                df
            )
        else:
            self._train_df, self._val_df, self._test_df = (
                self._leave_one_out_split(df)
            )

        self._pop_weights = self._build_popularity_weights(self._train_df)

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
        genre_cols_df = items[GENRE_NAMES]
        items["genres"] = genre_cols_df.apply(
            lambda row: [g for g, v in zip(GENRE_NAMES, row) if v == 1], axis=1
        )

        # Item-level derived features (purely from item metadata)
        items["_release_dt"] = pd.to_datetime(
            items["release_date"], format="%d-%b-%Y", errors="coerce"
        )
        items["release_year_bucket"] = (
            items["_release_dt"].dt.year.apply(_bucket_release_year)
        )
        items["num_genres"] = items[GENRE_NAMES].sum(axis=1).astype(str)

        # Store item features for negative sampling (includes new fields + _release_dt)
        self._item_features = items[
            ["movie_id", "genres", "release_year_bucket", "num_genres", "_release_dt"]
        ].copy()
        self._user_features = users[
            ["user_id", "age", "gender", "occupation", "zip_prefix"]
        ].copy()
        self._all_movie_ids = set(items["movie_id"].tolist())

        # Merge (bring in item-level features except private _release_dt for df)
        items_for_merge = items[
            ["movie_id", "genres", "release_year_bucket", "num_genres", "_release_dt"]
        ]
        df = ratings.merge(users, on="user_id").merge(items_for_merge, on="movie_id")

        # Binary label
        df["label"] = (df["rating"] >= self.config.label_threshold).astype(
            np.float32
        )

        # Context features from rating timestamp
        df["_rating_dt"] = pd.to_datetime(df["timestamp"], unit="s")
        weekday = df["_rating_dt"].dt.weekday.astype(float)
        df["dow_sin"] = np.sin(2 * np.pi * weekday / 7).astype(np.float32)
        df["dow_cos"] = np.cos(2 * np.pi * weekday / 7).astype(np.float32)
        hour = df["_rating_dt"].dt.hour.astype(float)
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)

        # Movie age at time of rating (years)
        df["movie_age_at_rating"] = (
            (df["_rating_dt"] - df["_release_dt"]).dt.days / 365.25
        ).apply(_bucket_movie_age)

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

    def _temporal_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Global temporal split: 80/10/10 by timestamp quantile."""
        df = df.sort_values("timestamp").reset_index(drop=True)

        val_ratio = self.config.temporal_val_ratio
        test_ratio = self.config.temporal_test_ratio

        train_cutoff = df["timestamp"].quantile(1 - val_ratio - test_ratio)
        val_cutoff = df["timestamp"].quantile(1 - test_ratio)

        train_df = df[df["timestamp"] <= train_cutoff]
        val_df_all = df[
            (df["timestamp"] > train_cutoff) & (df["timestamp"] <= val_cutoff)
        ]
        test_df_all = df[df["timestamp"] > val_cutoff]

        # Build _user_items from ALL interactions (prevents negative collisions)
        self._user_items = (
            df.groupby("user_id")["movie_id"].apply(set).to_dict()
        )

        train_users = set(train_df["user_id"].unique())

        # For val/test: keep only positives, 1 per user (first chronologically),
        # restricted to users seen in train
        def _first_positive_per_user(split_df: pd.DataFrame) -> pd.DataFrame:
            positives = split_df[split_df["label"] == 1.0]
            positives = positives[positives["user_id"].isin(train_users)]
            return positives.groupby("user_id").first().reset_index()

        val_df = _first_positive_per_user(val_df_all)
        test_df = _first_positive_per_user(test_df_all)

        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _fit_encoders(self, train_df: pd.DataFrame) -> None:
        """Fit label encoders, multi-hot encoder, and scalers on training data."""
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

        # New SPARSE fields
        for col in ["release_year_bucket", "movie_age_at_rating", "num_genres"]:
            enc = LabelEncoder()
            enc.fit(train_df[col].tolist())
            self._encoders[col] = enc

        # DENSE count features: fit scalers on training positives only (no leakage)
        pos_train = train_df[train_df["label"] == 1]
        self._user_counts = pos_train.groupby("user_id").size()
        self._item_counts = pos_train.groupby("movie_id").size()

        self._scalers["user_rating_count"] = MinMaxScaler().fit(
            np.log1p(self._user_counts.values)
        )
        self._scalers["item_rating_count"] = MinMaxScaler().fit(
            np.log1p(self._item_counts.values)
        )

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

        # New SPARSE fields
        new_sparse_specs = [
            ("release_year_bucket", 4, "item"),
            ("movie_age_at_rating", 4, "context"),
            ("num_genres", 4, "item"),
        ]
        for name, embed_dim, group in new_sparse_specs:
            enc = self._encoders[name]
            fields[name] = FieldSchema(
                name=name,
                feature_type=FeatureType.SPARSE,
                vocabulary_size=enc.vocabulary_size,
                embedding_dim=embed_dim,
                group=group,
            )

        # New DENSE cyclical context features (already in [-1, 1])
        for name in ["dow_sin", "dow_cos", "hour_sin", "hour_cos"]:
            fields[name] = FieldSchema(
                name=name,
                feature_type=FeatureType.DENSE,
                embedding_dim=4,
                group="context",
            )

        # New DENSE activity/popularity features
        fields["user_rating_count"] = FieldSchema(
            name="user_rating_count",
            feature_type=FeatureType.DENSE,
            embedding_dim=8,
            group="user",
        )
        fields["item_rating_count"] = FieldSchema(
            name="item_rating_count",
            feature_type=FeatureType.DENSE,
            embedding_dim=8,
            group="item",
        )

        return DatasetSchema(fields=fields, label_field="label")

    def _transform(self, df: pd.DataFrame) -> TabularDataset:
        """Apply fitted encoders/scalers and return a TabularDataset."""
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

        # New SPARSE fields
        for col in ["release_year_bucket", "movie_age_at_rating", "num_genres"]:
            features[col] = self._encoders[col].transform(df[col].tolist())

        # DENSE cyclical features — pass through directly
        for col in ["dow_sin", "dow_cos", "hour_sin", "hour_cos"]:
            features[col] = df[col].values.astype(np.float32)

        # DENSE count features — log1p + MinMaxScale, OOV → 0
        user_log = np.log1p(
            df["user_id"].map(self._user_counts).fillna(0).values
        )
        features["user_rating_count"] = (
            self._scalers["user_rating_count"].transform(user_log).astype(np.float32)
        )
        item_log = np.log1p(
            df["movie_id"].map(self._item_counts).fillna(0).values
        )
        features["item_rating_count"] = (
            self._scalers["item_rating_count"].transform(item_log).astype(np.float32)
        )

        labels = df["label"].values.astype(np.float32)
        return TabularDataset(features, labels)

    # ------------------------------------------------------------------
    # Negative sampling
    # ------------------------------------------------------------------

    def _build_popularity_weights(
        self, train_df: pd.DataFrame
    ) -> dict[int, float]:
        """Popularity-stratified weights: count(item)^alpha, min count=1."""
        alpha = self.config.neg_sampling_alpha
        counts = (
            train_df[train_df["label"] == 1.0]["movie_id"]
            .value_counts()
            .to_dict()
        )
        return {
            item: max(counts.get(item, 1), 1) ** alpha
            for item in self._all_movie_ids
        }

    def _sample_negatives_for_user(
        self, user_id: int, num_neg: int
    ) -> list[int]:
        """Sample movie IDs the user has NOT interacted with (uniform)."""
        seen = self._user_items[user_id]
        candidates = list(self._all_movie_ids - seen)
        if len(candidates) < num_neg:
            return candidates
        return random.sample(candidates, num_neg)

    def _build_neg_rows(
        self,
        user_id: int,
        neg_movie_ids: list[int],
        user_row: pd.Series,
        pos_row: pd.Series,
    ) -> list[dict]:
        """Build negative sample rows with full user + item + context features."""
        item_lookup = self._item_features.set_index("movie_id")
        rating_dt = pos_row["_rating_dt"]
        rows = []
        for mid in neg_movie_ids:
            item = item_lookup.loc[mid]
            release_dt = item["_release_dt"]
            if pd.notna(rating_dt) and pd.notna(release_dt):
                age_days = (rating_dt - release_dt).days / 365.25
            else:
                age_days = float("nan")

            row = {
                "user_id": user_id,
                "movie_id": mid,
                "gender": user_row["gender"],
                "age": user_row["age"],
                "occupation": user_row["occupation"],
                "zip_prefix": user_row["zip_prefix"],
                "genres": item["genres"],
                "release_year_bucket": item["release_year_bucket"],
                "num_genres": item["num_genres"],
                "movie_age_at_rating": _bucket_movie_age(age_days),
                "dow_sin": float(pos_row["dow_sin"]),
                "dow_cos": float(pos_row["dow_cos"]),
                "hour_sin": float(pos_row["hour_sin"]),
                "hour_cos": float(pos_row["hour_cos"]),
                "_rating_dt": rating_dt,
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
            neg_rows.extend(self._build_neg_rows(uid, neg_movies, user_row, pos_row))

        neg_df = pd.DataFrame(neg_rows)
        keep_cols = [
            "user_id",
            "movie_id",
            "gender",
            "age",
            "occupation",
            "zip_prefix",
            "genres",
            "release_year_bucket",
            "movie_age_at_rating",
            "num_genres",
            "dow_sin",
            "dow_cos",
            "hour_sin",
            "hour_cos",
            "label",
        ]
        combined = pd.concat(
            [train_df[keep_cols], neg_df[keep_cols]], ignore_index=True
        )
        return combined.sample(frac=1.0).reset_index(drop=True)

    def _add_eval_negatives(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        """Add num_neg_eval negatives per positive sample for ranking eval.

        Uses popularity-stratified sampling (alpha from config) so harder
        negatives are included, making model differences easier to detect.
        """
        user_lookup = self._user_features.set_index("user_id")
        neg_rows = []
        for _, pos_row in eval_df.iterrows():
            uid = pos_row["user_id"]
            candidates = list(self._all_movie_ids - self._user_items[uid])
            weights = [self._pop_weights[c] for c in candidates]
            num_neg = min(self.config.num_neg_eval, len(candidates))
            neg_movies = random.choices(candidates, weights=weights, k=num_neg)
            user_row = user_lookup.loc[uid]
            neg_rows.extend(self._build_neg_rows(uid, neg_movies, user_row, pos_row))

        neg_df = pd.DataFrame(neg_rows)
        keep_cols = [
            "user_id",
            "movie_id",
            "gender",
            "age",
            "occupation",
            "zip_prefix",
            "genres",
            "release_year_bucket",
            "movie_age_at_rating",
            "num_genres",
            "dow_sin",
            "dow_cos",
            "hour_sin",
            "hour_cos",
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
