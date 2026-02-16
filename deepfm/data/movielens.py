from __future__ import annotations

import io
import logging
import os
import urllib.request
import zipfile
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from deepfm.data.dataset import EvalRankingDataset, NegativeSamplingDataset
from deepfm.data.schema import DatasetSchema, FeatureType, FieldSchema
from deepfm.data.transforms import LabelEncoder, MultiHotEncoder

logger = logging.getLogger(__name__)

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


class MovieLensAdapter:
    """Loads MovieLens-100K and maps it to the generic pipeline.

    Features:
        SPARSE: user_id(16), movie_id(16), gender(4), age(4), occupation(8), zip_prefix(8)
        SEQUENCE: genres(8, mean pooling, max_length=6)
        LABEL: rating >= threshold -> 1, else 0

    Split: leave-one-out per user by timestamp.
    Negative sampling: dynamic per epoch (train), pre-computed (eval).
    """

    def __init__(self, data_dir: str = "./data/ml-100k", label_threshold: float = 4.0):
        self.data_dir = data_dir
        self.label_threshold = label_threshold
        self.encoders: Dict[str, LabelEncoder | MultiHotEncoder] = {}
        self.schema: Optional[DatasetSchema] = None

        # Item feature lookup (encoded), indexed by position
        self._item_data: Optional[Dict[str, np.ndarray]] = None
        # User-item interaction sets (encoded item ids)
        self._user_interacted: Optional[Dict[int, Set[int]]] = None
        # Item feature column names (for negative sampling)
        self._item_feature_cols: List[str] = []

    def _auto_download(self):
        """Download and extract ML-100K if not present."""
        if os.path.exists(os.path.join(self.data_dir, "u.data")):
            return

        logger.info(f"Downloading MovieLens-100K to {self.data_dir}...")
        parent_dir = os.path.dirname(self.data_dir) or "."
        os.makedirs(parent_dir, exist_ok=True)

        import ssl
        import subprocess

        # Try curl first (handles system SSL certs better on macOS)
        zip_path = os.path.join(parent_dir, "ml-100k.zip")
        try:
            subprocess.run(
                ["curl", "-fSL", "-o", zip_path, ML100K_URL],
                check=True,
                capture_output=True,
                timeout=120,
            )
            with open(zip_path, "rb") as f:
                zip_data = io.BytesIO(f.read())
            os.remove(zip_path)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            # Fallback to urllib with unverified SSL
            ctx = ssl._create_unverified_context()
            response = urllib.request.urlopen(ML100K_URL, context=ctx)
            zip_data = io.BytesIO(response.read())

        with zipfile.ZipFile(zip_data) as zf:
            zf.extractall(parent_dir)

        logger.info("Download complete.")

    def _load_raw_data(self) -> pd.DataFrame:
        """Load and join ML-100K raw files."""
        # Ratings: user_id, item_id, rating, timestamp
        ratings = pd.read_csv(
            os.path.join(self.data_dir, "u.data"),
            sep="\t",
            names=["user_id", "movie_id", "rating", "timestamp"],
            encoding="latin-1",
        )

        # Users: user_id, age, gender, occupation, zip_code
        users = pd.read_csv(
            os.path.join(self.data_dir, "u.user"),
            sep="|",
            names=["user_id", "age", "gender", "occupation", "zip_code"],
            encoding="latin-1",
        )

        # Movies: movie_id | title | release_date | video_date | imdb_url | genre_flags...
        genre_names = [
            "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
        ]
        movie_cols = ["movie_id", "title", "release_date", "video_date", "imdb_url"] + genre_names
        movies = pd.read_csv(
            os.path.join(self.data_dir, "u.item"),
            sep="|",
            names=movie_cols,
            encoding="latin-1",
        )

        # Convert genre flags to list of genre names
        movies["genres"] = movies[genre_names].apply(
            lambda row: [g for g, v in zip(genre_names, row) if v == 1], axis=1
        )

        # Extract zip prefix (first 3 chars)
        users["zip_prefix"] = users["zip_code"].astype(str).str[:3]

        # Join
        df = ratings.merge(users, on="user_id").merge(
            movies[["movie_id", "genres"]], on="movie_id"
        )

        # Binarize label
        df["label"] = (df["rating"] >= self.label_threshold).astype(np.float32)

        return df

    def _leave_one_out_split(
        self, df: pd.DataFrame, min_interactions: int = 3
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Leave-one-out split per user, ordered by timestamp."""
        # Sort by user and timestamp
        df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

        train_rows, val_rows, test_rows = [], [], []

        for _, group in df.groupby("user_id"):
            if len(group) < min_interactions:
                # Not enough interactions — keep all in train
                train_rows.append(group)
            else:
                train_rows.append(group.iloc[:-2])
                val_rows.append(group.iloc[-2:-1])
                test_rows.append(group.iloc[-1:])

        train_df = pd.concat(train_rows, ignore_index=True)
        val_df = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame()
        test_df = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()

        logger.info(
            f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
        return train_df, val_df, test_df

    def _fit_encoders(self, train_df: pd.DataFrame):
        """Fit all encoders on training data only."""
        # Sparse features
        for col in ["user_id", "movie_id", "gender", "age", "occupation", "zip_prefix"]:
            enc = LabelEncoder()
            enc.fit(train_df[col].values)
            self.encoders[col] = enc

        # Sequence feature
        genre_enc = MultiHotEncoder(max_length=6)
        genre_enc.fit(train_df["genres"].tolist())
        self.encoders["genres"] = genre_enc

    def _encode_df(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Encode a dataframe into a dict of numpy arrays using fitted encoders."""
        data = {}
        for col in ["user_id", "movie_id", "gender", "age", "occupation", "zip_prefix"]:
            data[col] = self.encoders[col].transform(df[col].values)

        data["genres"] = self.encoders["genres"].transform(df["genres"].tolist())
        data["label"] = df["label"].values.astype(np.float32)

        return data

    def _build_item_lookup(self, df: pd.DataFrame):
        """Build item feature lookup table for negative sampling."""
        # Deduplicate items — one row per movie_id
        items_df = df.drop_duplicates(subset="movie_id").reset_index(drop=True)

        self._item_data = {
            "movie_id": self.encoders["movie_id"].transform(items_df["movie_id"].values),
            "genres": self.encoders["genres"].transform(items_df["genres"].tolist()),
        }
        self._item_feature_cols = ["movie_id", "genres"]

    def _build_user_interacted(self, full_data: Dict[str, np.ndarray]):
        """Build user -> set of interacted item_ids mapping."""
        self._user_interacted = {}
        users = full_data["user_id"]
        items = full_data["movie_id"]
        for u, i in zip(users, items):
            if u not in self._user_interacted:
                self._user_interacted[u] = set()
            self._user_interacted[u].add(i)

    def _build_schema(self) -> DatasetSchema:
        """Construct schema from fitted encoders."""
        fields = {
            "user_id": FieldSchema(
                name="user_id",
                feature_type=FeatureType.SPARSE,
                vocabulary_size=self.encoders["user_id"].vocabulary_size,
                embedding_dim=16,
                group="user",
            ),
            "movie_id": FieldSchema(
                name="movie_id",
                feature_type=FeatureType.SPARSE,
                vocabulary_size=self.encoders["movie_id"].vocabulary_size,
                embedding_dim=16,
                group="item",
            ),
            "gender": FieldSchema(
                name="gender",
                feature_type=FeatureType.SPARSE,
                vocabulary_size=self.encoders["gender"].vocabulary_size,
                embedding_dim=4,
                group="user",
            ),
            "age": FieldSchema(
                name="age",
                feature_type=FeatureType.SPARSE,
                vocabulary_size=self.encoders["age"].vocabulary_size,
                embedding_dim=4,
                group="user",
            ),
            "occupation": FieldSchema(
                name="occupation",
                feature_type=FeatureType.SPARSE,
                vocabulary_size=self.encoders["occupation"].vocabulary_size,
                embedding_dim=8,
                group="user",
            ),
            "zip_prefix": FieldSchema(
                name="zip_prefix",
                feature_type=FeatureType.SPARSE,
                vocabulary_size=self.encoders["zip_prefix"].vocabulary_size,
                embedding_dim=8,
                group="user",
            ),
            "genres": FieldSchema(
                name="genres",
                feature_type=FeatureType.SEQUENCE,
                vocabulary_size=self.encoders["genres"].vocabulary_size,
                embedding_dim=8,
                max_length=6,
                combiner="mean",
                group="item",
            ),
            "label": FieldSchema(
                name="label",
                feature_type=FeatureType.DENSE,
                is_label=True,
            ),
        }
        return DatasetSchema(fields=fields, label_field="label")

    def build_datasets(
        self,
        min_interactions: int = 3,
        num_neg_train: int = 4,
        num_neg_eval: int = 999,
        auto_download: bool = True,
    ) -> Tuple[NegativeSamplingDataset, EvalRankingDataset, EvalRankingDataset]:
        """Load data, split, encode, and return train/val/test datasets."""
        if auto_download:
            self._auto_download()

        # Load raw data
        df = self._load_raw_data()
        logger.info(f"Loaded {len(df)} interactions from MovieLens-100K")

        # Split
        train_df, val_df, test_df = self._leave_one_out_split(df, min_interactions)

        # Fit encoders on train
        self._fit_encoders(train_df)

        # Encode all splits
        train_data = self._encode_df(train_df)
        val_data = self._encode_df(val_df)
        test_data = self._encode_df(test_df)

        # Build schema
        self.schema = self._build_schema()

        # Build item lookup (from full data for complete item features)
        self._build_item_lookup(df)

        # Build user interaction history (from all data for proper neg filtering)
        full_data = self._encode_df(df)
        self._build_user_interacted(full_data)

        num_items = len(self._item_data["movie_id"])

        # Training dataset with dynamic negative sampling
        train_ds = NegativeSamplingDataset(
            positive_data=train_data,
            schema=self.schema,
            user_col="user_id",
            item_col="movie_id",
            all_item_data=self._item_data,
            user_interacted_items=self._user_interacted,
            num_items=num_items,
            num_neg=num_neg_train,
            item_feature_cols=self._item_feature_cols,
        )

        # Validation dataset with ranking evaluation
        val_ds = EvalRankingDataset(
            eval_data=val_data,
            schema=self.schema,
            user_col="user_id",
            item_col="movie_id",
            all_item_data=self._item_data,
            user_interacted_items=self._user_interacted,
            num_neg_eval=num_neg_eval,
            item_feature_cols=self._item_feature_cols,
        )

        # Test dataset with ranking evaluation
        test_ds = EvalRankingDataset(
            eval_data=test_data,
            schema=self.schema,
            user_col="user_id",
            item_col="movie_id",
            all_item_data=self._item_data,
            user_interacted_items=self._user_interacted,
            num_neg_eval=num_neg_eval,
            item_feature_cols=self._item_feature_cols,
        )

        logger.info(
            f"Datasets: train={len(train_ds)} (with neg), "
            f"val={len(val_ds)} ({val_ds.num_users} users × {val_ds.candidates_per_user}), "
            f"test={len(test_ds)} ({test_ds.num_users} users × {test_ds.candidates_per_user})"
        )

        return train_ds, val_ds, test_ds
