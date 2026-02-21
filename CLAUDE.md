# DeepFM Project

## Overview

Production-grade DeepFM and variants (xDeepFM, AttentionDeepFM) for CTR prediction in PyTorch. Targets Apple M2 with MPS backend.

## Key Abstractions

- `FieldSchema` / `DatasetSchema` (`deepfm/data/schema.py`): Generic feature contract. All modules consume the schema — no hard-coded feature names.
- `FeatureEmbedding` (`deepfm/models/layers/embedding.py`): Shared embedding layer returning 3 views: first_order (B,1), field_embeddings (B,F,D), flat_embeddings (B,total_dim).
- `BaseCTRModel` (`deepfm/models/base.py`): Abstract base. Subclasses implement `_build_components()` and `_forward_components()`.
- `MovieLensAdapter` (`deepfm/data/movielens.py`): Dataset adapter pattern — each dataset implements this to plug into the pipeline.

## Conventions

- Device: MPS-first, no CUDA, no mixed precision. Use `resolve_device()` from cli.py.
- Loss: BCEWithLogitsLoss (raw logits from model, sigmoid only at predict time).
- Embedding: Per-field custom dims with projection to common fm_embed_dim for FM/CIN/Attention.
- OOV: padding_idx=0, unknown features contribute zero.

## Dataset

### MovieLens (ML-100K)

- **Label**: binary — rating ≥ 4.0 → 1 (liked), else 0.
- **Features** (7 fields, total_embedding_dim=64):
  - User: `user_id` (dim=16), `gender` (4), `age` (4, bucketed to [1,18,25,35,45,50,56]), `occupation` (8), `zip_prefix` (8, first 3 digits)
  - Item: `movie_id` (dim=16), `genres` (8, SEQUENCE multi-hot, mean-pooled, max_len=6)

## Training Setup

- **Split**: global temporal split by timestamp — 80% train / 10% val / 10% test (by quantile). For val/test, the first chronologically positive interaction per user in that window is kept (1 positive per user); users with no positive in a window or not seen in train are excluded from that split's eval. `_user_items` is built from all interactions to prevent negative collisions. Legacy leave-one-out is still available via `data.split_strategy=leave_one_out`.
- **Negative sampling**:
  - Train: 4 negatives per positive, uniform random, re-sampled every epoch (dynamic) to act as regularizer.
  - Val/Test: 999 negatives per positive, popularity-stratified (`random.choices` with weights `count(item)^0.75`). Items unseen in train get count=1 (minimum weight). Simulates ranking 1 relevant item among 1000 candidates with harder negatives.
  - To restore uniform eval negatives: `data.neg_sampling_alpha=0.0`.
- **Metrics**:
  - AUC, LogLoss — computed over all rows flattened (global discrimination / calibration).
  - HR@K, NDCG@K (K=1,5,10,20) — computed per user. Scores for each user's 1+999 items are ranked; HR@K = fraction of users where the positive lands in top-K; NDCG@K = mean 1/log₂(rank+1), rewarding higher ranks more.

## Commands

- `make install` — install with uv
- `make train [ARGS="key=value ..."]` — train DeepFM on ML-100K
- `make evaluate [ARGS="..."]` — evaluate saved checkpoint
- `make compare [RUNS_DIR=outputs]` — print side-by-side metric table
- `make test` — run pytest
- `make lint` — ruff check + format check
- `make format` — auto-fix lint + format

## Package Manager

uv

## Testing

- Unit tests use synthetic data (fast, no network)
- Integration test downloads real ML-100K
- Run: `pytest tests/ -v`
