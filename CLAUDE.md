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

- **Split**: leave-one-out per user by timestamp — last → test, 2nd-to-last → val, rest → train. Users with <3 interactions go entirely to train.
- **Negative sampling**: The training dataset has only observed positives; negatives are sampled from movies the user has never rated (across the full dataset).
  - Train: 4 negatives per positive, re-sampled every epoch (dynamic) to act as regularizer.
  - Val/Test: 999 negatives per positive, fixed pool — simulates ranking 1 relevant item among 1000 candidates.
- **Metrics**:
  - AUC, LogLoss — computed over all rows flattened (global discrimination / calibration).
  - HR@K, NDCG@K (K=5,10,20) — computed per user. Scores for each user's 1+999 items are ranked; HR@K = fraction of users where the positive lands in top-K; NDCG@K = mean 1/log₂(rank+1), rewarding higher ranks more.

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
