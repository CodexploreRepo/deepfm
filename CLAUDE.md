# DeepFM Project

## Overview
Production-grade DeepFM and variants (xDeepFM, AttentionDeepFM) for CTR prediction in PyTorch. Targets Apple M2 with MPS backend.

## Key Abstractions
- `FieldSchema` / `DatasetSchema` (`deepfm/data/schema.py`): Generic feature contract. All modules consume the schema — no hard-coded feature names.
- `FeatureEmbedding` (`deepfm/models/layers/embedding.py`): Shared embedding layer returning 3 views: first_order (B,1), field_embeddings (B,F,D), flat_embeddings (B,total_dim).
- `BaseCTRModel` (`deepfm/models/base.py`): Abstract base. Subclasses implement `_build_components()` and `_forward_components()`.
- `MovieLensAdapter` (`deepfm/data/movielens.py`): Dataset adapter pattern — each dataset implements this to plug into the pipeline.

## Conventions
- Config: Python dataclasses in `deepfm/config.py`, loaded from YAML via dacite.
- Device: MPS-first, no CUDA, no mixed precision. Use `resolve_device()` from cli.py.
- Loss: BCEWithLogitsLoss (raw logits from model, sigmoid only at predict time).
- Split: Leave-one-out per user by timestamp. Negative sampling: 4 neg/pos train, 999 neg/pos eval.
- Metrics: AUC, LogLoss, HR@K, NDCG@K for K=5,10,20.
- Embedding: Per-field custom dims with projection to common fm_embed_dim for FM/CIN/Attention.
- OOV: padding_idx=0, unknown features contribute zero.

## Commands
- `make install` — install with uv
- `make train` — train DeepFM on ML-100K
- `make test` — run pytest
- `make lint` — ruff check + format check
- `make format` — auto-fix lint + format

## Package Manager
uv

## Testing
- Unit tests use synthetic data (fast, no network)
- Integration test downloads real ML-100K
- Run: `pytest tests/ -v`
