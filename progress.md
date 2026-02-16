# DeepFM Implementation Progress

## Task Dependency Graph

```
#1 Foundation
├── #2 Data Pipeline  ──┐
├── #3 Model Layers   ──┼── #4 Models ── #5 Training & CLI ── #6 Tests & Docs
```

## Tasks

### Phase 1: Foundation `[completed]`

> **Task #1**: Set up project foundation

- [x] `pyproject.toml` — uv, deps (torch, numpy, pandas, scikit-learn, pyyaml, dacite), dev deps (pytest, pytest-cov, ruff)
- [x] `Makefile` — targets: install, train, test, lint, format, clean
- [x] `CLAUDE.md` — project conventions for AI-assisted dev
- [x] `deepfm/__init__.py`
- [x] `deepfm/config.py` — nested dataclasses (ExperimentConfig, DataConfig, FeatureConfig, FMConfig, DNNConfig, CINConfig, AttentionConfig, TrainingConfig) + `load_config()` via dacite
- [x] `deepfm/data/__init__.py`
- [x] `deepfm/data/schema.py` — FieldSchema, DatasetSchema with property helpers
- [x] `deepfm/data/transforms.py` — LabelEncoder (OOV=0), MinMaxScaler, MultiHotEncoder
- [x] `deepfm/utils/__init__.py`
- [x] `deepfm/utils/seeding.py` — `seed_everything()`
- [x] `deepfm/utils/logging.py` — stdout + file logging
- [x] `deepfm/utils/io.py` — checkpoint save/load

### Phase 2: Data Pipeline `[completed]`

> **Task #2**: Build generic data pipeline with MovieLens-100K adapter

- [x] `deepfm/data/dataset.py` — TabularDataset, NegativeSamplingDataset, EvalRankingDataset
- [x] `deepfm/data/movielens.py` — MovieLensAdapter:
  - [x] Auto-download ML-100K via curl (SSL fallback)
  - [x] Parse native tab-separated files (u.data, u.user, u.item)
  - [x] Feature mapping: user_id, movie_id, gender, age, occupation, zip_prefix (SPARSE); genres (SEQUENCE, mean pooling)
  - [x] Binarize label: rating >= 4.0 → positive
  - [x] Leave-one-out split (last → test, second-to-last → val, rest → train, min 3 interactions)
  - [x] Dynamic negative sampling: 4 neg/pos (train), 999 neg/pos (eval), per-user filtered
  - [x] Full item features for negatives (genres, etc.)
  - [x] Encoders fit on train split only
  - [x] Verified: 943 users in val/test, 1000 candidates each

### Phase 3: Model Layers `[completed]`

> **Task #3**: Implement model layers

- [x] `deepfm/models/__init__.py` — MODEL_REGISTRY + build_model() factory
- [x] `deepfm/models/layers/__init__.py`
- [x] `deepfm/models/layers/embedding.py` — FeatureEmbedding with per-field dims + projection to fm_embed_dim
- [x] `deepfm/models/layers/fm.py` — FMInteraction: verified efficient vs naive max_diff=3.73e-08
- [x] `deepfm/models/layers/dnn.py` — DNN: Linear→BN→ReLU→Dropout, He init
- [x] `deepfm/models/layers/cin.py` — CIN: einsum outer product → Conv1d compression → sum pool, split_half
- [x] `deepfm/models/layers/attention.py` — FieldAttention: multi-head self-attention + residual + LayerNorm
- [x] All ops verified on CPU (MPS-compatible)

### Phase 4: Models `[completed]`

> **Task #4**: Implement DeepFM, xDeepFM, and AttentionDeepFM

- [x] `deepfm/models/base.py` — BaseCTRModel (ABC): FeatureEmbedding, forward(), predict(), get_l2_reg_loss()
- [x] `deepfm/models/deepfm.py` — DeepFM: y = y_FM + y_DNN (shared embeddings)
- [x] `deepfm/models/xdeepfm.py` — xDeepFM: y = y_linear + y_CIN + y_DNN
- [x] `deepfm/models/attention_deepfm.py` — AttentionDeepFM: y = y_FM + DNN(Attention(field_emb))
- [x] `deepfm/models/__init__.py` — MODEL_REGISTRY + build_model() factory
- [x] Verified: DeepFM(108K), xDeepFM(172K), AttentionDeepFM(124K) params, gradient flow OK

### Phase 5: Training & CLI `[completed]`

> **Task #5**: Implement training loop, metrics, and CLI

- [x] `deepfm/training/__init__.py`
- [x] `deepfm/training/metrics.py` — AUC, LogLoss (sklearn) + HR@K, NDCG@K for K=5,10,20
- [x] `deepfm/training/trainer.py` — BCEWithLogitsLoss, Adam/AdamW/SGD, ReduceLROnPlateau/Cosine, gradient clipping, early stopping, checkpointing, dynamic neg re-sampling per epoch
- [x] `deepfm/cli.py` — argparse: train/evaluate subcommands, --config, --override
- [x] `deepfm/__main__.py` — entry point for `python -m deepfm`
- [x] `configs/deepfm_movielens.yaml`
- [x] `configs/xdeepfm_movielens.yaml`
- [x] `configs/attention_deepfm_movielens.yaml`
- [x] 1-epoch smoke test on MPS: AUC=0.762, HR@10=0.375, NDCG@10=0.198

### Phase 6: Tests & Docs `[completed]`

> **Task #6**: Write tests and documentation

- [x] `tests/__init__.py`
- [x] `tests/test_schema.py` — schema construction, field filtering, empty schema
- [x] `tests/test_transforms.py` — LabelEncoder OOV, MinMaxScaler range, MultiHotEncoder padding/truncation
- [x] `tests/test_dataset.py` — TabularDataset shapes, NegativeSamplingDataset negatives, EvalRankingDataset candidates
- [x] `tests/test_layers.py` — FM efficient vs naive (atol=1e-5), DNN/CIN/Attention shapes, gradient flow, padding_idx=0
- [x] `tests/test_models.py` — DeepFM/xDeepFM/AttentionDeepFM forward, predict range, gradient flow, registry
- [x] `tests/test_trainer.py` — MetricCalculator, ranking metrics (HR/NDCG), 1-epoch smoke test
- [x] `tests/test_integration.py` — end-to-end on real ML-100K (marked @slow)
- [x] `README.md` — overview, quickstart (uv), usage, project structure, design decisions
- [x] 75/75 unit tests passing in 6.8s

---

## Decision Log

| Decision             | Choice                                        | Rationale                                        |
| -------------------- | --------------------------------------------- | ------------------------------------------------ |
| Dataset              | MovieLens-100K                                | Fast iteration, small enough for M2              |
| Label threshold      | rating >= 4.0                                 | Standard in literature, ~55% positive rate       |
| Age encoding         | SPARSE (7 buckets, embed_dim=4)               | Learns non-linear age effects                    |
| Embedding dims       | Per-field custom                              | Parameter-efficient, projection for FM alignment |
| Sequence pooling     | Mean                                          | Normalizes by length                             |
| DNN activation       | ReLU                                          | Standard, used in original paper                 |
| Split strategy       | Leave-one-out                                 | Standard RecSys protocol (NCF, DeepFM papers)    |
| Neg sampling (train) | 4 neg/pos, dynamic per epoch                  | Better generalization                            |
| Neg sampling (eval)  | 999 neg/pos, per-user filtered                | Reliable ranking metrics                         |
| Eval metrics         | AUC, LogLoss, HR@K, NDCG@K (K=5,10,20)        | Classification + ranking quality                 |
| Device               | MPS → CPU (no CUDA)                           | M2 MacBook target                                |
| Mixed precision      | Disabled                                      | Limited MPS support                              |
| Experiment tracking  | Python logging only                           | Lightweight, no external deps                    |
| Package manager      | uv                                            | Fast, modern                                     |
| Linting              | ruff only                                     | No strict type checking                          |
| Build tool           | Makefile                                      | Convenient, self-documenting                     |
| Data download        | Auto-download in adapter                      | Seamless first-run                               |
| Tests                | Unit (synthetic) + integration (real ML-100K) | Fast CI + end-to-end validation                  |
