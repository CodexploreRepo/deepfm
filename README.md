# DeepFM

Production-grade DeepFM and variants for Click-Through Rate (CTR) prediction, built with PyTorch.

## Models

| Model               | Paper                                   | Key Idea                                     |
| ------------------- | --------------------------------------- | -------------------------------------------- |
| **DeepFM**          | Guo et al., 2017                        | FM second-order + DNN with shared embeddings |
| **xDeepFM**         | Lian et al., 2018                       | Compressed Interaction Network (CIN) + DNN   |
| **AttentionDeepFM** | Inspired by AutoInt (Song et al., 2019) | Multi-head self-attention + FM + DNN         |

## Features

- Schema-driven data pipeline — zero hard-coded feature names
- Per-field custom embedding dimensions with projection to common FM dim
- Leave-one-out evaluation with HR@K, NDCG@K ranking metrics
- Dynamic negative sampling (re-sampled each epoch)
- Early stopping, LR scheduling, gradient clipping
- Apple MPS support (M1/M2/M3)

## Quickstart

```bash
# Install with uv
python3 -m uv pip install -e ".[dev]"

# Train DeepFM on MovieLens-100K (auto-downloads data)
python -m deepfm train --config configs/deepfm_movielens.yaml

# Train xDeepFM
python -m deepfm train --config configs/xdeepfm_movielens.yaml

# Train AttentionDeepFM
python -m deepfm train --config configs/attention_deepfm_movielens.yaml

# Evaluate a saved checkpoint
python -m deepfm evaluate --config configs/deepfm_movielens.yaml \
    --checkpoint outputs/deepfm_movielens/best_model.pt

# Override config values from CLI
python -m deepfm train --config configs/deepfm_movielens.yaml \
    --override training.batch_size=2048 training.learning_rate=5e-4
```

## Running Tests

```bash
# Unit tests (fast, no data download)
python3 -m pytest tests/ -m "not slow" -v

# All tests including integration (downloads ML-100K)
python3 -m pytest tests/ -v

# With coverage
python3 -m pytest tests/ -m "not slow" --cov=deepfm --cov-report=term-missing
```

## Using the Makefile

```bash
make install      # Install package in editable mode
make train        # Train default DeepFM config
make test         # Run unit tests
make lint         # Lint with ruff
make format       # Auto-format with ruff
make clean        # Remove outputs and caches
```

## Project Structure

```
deepfm/
├── configs/                    # YAML experiment configs
│   ├── deepfm_movielens.yaml
│   ├── xdeepfm_movielens.yaml
│   └── attention_deepfm_movielens.yaml
├── deepfm/
│   ├── cli.py                  # CLI: train / evaluate
│   ├── config.py               # Dataclass configs + YAML loader
│   ├── data/
│   │   ├── schema.py           # FieldSchema, DatasetSchema
│   │   ├── dataset.py          # TabularDataset, NegativeSamplingDataset, EvalRankingDataset
│   │   ├── transforms.py       # LabelEncoder, MinMaxScaler, MultiHotEncoder
│   │   └── movielens.py        # MovieLens-100K adapter
│   ├── models/
│   │   ├── base.py             # BaseCTRModel (abstract)
│   │   ├── deepfm.py           # DeepFM
│   │   ├── xdeepfm.py          # xDeepFM
│   │   ├── attention_deepfm.py # AttentionDeepFM
│   │   └── layers/
│   │       ├── embedding.py    # FeatureEmbedding (sparse/dense/sequence)
│   │       ├── fm.py           # FMInteraction (efficient O(n*d))
│   │       ├── dnn.py          # MLP with BN + dropout
│   │       ├── cin.py          # Compressed Interaction Network
│   │       └── attention.py    # Multi-head field self-attention
│   ├── training/
│   │   ├── trainer.py          # Training loop + early stopping
│   │   └── metrics.py          # AUC, LogLoss, HR@K, NDCG@K
│   └── utils/
│       ├── seeding.py          # seed_everything()
│       ├── logging.py          # Logging setup
│       └── io.py               # Checkpoint save/load
└── tests/
    ├── test_schema.py          # Schema construction, field filtering
    ├── test_transforms.py      # Encoder and scaler correctness
    ├── test_dataset.py         # Dataset shapes and dtypes
    ├── test_layers.py          # FM math, DNN/CIN/Attention shapes
    ├── test_models.py          # Forward pass, gradient flow
    ├── test_trainer.py         # Metrics + 1-epoch smoke test
    └── test_integration.py     # End-to-end on real ML-100K
```

## Configuration

All hyperparameters are in YAML config files. Key sections:

- `data` — dataset, split strategy, negative sampling
- `feature` — embedding L2 reg, FM embedding dim
- `fm` — first/second order toggles
- `dnn` — hidden units, activation, dropout, batch norm
- `cin` — layer sizes, split half (xDeepFM only)
- `attention` — heads, dim, layers (AttentionDeepFM only)
- `training` — batch size, LR, optimizer, scheduler, early stopping

## Design Decisions

| Decision             | Choice             | Rationale                                        |
| -------------------- | ------------------ | ------------------------------------------------ |
| Label threshold      | rating >= 4.0      | ~55% positive rate, standard in literature       |
| Split strategy       | Leave-one-out      | Standard RecSys protocol (NCF, DeepFM papers)    |
| Neg sampling (train) | 4 neg/pos, dynamic | Better generalization via re-sampling            |
| Neg sampling (eval)  | 999 neg/pos        | Reliable ranking metrics                         |
| Embedding dims       | Per-field custom   | Parameter-efficient, projection for FM alignment |
| Loss                 | BCEWithLogitsLoss  | Numerically stable, raw logits from model        |
| Device               | MPS > CPU          | Targets Apple Silicon, no CUDA                   |
