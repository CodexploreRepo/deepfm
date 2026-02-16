# DeepFM: Production-Grade CTR Prediction Framework

## Overview

A PyTorch implementation of DeepFM and variants (xDeepFM, AttentionDeepFM) for Click-Through Rate prediction. Features a schema-driven generic data pipeline that works with any tabular dataset, starting with MovieLens-100K.

## Goals

- Clean, modular, production-grade codebase with clear separation of concerns
- Generic data pipeline: any dataset plugs in via a schema adapter (no hard-coded features)
- Reference-quality model implementations faithful to the original papers
- Configurable training loop with industry-standard practices (early stopping, LR scheduling)
- Model variants for architectural comparison

## Target Platform

- **Hardware**: MacBook with Apple M2 chip
- **Accelerator**: MPS (Metal Performance Shaders) backend via `torch.device("mps")`
- Device resolution order: MPS → CPU (no CUDA)
- All tensor operations and model layers must be MPS-compatible (avoid ops not yet supported on MPS backend)
- No mixed precision (limited MPS support)

## Non-Goals

- Model serving / ONNX export (future work)
- Distributed training
- Real-time feature engineering / online learning

---

## Architecture

### Project Structure

```
deepfm/
├── pyproject.toml
├── Makefile
├── README.md
├── CLAUDE.md
├── prd.md
├── configs/
│   ├── deepfm_movielens.yaml
│   ├── xdeepfm_movielens.yaml
│   └── attention_deepfm_movielens.yaml
├── deepfm/
│   ├── __init__.py
│   ├── cli.py                          # CLI entry point: train / evaluate
│   ├── config.py                       # Dataclass-based config + YAML loader
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schema.py                   # FieldSchema, DatasetSchema
│   │   ├── dataset.py                  # TabularDataset (generic torch Dataset)
│   │   ├── transforms.py              # LabelEncoder, MinMaxScaler, MultiHotEncoder
│   │   └── movielens.py               # MovieLens-100K adapter (auto-downloads)
│   ├── models/
│   │   ├── __init__.py                 # Model registry + factory
│   │   ├── base.py                     # BaseCTRModel (abstract base)
│   │   ├── deepfm.py                   # DeepFM = FM + DNN
│   │   ├── xdeepfm.py                 # xDeepFM = Linear + CIN + DNN
│   │   ├── attention_deepfm.py        # AttentionDeepFM = FM + Attention + DNN
│   │   └── layers/
│   │       ├── __init__.py
│   │       ├── embedding.py            # FeatureEmbedding (sparse/dense/sequence)
│   │       ├── fm.py                   # FM second-order interaction
│   │       ├── dnn.py                  # MLP with BatchNorm + Dropout
│   │       ├── cin.py                  # Compressed Interaction Network
│   │       └── attention.py            # Multi-head field self-attention
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                  # Training loop + early stopping + checkpointing
│   │   └── metrics.py                  # AUC, LogLoss, HR@K, NDCG@K
│   └── utils/
│       ├── __init__.py
│       ├── seeding.py
│       ├── logging.py
│       └── io.py                       # Checkpoint save/load
└── tests/
    ├── __init__.py
    ├── test_schema.py
    ├── test_dataset.py
    ├── test_layers.py
    ├── test_models.py
    ├── test_trainer.py
    └── test_integration.py             # End-to-end test on real ML-100K
```

---

## Data Pipeline Design

### Schema System (Generic Contract)

The core abstraction is `FieldSchema` — a declarative description of a single input feature. Every module in the system (embedding layer, dataset, model) is constructed from the schema. Zero hard-coded feature names anywhere.

```
FieldSchema:
  name: str                    # unique identifier ("user_id", "age", "genres")
  feature_type: SPARSE|DENSE|SEQUENCE
  vocabulary_size: int         # for SPARSE/SEQUENCE, set after fitting encoder
  embedding_dim: int           # learned embedding dimension (per-field custom dims)
  group: str                   # semantic grouping ("user", "item", "context")
  max_length: int              # for SEQUENCE features, padded length
  combiner: str                # for SEQUENCE: "mean", "sum", "max" pooling

DatasetSchema:
  fields: Dict[str, FieldSchema]
  label_field: str
  Properties: sparse_fields, dense_fields, sequence_fields, total_embedding_dim
```

### Embedding Dimension Strategy

Per-field custom embedding dimensions. Each field gets its own `embedding_dim` set in the adapter:

- High-cardinality fields (user_id, movie_id): larger dims (e.g., 16)
- Low-cardinality fields (gender, age): smaller dims (e.g., 4)
- Sequence fields (genres): moderate dims (e.g., 8)

Since FM second-order interaction requires same-dimension vectors for Hadamard products, a **projection layer** aligns all field embeddings to a common `fm_embedding_dim` before the FM component. The DNN receives the raw (non-projected) concatenated embeddings.

### MovieLens-100K Feature Mapping

Dataset: MovieLens-100K (~100K ratings, 943 users, 1682 movies)
File format: Tab-separated files (u.data, u.user, u.item) — parsed natively.
Auto-download: Adapter downloads and extracts from grouplens.org if `data_dir` doesn't exist.

| Feature    | Type     | Details                                                    | Embedding Dim |
| ---------- | -------- | ---------------------------------------------------------- | ------------- |
| user_id    | SPARSE   | 943 users, LabelEncoded                                    | 16            |
| movie_id   | SPARSE   | 1682 movies, LabelEncoded                                  | 16            |
| gender     | SPARSE   | 2 values (M/F)                                             | 4             |
| age        | SPARSE   | 7 buckets (1,18,25,35,45,50,56), LabelEncoded              | 4             |
| occupation | SPARSE   | 21 values                                                  | 8             |
| zip_prefix | SPARSE   | 3-digit zip prefixes                                       | 8             |
| genres     | SEQUENCE | 19 genres, multi-hot, padded to max_length=6, mean pooling | 8             |
| label      | -        | rating >= 4.0 → 1, else 0                                  | -             |

### Split Strategy: Leave-One-Out

Per user, ordered by timestamp:

- **Last interaction** → test set (1 sample per user)
- **Second-to-last interaction** → validation set (1 sample per user)
- **All remaining interactions** → training set
- Users with fewer than 3 interactions → training only (no val/test entry)
- Encoders fit on training set only. OOV maps to index 0.

### Negative Sampling

**Training**: Dynamic negative sampling, re-sampled each epoch.

- For each positive interaction, sample **4 negative items** (items the user has NOT interacted with)
- Full features looked up for negatives (user features + all item features including genres)
- Negatives get label=0, positives get label=1

**Evaluation**: Sampled ranking protocol.

- For each user in val/test, rank the 1 held-out positive item against **999 randomly sampled negative items** (per-user filtered: exclude all items the user has interacted with)
- Compute ranking metrics on this list of 1000 items

### Adding a New Dataset

Write an adapter class that:

1. Reads raw data files (auto-download optional)
2. Defines `FieldSchema` for each feature with custom embedding dims
3. Fits encoders on train split
4. Implements leave-one-out split + negative sampling
5. Returns `(DatasetSchema, TabularDataset)` for train/val/test

---

## Model Architecture

### Shared: FeatureEmbedding Layer

All models share a single `FeatureEmbedding` layer that produces three tensor views from one forward pass:

| Output             | Shape                         | Consumer                       |
| ------------------ | ----------------------------- | ------------------------------ |
| `first_order`      | (B, 1)                        | FM linear term                 |
| `field_embeddings` | (B, num_fields, fm_embed_dim) | FM interaction, CIN, Attention |
| `flat_embeddings`  | (B, total_dim)                | DNN input                      |

- Sparse features: `nn.Embedding(vocab_size, field_embed_dim)` + first-order `nn.Embedding(vocab_size, 1)`
- Per-field embeddings projected to common `fm_embed_dim` via `nn.Linear` for FM/CIN/Attention
- Dense features: optional `nn.Linear(1, embed_dim)` projection, or raw pass-through
- Sequence features: `nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")`
- `padding_idx=0` for OOV (zero contribution)
- Xavier uniform initialization

### DeepFM (Guo et al., 2017)

```
y = sigmoid(y_FM + y_DNN)

y_FM  = bias + <w, x> + 0.5 * (||sum(v_i)||^2 - sum(||v_i||^2))
y_DNN = W_out * DNN(concat(all_field_embeddings))
```

Key: FM and DNN **share the same embedding vectors**. No separate feature engineering for the wide component (unlike Wide&Deep).

FM second-order uses the efficient O(n\*d) identity:

```
sum_{i<j} <v_i, v_j> = 0.5 * (square_of_sum - sum_of_squares)
```

### xDeepFM (Lian et al., 2018)

```
y = sigmoid(y_linear + y_CIN + y_DNN)
```

Replaces FM's 2nd-order interaction with CIN (Compressed Interaction Network):

- CIN captures **arbitrary-order explicit interactions** (layer k = k+1 order)
- Operates at **vector-wise** level (preserves embedding structure)
- Implementation: outer product via einsum → compress via Conv1d(kernel=1) → sum pool
- `split_half` option: first half feeds next layer, second half goes to output

### AttentionDeepFM (inspired by AutoInt, Song et al. 2019)

```
y = sigmoid(y_FM + y_DNN_attn)

y_DNN_attn = W_out * DNN(flatten(MultiHeadAttention(field_embeddings)))
```

- Field embeddings pass through N stacked multi-head self-attention layers
- Attention learns **which field interactions matter** (weighted, selective)
- Residual connections + LayerNorm for training stability
- DNN operates on attention-refined representations

### BaseCTRModel (Abstract Base)

All models inherit from `BaseCTRModel` and implement:

- `_build_components()` — initialize model-specific layers
- `_forward_components(first_order, field_embeddings, flat_embeddings) → logit`

Base provides: embedding layer, `forward()`, `predict()` (with sigmoid), `get_l2_reg_loss()`.

---

## Config System

Nested Python dataclasses with YAML override via `dacite`:

```
ExperimentConfig
├── model_name: str                    # "deepfm" | "xdeepfm" | "attention_deepfm"
├── seed: int
├── device: str                        # "auto" (MPS → CPU) | "cpu" | "mps"
├── output_dir: str
├── DataConfig
│   ├── dataset_name, data_dir
│   ├── split_strategy: "leave_one_out"
│   ├── min_interactions: 3 (users with fewer are train-only)
│   ├── label_threshold: 4.0
│   ├── num_neg_train: 4              # negatives per positive during training
│   ├── num_neg_eval: 999             # negatives per positive during eval
│   └── auto_download: true
├── FeatureConfig
│   └── embedding_l2_reg: 1e-5
├── FMConfig
│   ├── use_first_order: true
│   └── use_second_order: true
├── DNNConfig
│   ├── hidden_units: [256, 128, 64]
│   ├── activation: "relu"
│   ├── dropout: 0.1
│   └── use_batch_norm: true
├── CINConfig
│   ├── layer_sizes: [128, 128]
│   └── split_half: true
├── AttentionConfig
│   ├── num_heads: 4, attention_dim: 64
│   ├── num_layers: 1
│   └── use_residual: true
└── TrainingConfig
    ├── batch_size: 4096, lr: 1e-3
    ├── optimizer: "adam", scheduler: "reduce_on_plateau"
    ├── early_stopping_patience: 5, metric: "auc"
    └── gradient_clip_norm: 1.0
```

---

## Training Loop

- **Loss**: `BCEWithLogitsLoss` (model outputs raw logits, numerically stable)
- **Optimizers**: Adam, AdamW, SGD (configurable)
- **LR Scheduling**: ReduceLROnPlateau or CosineAnnealing
- **Early Stopping**: monitors validation AUC (or logloss), patience-based
- **Gradient Clipping**: max norm (default 1.0)
- **No mixed precision** (MPS compatibility)
- **Checkpointing**: saves best model by validation metric
- **Negative sampling**: re-sampled dynamically each epoch during training

---

## Evaluation

### Metrics

**Classification metrics** (computed on all val/test samples):

- AUC (via sklearn `roc_auc_score`)
- LogLoss (via sklearn `log_loss`)

**Ranking metrics** (leave-one-out with 999 sampled negatives, per-user filtered):

- HR@K (Hit Rate at K) for K = 5, 10, 20
- NDCG@K (Normalized Discounted Cumulative Gain at K) for K = 5, 10, 20

### Evaluation Protocol

For each user in val/test:

1. Take the 1 held-out positive item
2. Sample 999 items the user has NOT interacted with
3. Score all 1000 items with the model
4. Rank by predicted score (descending)
5. Check if positive item appears in top K → HR@K
6. Compute NDCG@K based on position of positive item

---

## Tooling

- **Package manager**: uv
- **Linting/formatting**: ruff (no strict type checking)
- **Build tool**: Makefile with targets: `train`, `test`, `lint`, `download-data`, `install`
- **Experiment tracking**: Python logging (stdout + file), no external tracking service
- **Tests**: Unit tests with synthetic data + integration test on real ML-100K

---

## Implementation Order

| Phase             | Files                                                                              | Dependencies |
| ----------------- | ---------------------------------------------------------------------------------- | ------------ |
| 1. Foundation     | pyproject.toml, Makefile, CLAUDE.md, config.py, schema.py, transforms.py, utils/\* | None         |
| 2. Data Pipeline  | dataset.py, movielens.py (with auto-download + neg sampling)                       | Phase 1      |
| 3. Model Layers   | embedding.py, fm.py, dnn.py, cin.py, attention.py                                  | Phase 1      |
| 4. Models         | base.py, deepfm.py, xdeepfm.py, attention_deepfm.py, registry                      | Phase 2, 3   |
| 5. Training & CLI | metrics.py (AUC, LogLoss, HR@K, NDCG@K), trainer.py, cli.py, YAML configs          | Phase 4      |
| 6. Tests & Docs   | test\_\*.py, test_integration.py, README.md                                        | Phase 5      |

---

## Expected Results (MovieLens-100K)

| Model           | AUC        | HR@10      | NDCG@10    |
| --------------- | ---------- | ---------- | ---------- |
| DeepFM          | ~0.78-0.80 | ~0.62-0.68 | ~0.36-0.42 |
| xDeepFM         | ~0.79-0.81 | ~0.64-0.70 | ~0.38-0.44 |
| AttentionDeepFM | ~0.78-0.81 | ~0.63-0.69 | ~0.37-0.43 |

---

## Dependencies

```
torch>=2.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
pyyaml>=6.0
dacite>=1.8
```

Dev: `pytest>=7.0, pytest-cov, ruff`

---

## Usage

```bash
# Install (using uv)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Train DeepFM (auto-downloads ML-100K on first run)
python -m deepfm train --config configs/deepfm_movielens.yaml

# Train xDeepFM with batch size override
python -m deepfm train --config configs/xdeepfm_movielens.yaml --override training.batch_size=2048

# Evaluate
python -m deepfm evaluate --config configs/deepfm_movielens.yaml --checkpoint outputs/best_model.pt

# Run tests
make test

# Lint
make lint
```
