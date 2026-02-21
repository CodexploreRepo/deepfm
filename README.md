# DeepFM

Production-grade DeepFM and variants for Click-Through Rate (CTR) prediction in PyTorch, targeting Apple Silicon (MPS backend).

## Models

| Model               | Components           | Key Idea                                                 |
| ------------------- | -------------------- | -------------------------------------------------------- |
| **DeepFM**          | FM + DNN             | 2nd-order explicit (FM) + implicit higher-order (DNN)    |
| **xDeepFM**         | CIN + DNN            | Explicit vector-wise higher-order (CIN) + implicit (DNN) |
| **AttentionDeepFM** | FM + Attention + DNN | Learned field interaction importance via self-attention  |

All models share a `FeatureEmbedding` layer that produces three views: first-order weights, projected field embeddings, and flat concatenated embeddings.

## Installation

```bash
make install
```

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv).

## Quick Start

### Train DeepFM on MovieLens-100K

```bash
make train
```

Or with custom config:

```bash
python -m deepfm train --config configs/deepfm_movielens.yaml
```

Override parameters:

```bash
python -m deepfm train --config configs/deepfm_movielens.yaml \
    --override training.num_epochs=10 training.batch_size=2048
```

### Train other variants

```bash
python -m deepfm train --config configs/xdeepfm_movielens.yaml
python -m deepfm train --config configs/attention_deepfm_movielens.yaml
```

### Evaluate a saved checkpoint

```bash
python -m deepfm evaluate --config configs/deepfm_movielens.yaml
```

## Project Structure

```
deepfm/
  config.py                     # Dataclass configs, YAML loading
  cli.py                        # CLI entry point
  data/
    schema.py                   # FieldSchema, DatasetSchema, FeatureType
    transforms.py               # LabelEncoder, MinMaxScaler, MultiHotEncoder
    dataset.py                  # TabularDataset (torch Dataset)
    movielens.py                # MovieLensAdapter (ML-100K)
  models/
    base.py                     # BaseCTRModel (abstract)
    deepfm.py                   # DeepFM
    xdeepfm.py                  # xDeepFM
    attention_deepfm.py         # AttentionDeepFM
    layers/
      embedding.py              # FeatureEmbedding (shared)
      fm.py                     # FMInteraction
      dnn.py                    # DNN (MLP)
      cin.py                    # CIN (Compressed Interaction Network)
      attention.py              # MultiHeadSelfAttention
  training/
    metrics.py                  # AUC, LogLoss, HR@K, NDCG@K
    trainer.py                  # Training loop, early stopping
  utils/
    seeding.py                  # seed_everything
    logging.py                  # get_logger
    io.py                       # save/load checkpoint
configs/
  deepfm_movielens.yaml
  xdeepfm_movielens.yaml
  attention_deepfm_movielens.yaml
tests/
  test_schema.py
  test_transforms.py
  test_dataset.py
  test_layers.py
  test_models.py
  test_trainer.py
  test_integration.py           # @slow — requires real ML-100K data
```

## Configuration

Configs are Python dataclasses loaded from YAML via [dacite](https://github.com/konradhalas/dacite). Key sections:

- **data**: dataset name, paths, split strategy, negative sampling counts
- **feature**: `fm_embed_dim`, L2 regularization weight
- **dnn**: hidden units, activation, dropout, batch norm
- **cin**: layer sizes, split_half (xDeepFM only)
- **attention**: num heads, attention dim, layers (AttentionDeepFM only)
- **training**: epochs, batch size, LR, optimizer, early stopping

## Evaluation Protocol

- **Split**: Leave-one-out per user by timestamp
- **Negative sampling**: 4 neg/pos (train), 999 neg/pos (eval)
- **Classification metrics**: AUC, LogLoss
- **Ranking metrics**: HR@K, NDCG@K for K = 5, 10, 20
- **Loss**: BCEWithLogitsLoss (sigmoid only at predict time)

## Expected Results (ML-100K, 2 epochs)

| Metric  | DeepFM |
| ------- | ------ |
| AUC     | ~0.86  |
| HR@10   | ~0.15  |
| NDCG@10 | ~0.08  |

## Testing

```bash
make test          # unit tests (fast, synthetic data)
make lint          # ruff check + format check
make format        # auto-fix lint + format

# Include slow integration tests
pytest tests/ -v -m slow
```

## Design Decisions

- **MPS-first**: No CUDA, no mixed precision. Uses `resolve_device()` for MPS → CPU fallback.
- **Schema-driven**: All modules consume `DatasetSchema` — no hard-coded feature names.
- **Adapter pattern**: Each dataset implements an adapter (e.g., `MovieLensAdapter`) that produces schema + datasets.
- **OOV handling**: `padding_idx=0` in all embeddings; unknown features contribute zero.
- **Per-field embedding dims**: Each field has its own `embedding_dim` based on cardinality, projected to common `fm_embed_dim` for FM/CIN/Attention.
