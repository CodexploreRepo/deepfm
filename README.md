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

## Hyperparameter Tuning

Use `make train ARGS="key=value ..."` to override any config value without editing YAML.
Always set `output_dir` so each run gets its own checkpoint and `results.json`.

```bash
# Baseline (saves to outputs/deepfm_movielens/)
make train

# Different learning rate
make train ARGS="output_dir=outputs/deepfm_lr5e4 training.lr=5e-4"

# Larger embedding dim
make train ARGS="output_dir=outputs/deepfm_emb32 feature.fm_embed_dim=32"

# Different model variant
make train ARGS="output_dir=outputs/xdeepfm model_name=xdeepfm"

# Multiple overrides at once
make train ARGS="output_dir=outputs/exp1 training.lr=5e-4 dnn.dropout=0.2 feature.fm_embed_dim=32"
```

### Common override keys

| Key                                | Default                    | Notes                                   |
| ---------------------------------- | -------------------------- | --------------------------------------- |
| `output_dir`                       | `outputs/deepfm_movielens` | Set per-run to avoid overwriting        |
| `model_name`                       | `deepfm`                   | `deepfm`, `xdeepfm`, `attention_deepfm` |
| `training.lr`                      | `1e-3`                     | Learning rate                           |
| `training.batch_size`              | `4096`                     |                                         |
| `training.num_epochs`              | `50`                       |                                         |
| `training.optimizer`               | `adam`                     | `adam`, `adamw`, `sgd`                  |
| `training.early_stopping_patience` | `5`                        |                                         |
| `feature.fm_embed_dim`             | `16`                       | Projected embedding dimension           |
| `feature.embedding_l2_reg`         | `1e-5`                     | L2 penalty on embeddings                |
| `dnn.dropout`                      | `0.1`                      |                                         |
| `dnn.use_batch_norm`               | `true`                     |                                         |
| `attention.num_heads`              | `4`                        | AttentionDeepFM only                    |

For list-valued fields (`dnn.hidden_units`, `cin.layer_sizes`), create a separate YAML config and pass it with `--config`.

### Comparing runs

Each completed run writes `results.json` to its `output_dir`. Compare all runs at once:

```bash
make compare                        # scans outputs/ recursively
make compare RUNS_DIR=outputs/exp1  # single run or subdirectory
```

Example output:

```
----------------------------------------------------------------------------------------------------------------------------
Run                         Model         LR·BS·Emb           Val AUC  Val LogL   Tst AUC  Tst LogL     HR@10   NDCG@10     BstEp
----------------------------------------------------------------------------------------------------------------------------
deepfm_movielens            deepfm        0.001·4096·16        0.8712    0.4231    0.8634    0.4312    0.1523    0.0812         8
deepfm_lr5e4                deepfm        0.0005·4096·16       0.8798    0.4102    0.8721    0.4189    0.1601    0.0867        12
deepfm_emb32                deepfm        0.001·4096·32        0.8841    0.4053    0.8763    0.4141    0.1648    0.0891        10
----------------------------------------------------------------------------------------------------------------------------
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
    io.py                       # save/load checkpoint, save_results
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
