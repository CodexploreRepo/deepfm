# DeepFM Implementation Todo

Each step has a clear deliverable and verification command. Complete in order.

---

## Phase 1: Foundation

### Step 1.1 — Package scaffolding & Makefile

**Files:** `Makefile`, `deepfm/__init__.py`, `deepfm/data/__init__.py`, `deepfm/models/__init__.py`, `deepfm/models/layers/__init__.py`, `deepfm/training/__init__.py`, `deepfm/utils/__init__.py`, `tests/__init__.py`
**Do:** Create all `__init__.py` (empty or minimal version string) and a Makefile with `install`, `train`, `test`, `lint`, `format` targets.
**Verify:** `make install && python -c "import deepfm; print('ok')"`

### Step 1.2 — Utils: seeding, logging, io

**Files:** `deepfm/utils/seeding.py`, `deepfm/utils/logging.py`, `deepfm/utils/io.py`
**Do:**

- `seeding.py`: `seed_everything(seed)` — sets `random`, `numpy`, `torch` seeds.
- `logging.py`: `get_logger(name)` — returns a configured Python logger (stdout + optional file handler).
- `io.py`: `save_checkpoint(state, path)`, `load_checkpoint(path, device)` — thin wrappers around `torch.save`/`torch.load`.
  **Verify:** `python -c "from deepfm.utils import seed_everything, get_logger, save_checkpoint, load_checkpoint; print('ok')"`

### Step 1.3 — Config dataclasses

**Files:** `deepfm/config.py`
**Do:** Define `DataConfig`, `FeatureConfig`, `FMConfig`, `DNNConfig`, `CINConfig`, `AttentionConfig`, `TrainingConfig`, `ExperimentConfig` as frozen dataclasses with defaults from the PRD. Add `load_config(yaml_path, overrides) -> ExperimentConfig` using dacite.
**Verify:** `python -c "from deepfm.config import ExperimentConfig; c = ExperimentConfig(); print(c.model_name, c.training.batch_size)"`

### Step 1.4 — Schema: FieldSchema & DatasetSchema

**Files:** `deepfm/data/schema.py`
**Do:** `FieldSchema` dataclass (name, feature_type enum SPARSE/DENSE/SEQUENCE, vocabulary_size, embedding_dim, group, max_length, combiner). `DatasetSchema` with fields dict, label_field, and properties: `sparse_fields`, `dense_fields`, `sequence_fields`, `num_fields`, `total_embedding_dim`.
**Verify:** `python -c "from deepfm.data.schema import FieldSchema, DatasetSchema, FeatureType; f = FieldSchema(name='test', feature_type=FeatureType.SPARSE, vocabulary_size=10, embedding_dim=8); print(f)"`

### Step 1.5 — Transforms: LabelEncoder, MinMaxScaler, MultiHotEncoder

**Files:** `deepfm/data/transforms.py`
**Do:**

- `LabelEncoder`: fit/transform with OOV→0 (reserves index 0 for unknown). `vocabulary_size` property = num_classes + 1.
- `MinMaxScaler`: fit/transform to [0,1].
- `MultiHotEncoder`: fit on lists of tokens, transform to padded integer sequences. `vocabulary_size` property = num_tokens + 1.
  **Verify:** `python -c "from deepfm.data.transforms import LabelEncoder, MultiHotEncoder; le = LabelEncoder(); le.fit(['a','b','c']); print(le.transform(['b','a','z'])); print(le.vocabulary_size)"`

---

## Phase 2: Data Pipeline

### Step 2.1 — TabularDataset

**Files:** `deepfm/data/dataset.py`
**Do:** `TabularDataset(torch.utils.data.Dataset)` — takes a dict of feature arrays (numpy) + labels. `__getitem__` returns `(feature_dict_tensors, label_tensor)`. Supports collation for DataLoader.
**Verify:** `python -c "
import numpy as np
from deepfm.data.dataset import TabularDataset
ds = TabularDataset({'f1': np.array([1,2,3])}, np.array([0,1,0]))
print(ds[0])
print(len(ds))
"`

### Step 2.2 — MovieLens adapter: loading & feature engineering

**Files:** `deepfm/data/movielens.py`
**Do:** `MovieLensAdapter` class:

- `__init__(data_dir, config)`: reads u.data, u.user, u.item, parses and merges.
- Defines all `FieldSchema` per the PRD (user_id, movie_id, gender, age, occupation, zip_prefix, genres).
- Fits encoders on train split, transforms all splits.
- Implements leave-one-out split by timestamp.
- Returns `DatasetSchema` + train/val/test `TabularDataset`.
  **Verify:** `python -c "
from deepfm.config import ExperimentConfig
from deepfm.data.movielens import MovieLensAdapter
adapter = MovieLensAdapter('data/ml-100k', ExperimentConfig().data)
schema, train_ds, val_ds, test_ds = adapter.build()
print(f'Schema fields: {list(schema.fields.keys())}')
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')
"`

### Step 2.3 — Negative sampling

**Files:** `deepfm/data/movielens.py` (extend)
**Do:** Add negative sampling to `MovieLensAdapter`:

- Training: for each positive, sample `num_neg_train` (4) negative items with full features, label=0. Re-sample each call.
- Evaluation: for each user, sample `num_neg_eval` (999) negatives filtered to exclude all user interactions.
- Return datasets include negatives.
  **Verify:** `python -c "
  from deepfm.config import ExperimentConfig
  from deepfm.data.movielens import MovieLensAdapter
  adapter = MovieLensAdapter('data/ml-100k', ExperimentConfig().data)
  schema, train_ds, val_ds, test_ds = adapter.build()
  print(f'Train (with neg): {len(train_ds)}')
  print(f'Val (with neg): {len(val_ds)}')

# Train should be ~5x positive count, val should be ~1000 per user

"`

---

## Phase 3: Model Layers

### Step 3.1 — FeatureEmbedding layer

**Files:** `deepfm/models/layers/embedding.py`
**Do:** `FeatureEmbedding(nn.Module)`:

- Takes `DatasetSchema` + `fm_embed_dim`.
- Creates per-field: second-order `nn.Embedding(vocab, field_dim)`, first-order `nn.Embedding(vocab, 1)`, projection `nn.Linear(field_dim, fm_embed_dim)`.
- Sequence fields: `nn.EmbeddingBag(vocab, dim, mode="mean")`.
- `padding_idx=0`, Xavier init.
- Forward returns: `first_order (B,1)`, `field_embeddings (B,F,fm_embed_dim)`, `flat_embeddings (B,total_dim)`.
  **Verify:** `python -c "
import torch
from deepfm.data.schema import FieldSchema, DatasetSchema, FeatureType
from deepfm.models.layers.embedding import FeatureEmbedding
fields = {
    'u': FieldSchema('u', FeatureType.SPARSE, 100, 8),
    'i': FieldSchema('i', FeatureType.SPARSE, 200, 16),
}
schema = DatasetSchema(fields=fields, label_field='label')
emb = FeatureEmbedding(schema, fm_embed_dim=16)
batch = {'u': torch.tensor([1,2]), 'i': torch.tensor([5,10])}
fo, fe, fl = emb(batch)
print(f'first_order: {fo.shape}, field_emb: {fe.shape}, flat_emb: {fl.shape}')
"`

### Step 3.2 — FM interaction layer

**Files:** `deepfm/models/layers/fm.py`
**Do:** `FMInteraction(nn.Module)`:

- Takes `field_embeddings (B, F, D)`.
- Computes 2nd-order: `0.5 * (square_of_sum - sum_of_squares)` → sum to scalar `(B, 1)`.
- Efficient O(n\*d), no explicit pairwise.
  **Verify:** `python -c "
import torch
from deepfm.models.layers.fm import FMInteraction
fm = FMInteraction()
x = torch.randn(4, 6, 16)
out = fm(x)
print(f'FM output: {out.shape}')  # (4, 1)
"`

### Step 3.3 — DNN layer

**Files:** `deepfm/models/layers/dnn.py`
**Do:** `DNN(nn.Module)`:

- `__init__(input_dim, hidden_units, activation, dropout, use_batch_norm)`.
- Stack of Linear → (BatchNorm) → Activation → Dropout.
- Output dim = last hidden unit.
  **Verify:** `python -c "
import torch
from deepfm.models.layers.dnn import DNN
dnn = DNN(64, [128, 64, 32], dropout=0.1, use_batch_norm=True)
x = torch.randn(4, 64)
print(f'DNN output: {dnn(x).shape}')  # (4, 32)
"`

### Step 3.4 — CIN layer (Compressed Interaction Network)

**Files:** `deepfm/models/layers/cin.py`
**Do:** `CIN(nn.Module)`:

- `__init__(num_fields, embed_dim, layer_sizes, split_half)`.
- Each layer: outer product via einsum → compress with Conv1d(kernel=1) → optional split_half.
- Output: sum-pooled from each layer, concatenated → `(B, output_dim)`.
  **Verify:** `python -c "
import torch
from deepfm.models.layers.cin import CIN
cin = CIN(num_fields=6, embed_dim=16, layer_sizes=[128, 128], split_half=True)
x = torch.randn(4, 6, 16)
out = cin(x)
print(f'CIN output: {out.shape}')  # (4, some_dim)
"`

### Step 3.5 — Multi-head self-attention layer

**Files:** `deepfm/models/layers/attention.py`
**Do:** `MultiHeadSelfAttention(nn.Module)`:

- `__init__(embed_dim, num_heads, attention_dim, num_layers, use_residual)`.
- Standard Q/K/V projections, scaled dot-product attention, residual + LayerNorm.
- Stacked N layers.
- Input: `(B, F, D)` → Output: `(B, F, D)`.
  **Verify:** `python -c "
import torch
from deepfm.models.layers.attention import MultiHeadSelfAttention
attn = MultiHeadSelfAttention(embed_dim=16, num_heads=4, attention_dim=64, num_layers=2, use_residual=True)
x = torch.randn(4, 6, 16)
out = attn(x)
print(f'Attention output: {out.shape}')  # (4, 6, 16)
"`

### Step 3.6 — Layers **init** re-exports

**Files:** `deepfm/models/layers/__init__.py`
**Do:** Re-export all layer classes from the layers package.
**Verify:** `python -c "from deepfm.models.layers import FeatureEmbedding, FMInteraction, DNN, CIN, MultiHeadSelfAttention; print('ok')"`

---

## Phase 4: Models

### Step 4.1 — BaseCTRModel

**Files:** `deepfm/models/base.py`
**Do:** `BaseCTRModel(nn.Module)`:

- `__init__(schema, config)`: builds `FeatureEmbedding`, calls `_build_components()`.
- `forward(batch) → logits (B,1)`: runs embedding, calls `_forward_components(fo, fe, fl)`.
- `predict(batch) → probabilities`: sigmoid on logits.
- `get_l2_reg_loss()`: L2 on embedding parameters.
- Abstract methods: `_build_components()`, `_forward_components()`.
  **Verify:** `python -c "from deepfm.models.base import BaseCTRModel; print('ok')"`

### Step 4.2 — DeepFM model

**Files:** `deepfm/models/deepfm.py`
**Do:** `DeepFM(BaseCTRModel)`:

- Components: FMInteraction + DNN + output linear.
- `_forward_components`: `logit = first_order + fm(field_emb) + linear(dnn(flat_emb))`.
  **Verify:** `python -c "
import torch
from deepfm.data.schema import FieldSchema, DatasetSchema, FeatureType
from deepfm.config import ExperimentConfig
from deepfm.models.deepfm import DeepFM
fields = {'u': FieldSchema('u', FeatureType.SPARSE, 100, 8), 'i': FieldSchema('i', FeatureType.SPARSE, 200, 16)}
schema = DatasetSchema(fields=fields, label_field='label')
model = DeepFM(schema, ExperimentConfig())
batch = {'u': torch.tensor([1,2]), 'i': torch.tensor([5,10])}
print(f'logits: {model(batch).shape}')  # (2, 1)
"`

### Step 4.3 — xDeepFM model

**Files:** `deepfm/models/xdeepfm.py`
**Do:** `xDeepFM(BaseCTRModel)`:

- Components: CIN + DNN + output linears.
- `_forward_components`: `logit = first_order + linear_cin(cin(field_emb)) + linear_dnn(dnn(flat_emb))`.
  **Verify:** `python -c "
import torch
from deepfm.data.schema import FieldSchema, DatasetSchema, FeatureType
from deepfm.config import ExperimentConfig
from deepfm.models.xdeepfm import xDeepFM
fields = {'u': FieldSchema('u', FeatureType.SPARSE, 100, 8), 'i': FieldSchema('i', FeatureType.SPARSE, 200, 16)}
schema = DatasetSchema(fields=fields, label_field='label')
model = xDeepFM(schema, ExperimentConfig())
batch = {'u': torch.tensor([1,2]), 'i': torch.tensor([5,10])}
print(f'logits: {model(batch).shape}')
"`

### Step 4.4 — AttentionDeepFM model

**Files:** `deepfm/models/attention_deepfm.py`
**Do:** `AttentionDeepFM(BaseCTRModel)`:

- Components: FMInteraction + MultiHeadSelfAttention + DNN + output linear.
- `_forward_components`: attention refines field_emb, flatten, concat with flat_emb for DNN.
  **Verify:** `python -c "
import torch
from deepfm.data.schema import FieldSchema, DatasetSchema, FeatureType
from deepfm.config import ExperimentConfig
from deepfm.models.attention_deepfm import AttentionDeepFM
fields = {'u': FieldSchema('u', FeatureType.SPARSE, 100, 8), 'i': FieldSchema('i', FeatureType.SPARSE, 200, 16)}
schema = DatasetSchema(fields=fields, label_field='label')
model = AttentionDeepFM(schema, ExperimentConfig())
batch = {'u': torch.tensor([1,2]), 'i': torch.tensor([5,10])}
print(f'logits: {model(batch).shape}')
"`

### Step 4.5 — Model registry & models **init**

**Files:** `deepfm/models/__init__.py`
**Do:** Registry dict mapping `"deepfm" → DeepFM`, `"xdeepfm" → xDeepFM`, `"attention_deepfm" → AttentionDeepFM`. Factory function `create_model(name, schema, config) → BaseCTRModel`.
**Verify:** `python -c "from deepfm.models import create_model; print('ok')"`

---

## Phase 5: Training & CLI

### Step 5.1 — Metrics: AUC, LogLoss, HR@K, NDCG@K

**Files:** `deepfm/training/metrics.py`
**Do:**

- `compute_auc(labels, scores)` — wraps sklearn `roc_auc_score`.
- `compute_logloss(labels, scores)` — wraps sklearn `log_loss`.
- `compute_hr_at_k(rankings, k)` — Hit Rate from ranked lists.
- `compute_ndcg_at_k(rankings, k)` — NDCG from ranked lists.
- `RankingEvaluator` class: takes per-user (1 positive + 999 negatives), scores them, computes HR@K and NDCG@K for K=5,10,20.
  **Verify:** `python -c "
import numpy as np
from deepfm.training.metrics import compute_auc, compute_logloss, compute_hr_at_k, compute_ndcg_at_k
labels = np.array([0,1,0,1])
scores = np.array([0.1, 0.9, 0.3, 0.8])
print(f'AUC: {compute_auc(labels, scores):.3f}')
print(f'LogLoss: {compute_logloss(labels, scores):.3f}')
"`

### Step 5.2 — Trainer

**Files:** `deepfm/training/trainer.py`
**Do:** `Trainer` class:

- `__init__(model, schema, config, train_ds, val_ds, test_ds)`.
- Configures optimizer (Adam/AdamW/SGD), LR scheduler, BCEWithLogitsLoss.
- `train()`: epoch loop with gradient clipping, L2 reg, early stopping, checkpointing.
- `evaluate(dataset, split_name)`: computes classification + ranking metrics.
- Dynamic negative re-sampling each epoch (calls adapter's resample method).
- Logging via `get_logger`.
  **Verify:** `python -c "from deepfm.training.trainer import Trainer; print('ok')"`

### Step 5.3 — CLI entry point

**Files:** `deepfm/cli.py`, `deepfm/__main__.py`
**Do:**

- `cli.py`: `resolve_device(config_device)` → MPS → CPU. `main()` with argparse: `train` and `evaluate` subcommands. Accepts `--config` and `--override key=value`.
- `__main__.py`: `from deepfm.cli import main; main()`.
  **Verify:** `python -m deepfm --help`

### Step 5.4 — YAML configs

**Files:** `configs/deepfm_movielens.yaml`, `configs/xdeepfm_movielens.yaml`, `configs/attention_deepfm_movielens.yaml`
**Do:** One YAML per model variant with PRD defaults. Point `data_dir` to `data/ml-100k`.
**Verify:** `python -c "from deepfm.config import load_config; c = load_config('configs/deepfm_movielens.yaml'); print(c.model_name)"`

### Step 5.5 — End-to-end training smoke test

**Files:** (none — verification only)
**Do:** Run a quick training pass with reduced epochs/data to confirm the full pipeline works.
**Verify:** `python -m deepfm train --config configs/deepfm_movielens.yaml --override training.num_epochs=2 training.batch_size=512`

---

## Phase 6: Tests & Docs

### Step 6.1 — Unit tests: schema & transforms

**Files:** `tests/test_schema.py`, `tests/test_transforms.py`
**Do:** Test FieldSchema/DatasetSchema properties, LabelEncoder OOV handling, MultiHotEncoder padding, MinMaxScaler boundaries.
**Verify:** `pytest tests/test_schema.py tests/test_transforms.py -v`

### Step 6.2 — Unit tests: dataset

**Files:** `tests/test_dataset.py`
**Do:** Test TabularDataset with synthetic data: indexing, length, DataLoader batching.
**Verify:** `pytest tests/test_dataset.py -v`

### Step 6.3 — Unit tests: layers

**Files:** `tests/test_layers.py`
**Do:** Test each layer (FeatureEmbedding, FM, DNN, CIN, Attention) with synthetic schemas and random tensors. Verify output shapes and gradient flow.
**Verify:** `pytest tests/test_layers.py -v`

### Step 6.4 — Unit tests: models

**Files:** `tests/test_models.py`
**Do:** Test DeepFM, xDeepFM, AttentionDeepFM: forward pass shapes, predict outputs in [0,1], L2 reg loss > 0.
**Verify:** `pytest tests/test_models.py -v`

### Step 6.5 — Unit tests: trainer

**Files:** `tests/test_trainer.py`
**Do:** Test Trainer with a tiny synthetic dataset and DeepFM: 1-2 epochs run without error, metrics are computed.
**Verify:** `pytest tests/test_trainer.py -v`

### Step 6.6 — Integration test

**Files:** `tests/test_integration.py`
**Do:** Full end-to-end test on real ML-100K (marked `@pytest.mark.slow`): load data → train DeepFM for 2 epochs → verify AUC > 0.5.
**Verify:** `pytest tests/test_integration.py -v`

### Step 6.7 — README

**Files:** `README.md`
**Do:** Project overview, install instructions, usage examples, model descriptions, expected results table.
**Verify:** `cat README.md` (manual review)

### Step 6.8 — Full test suite & lint

**Files:** (none — verification only)
**Do:** Run all tests and lint checks.
**Verify:** `make test && make lint`
