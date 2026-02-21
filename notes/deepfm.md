# DeepFM — Today I Learned

Notes captured while implementing a production-grade DeepFM framework.

---

## 1. First-Order vs Second-Order Embeddings

In FM (Factorization Machine), each feature has **two** separate embeddings:

### First-Order — `nn.Embedding(vocab, 1)`

- A single **scalar weight** per feature value
- Captures: "how important is this feature value on its own?"
- Analogous to logistic regression coefficients
- Example: `user_id=42 → w = 0.3`
- All first-order values are summed across fields → `(B, 1)`

### Second-Order — `nn.Embedding(vocab, embedding_dim)`

- A **dense vector** per feature value (e.g., dim=16)
- Captures: "how does this feature interact with others?"
- Used to compute pairwise interactions between fields
- Example: `user_id=42 → v = [0.1, -0.2, 0.5, ...]`

### The FM prediction combines both:

```
y = bias + Σ(first_order) + 0.5 * (square_of_sum - sum_of_squares)
           ↑ linear          ↑ pairwise interactions from second-order vectors
```

---

## 2. The Square-of-Sum Minus Sum-of-Squares Trick

The FM second-order term needs all pairwise dot products between field vectors. Naively O(n²), but an algebraic identity makes it O(n·d).

### What we want:

```
Σ_{i<j} <v_i, v_j>    (sum of all pairwise dot products)
```

### The identity:

When you square the sum of all vectors element-wise:

```
(v_1 + v_2 + v_3)² = v_1² + v_2² + v_3² + 2·v_1·v_2 + 2·v_1·v_3 + 2·v_2·v_3
                      \________________/   \_________________________________/
                       sum_of_squares           2 × (what we want)
```

Rearranging:

```
Σ_{i<j} <v_i, v_j> = 0.5 × (square_of_sum − sum_of_squares)
```

### In code:

```python
# field_embeddings: (B, F, D)
sum_vec = field_embeddings.sum(dim=1)              # (B, D)
square_of_sum = sum_vec ** 2                       # (B, D)
sum_of_squares = (field_embeddings ** 2).sum(dim=1) # (B, D)
interaction = 0.5 * (square_of_sum - sum_of_squares) # (B, D)
output = interaction.sum(dim=1, keepdim=True)         # (B, 1)
```

### Numeric example (3 fields, D=2):

```
v_user  = [1, 2]
v_movie = [3, 4]
v_genre = [5, 6]

Brute force pairwise:
  <v_user, v_movie> = 1×3 + 2×4 = 11
  <v_user, v_genre> = 1×5 + 2×6 = 17
  <v_movie, v_genre> = 3×5 + 4×6 = 39
  total = 67

With the trick:
  sum_vec        = [9, 12]
  square_of_sum  = [81, 144]
  sum_of_squares = [1+9+25, 4+16+36] = [35, 56]
  0.5 × [81−35, 144−56] = 0.5 × [46, 88] = [23, 44]
  sum = 67  ✓
```

Same result, O(n·d) instead of O(n²·d).

---

## 3. FeatureEmbedding — Three Output Views

The shared `FeatureEmbedding` layer returns three tensors from one forward pass:

| Output             | Shape            | Consumer                       |
| ------------------ | ---------------- | ------------------------------ |
| `first_order`      | `(B, 1)`         | FM linear term                 |
| `field_embeddings` | `(B, F, fm_dim)` | FM interaction, CIN, Attention |
| `flat_embeddings`  | `(B, total_dim)` | DNN input                      |

### Key design decisions:

- **Per-field custom dims**: high-cardinality fields (user_id: 16) get larger embeddings than low-cardinality ones (gender: 4)
- **Projection layer**: since FM needs same-dim vectors for dot products, a `nn.Linear(field_dim, fm_embed_dim)` aligns fields to a common dimension
- **Shared embeddings**: FM and DNN share the same embedding vectors — DNN gets the raw concat, FM gets the projected view. This is the core DeepFM insight (unlike Wide&Deep which has separate feature engineering for the wide component)

### Data flow:

```
FeatureEmbedding(batch)
    │
    ├── first_order (B, 1)  ──────────> FM linear term
    │
    ├── field_embeddings (B, F, fm_dim)
    │     ├──> FM 2nd order interaction
    │     ├──> CIN vector-wise interactions (xDeepFM)
    │     └──> Multi-head self-attention (AttentionDeepFM)
    │
    └── flat_embeddings (B, total_dim) ──> DNN (MLP)
```

---

## 4. OOV / Padding Strategy

- Index 0 is reserved for unknown/OOV across all encoders
- `padding_idx=0` in `nn.Embedding` and `nn.EmbeddingBag` ensures zero contribution
- LabelEncoder reserves index 0: known values mapped to 1, 2, 3, ...
- MultiHotEncoder pads sequences with 0s to `max_length`
- Result: unseen features contribute nothing to the prediction (safe degradation)

---

## 5. Leave-One-Out Split

Per user, ordered by timestamp:

- **Last interaction** → test (1 sample per user)
- **Second-to-last** → validation (1 sample per user)
- **All remaining** → training
- Users with < `min_interactions` (3) → training only

Note: the held-out interaction can have any rating. Some val/test "positives" have label=0 (rating < 4.0). This is expected — the split is temporal, not label-based.

---

## 6. Negative Sampling

### Training (dynamic, re-sampled each epoch):

- For each positive, sample `num_neg_train` (4) random items the user hasn't interacted with
- Full features (user + item including genres) are looked up for negatives
- Result: ~5× the positive count

### Evaluation (sampled ranking protocol):

- For each user, 1 held-out positive + `num_neg_eval` (999) sampled negatives
- Negatives filtered to exclude all user interactions
- Score all 1000 items → rank → HR@K, NDCG@K
