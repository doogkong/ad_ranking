# Wukong

PyTorch reference implementation of **Wukong: Towards a Scaling Law for Large-Scale Recommendation** (Meta AI, ICML 2024).

Paper: https://arxiv.org/abs/2403.02545

---

## Key Ideas

### Problem: sparse scaling hits a wall

Existing recommendation models (DLRM, DCNv2, AutoInt+, FinalMLP) scale almost exclusively by expanding embedding tables — more rows, higher dimensions. This has two fundamental problems:

1. **Doesn't improve feature interaction capture.** Larger embeddings don't help the model discover higher-order correlations among features.
2. **Hardware mismatch.** Next-generation accelerators (H100, etc.) gain compute capacity, not memory bandwidth. Embedding lookups are memory-bound and can't use the new compute.

As a result, existing models plateau early — AutoInt+ and DCNv2 collapse in quality beyond ~50 GFLOP/example (Figure 1 in the paper).

### Wukong's solution: dense scaling via stacked FMs

Wukong focuses entirely on scaling the **interaction component** (dense layers), using a stacked Factorization Machine design inspired by **binary exponentiation**:

> Layer i captures interactions of order 1 to **2^i**.
> A stack of l layers captures interactions up to order **2^l**.

With l=8 layers: up to 256th-order feature interactions — something no prior model approaches.

| l (layers) | Max interaction order |
|---|---|
| 1 | 2 |
| 2 | 4 |
| 3 | 8 |
| 4 | 16 |
| 6 | 64 |
| 8 | 256 |

### Scaling law

Quality scales as a power law in compute:

```
AUC_gain ≈ -100 + 99.56 · x^0.00071    (x = GFLOP/example)
```

~0.1% LogLoss improvement per 4× compute, sustained across **two orders of magnitude** (1 → 100+ GFLOP/example). No other model in the paper maintains this trend beyond ~36 GFLOP/example.

---

## Architecture

```
Feature groups  [user, item, context, ...]   each (B, raw_dim_i)
        │
        ▼
EmbeddingLayer
  └─ per-group MLP  →  d-dim token          (one token per feature group)
        │
        │  X_0 ∈ ℝ^(B, n, d)              n = number of feature groups
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  WukongLayer  × l                                               │
│                                                                 │
│   ┌─────────────────────────┐  ┌───────────────────────────┐   │
│   │   FMBlock                │  │   LCBBlock                │   │
│   │                         │  │                           │   │
│   │  FM(X) = X·(X^T·Y)      │  │  W_L · X                  │   │
│   │  (attentive Y via MLP)  │  │  (linear recombination)   │   │
│   │  flatten → LN → MLP     │  │                           │   │
│   │  → n_F tokens           │  │  → n_L tokens             │   │
│   └────────────┬────────────┘  └──────────┬────────────────┘   │
│                └──────────┬───────────────┘                     │
│                           │  concat                             │
│                           ▼                                     │
│                  + residual(X_i)  →  LayerNorm                  │
│                           │                                     │
│                       X_{i+1}  ∈ ℝ^(B, n_F+n_L, d)            │
└───────────────────────────┼─────────────────────────────────────┘
                            │
                        flatten
                            │
                        Final MLP  →  logit(s)   (B, num_tasks)
```

---

## Key Components

### 1. EmbeddingLayer (§3.2)

Each feature group (user features, item features, context, etc.) is projected to a common `d_model`-dimensional token via a group-specific two-layer MLP. This gives a token matrix X_0 ∈ ℝ^(n×d) where n = number of groups.

Unlike DCN which flattens all embeddings into one vector (losing group structure), Wukong treats each embedding as an atomic unit and interacts them whole — dramatically reducing compute vs. element-wise approaches like xDeepFM.

### 2. Optimized FM (§3.6)

Standard FM computes XX^T ∈ ℝ^(n×n), which is O(n²d) and becomes prohibitive with hundreds of features.

**Key insight**: when d ≤ n (common in production — embedding dim < feature count), XX^T has rank ≤ d. So it's lossless to project it to n×k with k ≤ d:

```
FM_opt(X) = X · (X^T · Y)      # associativity saves compute: O(n²d) → O(nkd)

where Y ∈ ℝ^(n×k) is the projection matrix, k << n
```

**Attentive Y**: instead of a fixed Y, Wukong derives it from the input via a small MLP:

```
Y = MLP(flatten(X))    # Y adapts to the current sample
```

This lets the projection focus on the most informative feature pairs per example.

### 3. FMBlock (§3.4)

```
FMB(X_i) = reshape(MLP(LN(flatten(FM_opt(X_i)))))

FM output:  (B, n, k)
flatten:    (B, n·k)
LN:         (B, n·k)
MLP:        (B, n_F · d)
reshape:    (B, n_F, d)     ← n_F new interaction tokens
```

The MLP inside FMB transforms raw pairwise interactions into new embedding representations that can be fed into the next layer's FM — this is the key mechanism for capturing higher-order interactions.

### 4. LCBBlock (§3.5)

```
LCB(X_i) = W_L · X_i      W_L ∈ ℝ^(n_L × n_i)
```

LCB performs a simple linear recombination of input tokens, preserving all interaction orders that X_i already contains. This is critical: without LCB, the output of FMB only contains interactions of the *new* order captured at this layer. LCB ensures the next layer also has access to all lower-order signals.

**Why this matters (ablation §7.3)**: removing LCB alone only drops quality by 0.03%; removing residual alone by 0.08%; removing both drops by 1.84% — they act synergistically.

### 5. WukongLayer (§3.3)

```
X_{i+1} = LN(concat(FMB(X_i), LCB(X_i)) + X_i)
```

- `concat(FMB, LCB)`: (B, n_F + n_L, d) — new interaction tokens
- Residual: X_i projected to match shape if n_i ≠ n_F + n_L
- LayerNorm stabilises training across l layers

### 6. Scaling Knobs (§3.8)

| Knob | What it controls | Impact |
|---|---|---|
| `num_layers` l | Stack depth; determines max interaction order 2^l | **Highest** — increasing l gives the most quality improvement |
| `n_F` | FMB output tokens; more = richer interaction representation | High |
| `n_L` | LCB output tokens | Moderate (set ≈ n_F in practice) |
| `k` | FM projection rank; higher k = less information loss | Moderate |
| `mlp_dims` | FMBlock internal MLP width/depth | Moderate |

Paper recommendation: scale l first, then jointly scale n_F, n_L, k, MLP.

---

## Usage

```python
from wukong import Wukong

model = Wukong(
    feature_dims=[64, 128, 32, 256],   # raw dim per feature group
    d_model=128,
    num_layers=8,        # l — primary scaling knob
    n_F=96,              # FMB output tokens
    n_L=96,              # LCB output tokens  (≈ n_F)
    k=96,                # FM projection rank
    mlp_dims=[8192],     # FMBlock internal MLP hidden dim
    top_mlp_dims=[512, 256],
    num_tasks=2,         # e.g. pCTR + pCVR
)

# Forward pass
groups = [user_feats, item_feats, ctx_feats, seq_feats]  # each (B, dim_i)
logits = model(groups)   # (B, 2)

# Training
loss = F.binary_cross_entropy_with_logits(logits, targets)
loss.backward()
```

### Pre-built scaling configurations

```python
from wukong import wukong_small, wukong_medium, wukong_large

# ~0.5 GFLOP/example (l=2, n_F=n_L=8,  k=24)
model = wukong_small(feature_dims=[64, 128, 32])

# ~2   GFLOP/example (l=8, n_F=n_L=32, k=24)
model = wukong_medium(feature_dims=[64, 128, 32])

# ~22  GFLOP/example (l=8, n_F=n_L=96, k=96, mlp=8192)
model = wukong_large(feature_dims=[64, 128, 32])
```

### Hyperparameter guidance from paper

| Setting | Value | Notes |
|---|---|---|
| `num_layers` | 2–8 | Start here when scaling; l=8 is the largest tested publicly |
| `n_F` = `n_L` | 8–192 | Scale together; equal values work well |
| `k` | 24–192 | Match to n_F; k=24 is a good default for small models |
| `mlp_dims` | `[2048]`–`[16384]` | Single hidden layer; scale width rather than depth |
| `d_model` | 128–160 | Paper fixes d=160 for internal dataset |
| `dropout` | 0.0 | Not used in paper; add if overfitting on small datasets |

---

## Files

```
wukong/
├── wukong.py              # full implementation + smoke test
│   ├── FeatureGroupEmbedding   # per-group MLP projection
│   ├── EmbeddingLayer          # all groups → X_0 ∈ ℝ^(n×d)
│   ├── OptimizedFM             # attentive low-rank FM: X(X^TY), O(nkd)
│   ├── FMBlock                 # FM → flatten → LN → MLP → n_F tokens
│   ├── LCBBlock                # W_L·X linear recombination → n_L tokens
│   ├── WukongLayer             # concat(FMB, LCB) + residual + LN
│   ├── Wukong                  # full model
│   └── wukong_small/medium/large  # pre-built scaling configs
├── test_wukong.py         # pytest test suite (37 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 wukong.py
```

Expected output:

```
logits:   torch.Size([4, 2])  [[0.1879, 0.2666], [-0.0488, 0.5262], ...]
loss:     0.8030
backward: OK
params:   276,674

Interaction order per layer (2^l):
  l=1: up to 2^1 = 2-order interactions
  l=2: up to 2^2 = 4-order interactions
  l=3: up to 2^3 = 8-order interactions
  l=4: up to 2^4 = 16-order interactions
  l=5: up to 2^5 = 32-order interactions
  l=6: up to 2^6 = 64-order interactions
```

### Test suite

```bash
python3 -m pytest test_wukong.py -v
```

Expected output:

```
collected 37 items

test_wukong.py::TestFeatureGroupEmbedding::test_output_shape PASSED
test_wukong.py::TestFeatureGroupEmbedding::test_different_input_dims PASSED
test_wukong.py::TestFeatureGroupEmbedding::test_gradient PASSED
test_wukong.py::TestEmbeddingLayer::test_output_shape PASSED
test_wukong.py::TestEmbeddingLayer::test_num_tokens PASSED
test_wukong.py::TestEmbeddingLayer::test_gradient_flows PASSED
test_wukong.py::TestEmbeddingLayer::test_different_groups_different_projections PASSED
test_wukong.py::TestOptimizedFM::test_output_shape PASSED
test_wukong.py::TestOptimizedFM::test_different_k PASSED
test_wukong.py::TestOptimizedFM::test_attentive_projection_varies_with_input PASSED
test_wukong.py::TestOptimizedFM::test_gradient PASSED
test_wukong.py::TestFMBlock::test_output_shape PASSED
test_wukong.py::TestFMBlock::test_different_n_F PASSED
test_wukong.py::TestFMBlock::test_gradient PASSED
test_wukong.py::TestLCBBlock::test_output_shape PASSED
test_wukong.py::TestLCBBlock::test_preserves_d_model PASSED
test_wukong.py::TestLCBBlock::test_linear_recombination PASSED
test_wukong.py::TestLCBBlock::test_gradient PASSED
test_wukong.py::TestWukongLayer::test_output_shape PASSED
test_wukong.py::TestWukongLayer::test_output_shape_first_layer_different_n PASSED
test_wukong.py::TestWukongLayer::test_subsequent_layer_same_shape PASSED
test_wukong.py::TestWukongLayer::test_residual_stabilises_output PASSED
test_wukong.py::TestWukongLayer::test_gradient PASSED
test_wukong.py::TestWukongLayer::test_layer_norm_applied PASSED
test_wukong.py::TestWukong::test_output_shape PASSED
test_wukong.py::TestWukong::test_single_task PASSED
test_wukong.py::TestWukong::test_three_tasks PASSED
test_wukong.py::TestWukong::test_single_feature_group PASSED
test_wukong.py::TestWukong::test_many_feature_groups PASSED
test_wukong.py::TestWukong::test_backward PASSED
test_wukong.py::TestWukong::test_grad_all_parameters PASSED
test_wukong.py::TestWukong::test_deterministic_with_seed PASSED
test_wukong.py::TestWukong::test_different_inputs_different_outputs PASSED
test_wukong.py::TestWukong::test_num_layers_scaling PASSED
test_wukong.py::TestWukong::test_interaction_order_growth PASSED
test_wukong.py::TestFactories::test_wukong_small PASSED
test_wukong.py::TestFactories::test_wukong_medium PASSED

37 passed in 1.14s
```

Useful variants:

```bash
python3 -m pytest test_wukong.py::TestWukongLayer -v
python3 -m pytest test_wukong.py -x
```
