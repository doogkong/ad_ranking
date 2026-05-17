# InterFormer

PyTorch reference implementation of **InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction** (UIUC + Meta AI, Sep 2025).

Paper: https://arxiv.org/abs/2411.09852

Deployed at Meta Ads: **+0.15% NE** and **+24% QPS** vs. prior SOTA.

---

## Summary

The key insight is **bidirectional heterogeneous interaction** — most CTR models that combine non-sequential features with user history only flow information one way. InterFormer solves this with three mutually-reinforcing arches per layer:

- **Interaction Arch**: non-seq tokens attend over a compact sequence summary (history → ad features)
- **Sequence Arch**: sequence tokens are transformed by a *personalized* d×d projection matrix derived from non-seq context (ad features → how we interpret history)
- **Cross Arch**: selectively summarizes both modalities using SelfGating + LCE (for non-seq) and CLS/PMA/recent tokens (for sequence)

The **PFFN (Personalized FFN)** is particularly elegant — each sample gets its own `d×d` weight matrix for sequence transformation, making behavior modeling fully query/ad-aware without a separate cross-attention module.

---

## Key Ideas

### Problem: two failure modes in existing CTR models

Existing models that combine non-sequential features (user demographics, item attributes) with sequential features (click history) suffer from two issues:

1. **Insufficient inter-modal interaction** — information flow is one-directional. The sequence model updates based on non-seq context, but non-seq features don't receive sequence feedback. Each modality evolves in its own silo.

2. **Aggressive information aggregation** — when summarizing features across modalities, most models flatten everything into a single vector, discarding structural information. Selective, dimensionality-preserving summarization is better.

### InterFormer's solution: three mutually-reinforcing architectures

InterFormer uses three parallel sub-architectures per layer, connected bidirectionally:

| Arch | Direction | What it does |
|---|---|---|
| **Interaction Arch** | Seq → Non-seq | Each non-seq feature attends over the sequence summary, gaining behavior context |
| **Sequence Arch** | Non-seq → Seq | Sequence tokens are transformed by a personalized projection derived from non-seq context |
| **Cross Arch** | Bridges both | Selectively summarizes each modality to produce compact, high-signal representations |

**Bidirectionality is the key**: non-seq features learn from user history; user history is personalized by item/user context.

### Application in Ad Ranking

| InterFormer Component | Ad Ranking Mapping |
|---|---|
| Non-seq features | User demographics, query, ad attributes, advertiser features |
| Sequence features | Click/view/conversion history, search history |
| MaskNet | Filters noisy behaviors, merges multiple behavior types |
| PFFN (Personalized FFN) | Makes historical behavior interpretation ad-query-aware |
| Interaction Arch | Fuses historical intent signal into each ad feature token |
| Cross Arch / PMA | Distills history into a small fixed-size context for efficient scoring |

---

## Architecture

```
Non-seq X^(0) [dense, sparse features]
    │  EmbeddingLayer (dense_proj + sparse_embs)
    ▼
X^(1) ∈ ℝ^(B, n_ns, d)          MaskNet(seq_feat)
                                       │
                                   S^(1) ∈ ℝ^(B, T, d)

X_sum_init = Gating(LCE(X^(1)))  ─── prepend as CLS tokens ──▶ S^(1) augmented

for l = 1..L:
    ┌─────────────────────────────────────────────────────────────────────┐
    │  Cross Arch                                                         │
    │    X_sum^(l) = Gating(LCE(X^(l)))                     [Eq. 10]    │
    │    S_sum^(l) = Gating([S_CLS^(l) ∥ PMA(S^(l)) ∥ S_recent^(l)])  [Eq. 11] │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Interaction Arch                                                   │
    │    X^(l+1) = MLP(inner_product_attn([X^(l) ∥ S_sum^(l)]))        [Eq. 7]  │
    ├─────────────────────────────────────────────────────────────────────┤
    │  Sequence Arch                                                      │
    │    S^(l+1) = LayerNorm(MHA(PFFN(X_sum^(l), S^(l))) + S^(l))      [Eq. 9]  │
    └─────────────────────────────────────────────────────────────────────┘

ŷ = MLP([ flatten(X_sum^(L)) ∥ flatten(S_sum^(L)) ])
```

---

## Key Components

### 1. MaskNet (§4.1, Eq. 6)

```
MaskNet(S) = lce_mlp(S ⊙ sigmoid(mask_mlp(S)))
```

Learns which sequence items and which feature dimensions are relevant. `mask_mlp` produces per-element gates; `lce_mlp` projects the merged dimension to `d_model`. Used to unify k input sequences (each with different embedding dims) into one `(B, T, d)` tensor.

### 2. Linear Compressed Embedding — LCE (Eq. 10)

```
LCE(X) ∈ ℝ^(B, n_sum, d)  where n_sum << n_ns
```

Projects the token axis (not the feature axis) from `n_ns` to `n_sum` via a weight matrix `W ∈ ℝ^(n_sum × n_ns)`. Reduces `n` feature tokens to a small summary without losing the shared `d`-dimensional structure.

### 3. SelfGating (Eq. 10, 11)

```
Gating(X) = sigmoid(gate_mlp(X)) ⊙ X
```

Sparse masking: learned per-dimension gates suppress noise and retain high-signal dimensions. Applied after LCE (for non-seq summarization) and after concatenation (for sequence summarization).

### 4. Pooling by Multi-Head Attention — PMA (§3.3, Eq. 4)

```
PMA(S) = MHA(Q_PMA, K=S, V=S)   Q_PMA ∈ ℝ^(n_pma × d) is learnable
```

Summarizes the full sequence into `n_pma` fixed-size tokens. Each query captures a different aspect of behavior (e.g., recency vs. diversity).

### 5. Sequence Summarization — S_sum (Eq. 11)

```
S_sum = Gating( [S_CLS ∥ PMA(S) ∥ S_recent] )
```

Three complementary views: CLS tokens (transformed by prior MHA layers), PMA tokens (learned summarization), and the K most recent items (explicit recency signal).

### 6. Personalized FFN — PFFN (§4.3, Eq. 8)

```
W_PFFN = MLP(flatten(X_sum))  ∈ ℝ^(B, d, d)   (sample-specific)
PFFN(X_sum, S) = S · W_PFFN
```

Each sample gets its own `d×d` projection matrix for sequence token transformation, derived from its non-seq summary. This makes the sequence arch fully conditional on the current user/item/query context.

### 7. Interaction Arch (§4.2, Eq. 7)

```
tokens = [X ∥ S_sum]    (concat on token axis)
scores = softmax(tokens · tokens^T / √d)
X_new  = MLP( (scores · tokens)[:, :n_ns, :] + X )
```

Inner-product-based attention lets every non-seq feature token directly attend over the sequence summary. Only the n_ns non-seq positions are retained (plus residual), keeping X dimensionally stable across layers.

### 8. Scaling via CLS prepend (Algorithm 1, step 2-3)

Before the main loop, `X_sum_init = Gating(LCE(X^(1)))` is computed and prepended to the sequence as CLS tokens. This gives the sequence model global context about non-seq features from the very first layer, without requiring an extra cross-attention module.

---

## Usage

```python
from interformer import InterFormer

model = InterFormer(
    dense_dim=256,           # total dim of dense features (concat'd)
    sparse_dims=[1M, 500K],  # vocabulary sizes per sparse feature
    seq_input_dim=64,        # embedding dim per sequence item (after multi-seq merge)
    d_model=64,
    num_layers=3,            # L — number of InterFormer blocks
    n_sum=4,                 # compressed non-seq token count (n_sum << n_ns)
    n_cls=4,                 # CLS tokens prepended to sequence (paper: 4)
    n_pma=2,                 # PMA learnable query count (paper: 2)
    k_recent=2,              # recent token count in S_sum (paper: 2)
    n_heads=4,
    top_mlp_dims=[256, 128],
    num_tasks=1,             # 1 for pCTR, 2 for pCTR+pCVR
)

# Forward pass
logits = model(
    dense_feat=torch.randn(B, 256),
    sparse_feats=[torch.randint(0, v, (B,)) for v in [1_000_000, 500_000]],
    seq_feat=torch.randn(B, T, 64),
)   # → (B, num_tasks)

loss = F.binary_cross_entropy_with_logits(logits, targets)
loss.backward()
```

### Hyperparameter guidance

| Parameter | Paper value | Notes |
|---|---|---|
| `d_model` | 64 | Embedding dimension |
| `num_layers` L | 3 | More layers = deeper non-seq ↔ seq interaction |
| `n_sum` | 4 | Compressed non-seq token count; small values (2–8) work well |
| `n_cls` | 4 | CLS tokens; balance with PMA count |
| `n_pma` | 2 | PMA queries; captures different aspects of behavior history |
| `k_recent` | 2 | Recent tokens; explicit recency bias in S_sum |
| `n_heads` | 4 | Attention heads for PMA and SequenceArch MHA |

---

## Files

```
interformer/
├── interformer.py         # full implementation + smoke test
│   ├── SelfGating             # σ(gate(X)) ⊙ X — dimensionality-preserving masking
│   ├── MaskNet                # S ⊙ sigmoid(mask_mlp(S)) → lce_mlp → d_model
│   ├── LinearCompressedEmbedding  # W^T X: compress n_in tokens → n_out
│   ├── PoolingByMHA           # learnable-query cross-attention summarization
│   ├── CrossArch              # ns_summarize + seq_summarize with SelfGating
│   ├── InteractionArch        # inner-product attention on [X ∥ S_sum]
│   ├── PersonalizedFFN        # sample-specific d×d sequence projection
│   ├── SequenceArch           # PFFN + MHA + LayerNorm
│   ├── InterFormerBlock       # one CrossArch + InteractionArch + SequenceArch block
│   └── InterFormer            # full model with preprocessing and prediction head
├── test_interformer.py    # pytest test suite (48 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 interformer.py
```

Expected output:

```
logits:   torch.Size([4, 1])  [-0.036..., -0.061..., -0.049..., -0.062...]
loss:     0.6672
backward: OK
params:   463,505
```

### Test suite

```bash
python3 -m pytest test_interformer.py -v
```

Expected output:

```
collected 48 items

test_interformer.py::TestSelfGating::test_output_shape PASSED
test_interformer.py::TestSelfGating::test_gate_range PASSED
test_interformer.py::TestSelfGating::test_gate_modulates_identity PASSED
test_interformer.py::TestSelfGating::test_gradient PASSED
test_interformer.py::TestMaskNet::test_output_shape PASSED
test_interformer.py::TestMaskNet::test_different_input_dims PASSED
test_interformer.py::TestMaskNet::test_mask_is_sigmoid PASSED
test_interformer.py::TestMaskNet::test_gradient PASSED
test_interformer.py::TestLinearCompressedEmbedding::test_output_shape PASSED
test_interformer.py::TestLinearCompressedEmbedding::test_compression PASSED
test_interformer.py::TestLinearCompressedEmbedding::test_linear_in_feature_dim PASSED
test_interformer.py::TestLinearCompressedEmbedding::test_gradient PASSED
test_interformer.py::TestPoolingByMHA::test_output_shape PASSED
test_interformer.py::TestPoolingByMHA::test_different_n_pma PASSED
test_interformer.py::TestPoolingByMHA::test_learned_query_batched PASSED
test_interformer.py::TestPoolingByMHA::test_gradient PASSED
test_interformer.py::TestCrossArch::test_ns_summarize_shape PASSED
test_interformer.py::TestCrossArch::test_seq_summarize_shape PASSED
test_interformer.py::TestCrossArch::test_forward_returns_pair PASSED
test_interformer.py::TestCrossArch::test_ns_summarize_gradient PASSED
test_interformer.py::TestCrossArch::test_seq_summarize_gradient PASSED
test_interformer.py::TestInteractionArch::test_output_shape PASSED
test_interformer.py::TestInteractionArch::test_retains_n_ns PASSED
test_interformer.py::TestInteractionArch::test_residual_included PASSED
test_interformer.py::TestInteractionArch::test_gradient PASSED
test_interformer.py::TestPersonalizedFFN::test_output_shape PASSED
test_interformer.py::TestPersonalizedFFN::test_personalization PASSED
test_interformer.py::TestPersonalizedFFN::test_gradient PASSED
test_interformer.py::TestSequenceArch::test_output_shape PASSED
test_interformer.py::TestSequenceArch::test_residual_connection PASSED
test_interformer.py::TestSequenceArch::test_layer_norm_applied PASSED
test_interformer.py::TestSequenceArch::test_gradient PASSED
test_interformer.py::TestInterFormerBlock::test_output_shapes PASSED
test_interformer.py::TestInterFormerBlock::test_sequence_length_preserved PASSED
test_interformer.py::TestInterFormerBlock::test_gradient_all_outputs PASSED
test_interformer.py::TestInterFormerBlock::test_different_inputs_different_outputs PASSED
test_interformer.py::TestInterFormer::test_output_shape_single_task PASSED
test_interformer.py::TestInterFormer::test_output_shape_multi_task PASSED
test_interformer.py::TestInterFormer::test_backward PASSED
test_interformer.py::TestInterFormer::test_grad_all_parameters PASSED
test_interformer.py::TestInterFormer::test_deterministic_with_seed PASSED
test_interformer.py::TestInterFormer::test_different_inputs_different_outputs PASSED
test_interformer.py::TestInterFormer::test_num_layers_scaling PASSED
test_interformer.py::TestInterFormer::test_finite_output PASSED
test_interformer.py::TestInterFormer::test_single_sparse_feature PASSED
test_interformer.py::TestInterFormer::test_many_sparse_features PASSED
test_interformer.py::TestInterFormer::test_batch_size_one PASSED
test_interformer.py::TestInterFormer::test_longer_sequence PASSED

48 passed in 0.78s
```

Useful variants:

```bash
python3 -m pytest test_interformer.py::TestInterFormerBlock -v
python3 -m pytest test_interformer.py -x
```
