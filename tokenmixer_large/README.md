# TokenMixer-Large

PyTorch reference implementation of **TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders** (ByteDance AML, SIGKDD 2026).

Paper: https://arxiv.org/abs/2602.06563

---

## Motivation

The original TokenMixer (RankMixer) has four failure modes when scaled to billions of parameters:

| Problem | Root cause |
|---|---|
| Sub-optimal residual | Mixing T tokens into H≠T tokens breaks semantic alignment — residuals no longer connect the same token positions |
| Impure model | Legacy ops (LHUC, DCNv2) pollute GPU MFU and memory bandwidth |
| Vanishing gradients | No mechanism to propagate signal across many stacked layers |
| Inadequate MoE | "Dense Train, Sparse Infer" is expensive; ReLU-MoE has unpredictable expert activation counts at inference |

TokenMixer-Large addresses all four with targeted architectural changes.

---

## Architecture

```
Input feature groups  [user, item, context, sequence, ...]
        │
        ▼
SemanticGroupTokenizer
  ├─ per-group MLP  →  token_i  ∈ ℝ^D          (one token per semantic group)
  └─ learnable global token  [G]  ∈ ℝ^D         (aggregates cross-group info)
        │
        │  tokens X ∈ ℝ^(T×D),   T = 1 + num_groups
        ▼
┌─────────────────────────────────────────────────────┐
│  TokenMixerLargeBlock  ×  L                         │
│                                                     │
│   RMSNorm                                           │
│      │                                              │
│      ▼                                              │
│   MixingReverting                                   │
│      ├─ Mixing:   T×D  →  H×(T·D/H)  →  S-P MoE   │
│      └─ Reverting: back to T×D,  residual to X      │
│      │                                              │
│   + residual add  (x = x + mix_out)                │
│                                                     │
│   RMSNorm                                           │
│      │                                              │
│      ▼                                              │
│   Sparse-Pertoken MoE                               │
│      │                                              │
│   + residual add  (x = x + moe_out)                │
└─────────────────────────────────────────────────────┘
        │
        │  inter-residual added every N blocks
        │  auxiliary loss from intermediate layer outputs
        ▼
   Final RMSNorm  →  mean pool over T  →  task heads
        │
        ▼
   logits  ∈ ℝ^(B × num_tasks)     (e.g. pCTR, pVTR)
```

---

## Key Components

### 1. Mixing & Reverting

The core fix over RankMixer. A symmetric two-step reshape ensures **Token Semantic Alignment**: residuals always connect the same token-position semantics end-to-end.

**Mixing** (cross-token information exchange):
```
X ∈ ℝ^(B,T,D)
  → reshape  →  H_heads ∈ ℝ^(B, H, T·D/H)    # head-major layout
  → per-head SwiGLU + residual + norm
```

Each head-token now contains all T original tokens' features for its D/H slice, enabling token mixing without attention's O(T²) cost.

**Reverting** (restore original layout, anchor residual to X):
```
H_heads ∈ ℝ^(B, H, T·D/H)
  → reshape  →  X_rev ∈ ℝ^(B, T, D)           # back to token-major
  → per-token SwiGLU + residual to *original X* + norm
```

The residual connects `X_rev` to the input `X` (not the mixed output), preserving semantic alignment across arbitrary depth.

**Why this matters vs. RankMixer:**

RankMixer mixes T tokens into H≠T new tokens, then tries to add back the original T-token residual — a position mismatch. TokenMixer-Large forces H=T by using the reshape/transpose trick, so every layer's residual is semantically consistent.

### 2. Sparse-Pertoken MoE (S-P MoE)

Replaces the dense PertokenSwiGLU with a mixture of experts, achieving **Sparse Train, Sparse Infer** (vs. RankMixer's Dense Train, Sparse Infer).

```
x ∈ ℝ^(B,T,D)
  → Router  →  logits ∈ ℝ^(B,T,E)
  → top-k selection per token  →  gate weights g ∈ ℝ^(B,T,k)
  → weighted sum of top-k expert outputs
  → × α  (Gate Value Scaling)
  → + shared expert output  (always active)
```

Four design details from the paper:

| Detail | Purpose |
|---|---|
| **Gate Value Scaling (α=2)** | Softmax forces gate weights to sum to 1, starving SwiGLU weights of gradient. α compensates by scaling up the routed contribution before adding the shared expert. |
| **Shared Expert** | Always-active expert provides a stable learning target, especially early in training when the router hasn't converged. |
| **Down-Matrix Small Init** | W_down initialised to 0.01 × xavier. The block starts near identity (F(x)+x ≈ x), preventing output-value explosion common with multiplicative SwiGLU interactions in deep stacks. |
| **"First Enlarge, Then Sparse"** | First train a dense model to find optimal capacity, then split SwiGLU hidden dim into E experts and sparsify. 1:2 sparsity achieves near-zero quality drop vs. dense, with half the active parameters. |

### 3. Inter-Residual & Auxiliary Loss

Two mechanisms to combat gradient vanishing in deep (L>6) models:

**Inter-residual**: a residual connection bridging every N blocks (paper: N=2 or 3), not just adjacent layers. At block i, if `i % N == 0`, the output of block `i-N` is added to the current representation before processing.

```
x_4 = block_4(x_3 + x_2)   # inter-residual from block 2
x_6 = block_6(x_5 + x_4)   # inter-residual from block 4
```

*Note*: inter-residuals are **not** applied at the final layer, whose job is to distill high-level abstract features for the task heads — injecting lower-level raw info there hurts performance.

**Auxiliary loss**: intermediate layers output logits via lightweight linear heads. These are trained jointly with the final task loss:

```
L_total = L_task(logits_final, y) + λ · Σ MSE(logits_layer_i, logits_final.detach())
```

This forces lower layers to produce meaningful representations, improving gradient flow and final model quality. λ=0.1 is a reasonable default.

### 4. Pertoken SwiGLU

Standard SwiGLU shared across all token positions would ignore the heterogeneity of user/item/context features. PertokenSwiGLU gives each position t its own weight matrices:

```
W_up^t, W_gate^t ∈ ℝ^(D × nD)
W_down^t         ∈ ℝ^(nD × D)

output_t = W_down^t · (SiLU(W_gate^t · x_t) ⊙ W_up^t · x_t)
```

Implemented via `torch.einsum("btd,tdn->btn", x, W)` — a batched matrix multiply over the token axis. This is the key operation targeted by MoEGroupedFFN + Token Parallel in production.

### 5. Scaling Laws

The paper establishes empirical scaling laws for TokenMixer-Large:

- **Balanced expansion** (width D, depth L, MoE scaling factor N together) yields better returns than scaling any single dimension — avoid creating bottlenecks.
- **Larger models need more data**: scaling from 500M to 2B parameters requires ~60 days of training samples to converge (vs. 14 days for 30M→90M).
- AUC scales as a power law: `AUC_gain = C · N^α` where N is the dense parameter count.

---

## Usage

```python
from tokenmixer_large import TokenMixerLarge

model = TokenMixerLarge(
    feature_groups=[user_dim, item_dim, context_dim, seq_dim],
    model_dim=256,
    num_heads=8,
    num_layers=12,
    num_tasks=2,                    # e.g. pCTR + pVTR
    num_experts=3,
    top_k=2,                        # 1:2 sparsity
    moe_expand=4,
    gate_scale=2.0,
    inter_residual_interval=2,
    aux_loss_weight=0.1,
)

# Forward pass
groups = [user_feats, item_feats, context_feats, seq_feats]  # each (B, dim)
logits, aux_loss = model(groups)   # logits: (B, 2)

# Training
task_loss = F.binary_cross_entropy_with_logits(logits, targets)
total_loss = task_loss + aux_loss   # aux_loss is None if aux_loss_weight=0
total_loss.backward()
```

### Hyperparameter guidance

| Setting | Value | Notes |
|---|---|---|
| `num_heads` | 4–8 | D must be divisible by num_heads |
| `top_k` | 2 | 1:2 sparsity has best ROI per paper |
| `gate_scale` | 2.0 | Optimal for 1:2 sparsity; use 4.0 for 1:4 |
| `inter_residual_interval` | 2–3 | 2 for deeper models (L≥8), 3 for shallower |
| `aux_loss_weight` | 0.1 | Increase if lower layers show poor gradient flow |
| `moe_expand` | 4 | Standard SwiGLU expansion; reduce to 2 for memory |

---

## File

```
tokenmixer_large/
└── tokenmixer_large.py     # full implementation + smoke test
    ├── RMSNorm
    ├── PertokenSwiGLU
    ├── MixingReverting
    ├── SparsePerTokenMoE
    ├── TokenMixerLargeBlock
    ├── SemanticGroupTokenizer
    └── TokenMixerLarge
```

Run the smoke test:
```bash
python3 tokenmixer_large.py
```
