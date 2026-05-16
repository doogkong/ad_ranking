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
## Overview 

  Key Ideas: TokenMixer-Large (ByteDance, Feb 2026)

  Problem: The original TokenMixer (RankMixer) has four failure modes at scale:
  1. Sub-optimal residual — mixing T tokens into H≠T new tokens breaks token semantic alignment in residuals
  2. Impure model — legacy ops (LHUC, DCNv2) pollute MFU and memory bandwidth
  3. Vanishing gradients in deep stacks — no mechanism to propagate signal across many layers
  4. Inadequate MoE — "Dense Train, Sparse Infer" is expensive; relu-MoE is unpredictable at inference

  Solution — three core contributions:

  1. Mixing & Reverting: A symmetric two-layer reshape. Layer 1 rearranges T×D tokens into H head-tokens of size T·(D/H), applies per-head SwiGLU — enabling cross-token interaction. Layer 2 reverses the reshape and
  applies a residual back to the original input X (not the mixed output), maintaining token semantic alignment throughout.
  2. Inter-residual + Auxiliary Loss: Residual connections every 2–3 blocks (not just adjacent) combat vanishing gradients. Lower-layer logits are combined with final logits as a joint auxiliary loss to force lower
  layers to learn meaningful representations.
  3. Sparse-Pertoken MoE: "First Enlarge, Then Sparse" — expand the pertoken SwiGLU into E experts, activate top-k via softmax routing + Gate Value Scaling (α multiplier) + one always-active shared expert +
  down-matrix small init (0.01×). Achieves 1:2 sparsity with near-zero quality drop.

  Results: 7B params online (15B offline), +2% ADSS (ads), +2.98% GMV (e-commerce), +1.4% pay revenue (live streaming) at ByteDance.

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

## Files

```
tokenmixer_large/
├── tokenmixer_large.py       # full implementation + smoke test
│   ├── RMSNorm
│   ├── PertokenSwiGLU
│   ├── MixingReverting
│   ├── SparsePerTokenMoE
│   ├── TokenMixerLargeBlock
│   ├── SemanticGroupTokenizer
│   └── TokenMixerLarge
├── test_tokenmixer_large.py  # pytest test suite (42 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 tokenmixer_large.py
```

Expected output:

```
logits:   torch.Size([4, 2])   tensor([[ 0.2164,  0.2532],
        [-0.2668, -0.1309],
        [-0.0556,  0.4862],
        [ 0.4110,  0.3442]], grad_fn=<CatBackward0>)
aux_loss: 0.0847
params:   4,810,756
```

### Test suite

```bash
python3 -m pytest test_tokenmixer_large.py -v
```

Expected output:

```
collected 42 items

test_tokenmixer_large.py::TestRMSNorm::test_output_shape PASSED
test_tokenmixer_large.py::TestRMSNorm::test_unit_rms_after_norm PASSED
test_tokenmixer_large.py::TestRMSNorm::test_weight_scales_output PASSED
test_tokenmixer_large.py::TestRMSNorm::test_gradient_flows PASSED
test_tokenmixer_large.py::TestPertokenSwiGLU::test_output_shape PASSED
test_tokenmixer_large.py::TestPertokenSwiGLU::test_pertoken_independence PASSED
test_tokenmixer_large.py::TestPertokenSwiGLU::test_small_init_wdown_magnitude PASSED
test_tokenmixer_large.py::TestPertokenSwiGLU::test_small_init_near_identity PASSED
test_tokenmixer_large.py::TestPertokenSwiGLU::test_gradient_flows PASSED
test_tokenmixer_large.py::TestMixingReverting::test_output_shape PASSED
test_tokenmixer_large.py::TestMixingReverting::test_dim_not_divisible_raises PASSED
test_tokenmixer_large.py::TestMixingReverting::test_residual_to_original_x PASSED
test_tokenmixer_large.py::TestMixingReverting::test_different_heads PASSED
test_tokenmixer_large.py::TestMixingReverting::test_gradient_flows PASSED
test_tokenmixer_large.py::TestMixingReverting::test_moe_factory PASSED
test_tokenmixer_large.py::TestSparsePerTokenMoE::test_output_shape PASSED
test_tokenmixer_large.py::TestSparsePerTokenMoE::test_gate_weights_sum_to_one PASSED
test_tokenmixer_large.py::TestSparsePerTokenMoE::test_top_k_experts_selected PASSED
test_tokenmixer_large.py::TestSparsePerTokenMoE::test_gate_scale_effect PASSED
test_tokenmixer_large.py::TestSparsePerTokenMoE::test_shared_expert_always_active PASSED
test_tokenmixer_large.py::TestSparsePerTokenMoE::test_top_k_equals_num_experts PASSED
test_tokenmixer_large.py::TestSparsePerTokenMoE::test_gradient_flows PASSED
test_tokenmixer_large.py::TestTokenMixerLargeBlock::test_output_shape PASSED
test_tokenmixer_large.py::TestTokenMixerLargeBlock::test_residual_preserves_scale PASSED
test_tokenmixer_large.py::TestTokenMixerLargeBlock::test_gradient_flows_through_both_sublayers PASSED
test_tokenmixer_large.py::TestSemanticGroupTokenizer::test_output_shape PASSED
test_tokenmixer_large.py::TestSemanticGroupTokenizer::test_global_token_prepended PASSED
test_tokenmixer_large.py::TestSemanticGroupTokenizer::test_num_tokens_attribute PASSED
test_tokenmixer_large.py::TestSemanticGroupTokenizer::test_global_token_shared_across_batch PASSED
test_tokenmixer_large.py::TestSemanticGroupTokenizer::test_different_groups_get_different_mlps PASSED
test_tokenmixer_large.py::TestSemanticGroupTokenizer::test_gradient_flows PASSED
test_tokenmixer_large.py::TestTokenMixerLarge::test_output_shapes PASSED
test_tokenmixer_large.py::TestTokenMixerLarge::test_single_task PASSED
test_tokenmixer_large.py::TestTokenMixerLarge::test_aux_loss_none_when_disabled PASSED
test_tokenmixer_large.py::TestTokenMixerLarge::test_aux_loss_positive PASSED
test_tokenmixer_large.py::TestTokenMixerLarge::test_no_inter_residual PASSED
test_tokenmixer_large.py::TestTokenMixerLarge::test_backward_through_total_loss PASSED
test_tokenmixer_large.py::TestTokenMixerLarge::test_deterministic_with_seed PASSED
test_tokenmixer_large.py::TestTokenMixerLarge::test_different_inputs_different_outputs PASSED
test_tokenmixer_large.py::TestTokenMixerLarge::test_many_feature_groups PASSED
test_tokenmixer_large.py::TestTokenMixerLarge::test_single_layer_no_inter_residual PASSED
test_tokenmixer_large.py::TestTokenMixerLarge::test_grad_exists_for_all_parameters PASSED

42 passed in 0.89s
```

Useful variants:

```bash
# Run a single test class
python3 -m pytest test_tokenmixer_large.py::TestSparsePerTokenMoE -v

# Run a single test
python3 -m pytest test_tokenmixer_large.py::TestTokenMixerLarge::test_backward_through_total_loss -v

# Stop on first failure
python3 -m pytest test_tokenmixer_large.py -x
```
