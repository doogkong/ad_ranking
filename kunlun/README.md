# Kunlun

PyTorch reference implementation of **Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems through Unified Architecture Design** (Meta Platforms, Feb 2026).

Paper: https://arxiv.org/abs/2602.10016

Deployed at Meta Ads: **+1.2% NE** on topline metrics. MFU: **17% → 37%** on NVIDIA B200 GPUs. **2× scaling efficiency** over InterFormer.

---

## Paper Summary

Scaling laws — the predictable relationship between model performance and compute — are well established for LLMs (Kaplan et al., 2020) but remain an open challenge for recommendation systems, especially those that jointly model **sequential user history** (click/conversion sequences) alongside **non-sequential context features** (user demographics, ad metadata, contextual signals). Naively scaling such models fails to produce consistent performance gains.

Kunlun establishes the **first predictable scaling laws for joint sequence–nonsequence modeling** by identifying *scaling efficiency* as the core missing ingredient. Scaling efficiency is defined as NE improvement per unit of log-scaled compute — the slope of the NE vs. log(Compute) curve:

```
NE(C) = NE_0 − η · log(C / C_0)          (power-law scaling)

Scaling Efficiency = η / η_baseline       (relative to Wukong baseline)
```

Kunlun achieves **η ≈ 2× InterFormer**, meaning every doubling of compute buys twice as much quality improvement.

### Why does scaling fail in recommendation systems?

Two root causes block predictable scaling:

**1. Inefficient modules (low MFU).** Recommendation feature spaces are heterogeneous: small embedding dimensions, irregular tensor shapes, and memory-bound operations (embedding lookups, back-to-back matrix multiplications). Prior SOTA models achieve only **3–15% MFU** on modern GPUs, compared to 40–60% for LLMs. Hardware sits idle while waiting for memory.

**2. Inefficient resource allocation.** Scaling all components uniformly (naively adding layers or widening everything equally) yields diminishing returns because different layers and event types have fundamentally different computational patterns. Impressions and clicks have different predictive value; shallow layers need different operations than deep layers.

### How does Kunlun solve it?

Kunlun's approach is **model-efficiency codesign** at two levels:

| Level | Root Cause Targeted | Innovations |
|---|---|---|
| **Low-level** (module optimization) | Low MFU | GDPA, HSP, Sliding Window Attention |
| **High-level** (resource reallocation) | Inefficient allocation | CompSkip, Event-Level Personalization, Wukong MoE |

### Measured results

| Component | Metric | Change |
|---|---|---|
| GDPA | MFU | 37% → 34% when removed (−3 pt) |
| GDPA | NE | −0.04% without GDPA |
| HSP vs PMA | NE | +0.08% for HSP; PMA is 8.8% faster but lower quality |
| SumKronLinear | NE | +0.03% |
| Sliding Window | QPS | +31.1%, FLOPs −29.5% |
| CompSkip | FLOPs | −43.1%, QPS +35% |
| Event Personalization | QPS | +13%, FLOPs −11% |
| Expert Parallelism | QPS | +4% |
| **Full Kunlun** | MFU | **37%** (vs 17% baseline) |
| **Full Kunlun** | Scaling efficiency | **2× InterFormer** |
| **Production** | NE | **+1.2% topline** |

---

## Key Innovations

### Low-level: Module Optimization

**GDPA — Generalized Dot-Product Attention** (§4.3.1)

The prior PFFN transforms sequence tokens using non-seq context: `PFFN(X_sum, S) = f(X_sum) · S`, where `f` is a two-layer MLP. The bottleneck: per-layer heavyweight activations and non-fusable back-to-back matrix multiplications keep the operation memory-bound.

Key insight: `f(X_sum) · S` is mathematically equivalent to cross-attention where the sequence is the query and the MLP weight matrices become keys and values. GDPA exploits this:

```
GDPA_h(Q, K, V) = Activation_h(Q K^T / τ) V

  Q  =  S^(l)              sequence tokens (queries)
  K  =  w1_h(X_sum^(l))    K projection from non-seq summary, per head h
  V  =  w2_h(X_sum^(l))    V projection from non-seq summary, per head h
  τ  =  maxlen(seq)         temperature (better than 1/√d for recs)
```

This single attention operator is fusable into a FlashAttention-style kernel, making it compute-bound. Residual connection enables stable stacking. **Result: 6× MFU improvement on PFFN.**

**HSP — Hierarchical Seed Pooling** (§4.3.2)

Prior PMA uses randomly initialized learnable queries to pool a sequence into tokens. HSP improves this with three stages:

1. **Overcomplete seeds**: `E_seed ∈ ℝ^(n_seeds × d)`, with `n_seeds > n_tokens` — more seeds than needed gives better initialization stability vs. direct random queries.
2. **Seed-level attention**: `H_seed = MHA(Norm(E_seed), S, S)` — seeds attend to the full sequence.
3. **SumKronLinear compression**: `H_seed → H_summary` via Kronecker decomposition.

SumKronLinear compresses `S` seeds to `T` output tokens:

```
Y_b = Σ_{i=1}^k  Z_i^T X_b W_i,    Z_i ∈ ℝ^(S×T),  W_i ∈ ℝ^(D×D)
```

- **14× fewer parameters** than full linear (S=256, T=32, D=384, k=8): `O(k·(ST + D²))` vs `O(S·D·T·D)`.
- Captures **cross-dimensional correlations** (joint S×D structure), unlike separable rank-1 factorization.
- Scales favorably: parameter savings grow as D increases.

**Sliding Window Attention** (§4.3.3)

Full self-attention is O(T²). For T > 1000, this becomes prohibitive. SWA restricts each position t to attend within `[t−w, t+w]`:

```
Attention(Q_t, K, V) = softmax(Q_t K[t−w:t+w]^T / √d) V[t−w:t+w]
```

Motivated by temporal locality bias: recent interactions have significantly higher predictive value than distant history even at adjacent positions. **Result: 31.1% QPS improvement, 29.5% FLOPs reduction, NE-neutral.**

**ROTE — Rotary Temporal Embeddings** (§4.2)

Standard RoPE treats token *position* as the primary signal. In recommendation sequences, the *temporal gap* between events matters more — a click yesterday and a click a month ago differ fundamentally, even if adjacent in the sequence. ROTE encodes log-scaled gaps:

```
τ_t = log(1 + Δt / t_scale)
```

Applied via the standard rotary rotation matrix, combining positional and temporal frequencies.

---

### High-level: Computation Reallocation

**CompSkip — Computation Skip** (§4.4.1)

Layer redundancy analysis (inspired by LLM findings, Men et al., 2024) shows that not every layer needs every operation. CompSkip implements an **every-other-layer alternating pattern**:

| Layer | Transformer Block | Interaction Block | What it learns |
|---|---|---|---|
| Even (0, 2, 4, ...) | GDPA (personalized PFFN) | Fresh HSP | Global: non-seq context → seq understanding |
| Odd  (1, 3, 5, ...) | SWA (local self-attention) | Reuse cached HSP | Local: within-sequence dependencies |

Each layer now does one thing well, rather than both things at half capacity. **Result: 43.1% FLOPs reduction, 35% QPS improvement, NE-neutral.**

**Event-Level Personalization** (§4.4.2)

Different event sequence types have different characteristics and importance. Clicks provide stronger signals than impressions; purchase sequences are shorter but higher-value. Personalization allocates compute proportional to importance:

```python
CLICK_CONFIG      = EventConfig(d_model=256, n_heads=8, n_tokens=32, n_layers=3, window=100)
IMPRESSION_CONFIG = EventConfig(d_model=128, n_heads=4, n_tokens=16, n_layers=2, window=50)
```

**Result: 13% QPS improvement, 11% FLOPs reduction.**

**Mixture of Wukong Experts** (§4.4.3)

The Global Interaction module uses M Wukong experts running in parallel, each processing a partition of `X_global = Concat(X_sum, H_summary)`. Each Wukong expert captures complementary interaction patterns:
- **DOT product**: linear/pairwise interactions via inner-product attention.
- **Deep MLP**: hierarchical interactions via a 2-layer network.

Expert parallelism enables horizontal scaling (more experts per layer) with minimal communication overhead. **Result: 4% additional QPS improvement.**

---

## Summary

The core problem: existing models that jointly handle sequential user history and non-sequential context features exhibit poor *scaling efficiency* — performance improves slowly per unit of compute. Kunlun identifies two root causes and fixes each:

**Problem 1 — Inefficient modules**: 3–15% MFU due to heterogeneous feature spaces, irregular tensor shapes, and memory-bound operations.

**Problem 2 — Inefficient resource allocation**: Naively scaling all components equally leads to diminishing returns.

**Solution — Model-efficiency codesign:**

| Level | Innovation | Gain |
|---|---|---|
| Low-level | **GDPA**: reformulates PFFN as fused multi-head attention | 6× MFU on PFFN block |
| Low-level | **HSP**: hierarchical seed pooling via SumKronLinear | Better quality than PMA |
| Low-level | **Sliding Window Attention**: O(Tw) instead of O(T²) | 31% QPS improvement |
| High-level | **CompSkip**: every-other-layer alternation | 43.1% FLOPs reduction, 35% QPS |
| High-level | **Event-Level Personalization**: per-event-type configs | 13% QPS improvement |
| High-level | **Mixture of Wukong Experts**: expert parallelism | 4% QPS improvement |

Together these bring MFU from 17% to 37% and achieve 2× scaling efficiency over the prior SOTA (InterFormer).

---

## Architecture

```
Non-seq features X^(0) [dense, sparse]
    │  EmbeddingLayer
    ▼
X^(1) ∈ ℝ^(B, n_ns, d)        seq_proj + ROTE
                                     │
                                S^(1) ∈ ℝ^(B, T, d)

for l = 0 .. L-1:

  ┌─ KunlunInteractionBlock ─────────────────────────────────────────────┐
  │  X_sum^(l) = WeightGeneration(X^(l))      [MLP token-axis compress]  │
  │  if even: H_summary = HSP(S^(l))          [fresh pooling]            │
  │  if odd:  H_summary = cache               [CompSkip reuse]           │
  │  X^(l+1) = GlobalInteraction(X_sum ∥ H_summary)  [Wukong MoE]       │
  └──────────────────────────────────────────────────────────────────────┘
  ┌─ KunlunTransformerBlock ─────────────────────────────────────────────┐
  │  if even: S^(l+1) = GDPA(S^(l), X_sum)   [personalized PFFN]        │
  │  if odd:  S^(l+1) = LN(SWA(S^(l)) + S)   [local SWA refinement]     │
  └──────────────────────────────────────────────────────────────────────┘

ŷ = σ(MLP(flatten(X^(L))))
```

**Bidirectional information flow** (inherited from InterFormer, §2.1):
- Non-seq → Seq: `X_sum` guides `GDPA` in the Transformer Block.
- Seq → Non-seq: `H_summary` from `HSP` feeds `GlobalInteraction`.

---

---

## Usage

```python
from kunlun import Kunlun, CLICK_CONFIG, IMPRESSION_CONFIG

model = Kunlun(
    dense_dim      = 256,         # total dim of dense features
    sparse_dims    = [1_000_000, 500_000],   # vocabulary sizes
    seq_input_dim  = 64,          # per-step sequence embedding dim
    d_model        = 128,         # global embedding dimension
    num_layers     = 4,           # L — CompSkip alternates every 2 layers
    n_sum          = 8,           # WeightGeneration output token count
    n_seeds        = 64,          # HSP seed count (must be > n_tokens)
    n_tokens       = 16,          # HSP output token count
    n_heads        = 8,
    window         = 100,         # SWA half-window size w
    n_experts      = 2,           # Wukong expert count
    kron_rank      = 8,           # SumKronLinear rank k
    max_seq_len    = 1000,        # GDPA temperature τ
    top_mlp_dims   = [256, 128],
    num_tasks      = 1,           # 1 for pCTR, 2 for pCTR+pCVR
)

logits = model(
    dense_feat   = torch.randn(B, 256),
    sparse_feats = [torch.randint(0, v, (B,)) for v in [1_000_000, 500_000]],
    seq_feat     = torch.randn(B, T, 64),
    timestamps   = None,          # (B, T) seconds; None → positional fallback
)  # → (B, num_tasks)
```

### Hyperparameter guidance

| Parameter | Typical value | Notes |
|---|---|---|
| `d_model` | 128–256 | Primary capacity knob |
| `num_layers` L | 3–6 | Each layer adds ~log(l+1) NE gain |
| `n_sum` | 4–16 | Compressed non-seq token count |
| `n_seeds` | 4×n_tokens | Overcomplete seeds for HSP |
| `n_tokens` | 16–32 | Sequence summary length |
| `window` | 50–200 | SWA receptive field |
| `kron_rank` k | 8 | Higher k → more SumKronLinear expressiveness |
| `n_experts` M | 2–4 | Wukong expert parallelism |

---

## Implementation

### What is implemented

Every component described in the paper is implemented as a standalone `nn.Module`:

| Module | Paper Section | Description |
|---|---|---|
| `ROTE` | §4.2 | Log-scaled temporal rotary embeddings. Falls back to position indices when no timestamps are provided. |
| `GDPA` | §4.3.1 | Multi-head cross-attention PFFN. Sequence is the query; non-seq summary provides K and V. Includes residual. |
| `SlidingWindowAttention` | §4.3.3 | Builds an explicit T×T attention mask for window `[t−w, t+w]`. Falls back to full attention when `T ≤ 2w+1`. |
| `SumKronLinear` | §4.3.2 | Kronecker compression `Y = Σ Z_i^T X W_i`. Parameters: `k×(S×T + D×D)`. |
| `HSP` | §4.3.2 | 3-stage: learnable seeds → MHA seed-attend → SumKronLinear compress. |
| `WeightGeneration` | §4.1 | Token-axis linear compression + 2-layer MLP to derive X_sum. |
| `KunlunTransformerBlock` | §4.1, §4.3 | GDPA on even layers, SWA+LN on odd layers. Driven by `is_even_layer`. |
| `WukongExpert` | §4.4.3 | Inner-product attention (DOT) + deep MLP + residual + LayerNorm. |
| `GlobalInteraction` | §4.4.3 | Partitions `X_global` across M Wukong experts; handles uneven splits. |
| `KunlunInteractionBlock` | §4.1, §4.4 | WeightGen + conditional HSP (fresh/cached) + GlobalInteraction. |
| `KunlunLayer` | §4.5 | Wraps InteractionBlock + TransformerBlock; manages CompSkip routing. |
| `EventConfig` | §4.4.2 | Dataclass for per-event-type architecture parameters. |
| `Kunlun` | §4 | Full model: preprocessing, ROTE, L layers with CompSkip cache, prediction head. |

### Design decisions and simplifications

**GDPA temperature:** Uses `τ = max_seq_len` (a constructor argument) rather than computing the actual runtime maximum sequence length. The paper recommends `maxlen(seq)` over `1/√d` empirically. Setting `max_seq_len` to your training sequence cap is correct.

**SlidingWindowAttention memory:** This reference implementation builds an explicit O(T²) attention mask. A production deployment would use a custom Triton/CUDA kernel for true O(Tw) memory. The computation is still windowed correctly — only memory is O(T²) in this reference.

**SumKronLinear initialization:** `Z` and `W` matrices are initialized with `trunc_normal(std=0.02)`, following the standard transformer initialization convention. The paper does not specify initialization.

**GlobalInteraction partitioning:** Tokens are partitioned evenly across experts (ceiling division). If `n_global` is not divisible by `n_experts`, the last expert gets fewer tokens. This matches the paper's description ("designated feature partition").

**CompSkip cache:** The main `Kunlun.forward` loop caches `H_summary` from each even layer and passes it to the next odd layer. The `KunlunInteractionBlock` uses `is_even_layer=False` + a non-None cache to reuse without recomputing HSP.

**Event-Level Personalization:** `EventConfig` and the two example configs (`CLICK_CONFIG`, `IMPRESSION_CONFIG`) are provided for constructing per-event-type `Kunlun` instances. The paper's full multi-event architecture (separate sequence towers combined at the Interaction Block level) is outside the scope of this single-model reference implementation.

**Non-seq token count after layer 0:** The first layer takes `n_ns = 1 + len(sparse_dims)` tokens; all subsequent layers take `n_global = n_sum + n_tokens` tokens (the shape of `X^(l+1)` output by `GlobalInteraction`). `Kunlun.__init__` creates each `KunlunLayer` with the correct `n_ns`.

### Relation to other models in this directory

| Model | Handles sequences? | Handles non-seq? | Bidirectional? | Scalable? |
|---|---|---|---|---|
| Wukong | No | Yes | N/A | Yes (FM stacking) |
| InterFormer | Yes | Yes | Yes | Limited MFU |
| **Kunlun** | **Yes** | **Yes** | **Yes** | **Yes (2× efficiency)** |

Kunlun builds on the InterFormer framework (Zeng et al., 2024) and uses Wukong modules (Zhang et al., 2024) inside its Global Interaction component.

---

## Files

```
kunlun/
├── kunlun.py           # full implementation + smoke test
│   ├── ROTE                    # log-scaled temporal rotary embeddings
│   ├── GDPA                    # multi-head cross-attention PFFN
│   ├── SlidingWindowAttention  # O(Tw) local self-attention
│   ├── SumKronLinear           # Kronecker compression (HSP Stage 3)
│   ├── HSP                     # hierarchical seed pooling (3 stages)
│   ├── WeightGeneration        # derives X_sum for GDPA + GlobalInteraction
│   ├── KunlunTransformerBlock  # GDPA (even) or SWA (odd) via CompSkip
│   ├── WukongExpert            # DOT + deep interaction expert
│   ├── GlobalInteraction       # Mixture of Wukong Experts
│   ├── KunlunInteractionBlock  # WeightGen + HSP/cache + GlobalInteraction
│   ├── KunlunLayer             # one InteractionBlock + TransformerBlock
│   ├── EventConfig             # per-event-type configuration dataclass
│   └── Kunlun                  # full model with ROTE, CompSkip, prediction head
├── test_kunlun.py      # pytest test suite (68 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 kunlun.py
```

Expected output:
```
logits:   torch.Size([4, 1])  [-0.385..., -0.372..., -0.376..., -0.379...]
loss:     0.5217
backward: OK
params:   206,721

CompSkip layer pattern:
  l=0: GDPA + fresh HSP
  l=1: SWA  + cached HSP
  l=2: GDPA + fresh HSP
  l=3: SWA  + cached HSP
```

### Test suite

```bash
python3 -m pytest test_kunlun.py -v
```

Expected: `68 passed`.
