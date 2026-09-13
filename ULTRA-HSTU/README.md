# ULTRA-HSTU

PyTorch reference implementation of **ULTRA-HSTU**, from *"Bending the Scaling Law Curve in Large-Scale Recommendation Systems"* (Meta Recommendation Systems, Feb 2026).

Paper: https://arxiv.org/abs/2602.16986

Deployed with 18 layers of self-attention over 16k-length user behavior sequences on hundreds of H100 GPUs, serving billions of users daily: **5.3x** faster training scaling efficiency and **21.4x** faster inference scaling efficiency than vanilla HSTU, with **4%-8%** online consumption/engagement gains and a **0.217%** topline metric lift.

---

## Summary

[`HSTU`](../HSTU) established that replacing hand-engineered feature-crossing with self-attention over a unified user-interaction-history (UIH) sequence gives recommendation models the same favorable compute scaling laws large language models enjoy. The catch: self-attention is `O(L^2)` in sequence length `L`, and production UIH sequences run to `O(10k)`-`O(100k)` events — quickly making full self-attention unaffordable. Prior industry work (STCA, Longer) responds by replacing self-attention with cheaper cross-attention against only the ranking candidates. This paper shows that trade loses real model quality once self-attention is stacked deeper or scaled up (their Table 1/3/5) — cross-attention's improvements *saturate* with depth, while self-attention's keep compounding.

ULTRA-HSTU's answer is to keep self-attention but bend its cost curve via three **stackable, complementary model changes** rather than by weakening the attention mechanism itself:

1. **Input sequence optimization** — merge item+action pairs into one token instead of interleaving them, halving `L` before attention ever runs.
2. **Semi-Local Attention (SLA)** — a sparse causal mask combining a short local window with a small set of global anchor positions, giving `O((K1+K2)*L)` attention cost instead of `O(L^2)`.
3. **Attention Truncation** — a dynamic topological design: run most layers over the full sequence, but only the last few layers over a short, most-recent segment, avoiding `O(depth * L)` cost for deep stacks.

---

## Key Ideas

### Problem: self-attention's `O(L^2)` cost is the real bottleneck, not self-attention itself

The paper's central empirical claim (Sec 1, Table 1/3/5) is that **self-attention remains superior to cross-attention** for this problem, even though cross-attention is asymptotically cheaper — cross-attention-based methods like STCA "achieve linear complexity" but "introduce performance regressions... without self-attention," and this gap *grows* with model depth/scale (App. B, Table 5: cross-attention quality saturates by ~9 layers, self-attention keeps improving through 12+). So the paper frames the problem as: **how do we make self-attention itself cheap enough**, rather than replacing it.

### ULTRA-HSTU's solution: three orthogonal cost reductions

```
X = Norm(Z)                                                    [Eq. 1]
U, Q, K, V = phi1(f1(X))                                       [Eq. 2]
A = (phi2(QK^T) elementwise* M) V                               [Eq. 3]
Y = f2(Norm(A) elementwise* U)                                 [Eq. 4]
Z_out = Y + Z                                                  [Eq. 5]
```

This is HSTU's pointwise-aggregated-attention block (`phi1 = phi2 = SiLU`, no softmax) in a pre-norm formulation, where `M` is now a *sparse* mask rather than a dense causal one.

| Piece | Cost reduction | Mechanism |
|---|---|---|
| **Input sequence optimization** (Sec 4.1) | `-32.5%` train / `-63.5%` inference FLOP at `L=3072` | `x_i = item_i + action_i` (addition, not interleaving) halves the effective sequence length; heterogeneous action encodings (implicit + explicit + context signals, summed) recover the `+0.45%` C-NE this would otherwise cost |
| **Semi-Local Attention** (Sec 4.2.1, Eq. 6-7) | `O((K1+K2)*L)` vs `O(L^2)`; `5x` inference scaling efficiency | Causal mask = local sliding window `K1` (recency) **union** global anchor positions `K2` (long-range summary). The paper finds *both* are necessary: local-only regresses `0.35%` C-NE, global-only regresses `0.03%` C-NE |
| **Attention Truncation** (Sec 4.3, Fig. 2d) | avoids `O(depth * L)` for deep stacks; `1.8x` inference scaling exponent improvement | Run `N1` layers over the full `L`-length sequence, then select the latest `L'` positions and run `N2` more layers only on that short segment — simplest of several segment-selection strategies tried, and empirically the best |

Two further system-level techniques from the paper are **out of scope** for this reference implementation (hardware/distributed-training concerns, not modeling contributions): the FP8/INT4 mixed-precision GEMM + custom FlashAttention-3-style SLA kernels (Sec 4.2.2), and the distributed Load-Balanced Stochastic Length training scheduler (Algorithm 1, Appendix C).

---

## Architecture

```
Merged UIH sequence: x_0, x_1, ..., x_(L-1),  x_i = item_i + action_i
    ▼
Z^(0) ∈ R^(B, L, D)

for l = 1..N1:                                          [UltraHSTULayer, full sequence]
    X = Norm(Z^(l-1))                                                [Eq. 1]
    U,Q,K,V = SiLU(f1(X))                                            [Eq. 2]
    A = SiLU(QK^T) elementwise* SLA_mask(L, K1, K2)                  [Eq. 3]
    Y = f2(Norm(A.V) * U)                                            [Eq. 4]
    Z^(l) = Y + Z^(l-1)                                              [Eq. 5]

h_full = Z^(N1)                                          [(B, L, D)]
h_trunc0 = h_full[:, -L':, :]                            [Attention Truncation, Fig. 2d]

for l = 1..N2:                                           [UltraHSTULayer, truncated segment]
    h_trunc = UltraHSTULayer(h_trunc, SLA_mask(L', K1, K2))

final_repr = h_trunc[:, -1, :]   (or h_full[:, -1, :] if N2 == 0)
    ▼
RankingHead(final_repr) -> multi-task logits (e.g. C-NE consumption, E-NE engagement)
```

---

## Key Components

### 1. Input sequence optimization (`build_uih_sequence`, `HeterogeneousActionEncoder`)

`x_i = item_i + action_i` merges each item/action pair into one sequence position instead of two, halving `L`. Action embeddings at positions that are themselves ranking candidates are masked to zero, per the paper's note that this prevents label leakage. `HeterogeneousActionEncoder` builds the action embedding itself from multiple signals (explicit engagement type, an implicit intensity bucket, a context bucket) summed together, rather than a single action-type id.

### 2. Semi-Local Attention mask (`semi_local_attention_mask`, `attention_complexity`)

```
keep(i, j) = causal(i, j)  AND  ( local(i, j)  OR  is_global_anchor(j) )
  causal(i, j)          = j <= i
  local(i, j)           = i - j < K1
  is_global_anchor(j)   = j < K2
```

Gives `O((K1+K2)*L)` nonzero attention entries instead of full causal attention's `O(L^2/2)`. This repo anchors the `K2` global window at the sequence's earliest positions (a fixed always-visible prefix, in the spirit of Longformer/BigBird global tokens) to realize the paper's stated linear complexity; the paper's own Eq. 7 index convention is underspecified in isolation, so treat the exact anchor placement as this repo's interpretation of the *mechanism*, not a byte-exact reproduction. `attention_complexity` reports the resulting sparsity, e.g. at `L=2048, K1=32, K2=8` the mask keeps under 4% of entries a full causal mask would.

### 3. `UltraHSTULayer` — Eq. 1-5

Pre-norm HSTU block: normalize the residual stream, project to `U,Q,K,V` via one `SiLU(Linear(.))`, compute pointwise-activated attention logits gated elementwise by the SLA mask (no softmax, no additive relative-position bias — this paper's formulation folds masking directly into `phi2(QK^T)`), then a `LayerNorm`-gated output projection, then residual add.

### 4. `UltraHSTU` — SLA + Attention Truncation encoder

Stacks `num_layers_full` `UltraHSTULayer`s over the full sequence, then (if `num_layers_truncated > 0`) slices out the latest `truncate_len` positions and stacks further layers over just that segment. `final_representation` reads off the last position of whichever stage ran last — the representation actually used for the paper's ranking read-off.

---

## Usage

```python
from ultra_hstu import (
    build_uih_sequence, HeterogeneousActionEncoder,
    UltraHSTU, RankingHead,
)

# 1. Build the merged item+action UIH sequence.
item_emb = torch.randn(B, L, 64)
action_emb = torch.randn(B, L, 64)
x = build_uih_sequence(item_emb, action_emb)   # (B, L, 64), half the length of interleaving

# 2. SLA-only encoder (no depth-scaling trick).
model = UltraHSTU(
    d_model=64, num_heads=4, dqk=16, dv=16,
    num_layers_full=8, local_window=256, global_window=32,
)
h_full, h_trunc = model(x)      # h_trunc is None

# 3. SLA + Attention Truncation (deep stack, cheap tail layers).
model = UltraHSTU(
    d_model=64, num_heads=4, dqk=16, dv=16,
    num_layers_full=12, num_layers_truncated=6,
    local_window=256, global_window=32, truncate_len=512,
)
h_full, h_trunc = model(x)      # h_full: (B, L, 64), h_trunc: (B, 512, 64)

head = RankingHead(d_model=64, hidden_dims=[64], num_tasks=2)  # e.g. C-NE, E-NE
logits = head(model.final_representation(h_full, h_trunc))
```

Heterogeneous action encoding:

```python
action_encoder = HeterogeneousActionEncoder(
    d_model=64, num_action_types=8, num_intensity_buckets=10, num_context_buckets=24,
)
action_emb = action_encoder(action_type_ids, intensity_bucket_ids, context_bucket_ids)
x = build_uih_sequence(item_emb, action_emb, is_candidate=is_candidate_mask)
```

---

## Files

```
ULTRA-HSTU/
├── ultra_hstu.py         # full implementation + smoke test
│   ├── build_uih_sequence           # Sec 4.1: merges item+action into one token
│   ├── HeterogeneousActionEncoder   # Sec 4.1: multi-signal action embedding
│   ├── semi_local_attention_mask    # Eq. 6-7: local-window + global-anchor sparse mask
│   ├── attention_complexity         # nnz/sparsity comparison vs full causal attention
│   ├── UltraHSTULayer                # Eq. 1-5: one pre-norm, mask-gated HSTU block
│   ├── UltraHSTU                     # stack of layers; SLA + Attention Truncation (Fig. 2d)
│   └── RankingHead                   # small MLP for multi-task ranking read-off
├── test_ultra_hstu.py    # pytest test suite (35 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 ultra_hstu.py
```

Expected output (values vary slightly by seed):

```
--- Input sequence optimization ---
merged UIH sequence: torch.Size([2, 256, 32])  (vs. 512 tokens if interleaved)
heterogeneous action embedding: torch.Size([2, 256, 32])

--- Semi-Local Attention complexity ---
L=  256  SLA nnz=   9460  full nnz=   32896  sparsity=0.2876
L=  512  SLA nnz=  19700  full nnz=  131328  sparsity=0.1500
L= 1024  SLA nnz=  40180  full nnz=  524800  sparsity=0.0766
L= 2048  SLA nnz=  81140  full nnz= 2098176  sparsity=0.0387

--- ULTRA-HSTU forward (SLA only) ---
h_full: torch.Size([2, 256, 32]), h_trunc: None

--- ULTRA-HSTU forward (SLA + Attention Truncation) ---
h_full: torch.Size([2, 256, 32]), h_trunc: torch.Size([2, 64, 32])
logits: torch.Size([2, 2])  loss: 0.6535  backward: OK
params: 27,040
```

### Test suite

```bash
python3 -m pytest test_ultra_hstu.py -v
```

Expected output:

```
collected 35 items
...
35 passed in ~1s
```

Useful variants:

```bash
python3 -m pytest test_ultra_hstu.py::TestUltraHSTU -v
python3 -m pytest test_ultra_hstu.py -x
```
