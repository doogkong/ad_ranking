# RankMixer

PyTorch reference implementation of **RankMixer: Scaling Up Ranking Models in Industrial Recommenders** (ByteDance, Jul 2025).

Paper: https://arxiv.org/abs/2507.15551

Deployed on Douyin feed ranking and ads at **1.1B dense parameters** (70x the prior production model) at roughly flat inference latency, lifting active days **+0.3%**, in-app duration **+1.08%**, and ads AUC **+0.73%**.

---

## Summary

The key insight is that **self-attention is the wrong primitive for ranking feature interaction**. Attention scores heterogeneous, high-cardinality ID feature subspaces against each other with an inner product — a comparison that doesn't mean much across incomparable semantic spaces, and one that is memory-bound (attention-weight matrices) rather than compute-bound on GPUs. RankMixer replaces it with two scalable pieces per layer:

- **Multi-head Token Mixing**: a *parameter-free* reshape/transpose that shuffles slices of every token across every other token position — global mixing without any learned similarity function.
- **Per-token FFN (PFFN)**: instead of one FFN shared by all tokens (or one expert shared by all inputs, as in MMoE), every token index gets its *own* private MLP — isolating capacity per feature subspace so high-frequency fields can't drown out long-tail ones.

A **Sparse-MoE** variant of the PFFN (ReLU routing + a Dense-Training/Sparse-Inference trick) pushes capacity to 1B+ parameters with near-flat serving cost.

---

## Key Ideas

### Problem: CPU-era feature-crossing modules don't scale on GPUs

Most ranking-model interaction layers (DCN, AutoInt, DHEN, ...) were designed when compute cost scaled with parameter count. On modern GPUs their core operators are memory-bound, so Model FLOPs Utilization (MFU) sits in the single digits — scaling them up buys little. Self-attention specifically struggles because:

1. Ranking features span hundreds of heterogeneous, high-cardinality ID subspaces (user IDs, item IDs, ...) — an inner-product similarity across such spaces is not well-defined the way it is for a shared NLP token embedding space.
2. Attention's O(T^2) score matrix is expensive in both compute and memory IO for the token counts used at this scale.
3. Sharing one FFN across heterogeneous feature groups lets high-frequency fields dominate, starving long-tail signals of capacity.

### RankMixer's solution

| Component | What it does |
|---|---|
| **Feature Tokenizer** | Groups hundreds of raw features into T semantically coherent groups, each projected to a shared width D |
| **Multi-head Token Mixing** | Splits each token into H=T heads and transposes token/head axes — every output token now contains a slice of every input token, at zero parameter cost |
| **Per-token FFN (PFFN)** | Gives every token index its own 2-layer MLP — capacity scales with #tokens without inflating FLOPs per token |
| **Sparse-MoE PFFN** | Replaces each token's dense PFFN with a bank of ReLU-routed experts for further capacity at near-constant compute |

**Bottom line**: keep the Transformer's parallelism and residual-block structure, drop the expensive learned similarity function, and spend the saved compute budget on more *isolated* per-token parameters instead.

---

## Architecture

```
Raw features (hundreds of embeddings: user, candidate, sequence-summary, cross)
    │  group into T semantic clusters, concat within each group
    ▼
FeatureTokenizer:  x_i = Proj(group_i)             [Eq. 2]
    ▼
X^(0) ∈ R^(B, T, D)

for n = 1..L:                                       [Eq. 1]
    ┌───────────────────────────────────────────────────────────┐
    │  Multi-head Token Mixing (parameter-free)                 │
    │    S^(n-1) = LN( TokenMixing(X^(n-1)) + X^(n-1) )   [Eq.3-5]│
    ├───────────────────────────────────────────────────────────┤
    │  Per-token FFN (dense or Sparse-MoE)                       │
    │    X^(n)   = LN( PFFN(S^(n-1)) + S^(n-1) )          [Eq.6-9]│
    └───────────────────────────────────────────────────────────┘

o_output = mean_pool(X^(L))
y_hat    = task_head(o_output)
```

---

## Key Components

### 1. Feature Tokenization (§3.2, Eq. 2)

```
x_i = Proj(e_input[d·(i-1) : d·i]),  i = 1..T
```

Groups hundreds of raw feature embeddings into T semantically coherent clusters (domain-knowledge driven), then projects each cluster to the shared model width D. Balances two failure modes: one-token-per-feature (thousands of tiny fragments, poor GPU utilization) versus one giant token (degenerates into a plain DNN, dominant features overshadow others).

### 2. Multi-head Token Mixing (§3.3.1, Eq. 3-5)

```
[x_t^(1) ‖ ... ‖ x_t^(H)] = SplitHead(x_t)
s^h = Concat(x_1^h, x_2^h, ..., x_T^h)
```

Each D-dim token is split into H=T heads of size D/H, then heads are gathered across token positions: the h-th output token concatenates the h-th head of *every* input token. Implemented as a single reshape + transpose — no learned parameters. This is the operator that replaces self-attention: the paper's ablation shows swapping it back in for self-attention loses a small amount of AUC while costing +72% FLOPs and +16% params, and removing it entirely (no cross-token mixing at all) is the single largest ablation loss (-0.50% AUC).

### 3. Per-token FFN — PFFN (§3.3.2, Eq. 6-9)

```
v_t = W2_t · Gelu(W1_t · s_t + b1_t) + b2_t
```

Every token index t owns its own weight matrices — distinct from a Transformer FFN (one MLP shared by all tokens) and from an MMoE expert (all experts see the same input). Parameters scale with #tokens; FLOPs per token stay the same as a shared FFN of equal width.

### 4. Sparse-MoE PFFN (§3.4, Eq. 10-11)

```
G_{t,j} = ReLU(h_t(s_t))_j
v_t     = Σ_j G_{t,j} · expert_{t,j}(s_t)
L = L_task + λ·L_reg,   L_reg = Σ_t Σ_j G_{t,j}
```

Replaces each token's single dense PFFN with a bank of experts. A **ReLU gate** (instead of Top-k + softmax) lets each token activate a variable number of experts — more for high-information tokens, fewer for low-information ones — while an L1 penalty on the gate values steers the average sparsity toward a budget. The paper additionally trains two routers — a dense `h_train` (no penalty, so gradients reach every expert and none starve) and a sparse `h_infer` (penalized, used only at serving time) — a **Dense-Training/Sparse-Inference (DTSI)** scheme; this repo's `dtsi=True` option is a reference approximation of that idea (dense router drives the forward pass in training, sparse router is distilled toward it and used at eval), not a reproduction of the exact production recipe.

### 5. Scaling axes (§3.5, Eq. 12)

RankMixer can be scaled along four independent axes — token count T, width D, layers L, expert count E:

```
#Param ≈ 2·k·L·T·D²      FLOPs ≈ 4·k·L·T·D²      (dense variant, k = PFFN expand ratio)
```

The paper finds quality tracks total parameter count regardless of *which* axis you scale (T, D, or L) — consistent with LLM-style scaling laws — while scaling width D reaches higher MFU than stacking more layers, because it produces larger GEMM shapes.

---

## Usage

```python
from rankmixer import RankMixer

# T=8 semantic feature groups with heterogeneous raw dims (user/candidate/seq/cross, etc.)
group_dims = [16, 8, 32, 4, 12, 20, 6, 10]

model = RankMixer(
    group_dims=group_dims,
    d_model=64,        # D
    num_layers=2,       # L
    expand_ratio=4,     # k, PFFN hidden = k*D
    moe=False,          # set True for the Sparse-MoE PFFN variant
    top_mlp_dims=[64],
    num_tasks=1,        # 1 for pCTR, 2+ for multi-task
)

groups = [torch.randn(B, g) for g in group_dims]  # one raw tensor per feature group
logits = model(groups)                             # (B, num_tasks)

loss = F.binary_cross_entropy_with_logits(logits, targets)
loss.backward()
```

Sparse-MoE variant:

```python
model = RankMixer(group_dims, d_model=64, num_layers=2,
                   moe=True, num_experts=4, dtsi=True, num_tasks=1)
model.train()
logits = model(groups)
loss = F.binary_cross_entropy_with_logits(logits, targets) + 1e-3 * model.moe_reg_loss()
loss.backward()
```

### Hyperparameter guidance

| Parameter | Paper's 100M config | Paper's 1B config | Notes |
|---|---|---|---|
| `d_model` (D) | 768 | 1536 | model width; larger D reaches higher MFU (bigger GEMM shapes) than more layers |
| `num_layers` (L) | 2 | 2 | paper finds depth/width/tokens scale ~interchangeably in quality |
| Token count (T) | 16 | 32 | must divide `d_model` (H = T heads) |
| `expand_ratio` (k) | — | — | PFFN hidden-dim multiplier; controls #Param/FLOPs via Eq. 12 |
| `num_experts` (MoE) | — | — | paper scales to >8x sparsity with ~no AUC loss via DTSI + ReLU routing |

---

## Files

```
RankMixer/
├── rankmixer.py         # full implementation + smoke test
│   ├── FeatureTokenizer        # groups raw features into T aligned D-dim tokens (Eq. 2)
│   ├── TokenMixing             # parameter-free head-transpose shuffle (Eq. 3-5)
│   ├── PerTokenFFN             # dense per-token-private 2-layer MLP (Eq. 6-9)
│   ├── SparseMoEPerTokenFFN    # ReLU-routed MoE variant + DTSI approximation (Eq. 10-11)
│   ├── RankMixerBlock          # one TokenMixing + PFFN block with residual + LN (Eq. 1)
│   └── RankMixer               # full model: tokenizer → L blocks → mean pool → task head(s)
├── test_rankmixer.py     # pytest test suite (42 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 rankmixer.py
```

Expected output:

```
logits:   torch.Size([4, 1])  [...]
loss:     0.7982
backward: OK
params:   818,817
Eq.12 estimate (PFFN-only, no tokenizer/head): params~=786,432 flops~=1,572,864

--- Sparse-MoE variant ---
logits:      torch.Size([4, 1])
reg_loss:    ...
backward:    OK
```

### Test suite

```bash
python3 -m pytest test_rankmixer.py -v
```

Expected output:

```
collected 42 items
...
42 passed in ~2s
```

Useful variants:

```bash
python3 -m pytest test_rankmixer.py::TestSparseMoEPerTokenFFN -v
python3 -m pytest test_rankmixer.py -x
```
