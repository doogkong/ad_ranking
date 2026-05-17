# Meta Lattice

PyTorch reference implementation of **Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations** (Meta Platforms, Dec 2025).

Paper: https://arxiv.org/abs/2512.09200

Deployed at Meta Ads: **+10% topline NE**, **+11.5% user satisfaction**, **+6% CVR**, **+20% capacity savings** over the baseline.

---

## Paper Summary

Large-scale ads recommendation systems must simultaneously handle **multiple domains** (Feed, Reels, Stories) and **multiple objectives** (CTR, CVR, quality). Maintaining separate specialized models for each domain-objective combination is economically unsustainable and operationally complex. Meta Lattice establishes a *unified model space* that consolidates this entire heterogeneous ecosystem into a single scalable architecture.

Three root challenges block cost-effective multi-domain multi-objective (MDMO) recommendation:

**Challenge 1 — Economic scalability.** Separate models for each (domain, objective) pair multiply infrastructure costs (training, serving, iteration) without proportional quality gains. As the business expands to new surfaces, costs grow super-linearly.

**Challenge 2 — Data fragmentation.** Conversion attribution windows differ across models (90-minute vs. 1-day vs. 7-day windows produce conflicting training labels for the same impression). Combining these datasets naively causes label noise and training instability.

**Challenge 3 — Feature set conflict.** Different domains surface different input features. Naively taking the union of all features creates noise; discarding features loses signal. No principled way to select a shared feature set across competing objectives existed.

**Challenge 4 — Deployment constraints.** Unified models must share the same hardware at inference time. Interference between domain-specific and objective-specific computation degrades quality without careful architectural design.

### How does Meta Lattice solve it?

Lattice introduces a **model space redesign** across three integrated components:

| Component | Challenge Targeted | Key Idea |
|---|---|---|
| **Lattice Partitioner** | Economic scalability | Portfolio consolidation — identify which models to merge; merge only when beneficial |
| **Lattice Zipper** | Data fragmentation | K window heads on shared backbone; train by impression hash routing; oracle head at serving |
| **Lattice Filter** | Feature set conflict | Pareto-optimal feature selection across all N tasks simultaneously |
| **Lattice Networks** | Architecture unification | Three-stage preprocessor-backbone-task with domain-specific FFNs and ECS residuals |
| **Lattice KTAP** | Deployment constraints | Async teacher precompute; student queries teacher KV cache at inference |

### Measured results

| Component | Metric | Change |
|---|---|---|
| Lattice Partitioner | Models in portfolio | −40% (consolidation) |
| Lattice Zipper | Training stability | Eliminates label conflicts across windows |
| Lattice Filter | Feature set size | Principled reduction while retaining signal |
| SwishRMSNorm | Training stability | Avoids catastrophic cancellation vs. LayerNorm |
| Bias-less layers | Embedding stability | Prevents unbounded drift in deep nets |
| Parameter untying | Cross-domain interference | Domain-specific FFNs prevent gradient conflict |
| **Full Meta Lattice** | Topline NE | **+10%** |
| **Full Meta Lattice** | User satisfaction | **+11.5%** |
| **Full Meta Lattice** | CVR | **+6%** |
| **Full Meta Lattice** | Capacity savings | **+20%** |

---

## Key Innovations

### Data Integration

**Lattice Partitioner** (§3.1)

Portfolio consolidation determines *when* to merge models. Not every pair benefits from unification; merging conflicting domain-objective pairs can hurt both. Lattice Partitioner uses a decision criterion based on feature overlap and training signal compatibility to identify which models should be combined, reducing the total portfolio size by ~40%.

**Lattice Zipper — Multi-Attribution-Window Heads** (§3.2.1)

Conversion signals arrive at different delays: a purchase might be attributed to a 90-minute window for CTR freshness, a 1-day window for CVR accuracy, or a 7-day window for LTV completeness. Maintaining three separate models wastes resources and prevents the backbone from learning across all these label regimes.

Lattice Zipper trains K prediction heads on a *single shared backbone*:

```
Training: each impression i is routed to window w by:
    w = hash(user_id, item_id, timestamp) mod K

    loss = BCE(head_w(backbone(x_i)), y_{i,w})

Serving: always use the oracle (longest-window) head:
    ŷ = head_{K-1}(backbone(x))
```

The oracle head benefits from the cleaner long-window labels while the shorter-window heads provide fresher gradients during backbone training. **Result: eliminates label conflicts, stable training on heterogeneous windows.**

**Lattice Filter — Pareto-Optimal Feature Selection** (§3.2.2)

Given N tasks and M candidate features, feature k *dominates* feature i (F_i ≺ F_k) if:

```
f_{k,j} ≥ f_{i,j}   for all j ∈ {1..N}
f_{k,j} > f_{i,j}   for some j ∈ {1..N}
```

where f_{i,j} is the permutation-based importance of feature i for task j. A feature on the Pareto frontier is not dominated by any other feature — it is uniquely important for at least one task.

Lattice Filter iteratively selects features from the Pareto frontier until a budget T is met. When more Pareto-optimal features exist than the remaining budget, random sampling avoids systematic bias. **Result: principled joint feature selection across all objectives simultaneously.**

---

### Architecture: Lattice Networks (§3.3)

**Three-Stage Architecture**

```
Stage 1: Feature Processors
  Categorical features → CategoricalProcessor → O_c  ∈ (B, n_cat,   d)
  Dense features       → DenseProcessor       → O_d  ∈ (B, n_dense, d)
  O_c + O_d            → MixingNetwork (QK-norm + LN) → O_cd ∈ (B, n_cd, d)
  Sequence features    → SequenceProcessor    → O_s  ∈ (B, T,       d)

Stage 2: Backbone (L interleaved layers with ECS)
  for l = 0..L-1:
    O_s^(l+1)  = TransformerBlock(O_s^(l), domain_ids)
    O_cd^(l+1) = DWFBlock(O_s^(l+1), O_cd^(l))
    ECS.push(O_cd^(l+1))
    O_cd^(l+1) += ECS.get_residual()   [DenseNet-style]

Stage 3: Predictions
  flat = O_cd^(L).reshape(B, -1)
  ŷ_k  = TaskModule_k(flat)     for each task k   [no zipper]
  ŷ    = LatticeZipper(flat, window_idx)            [with zipper]
```

**TransformerBlock with Domain-Specific FFN** (§3.3.2)

The backbone Transformer uses standard MHA but replaces the shared FFN with one FFN per domain, preventing gradient conflicts across domains:

```
O_s' = O_s + MHA(LN(O_s), LN(O_s), LN(O_s))
O_s  = O_s' + FFN_domain[d](LN(O_s'))
```

where `FFN_domain[d]` is selected by the domain index of each sample. This *parameter untying* (§3.3.4) is the critical design choice for multi-domain stability.

**DWFBlock — DHEN/Wukong Fusion** (§3.3.2)

Cross-modal interaction between sequence context O_s and non-sequence context O_cd uses Wukong's proven FM+LCB stack:

```
combined = Concat(O_s_pooled, O_cd)  ∈ (B, 1+n_cd, d)

FMBlock(combined)  → O_F ∈ (B, n_F, d)   [pairwise FM interactions]
LCBBlock(combined) → O_L ∈ (B, n_L, d)   [linear token recombination]

O'_cd = LN(Concat(O_F, O_L) + residual)  ∈ (B, n_out, d)
```

FM captures pairwise token interactions (sequence–context cross-signals). LCB preserves all-order information. Together they replace the interformer-style cross-attention with a cheaper interaction stack.

**Extended Context Storage (ECS)** (§3.3.2)

ECS implements DenseNet-style residuals: every layer stores its output, and subsequent layers receive a projected summary of all prior activations:

```
ECS.push(O_cd^(l))                      → store mean-pooled (B, d)
ECS.get_residual() = Proj(concat(all))  → (B, d) added back to O_cd
```

This enables high-bandwidth information flow from early layers (coarse features) to late layers (fine-grained interaction outputs) without gradient vanishing. Analogous to DenseNet skip connections in vision models.

**SwishRMSNorm** (§3.3.4)

A stability primitive replacing LayerNorm in deep FFNs:

```
SwishRMSNorm(X) = RMSNorm(X) ⊙ Sigmoid(RMSNorm(X))
```

LayerNorm shifts to zero mean, which can cause catastrophic cancellation when elements are near the mean in deep ensembles. RMSNorm avoids the shift; the Sigmoid self-gate smooths activations without introducing sign-flipping near zero. **Used in all FFN layers within TransformerBlock.**

Additional stability mechanisms (§3.3.4):
- **Bias-less linear layers**: removes additive bias from all Linear and LayerNorm layers to prevent unbounded growth in the shared embedding space during joint multi-domain training.
- **QK-norm**: LayerNorm applied to the combined O_cd token matrix before attention in MixingNetwork to prevent modality contention between categorical and dense feature spaces.

---

### Knowledge Transfer: Lattice KTAP (§3.4)

Traditional knowledge distillation provides soft labels during training only. KTAP (**K**nowledge **T**ransfer **A**t **P**rediction time) extends teacher knowledge into the inference path:

```
Background (async):
  Teacher model runs on recent (user, item) pairs.
  Embeddings + logits stored in KV cache with TTL ≈ 6 hours.

Serving (per-request):
  Student queries cache: key = hash(user_id * 1e6 + item_id)
  Hit  → teacher_emb projected to student space, injected into O_cd
  Miss → zero vector placeholder (graceful degradation)

Training:
  Dual distillation loss:
    L_task  = BCE(student_logits, labels)
    L_dist  = KL(softmax(student/T), softmax(teacher/T)) × T²
    L_total = L_task + λ · L_dist
```

KTAP provides **feature-level transfer** (teacher backbone embeddings enrich student inputs) and **label-level transfer** (teacher logits provide soft supervision) simultaneously. The async architecture means zero serving latency overhead for the teacher computation.

---

## Summary

| Component | What it solves | Key Mechanism |
|---|---|---|
| **Lattice Partitioner** | Portfolio explosion | Merge-decision criterion; ~40% model reduction |
| **Lattice Zipper** | Conflicting attribution windows | K heads on shared backbone; hash routing; oracle at inference |
| **Lattice Filter** | Multi-task feature selection | Iterative Pareto frontier selection |
| **CategoricalProcessor** | Sparse feature unification | Embedding tables → uniform d-dim tokens |
| **DenseProcessor** | Dense feature unification | Bias-less projection per feature group |
| **SequenceProcessor** | Behavioral history | Attention-based event model |
| **MixingNetwork** | Categorical+dense fusion | QK-norm + LayerNorm |
| **TransformerBlock** | Sequence contextualization | Domain-specific FFNs (parameter untying) |
| **DWFBlock** | Cross-modal interaction | FM + LCB (Wukong fusion) |
| **ECS** | Cross-layer information flow | DenseNet-style residual store |
| **SwishRMSNorm** | Training stability | RMSNorm ⊙ Sigmoid(RMSNorm) |
| **TaskModule** | Multi-objective heads | Bias-less per-task MLP |
| **LatticeKTAP** | Inference-time knowledge transfer | Async teacher KV cache; dual distillation |

---

## Architecture

```
Categorical feats [B, n_cat]    Dense feats [B, n_dense_i]    Seq [B, T, seq_dim]
        │                                  │                          │
 CategoricalProcessor              DenseProcessor             SequenceProcessor
        │                                  │                          │
    O_c (B, n_cat, d)              O_d (B, n_dense, d)           O_s (B, T, d)
        │___________________________|                               │
                    │                                               │
              MixingNetwork (QK-norm + LN)                         │
                    │                                               │
              O_cd (B, n_cd, d)                                     │
                                                                    │
for l = 0..L-1:                                                     │
  ┌─ TransformerBlock ──────────────────────────────────────────────┤
  │  MHA(O_s) + domain-specific FFN (per-domain weights)            │
  │  O_s^(l+1) ∈ (B, T, d)                                         │
  └──────────────────────────────────────────────────────────────────┘
  ┌─ DWFBlock ───────────────────────────────────────────────────────┐
  │  Concat(pool(O_s), O_cd) → FMBlock ∥ LCBBlock → LN             │
  │  O_cd^(l+1) ∈ (B, n_out, d)                                    │
  └──────────────────────────────────────────────────────────────────┘
  ECS.push(O_cd^(l+1))
  O_cd^(l+1) += ECS.get_residual()         [DenseNet residual]

flat = O_cd^(L).reshape(B, -1)

LatticeZipper(flat, window_idx) → (B, n_tasks)   [training: routed head]
                                 → (B, n_tasks)   [inference: oracle head]
```

---

## Usage

```python
from meta_lattice import LatticeNetwork, LatticeKTAP, lattice_filter

model = LatticeNetwork(
    vocab_sizes   = [1_000_000, 500_000, 200_000],  # categorical vocabulary sizes
    dense_dims    = [256, 64],                        # dense feature group dims
    seq_input_dim = 64,                               # per-step sequence embedding dim
    d_model       = 64,                               # global embedding dimension
    n_layers      = 3,                                # L — backbone depth
    n_domains     = 4,                                # number of domains (Feed/Reels/Stories/...)
    n_out_tokens  = 16,                               # DWFBlock output token count
    fm_rank       = 16,                               # FM projection rank
    n_heads       = 4,                                # attention heads
    task_hidden   = [128],                            # TaskModule hidden dims
    n_tasks       = 2,                                # e.g., CTR + CVR
    n_windows     = 3,                                # K attribution windows (1 = no zipper)
    ktap_dim      = 0,                                # teacher dim for KTAP (0 = disabled)
)

logits = model(
    cat_feats   = [torch.randint(0, v, (B,)) for v in [1_000_000, 500_000, 200_000]],
    dense_feats = [torch.randn(B, 256), torch.randn(B, 64)],
    seq_feat    = torch.randn(B, T, 64),
    domain_ids  = torch.randint(0, 4, (B,)),  # domain index per sample
    window_idx  = torch.randint(0, 3, (B,)),  # attribution window per sample (training only)
    ktap_keys   = None,                        # (B,) int64 for KTAP; None to skip
)  # → (B, n_tasks)

# Lattice Filter: select 50 Pareto-optimal features from 200 candidates across 3 tasks
importance = torch.rand(200, 3)   # (n_features, n_tasks)
selected = lattice_filter(importance, target_count=50, seed=42)
# selected: sorted list of 50 feature indices
```

### Hyperparameter guidance

| Parameter | Typical value | Notes |
|---|---|---|
| `d_model` | 32–128 | Primary capacity knob; balance with feature count |
| `n_layers` | 2–4 | More layers → deeper cross-modal interaction |
| `n_domains` | 1–8 | Match number of distinct ad surfaces |
| `n_out_tokens` | 8–32 | DWFBlock output width; grows backbone capacity |
| `fm_rank` | 8–32 | FM interaction rank; higher = more expressiveness |
| `n_heads` | 4–8 | Multi-head attention for sequence processing |
| `n_tasks` | 1–4 | One per prediction objective (CTR, CVR, quality) |
| `n_windows` | 1–5 | Attribution windows; 1 disables zipper |
| `ktap_dim` | 0 or teacher `d` | 0 disables KTAP; match teacher's embedding dim |

---

## Implementation

### What is implemented

Every component described in the paper is implemented as a standalone `nn.Module`:

| Module | Paper Section | Description |
|---|---|---|
| `SwishRMSNorm` | §3.3.4 | RMSNorm(x) ⊙ Sigmoid(RMSNorm(x)). Stability primitive for deep FFNs. |
| `lattice_filter` | §3.2.2 | Pareto-optimal feature selection. Iterative frontier; random fill on oversized frontiers. |
| `LatticeZipper` | §3.2.1 | K window-specific prediction heads. Hash-routed during training; oracle head at inference. |
| `CategoricalProcessor` | §3.3.1 | Embedding tables → (B, n_cat, d). One embedding per sparse feature. |
| `DenseProcessor` | §3.3.1 | Bias-less linear projections → (B, n_dense, d). One projection per dense group. |
| `SequenceProcessor` | §3.3.1 | Linear proj + MHA + LayerNorm → (B, T, d). Basic attention-based event encoder. |
| `MixingNetwork` | §3.3.1 | Concatenates O_c and O_d; applies QK-norm + LayerNorm → O_cd. |
| `ExtendedContextStorage` | §3.3.2 | DenseNet-style residual store. Pushes mean-pooled (B, d); projects concatenated history back to d. |
| `TransformerBlock` | §3.3.2 | Pre-norm MHA + domain-specific FFN (parameter untying). SwishRMSNorm in FFN. |
| `DWFBlock` | §3.3.2 | DHEN/Wukong fusion: FMBlock + LCBBlock + residual + LayerNorm. |
| `TaskModule` | §3.3.3 | Per-objective bias-less MLP head. |
| `LatticeKTAP` | §3.4 | In-memory KV cache simulating async teacher store. `store()`, `query()`, `distillation_loss()`. |
| `LatticeNetwork` | §3.3 | Full model: preprocessors → backbone (TB+DWF+ECS) → task modules → zipper. |

### Design decisions and simplifications

**DWFBlock sequence pooling:** The backbone operates on variable-length sequences O_s. Rather than fixing a sequence length at construction time, the DWFBlock receives the mean-pooled sequence `O_s.mean(dim=1, keepdim=True)` as a single token. This avoids hard-coded `n_seq` in the architecture and is compatible with variable-length batches. A production deployment might use learned pooling (e.g., HSP from Kunlun).

**ECS memory:** This implementation pushes mean-pooled `(B, d)` vectors rather than the full token matrix. The paper does not specify pooling strategy; mean pooling is chosen for memory efficiency. The ECS projection is fixed to `max_layers × d_model` input, zero-padded for early layers.

**LatticeZipper inference detection:** The oracle head is activated when `window_idx is None` or `self.training == False`. This correctly handles both explicit inference (no window assignment) and `model.eval()` mode.

**LatticeFilter random fill:** When the Pareto frontier is larger than the remaining budget, `random.sample(pareto, budget)` fills the remainder. The paper notes this is to avoid systematic bias. The `seed` parameter makes this deterministic for reproducibility.

**KTAP cache eviction:** The reference implementation evicts the oldest cache entry (FIFO) when at capacity. A production deployment would use a distributed KV store with TTL-based expiration (~6 hours as noted in the paper). The `distillation_loss` method implements standard KD (KL divergence, temperature scaling) for the label-level transfer component.

**Lattice Partitioner:** The paper's portfolio consolidation decision criterion (based on feature overlap and training compatibility analysis) requires comparing multiple existing trained models. This offline analysis step is not implemented; `LatticeNetwork` is the architecture used after consolidation has been decided. The `EventConfig`-style per-model configuration is left to the caller.

**TransformerBlock bias-less attention:** Following §3.3.4, `nn.MultiheadAttention` is constructed with `bias=False` to prevent unbounded drift in the shared embedding space during joint training.

### Relation to other models in this directory

| Model | Domain | Sequential | Multi-domain | Multi-task | Scaling |
|---|---|---|---|---|---|
| Wukong | Non-seq | No | No | No | FM stacking |
| InterFormer | Non-seq + Seq | Yes | No | No | Bidirectional |
| Kunlun | Non-seq + Seq | Yes | No | Limited | 2× efficiency |
| **Meta Lattice** | **Non-seq + Seq** | **Yes** | **Yes** | **Yes** | **Portfolio-scale** |

Meta Lattice reuses Wukong's FM+LCB blocks (as `DWFBlock`) for cross-modal interaction, and introduces Lattice-specific components (Zipper, Filter, KTAP, domain-specific FFNs) for the MDMO problem.

---

## Files

```
meta_lattice/
├── meta_lattice.py     # full implementation + smoke test
│   ├── SwishRMSNorm            # stability primitive: RMSNorm ⊙ Sigmoid(RMSNorm)
│   ├── lattice_filter          # Pareto-optimal feature selection function
│   ├── LatticeZipper           # K attribution-window heads with hash routing
│   ├── CategoricalProcessor    # sparse embedding tables → token matrix
│   ├── DenseProcessor          # dense feature projection → token matrix
│   ├── SequenceProcessor       # attention-based event encoder
│   ├── MixingNetwork           # QK-norm categorical+dense fusion
│   ├── ExtendedContextStorage  # DenseNet-style cross-layer residual store
│   ├── TransformerBlock        # MHA + domain-specific FFN with SwishRMSNorm
│   ├── DWFBlock                # FM + LCB cross-modal interaction (Wukong)
│   ├── TaskModule              # per-objective bias-less MLP head
│   ├── LatticeKTAP             # async teacher KV cache with distillation loss
│   └── LatticeNetwork          # full model: preprocessors + backbone + task heads
├── test_meta_lattice.py  # pytest test suite (73 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 meta_lattice.py
```

Expected output:
```
train logits: torch.Size([4, 2])  [...]
loss:         0.7877
backward:     OK
infer logits: torch.Size([4, 2])
params:       192,928

LatticeFilter: selected 4 from 10 → [...]

LatticeZipper training: routes to assigned window head
LatticeZipper inference: always uses oracle (longest-window) head
```

### Test suite

```bash
python3 -m pytest test_meta_lattice.py -v
```

Expected: `73 passed`.
