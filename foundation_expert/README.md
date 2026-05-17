# Foundation-Expert Paradigm

PyTorch reference implementation of **Realizing Scaling Laws in Recommender Systems: A Foundation-Expert Paradigm for Hyperscale Model Deployment** (Meta AI, Aug 2025).

Paper: https://arxiv.org/abs/2508.02929

Deployed at Meta handling **tens of billions of daily requests** across multiple surfaces (Feed, Reels, Groups, etc.) with expert latency **neutral vs. one-stage baseline** while achieving consistent NE improvements through FM-to-Expert knowledge transfer (Transfer Ratio 0.64–1.0).

---

## Motivation

Scaling a single large model across all recommendation surfaces hits two fundamental walls simultaneously:

| Problem | Root cause |
|---|---|
| **Scaling cost** | A fully-shared cross-surface model must be large enough to serve every surface, but all that capacity must be invoked at inference on every request — even for surfaces that only need a fraction of it |
| **Surface divergence** | Different surfaces (Feed vs. Reels vs. Groups) have different behavioral distributions, label sparsity, and session lengths. A single model must compromise between all of them |
| **Training coupling** | Joint training couples surface-specific and cross-surface gradients. Surface A's data affects surface B's parameters whether or not that helps surface B |
| **Infrastructure rigidity** | Monolithic models cannot be deployed, updated, or A/B tested per surface without coordinating all surfaces simultaneously |

The insight behind Foundation-Expert: **the cross-surface user understanding (general knowledge) and the surface-specific engagement prediction (local knowledge) require fundamentally different amounts of compute and data, and should be learned and deployed separately.**

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Stage 1 — Foundation Model  (large, cross-surface, trained once)             │
│                                                                              │
│  History items [i=1..N]            Candidate items [j=1..M]                 │
│  (prod_i, ctx_i, act_i)            (prod_j, ctx_j)                          │
│         │                                  │                                 │
│         └─────────── ItemEmbedding ────────┘                                 │
│                        Emb_x_i = f(prod_i, ctx_i) + act_i    (history)      │
│                        Emb_y_j = f(prod_j, ctx_j)            (candidate)    │
│                        summation (not interleaving) → N+M tokens            │
│                                  │                                           │
│               concat([hist_embs, cand_embs])  ∈ (B, N+M, d)                │
│                                  │                                           │
│                               HSTU stack                                     │
│                         (target-aware mask)                                  │
│                     history: causal attention                                │
│                     candidate: full history attention,                       │
│                                self-only across candidates                   │
│                                  │                                           │
│                    out ∈ (B, N+M, d) → TAE = out[:, N:, :]                  │
│                    Target-Aware Embeddings ∈ (B, M, d)                      │
│                      │               │                                       │
│              MultiTaskHead       AlignmentModule                             │
│              (cross-surface)    (surface-specific aux)                       │
│              L_main (§3.1.3)    L_aux (§3.1.3)                              │
└──────────────────────────────────────────────────────────────────────────────┘
                              │
                    TAE materialized as features
                    in expert training data
                              │
┌──────────────────────────────────────────────────────────────────────────────┐
│ Stage 2 — Expert Model  (lightweight, one per surface, 20-40% FM compute)   │
│                                                                              │
│   TAE (from FM)                 Short-term surface history                  │
│       │                         (prod, ctx, act — recent T events)          │
│       ▼                                   │                                 │
│  FMEmbeddingModule                        ▼                                 │
│  LayerNorm → Dropout                LightweightHSTU                         │
│  → optional proj                    (causal, 1-2 layers)                    │
│       │                                   │                                 │
│       │                             mean pool → short_rep (B, d)            │
│       │                                   │                                 │
│       └──────── FMFusionModule ───────────┘                                 │
│                 concat(fm_emb, short_rep) → MLP → fused (B, M, d)           │
│                              │                                              │
│                    ExpertFusionModule                                        │
│                    concat(fused, surface_feats) → MLP                       │
│                              │                                              │
│                    logits ∈ (B, M, n_tasks)                                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. Item Embedding — §3.1.1

Items have three feature groups: product features **p** (item ID, category, LLM representation), contextual features **c** (timestamp, surface, query), and action features **a** (like, share, watch duration). The critical design choice is how to combine them:

```
History:   Emb_x_i = f(Emb_p_i, Emb_c_i) + Emb_a_i
Candidate: Emb_y_j = f(Emb_p_j, Emb_c_j)             (no action — target has no label)

f = two-layer MLP over concat([prod, ctx])
```

**Why summation, not interleaving?** An earlier design interleaved item and action tokens: `[item_1, action_1, item_2, action_2, ..., candidate_1, ...]`, producing a sequence of length `2(N+M)`. Summing action into the item token cuts this to `N+M`:

| Design | Sequence length | Projection FLOPs | Attention FLOPs |
|---|---|---|---|
| Interleaved (item + action tokens) | 2(N+M) | 2× | 4× |
| **Summation (this paper)** | **N+M** | **1×** | **1×** |

50% fewer projection FLOPs, 25% fewer attention FLOPs relative to interleaving — a significant saving at billions-of-requests scale.

---

### 2. Target-Aware Attention Mask — §3.1.2

The HSTU processes a unified sequence of `N` history items followed by `M` candidate items. The mask structure enforces two critical properties:

```
Positions:  [0 .. N-1 | N .. N+M-1]
             history  | candidates

mask[i, j] = 0.0    (attend)
mask[i, j] = -inf   (blocked)

History  → history:   causal (lower-triangular). pos i sees j ≤ i only.
Candidate → history:  0.0 (full). Each candidate sees the complete history.
Candidate → candidate: diagonal only. Candidate i sees only itself.
History  → candidate: -inf (blocked). History cannot see future candidates.
```

**Why candidates see all history, not just causal:** Each candidate's TAE should represent the user's interest in *that specific item* given their *entire* history. Full history access is what makes them "target-aware" rather than just sequential.

**Why candidates cannot see each other:** Ranking requires that candidate j's score must not depend on which other candidates happen to be in the same batch. Blocking cross-candidate attention preserves this independence invariant.

---

### 3. HSTU — §3.1.2

Hierarchical Sequential Transduction Units: a transformer stack with RMSNorm pre-normalization, bias-free attention and FFN layers, and SiLU activation. Pre-norm and bias removal follow training stability best practices for deep sequential models at scale.

```
For each layer l:
  x = x + Dropout(MHA(RMSNorm(x), mask=tae_mask))
  x = x + Dropout(FFN(RMSNorm(x)))

FFN: Linear(d, 4d) → SiLU → Linear(4d, d)   (bias=False throughout)
```

**TAE extraction:** after `L` HSTU layers and a final RMSNorm, the candidate positions `out[:, N:, :]` are the Target-Aware Embeddings (TAE). These are materialized and stored as features in the expert's training data — the FM and expert train **independently** on separate data pipelines.

---

### 4. Foundation Model Loss — §3.1.3

The FM is trained with two complementary objectives:

```
L_total = Σ_s ω_s · L_main_s  +  Σ_t ω_t · L_aux_t
```

**Main loss** (`L_main_s`): cross-surface generalizable objectives (likes, shares, video completions) applied directly to the TAE via `MultiTaskHead`. These tasks are shared across all surfaces, pushing the FM to learn universal user-interest representations.

**Auxiliary loss** (`L_aux_t`): surface-specific alignment applied via `AlignmentModule`. Not all samples are valid for all tasks (e.g., watch-time only applies to video surfaces). The **δ mask** restricts each task's loss to its valid sample space:

```
L_aux_t = (1 / Σ_i δ_i^t) · Σ_i δ_i^t · loss_t(ŷ_i^t, y_i^t)

δ_i^t ∈ {0, 1}  — 1 if sample i is in task t's valid sample space
```

Without the mask, tasks with sparse valid samples would be overwhelmed by zero-gradient updates from invalid samples, degrading their signal.

---

### 5. Expert Model — §3.2

Each surface deploys its own lightweight expert, using FM's TAE as precomputed features. The expert has **20–40% of FM compute** while matching or exceeding one-stage baseline quality.

**FMEmbeddingModule**: bridges the FM and expert training data distributions. LayerNorm aligns the scale and mean of TAE (which may shift between FM training runs); Dropout provides regularization against FM distribution shift. Optional linear projection reduces FM dimension to a smaller expert dimension.

**LightweightHSTU**: encodes recent, surface-specific interaction history. This captures short-term behavioral patterns (last 10–20 actions on this surface) that the FM's lifelong cross-surface history may dilute. Uses standard causal attention (no target-aware mask — it processes history only, not candidates).

**FMFusionModule**: combines long-term FM knowledge with short-term surface-specific patterns:
```
fused = MLP(concat(fm_emb, mean_pool(light_hstu_output)))
```
The mean-pool collapses the short-term sequence to a single vector, then broadcasts it across all `M` candidate positions before concatenation.

**ExpertFusionModule**: final prediction from fused representation + surface-specific features (e.g., device type, locale, session context):
```
logits = MLP(concat(fused, surface_feats))   # (B, M, n_tasks)
```

**Decoupled training**: experts consume TAE from the FM as static input features. `tae.detach()` ensures no gradient flows from expert training back to the FM. The FM and experts can be updated, versioned, and A/B tested independently.

---

### 6. Transfer Ratio — §4.3

The Transfer Ratio (TR) quantifies how much of the FM's quality improvement flows through to the expert:

```
TR = (NE(Expert_FM1) − NE(Expert_FM2)) / (NE(FM1) − NE(FM2))

NE = Normalized Entropy (lower is better; normalized cross-entropy)
FM1 = stronger FM,  FM2 = weaker FM
```

| TR value | Interpretation |
|---|---|
| 1.0 | Expert fully inherits FM improvement |
| 0.64–1.0 | Range achieved at Meta across surfaces/tasks |
| > 1.0 | Expert benefits more than FM alone (cross-surface features improve local tasks via higher-order interactions) |
| 0.0 | Expert gain is zero — FM improvement did not transfer |

TR allows infrastructure planning: if the FM improves NE by 0.5% and TR = 0.8, the expert on that surface will improve by ~0.4% without any expert retraining — just by swapping the FM.

---

## Measured Results

| Component | Metric | Result |
|---|---|---|
| Transfer Ratio | Across surfaces/tasks at Meta | 0.64–1.0 |
| Expert compute | vs. equivalent one-stage model | 20–40% |
| Expert latency | vs. one-stage baseline | Neutral (0 overhead) |
| Data-to-trainer latency | HyperCast streaming pipeline | ~30 minutes |
| FM scale | HSTU-0.5B / HSTU-1B | Deployed in production |

Transfer Ratio near 1.0 means each FM scaling step yields near-proportional gains across all deployed surfaces without retraining experts — the key mechanism that makes FM scaling financially viable at hyperscale.

---

## Usage

```python
from foundation_expert import FoundationModel, ExpertModel, transfer_ratio

# --- Stage 1: Foundation Model ---
fm = FoundationModel(
    prod_dim     = 64,    # item product feature dimension
    ctx_dim      = 32,    # contextual feature dimension
    act_dim      = 16,    # user action feature dimension
    aux_dim      = 24,    # auxiliary alignment feature dimension
    d_model      = 256,   # HSTU hidden dimension (HSTU-0.5B uses ~512)
    n_layers     = 6,     # HSTU depth
    n_heads      = 8,     # attention heads
    n_main_tasks = 4,     # cross-surface tasks: like, share, save, complete
    n_aux_tasks  = 3,     # surface-specific aux tasks
)

tae, main_logits, aux_logits = fm(
    hist_prod    = torch.randn(B, N, 64),   # (B, N, prod_dim)
    hist_ctx     = torch.randn(B, N, 32),   # (B, N, ctx_dim)
    hist_act     = torch.randn(B, N, 16),   # (B, N, act_dim)
    cand_prod    = torch.randn(B, M, 64),   # (B, M, prod_dim)
    cand_ctx     = torch.randn(B, M, 32),   # (B, M, ctx_dim)
    aux_features = torch.randn(B, M, 24),   # (B, M, aux_dim)
)
# tae: (B, M, 256)  → materialized and stored as expert training features

# FM training loss
fm_loss = fm.compute_loss(
    main_logits, main_labels,       # (B, M, 4)
    aux_logits, aux_labels,         # (B, M, 3)
    aux_delta=valid_sample_mask,    # (B, M, 3) optional validity mask per task
)
fm_loss.backward()

# --- Stage 2: Expert Model (trains independently, consumes TAE as features) ---
expert = ExpertModel(
    fm_dim    = 256,   # must match FM d_model
    prod_dim  = 64,
    ctx_dim   = 32,
    act_dim   = 16,
    surf_dim  = 48,    # surface-specific features (device, locale, session)
    d_expert  = 64,    # expert hidden dim (~25% of FM d_model)
    n_layers  = 2,     # lightweight HSTU: 1-2 layers
    n_heads   = 4,
    n_tasks   = 3,     # surface-specific tasks
)

expert_logits = expert(
    tae           = tae.detach(),           # (B, M, 256)  detach — no FM gradient
    short_prod    = torch.randn(B, T, 64),  # recent T events on this surface
    short_ctx     = torch.randn(B, T, 32),
    short_act     = torch.randn(B, T, 16),
    surface_feats = torch.randn(B, M, 48),  # surface context per candidate
)
# expert_logits: (B, M, 3)

expert_loss = F.binary_cross_entropy_with_logits(expert_logits, labels)
expert_loss.backward()   # gradients stay in expert; FM is unaffected

# --- Transfer Ratio: quantify FM → Expert knowledge transfer ---
tr = transfer_ratio(
    ne_expert_fm1 = 0.9810,   # NE of expert using new FM
    ne_expert_fm2 = 0.9874,   # NE of expert using old FM
    ne_fm1        = 0.9780,   # NE of new FM
    ne_fm2        = 0.9850,   # NE of old FM
)
print(f"Transfer Ratio: {tr:.3f}")  # target: 0.64–1.0
```

### Hyperparameter guidance

| Parameter | Typical value | Notes |
|---|---|---|
| `d_model` (FM) | 256–512 | HSTU-0.5B uses ~512; scale this to hit your FLOP budget |
| `n_layers` (FM) | 4–8 | More layers = longer lifelong history context |
| `d_expert` | 25–40% of FM `d_model` | Expert should be 20–40% of FM compute |
| `n_layers` (Expert) | 1–2 | Lightweight — captures only recent surface-specific history |
| `n_main_tasks` | 2–6 | Cross-surface tasks shared across all expert surfaces |
| `n_aux_tasks` | 1–4 | Surface-specific auxiliary tasks; use δ mask for sparse tasks |
| `aux_delta` | bool mask | Always pass a validity mask for tasks with sparse label coverage |
| short history `T` | 16–64 | Recent events on this surface; much shorter than FM's lifelong N |

---

## Files

```
foundation_expert/
├── foundation_expert.py        # full implementation + smoke test
│   ├── ItemEmbedding               # f(prod, ctx) + act summation (§3.1.1)
│   ├── _build_tae_mask             # target-aware additive attention mask (§3.1.2)
│   ├── HSTULayer                   # single HSTU transformer layer: RMSNorm + MHA + FFN
│   ├── HSTU                        # HSTU stack; extracts TAE at candidate positions
│   ├── MultiTaskHead               # cross-surface main task prediction head (§3.1.3)
│   ├── AlignmentModule             # surface-specific auxiliary loss with δ mask (§3.1.3)
│   ├── FoundationModel             # full FM: ItemEmbedding → HSTU → heads (§3.1)
│   ├── FMEmbeddingModule           # TAE normalization + optional projection (§3.2)
│   ├── FMFusionModule              # concat(FM, short_term) → MLP fusion (§3.2)
│   ├── ExpertFusionModule          # concat(fused, surface_feats) → predictions (§3.2)
│   ├── ExpertModel                 # full expert: FM TAE + lightweight HSTU (§3.2)
│   └── transfer_ratio              # TR metric: FM scaling gain → Expert gain (§4.3)
├── test_foundation_expert.py   # pytest test suite (59 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 foundation_expert.py
```

Expected output:
```
FM TAE:         torch.Size([4, 8, 32])
FM main logits: torch.Size([4, 8, 3])
FM aux logits:  torch.Size([4, 8, 2])
FM loss:        3.5081
FM backward:    OK
FM params:      29,312

Expert logits:  torch.Size([4, 8, 2])
Expert loss:    0.6886
Expert backward: OK
Expert params:  5,648
Expert/FM ratio: 19.3%  (paper: 20-40%)

Transfer Ratio: 1.0047  (paper range: [0.64, 1.0])

Foundation-Expert paradigm:
  FM:     generalizable, lifelong, cross-surface → TAE
  Expert: lightweight, surface-specific → surface predictions
  TAE transfers FM scaling gains to experts without joint training
```

### Test suite

```bash
python3 -m pytest test_foundation_expert.py -v
```

Expected output:

```
collected 59 items

test_foundation_expert.py::TestItemEmbedding::test_shape_with_action PASSED
test_foundation_expert.py::TestItemEmbedding::test_shape_without_action PASSED
test_foundation_expert.py::TestItemEmbedding::test_action_changes_output PASSED
test_foundation_expert.py::TestItemEmbedding::test_gradient_flows PASSED
test_foundation_expert.py::TestItemEmbedding::test_summation_not_concat_same_output_dim PASSED
test_foundation_expert.py::TestTAEMask::test_shape PASSED
test_foundation_expert.py::TestTAEMask::test_history_causal PASSED
test_foundation_expert.py::TestTAEMask::test_history_self_attend PASSED
test_foundation_expert.py::TestTAEMask::test_candidates_see_all_history PASSED
test_foundation_expert.py::TestTAEMask::test_candidates_blocked_from_each_other PASSED
test_foundation_expert.py::TestTAEMask::test_candidates_self_attend PASSED
test_foundation_expert.py::TestTAEMask::test_history_cannot_see_candidates PASSED
test_foundation_expert.py::TestHSTULayer::test_output_shape PASSED
test_foundation_expert.py::TestHSTULayer::test_with_mask PASSED
test_foundation_expert.py::TestHSTULayer::test_gradient_flows PASSED
test_foundation_expert.py::TestHSTU::test_output_shape PASSED
test_foundation_expert.py::TestHSTU::test_tae_extraction PASSED
test_foundation_expert.py::TestHSTU::test_candidate_independence PASSED
test_foundation_expert.py::TestHSTU::test_history_affects_tae PASSED
test_foundation_expert.py::TestHSTU::test_gradient_flows_through_history PASSED
test_foundation_expert.py::TestMultiTaskHead::test_output_shape PASSED
test_foundation_expert.py::TestAlignmentModule::test_output_shape PASSED
test_foundation_expert.py::TestAlignmentModule::test_masked_loss_with_delta PASSED
test_foundation_expert.py::TestAlignmentModule::test_masked_loss_no_delta PASSED
test_foundation_expert.py::TestAlignmentModule::test_masked_loss_zero_delta PASSED
test_foundation_expert.py::TestFoundationModel::test_output_shapes PASSED
test_foundation_expert.py::TestFoundationModel::test_backward PASSED
test_foundation_expert.py::TestFoundationModel::test_loss_with_delta_mask PASSED
test_foundation_expert.py::TestFoundationModel::test_different_histories_different_tae PASSED
test_foundation_expert.py::TestFoundationModel::test_different_candidates_different_tae PASSED
test_foundation_expert.py::TestFoundationModel::test_single_candidate PASSED
test_foundation_expert.py::TestFoundationModel::test_param_count PASSED
test_foundation_expert.py::TestFoundationModel::test_task_weights_applied PASSED
test_foundation_expert.py::TestFMEmbeddingModule::test_output_shape_same_dim PASSED
test_foundation_expert.py::TestFMEmbeddingModule::test_output_shape_projected PASSED
test_foundation_expert.py::TestFMEmbeddingModule::test_normalizes_input PASSED
test_foundation_expert.py::TestFMEmbeddingModule::test_gradient_flows PASSED
test_foundation_expert.py::TestFMFusionModule::test_output_shape PASSED
test_foundation_expert.py::TestFMFusionModule::test_short_rep_broadcast PASSED
test_foundation_expert.py::TestFMFusionModule::test_gradient_flows PASSED
test_foundation_expert.py::TestExpertFusionModule::test_output_shape PASSED
test_foundation_expert.py::TestExpertFusionModule::test_gradient_flows PASSED
test_foundation_expert.py::TestExpertModel::test_output_shape PASSED
test_foundation_expert.py::TestExpertModel::test_backward PASSED
test_foundation_expert.py::TestExpertModel::test_different_tae_different_output PASSED
test_foundation_expert.py::TestExpertModel::test_different_short_history_different_output PASSED
test_foundation_expert.py::TestExpertModel::test_expert_compute_fraction PASSED
test_foundation_expert.py::TestExpertModel::test_single_candidate PASSED
test_foundation_expert.py::TestExpertModel::test_surface_feats_matter PASSED
test_foundation_expert.py::TestExpertModel::test_multi_task PASSED
test_foundation_expert.py::TestTransferRatio::test_perfect_transfer PASSED
test_foundation_expert.py::TestTransferRatio::test_partial_transfer PASSED
test_foundation_expert.py::TestTransferRatio::test_paper_range PASSED
test_foundation_expert.py::TestTransferRatio::test_zero_fm_delta_raises PASSED
test_foundation_expert.py::TestTransferRatio::test_symmetric PASSED
test_foundation_expert.py::TestFMExpertPipeline::test_end_to_end_forward PASSED
test_foundation_expert.py::TestFMExpertPipeline::test_decoupled_training PASSED
test_foundation_expert.py::TestFMExpertPipeline::test_fm_improvement_transfers PASSED
test_foundation_expert.py::TestFMExpertPipeline::test_multiple_surfaces_same_fm PASSED

59 passed in 0.72s
```

Useful variants:

```bash
# Run a single test class
python3 -m pytest test_foundation_expert.py::TestTAEMask -v

# Run a single test
python3 -m pytest test_foundation_expert.py::TestFMExpertPipeline::test_decoupled_training -v

# Stop on first failure
python3 -m pytest test_foundation_expert.py -x
```
