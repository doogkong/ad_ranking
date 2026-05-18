# Ad Ranking Model: Technical Proposal
# Leveraging Scaling Laws and Advanced Architecture Under GPU/Infra Constraints

**Version:** 2.0  
**Date:** 2026-05-17

---

## Executive Summary

This proposal outlines an architecture for a next-generation ad ranking model that achieves
predictable quality scaling within a fixed GPU budget by synthesizing insights from six
recent Meta/industry papers implemented in this directory:

| Paper | Core Contribution | Our Takeaway |
|---|---|---|
| **Foundation-Expert** (2508.02929) | Two-stage FM+Expert; TR transfers FM scaling gains | FM trains once cross-surface; experts are lightweight (20–40% FM compute) |
| **Kunlun** (2602.10016) | Scaling laws for joint seq+non-seq ranking | Maximize MFU; CompSkip + SWA for 43%+31% FLOPs reduction |
| **Meta Lattice** (2512.09200) | Multi-domain multi-task unification | Consolidate model portfolio; KTAP for inference transfer |
| **PEPNet** (2302.01115) | Plug-and-play domain/task personalization | EPNet in FM (domain gates); PPNet in Expert (task gates) |
| **OneTrans** (2510.26104) | Unified seq + feature interaction | Cross-request KV cache; massive FM serving cost reduction |
| **InterFormer** (2411.09852) | Bidirectional seq ↔ non-seq info flow | Foundation for fused FM backbone |

**Proposed architecture:** Two-stage **Foundation-Expert Unified Ad Ranking Network (FE-UARN)**:
1. **Foundation Model (FM):** One large cross-surface HSTU model trains on lifelong user history
   and produces Target-Aware Embeddings (TAE) — one per candidate per user.
2. **Expert Models:** One lightweight model per surface consumes TAE as precomputed features,
   adds short-term surface-specific signals, and produces surface-specific ranking logits.

**Key scaling lever:** Transfer Ratio (TR = 0.64–1.0 at Meta). Every FM improvement propagates
to all surfaces without retraining experts. FM scaling is amortized across N surfaces.

**Projected outcome:** +4–7% NE improvement over current baseline, 2× scaling efficiency
(NE gain per log-compute), 30–40% reduction in serving FLOPs, N × 0.5 expert model count.

---

## 1. Problem Statement

### 1.1 Current Pain Points

**Quality plateau.** Adding more capacity (layers, width) yields diminishing NE returns.
Root cause: low MFU (estimated 5–15% on current hardware) means most GPU cycles are
wasted on memory-bound operations rather than compute. Scaling model size without fixing
MFU first is expensive and inefficient.

**Model proliferation.** Separate models per surface (Search, Feed, Notifications) and
per objective (pCTR, pCVR, pRelevance) multiply training infrastructure, create stale
embeddings, and fragment training data. The **imperfectly double seesaw** (PEPNet §1)
means naive joint training makes cross-domain and cross-task tradeoffs worse, not better.

**Serving bottleneck.** For every ad candidate, the user's behavioral sequence (500+ items)
is re-encoded from scratch. At 1000 candidates per request, sequence encoding dominates
serving latency and cost.

**Scaling cost amortization.** Growing a per-surface model by 2× costs 2× compute per
surface — N surfaces means N× the cost. With a shared FM, growing the FM by 2× benefits
all N surfaces through the Transfer Ratio mechanism.

**Attribution window fragmentation.** pCVR models trained on different attribution windows
(1-hour, 1-day, 7-day) produce conflicting training signals. Running separate models for
each wastes resources; naive mixing causes label noise.

### 1.2 Design Constraints

| Constraint | Implication |
|---|---|
| Fixed GPU cluster (no immediate hardware upgrade) | Must reduce FLOPs per QPS before scaling model size |
| Multi-surface (Search, Feed, Notifications) | FM trained cross-surface; lightweight experts per surface |
| Multi-task (CTR, CVR, Relevance, Engagement) | FM: generalizable main tasks; Expert: surface-specific tasks |
| Latency budget: < 50ms p99 | FM TAE materialized offline; Expert latency equals one-stage |
| Training budget: K GPU-hours/day | FM trained once; experts are 20–40% FM compute |

---

## 2. Architectural Principles

Four principles — derived directly from the scaling law literature — guide every design
decision:

**Principle 1: Fix MFU before scaling size.**
Kunlun demonstrates that going from 17% → 37% MFU doubles real NE gain per dollar. Memory-
bound operations (embedding lookups, back-to-back matmuls, irregular tensor shapes) are
the primary bottleneck. Every architectural choice below is evaluated against its MFU impact.

**Principle 2: Compute should be proportional to signal value.**
CompSkip (Kunlun) shows that not all positions, layers, and event types deserve equal compute.
Selective attention (SWA), alternating computation (CompSkip), and per-event capacity
allocation yield NE-neutral FLOPs reduction.

**Principle 3: Consolidate, then personalize.**
One shared FM + lightweight personalization gates (EPNet in FM; PPNet in Expert) + separate
lightweight Expert heads outperforms N separate specialized models. The savings from
consolidation fund the quality investment in a deeper shared FM.

**Principle 4: Scale FM once, transfer to N surfaces.**
Foundation-Expert Transfer Ratio (TR ∈ [0.64, 1.0]) means each FM improvement propagates
to all surfaces nearly for free. Amortize the cost of scaling over the surface portfolio.
A 0.5% FM NE gain at TR = 0.8 yields ~0.4% gain on every expert surface without retraining.

---

## 3. Proposed Architecture: Foundation-Expert Unified Ad Ranking Network (FE-UARN)

### 3.1 High-Level Design

```
                    FE-UARN Two-Stage Architecture
┌───────────────────────────────────────────────────────────────────────┐
│ STAGE 1 — Foundation Model  (large, cross-surface, trained once)      │
│                                                                       │
│  Lifelong history [N items]          Candidate items [M items]        │
│  (prod_i, ctx_i, act_i)              (prod_j, ctx_j)                  │
│         │                                   │                         │
│   ItemEmbedding (§3.1.1)             ItemEmbedding (no act)           │
│   f(prod, ctx) + act                 f(prod, ctx)                     │
│   → halves seq len vs. interleaving                                   │
│         │                                   │                         │
│   concat([hist_embs, cand_embs])  ∈ (B, N+M, d)                      │
│                       │                                               │
│         EPNet domain gate on shared embeddings                        │
│         O_ep = GateNU(domain_emb ⊕ ∅(E)) ⊗ E                        │
│                       │                                               │
│            HSTU stack (Kunlun-optimized)                              │
│            Even layers: GDPA + CompSkip                               │
│            Odd layers:  SWA (w=100) + cached HSP                      │
│            ROTE temporal embeddings throughout                        │
│            Target-aware mask:                                         │
│              history → causal                                         │
│              candidates → full history, self-only cross-cand          │
│                       │                                               │
│             TAE = out[:, N:, :]  ∈ (B, M, d)                         │
│             Target-Aware Embeddings → materialized as features        │
│             in Expert training data via HyperCast pipeline            │
│                       │                                               │
│         ┌─────────────┤                                               │
│  MultiTaskHead   AlignmentModule                                      │
│  (cross-surface) (surface-specific aux, δ mask)                       │
│  L_main          L_aux (§3.1.3)                                       │
│         │                                                             │
│  OneTrans cross-request KV cache: user K/V computed once per          │
│  request, reused across all M candidates (50× serving FLOP reduction) │
└──────────────────────────────────────────────────────────────────────-┘
                              │
                         TAE features
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│ Expert: Search │  │ Expert: Feed   │  │ Expert: Notif  │
│  (20-40% FM)   │  │  (20-40% FM)   │  │  (20-40% FM)   │
│                │  │                │  │                │
│ FMEmbedModule  │  │ FMEmbedModule  │  │ FMEmbedModule  │
│ LightHSTU(1-2) │  │ LightHSTU(1-2) │  │ LightHSTU(1-2) │
│ FMFusionModule │  │ FMFusionModule │  │ FMFusionModule │
│ PPNet gates    │  │ PPNet gates    │  │ PPNet gates    │
│ LatticeZipper  │  │ LatticeZipper  │  │                │
│ pCTR,pCVR,pRel │  │ pCTR,pEng,pQ  │  │ pCTR,pCVR      │
└────────────────┘  └────────────────┘  └────────────────┘
```

---

### 3.2 Foundation Model Stage

The FM is a large cross-surface HSTU that generates Target-Aware Embeddings (TAE)
for every candidate item given a user's lifelong interaction history.

#### 3.2.1 Item Embedding (Foundation-Expert §3.1.1)

Items have three feature groups: product **p** (item ID, category, LLM representation),
contextual **c** (timestamp, surface, query), and action **a** (like, share, watch duration):

```
History:   Emb_x_i = f(Emb_p_i, Emb_c_i) + Emb_a_i     two-layer MLP + action
Candidate: Emb_y_j = f(Emb_p_j, Emb_c_j)                no action (unlabeled target)
```

**Why summation, not interleaving?** Interleaving item and action tokens doubles sequence
length to 2(N+M). Summing action into the item token at dim-d costs nothing in sequence
length and halves it relative to interleaving:

| Design | Sequence length | Projection FLOPs | Attention FLOPs |
|---|---|---|---|
| Interleaved | 2(N+M) | 2× | 4× |
| **Summation (this paper)** | **N+M** | **1×** | **1×** |

#### 3.2.2 Target-Aware Attention Mask (Foundation-Expert §3.1.2)

The HSTU processes a unified `[hist_embs ∥ cand_embs]` sequence with a structured mask:

```
mask[i, j] = 0.0    (attend)   or   -inf  (blocked)

History  → history:   causal. Position i sees only j ≤ i.
Candidate → history:  0.0 (full). Each candidate sees the complete history.
Candidate → candidate: diagonal only. Candidate i sees only itself.
History  → candidate: -inf (blocked). No future leakage into history.
```

**Why candidates see all history:** TAE_j captures the user's contextual interest in
candidate j given their *complete* history — full access is what makes them "target-aware."

**Why candidates cannot see each other:** Ranking requires that candidate j's score must
not depend on which other candidates happen to be batched together. Blocking cross-candidate
attention preserves the score independence invariant essential for fair auction mechanics.

#### 3.2.3 FM Loss (Foundation-Expert §3.1.3)

```
L_total = Σ_s ω_s · L_main_s  +  Σ_t ω_t · L_aux_t

L_main_s:  cross-surface BCE per generalizable objective (like, share, complete)
           applied directly to TAE via MultiTaskHead

L_aux_t:   surface-specific auxiliary BCE, restricted to valid samples via δ mask:
           L_aux_t = (1 / Σ_i δ_i^t) · Σ_i δ_i^t · BCE(ŷ_i^t, y_i^t)
           δ_i^t ∈ {0,1}: 1 if sample i is in task t's valid label space
```

The δ mask prevents sparse tasks (e.g., watch-time on video-only surfaces) from being
overwhelmed by zero-gradient updates from samples where the label is undefined.

---

### 3.3 FM Backbone: Kunlun with CompSkip

The FM's HSTU backbone follows the Kunlun alternating architecture (§4.4.1), which achieves
43.1% FLOPs reduction over a uniform stack at NE-neutral:

```
Even layers (l = 0, 2, 4, ...):   GLOBAL — "what context matters for this sequence?"
  X_sum       = WeightGeneration(X^(l))        compress non-seq tokens → summary
  H_summary   = HSP(S^(l))                     fresh hierarchical seed pooling
  X^(l+1)     = GlobalInteraction(X_sum ∥ H_summary)  Wukong MoE
  S^(l+1)     = GDPA(S^(l), X_sum)             fused cross-attention PFFN

Odd layers (l = 1, 3, 5, ...):    LOCAL — "what's the local sequence structure?"
  X_sum       = WeightGeneration(X^(l))
  H_summary   = cached                         reuse from previous even layer
  X^(l+1)     = GlobalInteraction(X_sum ∥ H_summary)
  S^(l+1)     = LN(SWA(S^(l)) + S^(l))        sliding window attention
```

**Why GDPA over standard PFFN:** GDPA reformulates the non-seq → seq interaction as
FlashAttention-compatible cross-attention, converting a memory-bound back-to-back matmul
into a compute-bound fused kernel. This single change accounts for 6× MFU improvement on
the PFFN block in Kunlun's ablations.

**SWA window sizing:** Set `window = 100` (covering ~30 days of daily behavior). Kunlun
shows NE is insensitive to window beyond this range while FLOPs drop 29.5% at T=1000.

---

### 3.4 Temporal Embeddings: ROTE

Replace positional embeddings with Rotary Temporal Embeddings (Kunlun §4.2). In ad ranking,
the temporal gap between a past interaction and the current request matters more than the
item's position in the sequence:

```
τ_t = log(1 + Δt / t_scale)     log-scaled gap in seconds
```

Applied via standard RoPE rotation. A click from yesterday and a click from last month are
adjacent in the sequence but fundamentally different signals; ROTE encodes this gap
explicitly, which is not possible with position-only RoPE.

---

### 3.5 FM Serving: Cross-Request KV Cache (OneTrans)

The costliest FM serving operation is re-encoding the user's behavioral sequence for every
ad candidate. OneTrans's cross-request KV cache eliminates this:

```
Per user (computed once per serving request, ~500 items):
  K_user, V_user = S-token KV projections of behavioral history
  → stored in request-scoped cache (shared across all candidates)

Per candidate (1000 candidates, each with cached user context):
  Q_ad = NS-token query from ad features
  attention(Q_ad, K_user, V_user) → ad-user relevance
  FLOPs: O(L_NS × d) instead of O((L_S + L_NS) × d)
```

**Serving FLOPs reduction estimate:** With T=500 user tokens and 1000 candidates, standard
serving runs 500×1000 = 500K token-pairs. With KV caching, each candidate runs only its
own NS-tokens (~10) against the cached user K/V: 10×1000 = 10K token-pairs. **50× reduction
in FM sequence attention FLOPs per serving request.**

The Foundation-Expert paradigm amplifies this further: TAE can be materialized offline
(e.g., hourly batch via HyperCast) for top-traffic users, making expert serving completely
FM-latency-free for those users.

---

### 3.6 Multi-Domain Personalization: EPNet in FM

Inject domain-specific personalization at the FM's embedding layer via EPNet (PEPNet §2.2.2).
The FM sees all surfaces — EPNet adapts the shared embedding output per surface:

```
E           = shared embedding (all surfaces share one table)
δ_domain    = GateNU_0(domain_emb ⊕ ∅(E))     γ = 2, stop-grad on E
O_ep        = δ_domain ⊗ E
```

Domain features: surface ID, user behavior distribution on that surface (CTR mean/variance),
item exposure statistics. Placing EPNet in the FM ensures domain-personalized TAE is
propagated to all expert surfaces automatically — experts inherit domain adaptation through
TAE without needing their own EPNet.

---

### 3.7 Expert Model per Surface (Foundation-Expert §3.2)

Each surface deploys its own lightweight expert at **20–40% of FM compute** with latency
neutral vs. the previous one-stage baseline. The expert encodes recent surface-specific
behavior that the FM's lifelong cross-surface history may dilute:

```
TAE (from FM, detached)           Short-term surface history (T_short events)
        │                          (prod, ctx, act — recent activity on this surface)
        ▼                                    │
  FMEmbeddingModule                          ▼
  LayerNorm → Dropout                  LightweightHSTU (1-2 layers, causal)
  → optional projection                      │
        │                              mean pool → short_rep (B, d_expert)
        │                                    │
        └──────── FMFusionModule ────────────┘
                  concat(fm_emb, short_rep) → MLP → fused (B, M, d_expert)
                               │
                  ExpertFusionModule
                  concat(fused, surface_feats) → MLP
                               │
                     PPNet task gates (per expert layer)
                     δ_task^(l) = GateNU_l(O_prior ⊕ ∅(fused))
                               │
                  task logits ∈ (B, M, n_tasks)
```

**FMEmbeddingModule** bridges the FM and expert training distributions. LayerNorm aligns
scale/mean of TAE; Dropout regularizes against FM distribution shift between training runs.
Optional projection from `d_fm` to `d_expert` reduces expert compute when `d_expert < d_fm`.

**LightweightHSTU** uses standard causal attention (no target-aware mask — history only)
to capture the last T_short interactions on this specific surface. Short-term local patterns
on Search look very different from Feed; this captures that divergence without overloading the FM.

**FMFusionModule** broadcasts the pooled short-term representation across all M candidates:
```
fused_j = MLP(concat(fm_emb_j, mean_pool(light_hstu_output)))   for each candidate j
```

**PPNet task gates** (PEPNet §2.2.3) apply within the expert's task heads. Expert-level
PPNet personalizes each surface task independently using user/item/author priors. The stop
gradient on `fused` ensures PPNet does not corrupt the FM-transferred representation:
```
O_prior      = concat(user_emb, item_emb, author_emb)
δ_task^(l)   = GateNU_l(O_prior ⊕ ∅(fused))    per task head layer
O_pp_t^(l)   = δ_task_t^(l) ⊗ H_t^(l)
```

**Decoupled training:** Expert forward uses `tae.detach()`. FM and experts train on
separate data pipelines with independent optimizers. Either can be updated, scaled, or
rolled back independently.

---

### 3.8 Attribution Window Handling: LatticeZipper in Expert

For pCVR in each expert, maintain K=3 heads (1-hour, 1-day, 7-day attribution windows)
on the shared expert backbone via LatticeZipper (Meta Lattice §3.2.1):

```
Training: impression i → window w = hash(user_id, item_id, timestamp) mod 3
          loss computed only against head_w

Serving:  always use head_2 (7-day oracle head)
          benefits from long-window label quality + short-window backbone gradients
```

LatticeZipper lives in the expert (not FM) because attribution windows are surface-specific.
Search ads have different conversion cycles than app install or video ads. Each expert can
configure K independently based on its surface's attribution characteristics.

---

### 3.9 Knowledge Transfer at Inference: KTAP (Optional, Phase 3)

If a large teacher FM exists (after FM is scaled in Phase 3), use LatticeKTAP (Meta Lattice
§3.4) to inject teacher embeddings into the expert at serving time:

```
Background: teacher FM precomputes TAE(user, item) every ~6 hours
            stores in distributed KV cache (Redis / feature store)

Expert serving: query cache by (user_id, item_id)
  hit  → inject teacher TAE as additional context into FMFusionModule
  miss → use online FM TAE (graceful degradation)
```

In the Foundation-Expert framework, KTAP operates as a second FM tier: the expert's
default input is the online FM's TAE, while KTAP provides a higher-quality teacher FM's TAE
as an optional enrichment. Zero serving latency overhead for teacher computation (async precompute).

---

## 4. Scaling Strategy Under GPU Constraints

### 4.1 The Scaling Efficiency Lens

Kunlun establishes the scaling law:

```
NE(C) = NE_0 − η · log(C / C_0)

Scaling efficiency = η / η_baseline
```

With limited GPU budget, the goal is to maximize η, not C. Every architectural choice
below is evaluated by its η impact, not just raw NE at current scale.

### 4.2 Transfer Ratio: The FM Scaling Multiplier

Foundation-Expert introduces a second efficiency metric — Transfer Ratio:

```
TR = (NE(Expert_FM1) − NE(Expert_FM2)) / (NE(FM1) − NE(FM2))

FM1 = stronger FM,  FM2 = weaker FM
NE = Normalized Entropy (lower is better)
```

| TR value | Interpretation |
|---|---|
| 1.0 | Expert fully inherits FM improvement |
| 0.64–1.0 | Range achieved at Meta across surfaces/tasks |
| > 1.0 | Expert benefits more than FM alone (higher-order cross-surface transfer) |
| 0.0 | Expert gain is zero; FM improvement did not transfer |

**Why TR matters for budget planning:** If the FM improves NE by 0.5% and TR = 0.8, every
expert surface gains ~0.4% without retraining. The cost of one FM training run is
amortized across N expert surfaces. At N=3 surfaces and TR=0.8:

```
Cost of 3 per-surface models growing 2× = 3 × 2× = 6× compute
Cost of FM growing 2× + TR transfer      = 2× compute + zero
```

Scale the FM, not the experts.

### 4.3 FLOPs Budget Allocation

Given a fixed compute budget C_total, allocate across components based on marginal η:

```
Component                    FLOPs share    η contribution    Priority
────────────────────────────────────────────────────────────────────────
FM backbone depth (L)            45%         High (log-linear)   1
FM sequence encoding             20%         High (KV cache fix)  2 → ~1% with cache
Expert models (N surfaces)       25%         Medium (each 20-40% FM)  3
Feature interaction              5%           Medium              4
Task heads + gates               4%           Low per head        5
Embedding tables                 1%           Fixed (vocab bound) 6
```

After OneTrans KV caching, FM sequence encoding per-candidate drops from 20% to ~0.4%.
This budget is reinvested into deeper FM layers, directly increasing η.

### 4.4 Model Sizing Guidance

| Phase | FM d_model | FM n_layers | Expert d_expert | Expert n_layers | Est. FM params | Est. MFU |
|---|---|---|---|---|---|---|
| Phase 1 (baseline) | 128 | 4 | 64 | 1 | ~15M | ~20% |
| Phase 2 (intermediate) | 256 | 6 | 96 | 2 | ~80M | ~30% |
| Phase 3 (full) | 512 | 8 | 128 | 2 | ~400M | ~35% |

Expert compute is 20–40% of FM for the same `d_model` / `n_layers` due to: no HSTU
target-aware cross-attention (only causal), no main-task heads, and smaller `d_expert`.

MFU estimates assume GDPA replaces PFFN (6× block improvement), CompSkip active (43%
FLOPs cut), and SWA at w=100 (29.5% FLOPs cut). Combined theoretical MFU uplift: from
~5% baseline to 30–37%.

### 4.5 What Not to Scale

- **Embedding table dimension beyond d=64:** Embedding lookups are memory-bound; wider
  embeddings do not improve MFU and increase parameter count without proportional NE gain.
  Use `d_embed=64` and project to `d_model` at the first layer.
- **Sequence length beyond T=1000:** SWA makes attention O(Tw) regardless of T, but HSP
  pooling and ROTE computation still scale. Beyond 1000 items, marginal NE gain is minimal
  per Kunlun's ablations.
- **Expert depth beyond 2 layers:** Expert lightweight HSTU should stay shallow (1–2 layers).
  Depth should go into the FM, not the expert — additional expert depth reduces the latency
  neutrality advantage and does not benefit from the TR mechanism.
- **PPNet gate layers beyond task head depth:** PPNet gates multiply parameter count by
  the number of task head layers. Keep task heads at 2 hidden layers + output.

---

## 5. Multi-Domain Multi-Task Design

### 5.1 Portfolio Consolidation

One FM is trained cross-surface; experts are lightweight per surface. The consolidation
question is: which surfaces share a single expert vs. separate experts?

Start with the Meta Lattice consolidation criterion: merge surfaces into a single expert
when they have >60% user overlap and compatible feature sets.

```
Group A — single expert:  Search Ads + App Install Ads
  Rationale: same user intent signal, overlapping item pool, compatible features

Group B — single expert:  Feed Ads + Video Ads
  Rationale: same behavioral sequence type (scroll/watch), similar attribution patterns

Group C — separate expert: Notification Ads
  Rationale: fundamentally different context (push vs. pull), distinct user state
```

This reduces expert count by ~40–50%. Each consolidated expert receives more training data
through the shared FM's cross-surface labels, improving sample efficiency.

### 5.2 Shared Feature Space via Lattice Filter

Run Pareto-optimal feature selection across all tasks before training:

```python
from meta_lattice import lattice_filter

# importance_scores: (n_features, n_tasks) from permutation importance
selected = lattice_filter(importance_scores, target_count=500, seed=42)
```

Features on the Pareto frontier are uniquely important for at least one task and cannot
be dropped without hurting some objective. The FM uses the full selected feature set;
each expert uses its surface-relevant subset.

### 5.3 Domain-Task Routing Summary

```
FM trains on: All surfaces simultaneously (cross-surface objectives)
              Main tasks: like, share, save, complete  (generalizable)
              Aux tasks: per-surface alignment (δ-masked)

Expert per group:  Tasks                      Attribution Windows
────────────────────────────────────────────────────────────────────────
Search (Group A)   pCTR, pCVR, pRelevance     1h / 1d / 7d (Zipper K=3)
Feed   (Group B)   pCTR, pEngagement, pQuality 1d / 7d (Zipper K=2)
Video  (Group B)   pCTR, pVTR, pCVR            1h / 7d (Zipper K=2)
Notif  (Group C)   pCTR, pCVR                  1d / 7d (Zipper K=2)
```

EPNet domain gates in FM adapt cross-surface embeddings per surface. PPNet task gates
in each expert adapt shared expert representations per task.

---

## 6. Stability and Training

### 6.1 Activation Function

Use **SwishRMSNorm** (Meta Lattice §3.3.4) in all FFN layers within the FM backbone:

```
SwishRMSNorm(X) = RMSNorm(X) ⊙ Sigmoid(RMSNorm(X))
```

Avoids catastrophic cancellation from LayerNorm's zero-mean shift in deep recommendation
networks. Critical when jointly training across domains with different activation scales.

### 6.2 Bias-less Layers

Remove additive bias from all linear layers in the FM backbone and Expert models.
Bias terms allow unbounded additive drift in the shared embedding space during joint
multi-domain training. Bias-less layers (b=0) constrain the parameter space and improve
training stability at no quality cost (verified in Meta Lattice ablations and reproduced
in the HSTU reference implementation: `nn.Linear(..., bias=False)` throughout).

### 6.3 Stop Gradients

Five critical stop-gradient points across the two-stage architecture:

1. `∅(E)` in **EPNet** gate input (FM): domain gate reads but does not write to embedding table
2. `∅(O_ep)` in **PPNet** (Expert): task gate reads but does not write to fused FM representation
3. `tae.detach()` at the **FM → Expert boundary**: expert training does not update FM parameters
4. `∅(O_ep)` in **PPNet gate** within Expert: task gate does not corrupt FMFusion output path
5. `∅(teacher_tae)` in **KTAP** injection: teacher embedding enriches without dominating gradient

### 6.4 Learning Rate Schedule

Two-phase learning rate following Kunlun / PEPNet engineering practice, extended for the
two-stage architecture:

```
FM training:
  Embedding tables:    AdaGrad,  lr = 0.05   (fast-moving, high-dimensional)
  HSTU backbone:       Adam,     lr = 5e-6   (slow-moving, requires precision)
  EPNet gates:         Adam,     lr = 1e-4   (gates converge faster than backbone)
  Main/aux heads:      Adam,     lr = 1e-4

Expert training (independent run):
  Expert backbone:     Adam,     lr = 2e-5   (slower than gates due to TAE anchoring)
  PPNet gates:         Adam,     lr = 1e-4
  LatticeZipper heads: Adam,     lr = 1e-4
```

---

## 7. Implementation Roadmap

### Phase 1 — Foundation (Weeks 1–4)
**Goal:** Establish baseline FM with measurable MFU improvement

- [ ] Implement `GDPA` replacing current PFFN module (+MFU)
- [ ] Implement `SlidingWindowAttention` with w=100 (−29.5% FLOPs)
- [ ] Implement `ROTE` replacing standard RoPE
- [ ] Add `CompSkip` alternating pattern to existing backbone
- [ ] Instrument MFU monitoring (target: from ~5–15% → ~25%)
- [ ] Implement `ItemEmbedding` summation (halves sequence length vs. interleaved)
- [ ] Implement `_build_tae_mask` + HSTU target-aware attention
- [ ] Implement FM `MultiTaskHead` + `AlignmentModule` with δ masking
- [ ] A/B test: CompSkip + SWA + target-aware FM vs. baseline (expect NE-neutral, QPS +30%)

**Deliverable:** Compute-equivalent baseline FM with higher throughput → free GPU budget for
Phase 2 scaling. FM TAE pipeline established (materialized daily to feature store).

### Phase 2 — FM Scale + Expert Deployment (Weeks 5–10)
**Goal:** Scale FM; deploy Expert models replacing per-surface models

- [ ] Implement `OneTrans` cross-request KV cache for FM sequence encoder (50× FLOPs reduction)
- [ ] Scale FM `d_model` and `n_layers` using freed FLOPs budget from Phase 1 savings
- [ ] Implement `EPNet` in FM for cross-surface domain embedding personalization
- [ ] Implement `ExpertModel` (FMEmbeddingModule + LightweightHSTU + FMFusionModule)
- [ ] Implement `PPNet` task gates within Expert heads
- [ ] Run Lattice Filter feature selection; prune low-Pareto features from FM input
- [ ] Consolidate 2–3 surface models into single Expert per group
- [ ] Implement `LatticeZipper` for CVR attribution window handling in each Expert
- [ ] Measure Transfer Ratio: TR = (NE_expert_FM2 − NE_expert_FM1) / (NE_FM2 − NE_FM1)
  - Target: TR ≥ 0.6 on all surfaces
- [ ] Set up HyperCast-style streaming pipeline for TAE materialization (~30 min latency)

**Deliverable:** FM + Expert per group replacing N surface-specific models.
Expected: +3–5% NE from FM scale + EPNet/PPNet + consolidation, −40% model count,
TR baseline established for Phase 3 FM scaling decisions.

### Phase 3 — FM Scaling + Advanced Transfer (Weeks 11–16)
**Goal:** Scale FM to maximize TR-multiplied gains across all expert surfaces

- [ ] Scale FM to Phase 3 sizing (d=512, L=8) if Phase 2 MFU targets are met
- [ ] Implement `HSP` (Hierarchical Seed Pooling) for richer FM sequence summaries
- [ ] Implement `Wukong MoE` experts in FM GlobalInteraction block
- [ ] Add `LatticeKTAP` for inference-time teacher FM → student Expert transfer
- [ ] Monitor TR per surface as FM scales; confirm TR ≥ 0.6 is maintained
- [ ] Implement `OneRec`-style generative re-ranking for top-K slot (optional)
- [ ] Evaluate offline TAE materialization for top-traffic users (zero FM latency)

**Deliverable:** Full FE-UARN with end-to-end quality optimization.
Expected: additional +1–2% NE on top of Phase 2 via FM scaling × TR.

---

## 8. Expected Outcomes

### Quality Projections

| Phase | ΔNE vs. Baseline | Mechanism |
|---|---|---|
| Phase 1: CompSkip + SWA + GDPA + TAE FM | +0.5–1.0% | MFU improvement + target-aware sequence modeling |
| Phase 2: FM scale + EPNet/PPNet + Expert deployment | +2.5–4.0% | FM depth + seesaw resolution + expert architecture |
| Phase 3: FM scaling × Transfer Ratio | +1.0–2.0% | FM scaling amortized across all surfaces via TR |
| **Total** | **+4.0–7.0%** | **Compounding** |

### Efficiency Projections

| Metric | Baseline | After Phase 1 | After Phase 2 |
|---|---|---|---|
| MFU | ~10% | ~25–30% | ~30–37% |
| FM serving FLOPs / request | 100% | ~65% | ~35% (KV cache) |
| Expert serving latency | N/A (new) | N/A | Neutral vs. one-stage baseline |
| Total model count | N | N | N × 0.5–0.6 |
| Training GPU-hours / NE point | 100% | ~50% | ~30% (FM amortized) |
| Transfer Ratio | N/A | N/A | 0.64–1.0 (target: ≥ 0.6) |

### Scaling Law Improvement

Current: η ≈ η_baseline (low, due to low MFU).
After Phase 2: η ≈ 2× η_baseline (matching Kunlun's reported 2× over InterFormer).
After Phase 3: η_eff = η × TR × N_surfaces (FM NE gain is multiplied by TR across all surfaces).

This means: **the same GPU investment that previously bought X NE on one surface will now
buy 2X NE across all surfaces.** Every future FM iteration benefits from this multiplier.

---

## 9. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Transfer Ratio < 0.5 on some surface | Medium | High | If TR < 0.5, add surface-specific HSTU layers in Expert; investigate FM auxiliary task coverage for that surface |
| FM TAE staleness (materialized hourly) | Low | Medium | For real-time use cases, run FM online with KV cache; offline TAE only for latency-sensitive top-traffic slots |
| GDPA FlashAttention kernel not available for current GPU gen | Medium | High | Fall back to standard MHA; GDPA logic is still correct, only MFU gain is partial |
| CompSkip NE regression in early training (odd/even desync) | Low | Medium | Warm-start from uniform-layer pretrained checkpoint; ramp CompSkip gradually |
| EPNet/PPNet stop gradients cause slow convergence | Low | Medium | Higher lr for gate networks (1e-4 vs. 5e-6 for backbone); gates converge in <1 pass |
| Expert consolidation hurts sparse-domain quality | Medium | High | Run consolidation ablation on smallest domain first; add domain-specific EPNet in Expert if GAUC drops |
| KV cache staleness (user behavior changes mid-request) | Low | Low | Cache per request, not per user session; TTL = one serving request |

---

## 10. Decision Points

Four explicit go/no-go checkpoints before advancing phases:

**After Phase 1 (Week 4):**
- Gate: MFU ≥ 20% AND serving QPS ≥ +20% AND NE within ±0.1% of baseline
- Gate: FM TAE materialization pipeline delivering TAE with < 2h latency
- If MFU < 20%: investigate kernel fusion gaps before scaling FM size

**After Phase 2 (Week 10):**
- Gate: Expert NE ≥ +2.5% vs. Phase 1 AND all-surface GAUC ≥ baseline per surface
- Gate: Transfer Ratio ≥ 0.6 on at least 2 of N surfaces
- If TR < 0.4 on a surface: increase FM aux task coverage for that surface before Phase 3

**After Phase 2 Expert Stability (Week 12):**
- Gate: Expert models stable in production ≥ 2 weeks with no regression
- Gate: LatticeZipper CVR heads stable (no oracle-head leakage)

**Before Phase 3 FM Scaling (Week 11):**
- Gate: TR ≥ 0.6 confirmed — without this, scaling the FM does not pay off
- Gate: KTAP teacher FM available and serving latency within budget

---

## Appendix: Component–Paper Mapping

| Component | Paper | Section | Key Equation / Note |
|---|---|---|---|
| **Foundation Model (FM)** | Foundation-Expert | §3.1 | Cross-surface HSTU; produces TAE per candidate |
| **ItemEmbedding** | Foundation-Expert | §3.1.1 | `Emb_x = f(prod, ctx) + act`; summation halves seq length |
| **Target-aware mask** | Foundation-Expert | §3.1.2 | Hist: causal; cand→hist: full; cand→cand: diagonal only |
| **FM loss (δ mask)** | Foundation-Expert | §3.1.3 | `L_aux_t = (1/Σδ) Σ δ·BCE`; valid sample masking |
| **Expert Model** | Foundation-Expert | §3.2 | FMEmbedModule → LightHSTU → FMFusion → ExpertFusion |
| **FMEmbeddingModule** | Foundation-Expert | §3.2 | LayerNorm + Dropout + optional projection on TAE |
| **Transfer Ratio (TR)** | Foundation-Expert | §4.3 | `TR = ΔNNE_expert / ΔNNE_FM`; target ≥ 0.6 |
| GDPA | Kunlun | §4.3.1 | `GDPA_h(Q,K,V) = Act(QK^T/τ)V`, Q=seq, K/V from X_sum |
| HSP | Kunlun | §4.3.2 | Seeds → MHA → SumKronLinear compression |
| SlidingWindowAttention | Kunlun | §4.3.3 | O(Tw) attention, window [t−w, t+w] |
| ROTE | Kunlun | §4.2 | `τ_t = log(1 + Δt/t_scale)` |
| CompSkip | Kunlun | §4.4.1 | Even: GDPA+HSP; Odd: SWA+cached HSP |
| Wukong MoE | Kunlun | §4.4.3 | M experts on X_global partition |
| LatticeZipper | Meta Lattice | §3.2.1 | K window heads; hash routing; oracle at inference |
| Lattice Filter | Meta Lattice | §3.2.2 | Pareto-optimal feature selection across N tasks |
| KTAP | Meta Lattice | §3.4 | Async teacher FM KV cache; dual distillation into Expert |
| SwishRMSNorm | Meta Lattice | §3.3.4 | `RMSNorm(x) ⊙ Sigmoid(RMSNorm(x))` |
| EPNet | PEPNet | §2.2.2 | In FM: `O_ep = GateNU(domain_emb ⊕ ∅(E)) ⊗ E` |
| PPNet | PEPNet | §2.2.3 | In Expert: `O_pp_t^(l) = GateNU_l(O_prior ⊕ ∅(fused))_t ⊗ H_t^(l)` |
| Cross-request KV cache | OneTrans | §3 | FM: user S-tokens encoded once; NS-token queries reuse cached K/V |
| Unified tokenization | OneTrans | §3 | S-tokens (seq, shared weights) + NS-tokens (non-seq, per-feature weights) |

All reference implementations are available in:
```
ranking/web/python/
  foundation_expert/  foundation_expert.py, test_foundation_expert.py, README.md
  kunlun/             kunlun.py, test_kunlun.py, README.md
  meta_lattice/       meta_lattice.py, test_meta_lattice.py, README.md
  pepnet/             pepnet.py, test_pepnet.py, README.md
  interformer/        interformer.py, test_interformer.py, README.md
  onetrans/           onetrans.py, test_onetrans.py, README.md
  wukong/             wukong.py, README.md
```
