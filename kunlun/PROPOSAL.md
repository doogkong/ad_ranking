# Ad Ranking Model: Technical Proposal
# Leveraging Scaling Laws and Advanced Architecture Under GPU/Infra Constraints

**Version:** 1.0  
**Date:** 2026-05-16

---

## Executive Summary

This proposal outlines an architecture for a next-generation ad ranking model that achieves
predictable quality scaling within a fixed GPU budget by synthesizing insights from five
recent Meta/industry papers implemented in this directory:

| Paper | Core Contribution | Our Takeaway |
|---|---|---|
| **Kunlun** (2602.10016) | Scaling laws for joint seq+non-seq ranking | Maximize MFU; CompSkip + SWA for 43%+31% FLOPs reduction |
| **Meta Lattice** (2512.09200) | Multi-domain multi-task unification | Consolidate model portfolio; KTAP for inference transfer |
| **PEPNet** (2302.01115) | Plug-and-play domain/task personalization | Gate NU on top of any backbone, low cost |
| **OneTrans** (2510.26104) | Unified seq + feature interaction | Cross-request KV cache; massive serving cost reduction |
| **InterFormer** (2411.09852) | Bidirectional seq ↔ non-seq info flow | Foundation for fused backbone |

**Projected outcome:** +3–6% NE improvement over current baseline, 2× scaling efficiency
(NE gain per log-compute), 30–40% reduction in serving FLOPs via architectural changes
before any hardware upgrade.

---

## 1. Problem Statement

### 1.1 Current Pain Points

**Quality plateau.** Adding more capacity (layers, width) yields diminishing NE returns.
Root cause: low MFU (estimated 5–15% on current hardware) means most GPU cycles are
wasted on memory-bound operations rather than compute. Scaling model size without fixing
MFU first is expensive and inefficient.

**Model proliferation.** Separate models per surface (Search, Feed, Notifications) and
per objective (pCTR, pCVR, pRelevance) multiply training infrastructure, create stale
model staleness, and fragment training data. The **imperfectly double seesaw** (PEPNet §1)
means naive joint training makes cross-domain and cross-task tradeoffs worse, not better.

**Serving bottleneck.** For every ad candidate, the user's behavioral sequence (500+ items)
is re-encoded from scratch. At 1000 candidates per request, sequence encoding dominates
serving latency and cost.

**Attribution window fragmentation.** pCVR models trained on different attribution windows
(1-hour, 1-day, 7-day) produce conflicting training signals. Running separate models for
each wastes resources; naive mixing causes label noise.

### 1.2 Design Constraints

| Constraint | Implication |
|---|---|
| Fixed GPU cluster (no immediate hardware upgrade) | Must reduce FLOPs per QPS before scaling model size |
| Multi-surface (Search, Feed, Notifications) | One consolidated model preferred over N separate ones |
| Multi-task (CTR, CVR, Relevance, Engagement) | Shared backbone, specialized heads |
| Latency budget: < 50ms p99 | Sequence encoding must be cacheable |
| Training budget: K GPU-hours/day | Every compute unit must contribute to NE |

---

## 2. Architectural Principles

Three principles — derived directly from the scaling law literature — guide every design
decision:

**Principle 1: Fix MFU before scaling size.**
Kunlun demonstrates that going from 17% → 37% MFU doubles real NE gain per dollar. Memory-
bound operations (embedding lookups, back-to-back matmuls, irregular tensor shapes) are
the primary bottleneck. Every architectural choice below is evaluated against its MFU impact.

**Principle 2: Compute should be proportional to signal value.**
CompSkip (Kunlun) and Event-Level Personalization show that not all positions, layers, and
event types deserve equal compute. Selective attention (SWA), alternating computation
(CompSkip), and per-event capacity allocation yield NE-neutral FLOPs reduction.

**Principle 3: Consolidate, then personalize.**
One shared backbone + lightweight personalization gates (PEPNet) + separate lightweight
task heads outperforms N separate specialized models. The savings from consolidation fund
the quality investment in deeper shared layers.

---

## 3. Proposed Architecture: Unified Ad Ranking Network (UARN)

### 3.1 High-Level Design

```
                        UARN Architecture
┌───────────────────────────────────────────────────────────┐
│  Input Features                                           │
│    Sparse (user_id, item_id, …) ──→ Embedding Tables     │
│    Dense (bid, quality score, …) ──→ Linear Projections   │
│    Behavioral sequences          ──→ Sequence Encoder     │
│    Domain ID + stats             ──→ EPNet domain gate    │
│    User/item/author priors       ──→ PPNet task gate      │
└───────────────┬───────────────────────────────────────────┘
                │  E ∈ (B, emb_flat)   O_ep after EPNet gate
                │  S ∈ (B, T, d)       sequence (KV-cached)
                ▼
┌───────────────────────────────────────────────────────────┐
│  Backbone: Kunlun-style L-layer stack with CompSkip       │
│                                                           │
│  for l = 0, 2, 4, …  (even — global context):            │
│    InteractionBlock: WeightGen + HSP + Wukong MoE         │
│    TransformerBlock: GDPA(S, X_sum)                       │
│                                                           │
│  for l = 1, 3, 5, …  (odd — local refinement):           │
│    InteractionBlock: WeightGen + cached HSP               │
│    TransformerBlock: SWA(S, window=w)                     │
└───────────────┬───────────────────────────────────────────┘
                │  X^(L), H_summary
                ▼
┌───────────────────────────────────────────────────────────┐
│  PPNet Task Personalization (PEPNet §2.2.3)               │
│    δ_task^(l) = GateNU_l(O_prior ⊕ ∅(X^(L)))            │
│    One gate per task head layer                           │
└───────────────┬───────────────────────────────────────────┘
                │
                ▼
        ┌───────┬───────┬───────┐
       pCTR   pCVR   pRel   pEng     Task heads (lightweight MLPs)
        └───────┴───────┴───────┘
              Multi-task output → auction blend
```

### 3.2 Backbone: Kunlun with CompSkip

The backbone follows the Kunlun alternating architecture (§4.4.1), which achieves 43.1%
FLOPs reduction over a uniform stack at NE-neutral:

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

### 3.3 Sequence Encoding with Cross-Request KV Cache (OneTrans)

The costliest serving operation is re-encoding the user's behavioral sequence for every ad
candidate. OneTrans's cross-request KV cache eliminates this:

```
Per user (computed once per serving request, ~500 items):
  K_user, V_user = S-token KV projections of behavioral history
  → stored in request-scoped cache

Per candidate (1000 candidates, each with cached user context):
  Q_ad = NS-token query from ad features
  attention(Q_ad, K_user, V_user) → ad-user relevance
  FLOPs: O(L_NS × d) instead of O((L_S + L_NS) × d)
```

**Serving FLOPs reduction estimate:** With T=500 user tokens and 1000 candidates, standard
serving runs 500×1000 = 500K token-pairs. With KV caching, each candidate runs only its
own NS-tokens (~10) against the cached user K/V: 10×1000 = 10K token-pairs. **50× reduction
in sequence attention FLOPs per serving request.**

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

### 3.5 Multi-Domain Personalization: EPNet

Inject domain-specific personalization at the embedding layer via EPNet (PEPNet §2.2.2):

```
E           = shared embedding (all surfaces share one table)
δ_domain    = GateNU_0(domain_emb ⊕ ∅(E))     γ = 2, stop-grad on E
O_ep        = δ_domain ⊗ E
```

Domain features include: surface ID, user behavior distribution on that surface (CTR
mean/variance), item exposure statistics. The stop gradient is critical: EPNet adjusts the
output of E for each surface without disrupting the shared embedding table's gradient
updates. This resolves the **domain seesaw** at minimal parameter cost (one 2-layer MLP).

### 3.6 Multi-Task Personalization: PPNet

Apply per-task gating to every backbone output layer (PEPNet §2.2.3):

```
O_prior     = concat(user_emb, item_emb, author_emb)
δ_task^(l)  = GateNU_l(O_prior ⊕ ∅(X^(L)))    per task head layer
O_pp_t^(l)  = δ_task_t^(l) ⊗ H_t^(l)
H_t^(l+1)   = ReLU(O_pp_t^(l) W_t^(l))
```

Task heads for pCTR, pCVR, pRelevance, pEngagement each receive independent scaling of
the shared backbone output. This resolves the **task seesaw** by allowing each task to
reweight shared features based on user/item/author identity.

### 3.7 Attribution Window Handling: LatticeZipper

For pCVR, maintain K=3 heads (1-hour, 1-day, 7-day attribution windows) on the shared
backbone via LatticeZipper (Meta Lattice §3.2.1):

```
Training: impression i → window w = hash(user_id, item_id, timestamp) mod 3
          loss computed only against head_w

Serving:  always use head_2 (7-day oracle head)
          benefits from long-window label quality + short-window backbone gradients
```

This eliminates the need for three separate pCVR models, reducing model infrastructure
by ~60% for the CVR task family.

### 3.8 Knowledge Transfer: KTAP (Optional, Phase 3)

If a large teacher model exists (or after Phase 2 is trained), use LatticeKTAP (Meta
Lattice §3.4) to transfer teacher embeddings into the student at serving time:

```
Background: teacher precomputes backbone(user, item) every ~6 hours
            stores in distributed KV cache (Redis / feature store)

Serving: student queries cache by (user_id, item_id)
         hit  → inject teacher embedding as additional context into O_ep
         miss → zero placeholder (graceful degradation)
```

Zero serving latency overhead for teacher computation (async precompute).

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

### 4.2 FLOPs Budget Allocation

Given a fixed compute budget C_total, allocate across components based on marginal η:

```
Component               FLOPs share    η contribution    Priority
──────────────────────────────────────────────────────────────────
Backbone depth (L)          40%         High (log-linear)   1
Sequence encoding           25%         High (KV cache fix)  2 (cache eliminates this)
Feature interaction         20%         Medium              3
Task heads + gates          10%         Low per head, high  4
                                        when multiplied by N tasks
Embedding tables             5%         Fixed (vocab bound) 5
```

**Key insight:** After implementing OneTrans-style KV caching, the 25% FLOPs allocated to
sequence encoding per-candidate drops to ~0.5% (50× reduction). This budget can be
reinvested into deeper backbone layers, directly increasing η.

### 4.3 Model Sizing Guidance

| Phase | d_model | n_layers | n_heads | Seq len | Est. params | Est. MFU |
|---|---|---|---|---|---|---|
| Phase 1 (baseline) | 128 | 4 | 8 | 256 | ~15M | ~20% |
| Phase 2 (intermediate) | 256 | 6 | 8 | 512 | ~80M | ~30% |
| Phase 3 (full) | 512 | 8 | 16 | 1000 | ~400M | ~35% |

MFU estimates assume GDPA replaces PFFN (6× block improvement), CompSkip active (43%
FLOPs cut), and SWA at w=100 (29.5% FLOPs cut). Combined theoretical MFU uplift: from
~5% baseline to 30–37%.

### 4.4 What Not to Scale

- **Embedding table dimension beyond d=64:** Embedding lookups are memory-bound; wider
  embeddings do not improve MFU and increase parameter count without proportional NE gain.
  Use `d_embed=64` and project to `d_model` at the first layer.
- **Sequence length beyond T=1000:** SWA makes attention O(Tw) regardless of T, but HSP
  pooling and ROTE computation still scale. Beyond 1000 items, marginal NE gain is minimal
  per Kunlun's ablations.
- **Task head depth beyond 2 layers:** PPNet gates on each layer; adding layers multiplies
  PPNet parameter count with diminishing returns. 2 hidden layers + output is optimal.

---

## 5. Multi-Domain Multi-Task Design

### 5.1 Portfolio Consolidation

Start with the Meta Lattice consolidation criterion: merge surfaces that have >60% user
overlap and compatible feature sets. For a typical ad system:

```
Group A (consolidate):  Search Ads + App Install Ads
  Rationale: same user intent signal, overlapping item pool, compatible features

Group B (consolidate):  Feed Ads + Video Ads
  Rationale: same behavioral sequence type (scroll/watch), similar attribution patterns

Group C (separate):     Notification Ads
  Rationale: fundamentally different context (push vs. pull), distinct user state
```

Single UARN model per group. This reduces total model count by ~40–50% while each
consolidated model receives more training data, improving sample efficiency.

### 5.2 Shared Feature Space via Lattice Filter

Run Pareto-optimal feature selection across all tasks before training:

```python
from meta_lattice import lattice_filter

# importance_scores: (n_features, n_tasks) from permutation importance
selected = lattice_filter(importance_scores, target_count=500, seed=42)
```

Features on the Pareto frontier are uniquely important for at least one task and cannot
be dropped without hurting some objective. This principled selection avoids both the
"union bloat" problem (too many features → noise) and the "intersection starvation"
problem (too few features → lost signal).

### 5.3 Domain-Task Routing Summary

```
Surface / Domain          Tasks                      Attribution Windows
──────────────────────────────────────────────────────────────────────
Search Ads (Group A)      pCTR, pCVR, pRelevance     1h / 1d / 7d (Zipper K=3)
Feed Ads (Group B)        pCTR, pEngagement, pQuality 1d / 7d (Zipper K=2)
Video Ads (Group B)       pCTR, pVTR, pCVR            1h / 7d (Zipper K=2)
```

One UARN backbone per group. EPNet gates per surface within the group. PPNet gates per
task within each UARN. LatticeZipper per CVR-type task.

---

## 6. Stability and Training

### 6.1 Activation Function

Use **SwishRMSNorm** (Meta Lattice §3.3.4) in all FFN layers within the backbone:

```
SwishRMSNorm(X) = RMSNorm(X) ⊙ Sigmoid(RMSNorm(X))
```

Avoids catastrophic cancellation from LayerNorm's zero-mean shift in deep recommendation
networks. Critical when jointly training across domains with different activation scales.

### 6.2 Bias-less Layers

Remove additive bias from all linear layers in the shared backbone and embedding projections.
Bias terms allow unbounded additive drift in the shared embedding space during joint
multi-domain training. Bias-less layers (b=0) constrain the parameter space and improve
training stability at no quality cost (verified in Meta Lattice ablations).

### 6.3 Stop Gradients

Three critical stop-gradient points:
1. `∅(E)` in EPNet gate input: domain gate reads but does not write to embedding table
2. `∅(O_ep)` in PPNet gate input: task gate reads but does not write to EPNet output path
3. `∅(O_ep)` in KTAP injection: teacher embedding enriches student without dominating gradient

### 6.4 Learning Rate Schedule

Two-phase learning rate following Kunlun / PEPNet engineering practice:

```
Embedding tables:    AdaGrad,  lr = 0.05   (fast-moving, high-dimensional)
DNN backbone:        Adam,     lr = 5e-6   (slow-moving, requires precision)
Gate networks:       Adam,     lr = 1e-4   (EPNet/PPNet gates converge faster than backbone)
Task heads:          Adam,     lr = 1e-4
```

---

## 7. Implementation Roadmap

### Phase 1 — Foundation (Weeks 1–4)
**Goal:** Establish baseline with measurable MFU improvement

- [ ] Implement `GDPA` replacing current PFFN module (+MFU)
- [ ] Implement `SlidingWindowAttention` with w=100 (−29.5% FLOPs)
- [ ] Implement `ROTE` replacing standard RoPE
- [ ] Add `CompSkip` alternating pattern to existing backbone
- [ ] Instrument MFU monitoring (target: from ~5–15% → ~25%)
- [ ] A/B test: CompSkip + SWA vs. baseline (expect NE-neutral, QPS +30%)

**Deliverable:** Compute-equivalent baseline with higher throughput → redeploy existing
model size at lower serving cost, freeing GPU budget for Phase 2.

### Phase 2 — Scale + Consolidate (Weeks 5–10)
**Goal:** Use freed compute to scale model size + consolidate model portfolio

- [ ] Implement OneTrans cross-request KV cache for sequence encoder
- [ ] Scale `d_model` and `n_layers` using freed FLOPs budget (Phase 1 savings)
- [ ] Implement `EPNet` + `PPNet` for multi-surface multi-task handling
- [ ] Run Lattice Filter feature selection; prune low-Pareto features
- [ ] Consolidate 2–3 surface models per group into single UARN
- [ ] Implement `LatticeZipper` for CVR attribution window handling

**Deliverable:** Single UARN per group replacing N surface-specific models.
Expected: +2–4% NE from consolidation + scale, −40% model count.

### Phase 3 — Advanced Features (Weeks 11–16)
**Goal:** Extract remaining quality headroom from knowledge transfer + HSP

- [ ] Implement `HSP` (Hierarchical Seed Pooling) for richer sequence summaries
- [ ] Implement `Wukong MoE` experts in GlobalInteraction block
- [ ] Add `LatticeKTAP` for inference-time teacher-to-student transfer
- [ ] Scale to Phase 3 sizing (d=512, L=8) if Phase 2 MFU targets are met
- [ ] Implement `OneRec`-style generative re-ranking for top-K slot (optional)

**Deliverable:** Full UARN with end-to-end quality optimization.
Expected: additional +1–2% NE from Phase 2 base.

---

## 8. Expected Outcomes

### Quality Projections

| Phase | ΔNE vs. Baseline | Mechanism |
|---|---|---|
| Phase 1: CompSkip + SWA + GDPA | +0.5–1.0% | MFU improvement → reallocate compute |
| Phase 2: Scale + EPNet/PPNet + Consolidation | +1.5–3.0% | More depth + seesaw resolution |
| Phase 3: HSP + MoE + KTAP | +0.5–1.0% | Marginal headroom from advanced features |
| **Total** | **+2.5–5.0%** | **Compounding** |

### Efficiency Projections

| Metric | Baseline | After Phase 1 | After Phase 2 |
|---|---|---|---|
| MFU | ~10% | ~25–30% | ~30–37% |
| Serving FLOPs / request | 100% | ~65% | ~35% (KV cache) |
| Total model count | N | N | N × 0.5–0.6 |
| Training GPU-hours / NE point | 100% | ~50% | ~30% |

### Scaling Law Improvement

Current: η ≈ η_baseline (low, due to low MFU).
After Phase 2: η ≈ 2× η_baseline (matching Kunlun's reported 2× over InterFormer).

This means: **the same GPU investment that previously bought X NE will now buy 2X NE.**
Every future model iteration benefits from this multiplier.

---

## 9. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GDPA FlashAttention kernel not available for current GPU gen | Medium | High | Fall back to standard MHA; GDPA logic is still correct, only MFU gain is partial |
| CompSkip NE regression in early training (odd/even desync) | Low | Medium | Warm-start from uniform-layer pretrained checkpoint; ramp CompSkip gradually |
| EPNet/PPNet stop gradients cause slow convergence | Low | Medium | Higher lr for gate networks (1e-4 vs. 5e-6 for backbone); gates converge in <1 pass |
| Model consolidation hurts sparse-domain quality | Medium | High | Run consolidation ablation with Domain C (smallest domain) first; rollback gate if GAUC drops |
| KV cache staleness (user behavior changes mid-request) | Low | Low | Cache per request, not per user session; TTL = one serving request |
| Sequence length assumption (T=1000) too short for long-horizon CVR | Medium | Medium | Use separate 7-day window head (LatticeZipper) to capture long-term signal |

---

## 10. Decision Points

Three explicit go/no-go checkpoints before advancing phases:

**After Phase 1 (Week 4):**
- Gate: MFU ≥ 20% AND serving QPS ≥ +20% AND NE within ±0.1% of baseline
- If MFU < 20%: investigate kernel fusion gaps before scaling model size

**After Phase 2 (Week 10):**
- Gate: Consolidated model NE ≥ +1.5% vs. Phase 1 AND all-domain GAUC ≥ baseline per domain
- If domain seesaw detected: increase EPNet capacity (larger domain_feat_dim) before Phase 3

**Before Phase 3 (Week 11):**
- Gate: Phase 2 model stable in production ≥ 2 weeks with no regression
- KTAP teacher model must be available and serving latency within budget

---

## Appendix: Component–Paper Mapping

| Component | Paper | Section | Key Equation |
|---|---|---|---|
| GDPA | Kunlun | §4.3.1 | `GDPA_h(Q,K,V) = Act(QK^T/τ)V`, Q=seq, K/V from X_sum |
| HSP | Kunlun | §4.3.2 | Seeds → MHA → SumKronLinear compression |
| SlidingWindowAttention | Kunlun | §4.3.3 | O(Tw) attention, window [t−w, t+w] |
| ROTE | Kunlun | §4.2 | `τ_t = log(1 + Δt/t_scale)` |
| CompSkip | Kunlun | §4.4.1 | Even: GDPA+HSP; Odd: SWA+cached HSP |
| Wukong MoE | Kunlun | §4.4.3 | M experts on X_global partition |
| LatticeZipper | Meta Lattice | §3.2.1 | K window heads; hash routing; oracle at inference |
| Lattice Filter | Meta Lattice | §3.2.2 | Pareto-optimal feature selection across N tasks |
| KTAP | Meta Lattice | §3.4 | Async teacher KV cache; dual distillation |
| SwishRMSNorm | Meta Lattice | §3.3.4 | `RMSNorm(x) ⊙ Sigmoid(RMSNorm(x))` |
| EPNet | PEPNet | §2.2.2 | `O_ep = GateNU(domain_emb ⊕ ∅(E)) ⊗ E` |
| PPNet | PEPNet | §2.2.3 | `O_pp_t^(l) = GateNU_l(O_prior ⊕ ∅(O_ep))_t ⊗ H_t^(l)` |
| Cross-request KV cache | OneTrans | §3 | User S-tokens encoded once; NS-token queries reuse cached K/V |
| Unified tokenization | OneTrans | §3 | S-tokens (seq, shared weights) + NS-tokens (non-seq, per-feature weights) |

All reference implementations are available in:
```
ranking/web/python/
  kunlun/        kunlun.py, test_kunlun.py, README.md
  meta_lattice/  meta_lattice.py, test_meta_lattice.py, README.md
  pepnet/        pepnet.py, test_pepnet.py, README.md
  interformer/   interformer.py, test_interformer.py, README.md
  onetrans/      onetrans.py, test_onetrans.py, README.md
  wukong/        wukong.py, README.md
```
