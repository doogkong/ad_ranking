# LLM Ads: Retrieval and Ranking with LLM Semantic Features

PyTorch reference implementations for:

- **`llm_retrieval.py`** — "LLM Retrieval for Stable and Predictable Ad Recommendations" (Meta Platforms, SIGIR Workshop AgentSearch 2026), arXiv: 2605.21969v1
- **`llm_ranking.py`** — Extension of the above to the ranking stage, combining with the Foundation-Expert paradigm (arXiv: 2508.02929)

---

## Problem

Traditional ad systems optimise for prediction accuracy (NE, recall, NDCG) but ignore **predictability** — the system's stability when minor ad input perturbations occur (creative copy variants, image swaps, minor description edits). Poor predictability causes three advertiser-facing problems:

| Problem | Description |
|---|---|
| **Repeatability** | Same ad gets very different delivery across otherwise identical auctions |
| **Cold start** | New creative variants of an existing ad are treated as entirely new, wasting ramp-up time |
| **Under-exploration** | System over-focuses on seen IDs, missing semantically equivalent ads |

Root cause: raw ad IDs (sparse) are the primary retrieval and ranking signal. Two creative variants of the same product have unrelated IDs, so the model sees them as unrelated.

**Solution:** Replace sparse ID-centric representations with **LLM-extracted semantic attributes** (hierarchical categories + dense caption embeddings). Semantically equivalent ads cluster together in the new space, making retrieval and ranking stable across creative variants.

---

## Part 1 — LLM Ad Retrieval (`llm_retrieval.py`)

### Architecture

```
Ad Creative (title, description, product)
         │
         ▼
AdSemanticExtractor  (fine-tuned Llama-3-8B; here: MLP approximation)
  │
  ├── category_logits  ∈ ℝ^(B, n_categories)   soft hierarchical category scores
  │     Category → Specific Category → Related Categories
  │
  └── caption_emb      ∈ ℝ^(B, d_llm)          L2-normalised dense semantic vector
         │
         ▼
SemanticGraph construction
  Nodes = ads
  Edges = S_R(Ad_i, Ad_j) ≥ min_edge_weight
    S_R = phrase Jaccard  if ≥ θ
          token Jaccard   otherwise
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│  Two-Stage Retrieval (LLMAdRetriever)                        │
│                                                              │
│  Stage 1 — Category Retrieval:                              │
│    cosine(caption_emb_query, caption_emb_cand)              │
│    → top-K candidates from the semantic space               │
│    (production: inverted index on category tokens + BFS)    │
│                                                              │
│  Stage 2 — Relevance Re-Ranking:                            │
│    RelevanceScorer(brand_product, temporal, personalized)   │
│    brand_product = query_emb ⊙ cand_emb                     │
│    → MLP → relevance score per candidate                    │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
  ranked candidates  (B, top_k_final)
```

### Key Components

#### 1. AdSemanticExtractor — §4.1

Maps ad creative features to hierarchical semantic attributes:

```
h = ReLU(W_2 · ReLU(W_1 · ad_feats))
category_logits = W_cat · h                     ∈ ℝ^(B, n_categories)
caption_emb     = L2_norm(W_cap · h)            ∈ ℝ^(B, d_llm)
```

In production this is a fine-tuned Llama-3-8B Instruct model with retrieval-specific instruction tuning on the ads dataset. The reference implementation uses an MLP with the same input/output contract.

#### 2. FuzzySetMatcher — §4.3

Implements the paper's `S_R(Ad1, Ad2)` fuzzy set similarity:

```
S_R(Ad1, Ad2) = S(P_Ad1, P_Ad2)   if S(P_Ad1, P_Ad2) ≥ θ
               S(T_Ad1, T_Ad2)    otherwise

S(A, B) = |A ∩ B| / |A ∪ B|   (Jaccard)
P = set of category phrases,  T = set of category tokens
```

Phrase-level matching is preferred; token-level fallback handles partial overlap (e.g. "running shoes" vs "trail running" share the "running" token).

#### 3. SemanticGraph — §4.3

```
Nodes:  ad_ids
Edges:  (ad_i, ad_j, weight=S_R(ad_i, ad_j))  if weight ≥ min_edge_weight

BFS expansion from seed ads:
  score(node at hop h) = Π edge_weights along path
  Returns top-K candidates ranked by cumulative score
```

BFS traversal finds clusters of contextually similar ads, enabling dynamic candidate expansion beyond the initial category match.

#### 4. RelevanceScorer — §4.4 Step 2

```
brand_product = query_emb ⊙ cand_emb           (element-wise; encodes cosine similarity)
relevance     = MLP(concat(brand_product, temporal, personalized))
```

Three relevance dimensions mirror the paper's Figure 1 (Brand-Product, Temporal, Personalized Relevance).

#### 5. PredictabilityMetrics — §2, §5

```
StatSigDiff(a_p, a_s):
  Δ = |conv(a_p) − conv(a_s)| / mean(conv(a_p), conv(a_s))
  SSD = max(0, Δ − 1.65 × √(2 / (conv(a_p) + conv(a_s))))

System StatSigDiff = Σ SSD × √(rev_p + rev_s) / Σ √(rev_p + rev_s)

rel_diff(day) = (imp_primary − imp_shadow) / imp_shadow × 100%
MAD = median(|rel_diff(day_i) − median(rel_diff)|)
```

Lower SSD and MAD = more stable system. The paper achieved −8.62% A/A' difference and −45% MAD vs. the ID-based baseline.

### Measured Results (paper §5)

| Metric | Result |
|---|---|
| Online top-line | **+0.45%** |
| Final stage recall | **+1.2%** |
| A/A' predictability (SSD) | **−8.62%** |
| MAD improvement | **−45%** |
| Incremental Recall @200 vs @5 | **+1.89×** |

---

## Part 2 — LLM-Enriched Ad Ranking (`llm_ranking.py`)

The paper targets retrieval. This extends LLM semantic features to the **ranking stage** by integrating them into the Foundation-Expert paradigm (arXiv 2508.02929).

### Architecture

```
LLM outputs (offline batch, per ad)
  category_logits  ∈ ℝ^(B, n_categories)
  caption_emb      ∈ ℝ^(B, d_llm)
         │
         ▼
LLMFeatureEncoder
  cat_proj(sigmoid(category_logits))  → (B, d_out)
  caption_proj(caption_emb)           → (B, d_out)
  fusion MLP                          → llm_feat ∈ ℝ^(B, d_model)
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1 — LLMEnrichedFoundationModel  (cross-surface, trained once) │
│                                                                     │
│  SemanticItemEmbedding:                                             │
│    Emb_x = f(prod, ctx, llm_feat) + act      (history)             │
│    Emb_y = f(prod, ctx, llm_feat)            (candidate)           │
│    LLM features replace raw ad ID → cold-start robust              │
│         │                                                           │
│  HSTU (target-aware mask) → TAE ∈ ℝ^(B, M, d)                     │
│         │                                                           │
│  Training loss:                                                     │
│    L_total = L_main + L_aux                                         │
│            + λ · L_consistency   (A/A' pair regularization)        │
│            + μ · L_semantic      (LLM category supervision)        │
│                                                                     │
│  ConsistencyRegularizer (A/A' pairs):                              │
│    L_consistency = (1/P) Σ_p w_p ||TAE(ad_p) − TAE(ad_p')||²     │
│    shadow TAE detached — primary TAE is pushed toward shadow        │
└─────────────────────────────────────────────────────────────────────┘
         │ TAE (detached)
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2 — LLMEnrichedExpertModel  (lightweight, per surface)        │
│                                                                     │
│  FMEmbeddingModule(TAE)         → fm_processed  (B, M, d_expert)   │
│  SemanticItemEmbedding(short)   → LightHSTU → short_rep (B, d)     │
│  FMFusionModule                 → fused (B, M, d_expert)           │
│                                                                     │
│  LLM candidate signal (new):                                        │
│    llm_surf = MLP(concat(cand_cat_logits, cand_captions))          │
│    → cold-start signal: "how typical is this ad for this surface"  │
│                                                                     │
│  ExpertFusionModule(fused, concat(surf_feats, llm_surf))           │
│    → logits (B, M, n_tasks)                                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. SemanticItemEmbedding — §3.1.1 + §4.1

```
llm_feat = fusion(cat_proj(σ(cat_logits)), caption_proj(caption_emb))

Emb_x = MLP(concat(prod, ctx, llm_feat)) + act_proj(act)   (history)
Emb_y = MLP(concat(prod, ctx, llm_feat))                   (candidate)
```

LLM features add semantic grounding that raw ad IDs cannot provide. New ads with no impression history receive meaningful embeddings from their LLM attributes alone — eliminating cold-start delay.

#### 2. ConsistencyRegularizer — §2 of 2605.21969v1

Ranking-stage analogue of the retrieval A/A' predictability metric:

```
L_consistency = (1/P) Σ_p w_p · ||TAE(ad_p) − TAE(ad_p')||²

ad_p  = primary ad
ad_p' = shadow ad (semantic variant: same product, different creative ID)
w_p   = LLM-derived semantic similarity between the pair ∈ [0, 1]
```

Shadow TAE is detached — only the primary TAE is pushed toward it. High `w_p` = strong consistency requirement (semantically near-identical ads). This directly optimises the MAD/SSD metrics from the retrieval paper at the ranking stage.

#### 3. PredictabilityAwareLoss

```
L_total = Σ_s ω_s · L_main_s           (cross-surface task loss)
        + Σ_t ω_t · L_aux_t            (surface-specific masked loss, δ-gated)
        + λ · L_consistency             (A/A' TAE stability, default λ=0.1)
        + μ · L_semantic                (LLM category supervision, default μ=0.05)

L_semantic = BCE(semantic_head(TAE), σ(cand_cat_labels))
  Teaches the model: "good ranking representations should predict LLM categories"
```

#### 4. LLM Expert Signal

The Expert model augments surface features with an LLM-derived candidate signal:

```
llm_surf = MLP(concat(cand_cat_logits, cand_captions))   ∈ ℝ^(B, M, d_expert)
surf_augmented = concat(surface_feats, llm_surf)
logits = ExpertFusionModule(fused, surf_augmented)
```

This gives the Expert a semantic prior on each candidate independent of FM TAE quality — valuable for new ads entering the system for the first time.

### Why extend to ranking

| Stage | LLM benefit |
|---|---|
| Retrieval (paper) | Recall: semantic candidates that sparse ID lookup misses |
| **Ranking (this)** | Precision: richer item features → better TAE; predictability: consistent scores across creative variants |

The ranking extension is higher leverage because it affects final scored positions, not just candidate inclusion.

---

## Usage

### Retrieval

```python
from llm_retrieval import (
    AdAttributes, AdSemanticExtractor, SemanticGraph,
    FuzzySetMatcher, LLMAdRetriever, PredictabilityMetrics,
)

# Build semantic graph from ad corpus
attrs = [AdAttributes(ad_id=..., categories=..., ...) for ad in corpus]
graph = SemanticGraph(min_edge_weight=0.05)
graph.build_from_attributes(attrs)

# Two-stage retrieval
retriever = LLMAdRetriever(
    ad_feat_dim  = 128,
    n_categories = 64,
    d_llm        = 48,
    d_temporal   = 16,
    d_personal   = 16,
    top_k_graph  = 50,
    max_hops     = 2,
)
retriever.set_graph(graph)

scores, candidate_indices = retriever.retrieve(
    query_ad_ids  = ["ad_123", "ad_456"],
    query_feats   = torch.randn(2, 128),
    cand_feats    = torch.randn(1000, 128),
    temporal      = torch.randn(2, 16),
    personalized  = torch.randn(2, 16),
    top_k_final   = 20,
)
# scores: (2, 20)   candidate_indices: list of 2 lists of 20 ints

# Measure predictability for A/A' ad pairs
report = PredictabilityMetrics.evaluate(
    daily_primary_impressions = [100, 105, 98, 110, ...],
    daily_shadow_impressions  = [100,  98, 102, 99, ...],
)
print(f"MAD: {report['mad']:.4f}")   # lower = more stable
```

### Ranking

```python
from llm_ranking import LLMEnrichedFoundationModel, LLMEnrichedExpertModel

# Foundation Model with LLM features
fm = LLMEnrichedFoundationModel(
    prod_dim           = 64,
    ctx_dim            = 32,
    act_dim            = 16,
    aux_dim            = 24,
    n_categories       = 128,   # LLM category vocabulary
    d_llm              = 48,    # LLM caption embedding dim
    d_model            = 256,
    n_layers           = 6,
    n_heads            = 8,
    n_main_tasks       = 4,
    n_aux_tasks        = 3,
    consistency_weight = 0.1,   # λ for A/A' consistency loss
    semantic_weight    = 0.05,  # μ for LLM category supervision
)

tae, main_logits, aux_logits, sem_logits = fm(
    hist_prod, hist_ctx, hist_act,
    hist_cat_logits, hist_captions,     # LLM features for history
    cand_prod, cand_ctx,
    cand_cat_logits, cand_captions,     # LLM features for candidates
    aux_features,
)

# A/A' consistency for P semantic variant pairs
fm_loss, breakdown = fm.compute_loss(
    main_logits, main_labels,
    aux_logits, aux_labels,
    sem_logits, cand_cat_labels,
    tae_primary  = tae[:P],             # (P, M, d) TAE for primary ads
    tae_shadow   = tae_shadow,          # (P, M, d) TAE for shadow variants
    pair_weights = llm_similarity[:P],  # (P,) similarity weights
)

# Expert model (trains independently, consumes TAE)
expert = LLMEnrichedExpertModel(
    fm_dim       = 256,
    prod_dim     = 64,
    ctx_dim      = 32,
    act_dim      = 16,
    surf_dim     = 48,
    n_categories = 128,
    d_llm        = 48,
    d_expert     = 64,
    n_layers     = 2,
    n_heads      = 4,
    n_tasks      = 3,
)

logits = expert(
    tae.detach(),                        # detach — no gradient to FM
    short_prod, short_ctx, short_act,
    short_cat, short_captions,           # LLM features for short history
    cand_cat_logits, cand_captions,      # LLM features for candidates
    surface_feats,
)  # (B, M, n_tasks)
```

### Hyperparameter guidance

| Parameter | Typical value | Notes |
|---|---|---|
| `n_categories` | 64–256 | LLM category vocabulary; larger = finer semantic granularity |
| `d_llm` | 32–128 | LLM caption embedding dim; match to the fine-tuned LLM output |
| `min_edge_weight` | 0.05 | Lower = denser graph, more BFS expansion; 0.05–0.1 works well |
| `phrase_threshold` θ | 0.1 | Minimum phrase Jaccard to prefer phrase over token similarity |
| `consistency_weight` λ | 0.05–0.2 | Start at 0.1; increase if A/A' MAD is high in production |
| `semantic_weight` μ | 0.01–0.1 | Controls how strongly TAE tracks LLM category space |
| `max_hops` | 2 | More hops = wider expansion but noisier candidates; 2 is sufficient |
| `top_k_graph` | 20–100 | Stage 1 recall pool size before Stage 2 re-ranking |

---

## Files

```
llm_ads/
├── llm_retrieval.py        # retrieval model + predictability metrics
│   ├── AdAttributes            # hierarchical semantic attributes dataclass
│   ├── AdSemanticExtractor     # LLM feature extractor: ad_feats → (cat_logits, caption_emb)
│   ├── FuzzySetMatcher         # S_R(Ad1, Ad2): phrase then token Jaccard
│   ├── SemanticGraph           # Jaccard adjacency graph + BFS expansion
│   ├── RelevanceScorer         # brand-product + temporal + personalized MLP
│   ├── LLMAdRetriever          # two-stage retrieval: cosine recall → relevance re-rank
│   └── PredictabilityMetrics   # StatSigDiff, system SSD, MAD for A/A' evaluation
├── llm_ranking.py          # LLM-enriched Foundation-Expert ranking model
│   ├── LLMFeatureEncoder       # (cat_logits, caption_emb) → dense ranking feature
│   ├── SemanticItemEmbedding   # Emb_x = f(prod, ctx, llm_feat) + act
│   ├── ConsistencyRegularizer  # ||TAE(ad) − TAE(ad')||² for A/A' pairs
│   ├── LLMEnrichedFoundationModel  # FM + LLM features + consistency + semantic loss
│   └── LLMEnrichedExpertModel  # Expert + LLM ad similarity as surface signal
└── README.md
```

---

## Running

### Smoke tests

```bash
python3 llm_retrieval.py
```

Expected output:
```
Category logits: torch.Size([4, 16])   caption emb: torch.Size([4, 24])
FuzzySetMatcher similarity(ad_0, ad_1): 0.1429
SemanticGraph BFS from ad_0: [('ad_4', 0.6), ('ad_3', 0.333...), ...]
Retrieval scores: torch.Size([4, 5])   indices[0]: [7, 10, 9, 5, 17]
Predictability MAD:            3.9216
Predictability median_rel_diff: 0.0000%
System StatSigDiff: 0.000000
Backward: OK
Retriever params: 4,928
```

```bash
python3 llm_ranking.py
```

Expected output:
```
FM TAE:          torch.Size([4, 6, 24])
FM main logits:  torch.Size([4, 6, 3])
FM sem logits:   torch.Size([4, 6, 16])
FM loss:         3.2890
  main=1.8916  aux=1.3598  sem=0.7371  consist=0.0077
FM backward:     OK
FM params:       19,296

Expert logits:   torch.Size([4, 6, 2])
Expert loss:     0.6705
Expert backward: OK
Expert params:   4,428
Expert/FM ratio: 22.9%  (target: 20-40%)

LLM-enriched ranking model:
  FM:     LLM category + caption → richer TAE; consistency loss for A/A' stability
  Expert: LLM similarity features → cold-start robust; surface-level predictability
```
