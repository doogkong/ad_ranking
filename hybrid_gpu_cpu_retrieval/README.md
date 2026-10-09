# Hybrid GPU–CPU Retrieval for Personalized Search at Ultra-Large Scale

Reference implementation of **Hybrid GPU–CPU Retrieval for Personalized Search at Ultra-Large Scale** (Fu, Sun, Zhu, Liu, Shi, Lu, Liu, Wang, Yao, Niu, Dong, Lyu, Shen, Huang, Chen, Ding, Fan, Kong; Meta Platforms; KDD '27; arXiv 2609.21281).

Paper: https://arxiv.org/abs/2609.21281

> **Read this first — what this is and isn't.** This is a systems-plus-modeling paper about a production deployment; there is no public code, data, hardware kernels or exact model dimensions. This folder implements **each described component at laptop scale** — lifecycle pool selection, INT8 ANN, the two-tower + interaction pre-ranker with joint training, the CPU embedding index with eager evaluation, the lightweight CPU model, co-serving/aggregation/versioned release, the paper's metrics and the capacity-plan arithmetic — and exercises them on a **synthetic personalized-search world**. A noisy oracle stands in for the shared downstream ranker. Synthetic numbers show the mechanisms behave as the paper argues; they are not production results.

---

## Summary

### The personalization–scale paradox

User-generated-content search needs two things that fight over hardware:

* **Depth** — rich, interaction-heavy scoring of (query, user, document) for queries with intent (discovery queries want behavioural history: "seafood over fast food"). This wants the inventory in GPU HBM, but HBM can't hold trillions of documents.
* **Breadth** — covering a massive, fast-changing inventory (new posts, expirations, deletions) so that obscure queries still find their one sparse relevant document. CPUs hold it cheaply but can't run the interaction model in the latency budget.

A single homogeneous tier must give up one side.

### The solution: heterogeneous co-serving, not a bigger model

Two **independently selected, independently versioned** pathways, forked per request and joined at an aggregator:

| | GPU pathway (depth) | CPU pathway (breadth) |
|---|---|---|
| Inventory | curated pool, ~1B docs, chosen for **search value** | ~20× larger (tens of billions), broad coverage; not a disjoint set — also contains popular docs |
| Retrieval | two-tower ANN **fused with interaction pre-ranking** inside the retrieval pass, on the accelerator | dedicated embedding index: centroids → inverted lists |
| Model | residual-MLP towers + DeepFM-style interaction module; joint training | lightweight unified two-tower (shared XLM-V backbone), dot-product only |
| Hardware | AMD MI300X; INT8 fused kernel; sparse tables on host over PCIe | commodity x86 DRAM; AVX-512 bulk scans |
| Refresh | daily model+index snapshot | base backfill + live updates |

The aggregator **deduplicates by doc id and keeps source attribution** before the shared downstream ranker. Each branch has its own deadline (a slow branch never discards the other's candidates), and either can be disabled for capacity control or rollback without touching the aggregation interface. A model and its index are published as one **matched snapshot** — a failed predecessor showed that an embedding alone is not a release unit (offline exact-NN vs. online quantized ANN, richer offline text, and a changed downstream filter explained a model that won offline and lost online).

### GPU pathway details (Sec. 4)

* **Pool construction — optimise search value, not virality.** Recommendation engagement is dominated by entertainment, while tutorials and local guides have low aggregate engagement but are essential to explicit intents. *Lifecycle-aware filtration*: ingest hard gates (language, originality, type) → week one early filters (spam, low value) → from day 7 taxonomy / search-usefulness / local-information / temporal-decay models estimate long-term utility. *Adaptive retention*: category-dependent decay (niche tutorials and local guides with high search value bypass engagement pruning; news and trends decay quickly); a highly selective evergreen pool retains under 0.1% of content older than a year.
* **Joint training** on search logs: `L = w₁·L_InfoNCE + w₂·L_SmoothL1 + w₃·L_BCE` (Eq. 1). InfoNCE (Eq. 2) uses in-batch negatives from other sessions, plus a **within-session InfoNCE**: engaged candidates are positives and relevant-but-not-engaged ones are session-local hard negatives. Smooth-L1 targets relevance, BCE targets engagement. A **value model** `S_final = Σ_t λ_t · p̂_t` sets the operating point per traffic segment without retraining.
* **Accelerator-resident serving** (SilverTorch-style): ANN search, inference and pre-ranking stay on the GPU; a **fused INT8 kernel** does quantisation and distance computation in one operator; sparse tables stay on the host. One MI300X: ~150–200 QPS at ~30–40 ms p99 model-server latency.

### CPU pathway details (Sec. 5)

* **Dedicated embedding index** — embedding search used to share lexical-index infrastructure and contend with latency-sensitive lexical work; a dedicated index expands the pool by more than 10×. Centroids are trained by an **independent daily distributed trainer** and published as a versioned centroid index (64k → 512k centroids would take prohibitively long inside index build); shards just ingest.
* **Eager evaluation** — legacy document-at-a-time (DAAT) execution jumps among scattered posting lists, thrashing the cache. Instead scan the nearest-neighbour lists **term-at-a-time**, take a global Top-K (K × prefetch multiplier), and only then apply the other matching/filtering rules. Heavy vector math finishes before Boolean processing, turning random reads into sequential scans. **Hit scanning** bypasses the per-document virtual-call iterator and passes raw vectors in bulk to AVX-512 list-scan kernels.
* **Lightweight personalized model** — one shared backbone for queries and docs with bidirectional InfoNCE; an attention module fuses the dense query vector with sparse contextual features; docs stay precomputable, so scans are SIMD-friendly. The deliberate loss of expressiveness (dot product only, coarse features) buys breadth.

### Reported results

| Experiment | Result |
|---|---|
| **Full-system A/B** vs legacy CPU-only (9 days, millions of accounts/arm) | DCG@20 **+4.51%** [+3.99, +5.04]; GSRR **+2.01%** [+1.71, +2.32] (conservative intervals; every daily interval > 0) |
| GPU pathway alone (7 days) | GSRR +1.04% ± 0.13, DCG@20 +0.73% ± 0.15 |
| CPU pathway alone (9 days, GPU disabled) | GSRR +1.20% ± 0.11, DCG@20 +3.16% ± 0.15 |
| Dedicated index + eager execution | retrieval-stage compute **−89.45% (9.48×)**; 64k→512k centroids (256 probes) cut CPU use a further **63.5%** with no detected GSRR regression |
| Branch latency (P50/P95/P99) | lexical 196/686/1184 ms; CPU pathway 41/439/859 ms; GPU pathway 151/675/1376 ms (GPU passes ~224 candidates to the ranker vs ~59 from CPU) |
| Candidate diversity | GPU-side overlap with CPU union only 2.1–2.4%; Jaccard 0.41–0.75%. Ranker input ≈ 781 candidates: GPU-only 209, dense-CPU-only 47, lexical-only 500, cross-group 26 |
| Capacity (matched d=256 SQ8 workload, 3 regions) | accelerator holds 2–3× vectors/unit, 3–4× QPS/unit → ~⅓ the units per region at ~12× unit-capacity cost ⇒ **~4× the CPU plan's cost** (storage-dominated; excludes ranking, networking, utilisation) |

### Statistical method worth reusing

Persistent assignment makes daily estimates dependent, so they aren't averaged as independent (no `/√9`). By Cauchy–Schwarz, the half-width of the mean is at most the mean of the daily half-widths — exact under perfect correlation. For each day take the *larger* side of its 95% interval and average those. `conservative_interval` implements it.

### Limits the paper states

Pathway experiments change several components at once (hardware, inventory, ANN point, model, filters), so gains can't be attributed to one; the capacity comparison isn't a latency or quality-equivalence test; shared-ranker input diversity isn't source-attributed Top-20 exposure; learned request-level routing is future work.

---

## What is implemented

| Component | Paper | Code |
|---|---|---|
| Lifecycle-aware filtration + category-adaptive retention + old-content cap; engagement-pool baseline; broad CPU inventory | Sec. 4.1, 5.2 | `select_search_value_pool`, `LifecycleConfig`, `select_engagement_pool`, `select_broad_inventory` |
| INT8 quantisation and fused integer distance; clustered INT8 ANN | Sec. 4.3 | `int8_quantize`, `int8_scores`, `Int8ClusterIndex` |
| Two-tower (residual-MLP towers), DeepFM-style interaction pre-ranker (FM + deep, rel & engagement heads), value model | Sec. 4.2, Fig. 2 | `GPUTwoTower`, `InteractionPreRanker`, `value_model` |
| Cross-session InfoNCE (Eq. 2), within-session InfoNCE, joint loss (Eq. 1) | Sec. 4.2 | `info_nce_cross_session`, `info_nce_within_session`, `train_gpu_models`, `JointWeights` |
| GPU pathway: ANN pre-fetch → interaction pre-rank → top-K, matched model-index snapshots | Fig. 3, Sec. 3.2 | `GPUPathway`, `ModelIndexSnapshot` |
| Versioned centroids trained separately from serving; live add/remove without re-clustering | Sec. 5.2 | `VersionedCentroids`, `train_centroids`, `EmbeddingIndex.add/remove` |
| Embedding index with coarse pruning; **eager (TAAT + prefetch)** vs DAAT; access instrumentation | Sec. 5.1–5.3, Fig. 4 | `EmbeddingIndex.search`, `AccessStats` |
| Recall–candidate frontier vs. #centroids | Fig. 5 | `recall_candidate_frontier` |
| Lightweight unified two-tower with context fusion, bidirectional InfoNCE | Sec. 5.4 | `LightweightTwoTower`, `bidirectional_info_nce`, `train_cpu_model` |
| Fork–join, per-branch deadlines, dedup + source attribution, enable/disable | Sec. 3 | `HybridRetriever`, `Aggregator`, `BranchResult` |
| DCG/nDCG, GSRR (Eqs. 3–4), relative lift, conservative cross-day interval | Sec. 6.1, App. A.5 | `ndcg_at_k`, `gsrr`, `relative_lift`, `conservative_interval` |
| GPU-side / CPU-side overlap, Jaccard, exclusive-source composition | App. A.2, Tables 4, 5, 8 | `overlap_metrics`, `source_composition` |
| Capacity-plan arithmetic (storage vs throughput bound, 3 regions) | Sec. 6.5, App. A.4 | `capacity_plan`, `compare_plans` |
| *(ours)* synthetic search world, noisy-oracle shared ranker | — | `SearchWorld`, `NoisyOracleRanker`, `evaluate_retrievers`, `modeling_depth_ndcg` |

### Choices the paper does not specify (ours)

* The synthetic world: documents have a topic, a "format" sub-topic, a category, hidden quality and virality (entertainment is viral/low-quality, tutorials and local the opposite), age and eligibility; users have a format taste; relevance = topic match × quality × (1 + taste match), news discounted by age. Selectors only see noisy proxies of quality/virality. GPU pool 5% of the corpus, CPU inventory ~17× larger (paper: ~20×).
* The pre-ranker is also given the retrieval cosine as a feature; without it the interaction module failed to learn topic matching in the short toy training run.
* CPU "lighter filter" = a minimum search-value estimate; the CPU branch emits a rank-derived score because it has no interaction scoring.
* Latency is injected (`latency_fn`) rather than measured, so deadline behaviour is deterministic.
* Not implemented: XLM-V, AVX-512 / MI300X kernels, lexical retrieval, query understanding, the capacity-routing layer, and the real downstream ranker.

---

## Demo

```bash
python3 hybrid_retrieval.py                      # ~10 s on CPU
python3 -m pytest test_hybrid_retrieval.py -v
```

```
corpus 20000; GPU pool 1000; CPU inventory 16711 (17x larger)
== pool selection (oracle nDCG@20 reachable from the pool) ==
  search-value pool 0.495   engagement pool 0.253
== end-to-end (shared noisy-oracle ranker; nDCG@20 against the whole corpus) ==
  gpu only  nDCG@20 0.510   candidates/query 30
  cpu only  nDCG@20 0.738   candidates/query 40
  hybrid    nDCG@20 0.822   candidates/query 67
== modeling depth on the SAME pool (own ordering, nDCG@20 vs best reachable from the pool) ==
  GPU two-tower + interaction pre-ranker 0.960   GPU two-tower only 0.969   lightweight CPU model 0.902
== candidate diversity == exclusive GPU 2739, exclusive CPU 3739, overlap 261 of 6739
== CPU index: eager (TAAT + prefetch) vs document-at-a-time, 100 filtered queries ==
  eager  distance ops   103594   sequential list scans   800   random list-to-list jumps       0
  daat   distance ops    94102   sequential list scans     0   random list-to-list jumps   88511
== capacity plan (storage-dominated, 3 regions) == accelerator units/region 0.33x, unit cost 12x, plan cost 4.0x the CPU plan
```

**Reading it honestly (single seed, toy scale):**

* **Reproduced in direction:** the search-value pool reaches ~2× the reachable nDCG of an engagement-driven pool of the same size (the Sec. 4.1 argument); the hybrid beats either pathway alone, and the two pathways' candidate sets are largely disjoint; eager execution trades ~10% *more* distance arithmetic for strictly sequential access (no list-to-list jumps vs ~88k for DAAT) — exactly the paper's trade; and the capacity arithmetic reproduces the ⅓-units × 12× ≈ 4× structure when storage dominates (an input to the function, not an independent finding).
* **Not reproduced:** the CPU-only configuration beats the GPU-only one here, because the GPU pool (1,000 docs, ~10 per topic) is too small to reach most relevant documents — in the paper the GPU pool holds ~1B docs and both pathways are positive against the legacy baseline. Likewise the GPU *interaction pre-ranker* adds nothing over the GPU two-tower in this world (0.960 vs 0.969): the depth advantage that shows up is two-tower-with-dense-features over the CPU's content-only dot product (0.969 vs 0.902), which is a weaker claim than the paper's.
* The end-to-end numbers lean on a noisy-oracle ranker, which rewards any candidate set that merely *contains* relevant documents; that favours breadth over ranking quality.

```python
from hybrid_retrieval import *

world = SearchWorld(seed=0)
pool = select_search_value_pool(world, LifecycleConfig(pool_size=1000, old_keep_fraction=0.2))
inventory = select_broad_inventory(world)
gpu = GPUPathway(world, train_gpu_models(world, pool, steps=300), pool)
cpu_model = train_cpu_model(world, inventory, steps=300)
cents = train_centroids(cpu_model.encode_doc(world.content[inventory]).detach(), 128, version=1)
cpu = CPUPathway(world, cpu_model, inventory, cents)
hybrid = HybridRetriever(gpu, cpu, deadlines_ms={"gpu": 200, "cpu": 200})
print(evaluate_retrievers(world, {"hybrid": hybrid}))
gpu.enabled = False        # rollback: aggregation interface unchanged
```

---

## Tests

`test_hybrid_retrieval.py` (64 tests, ~3 s): nDCG/GSRR/lift; the conservative interval (mean of larger-side daily half-widths, wider than an independence assumption); overlap/Jaccard/composition; capacity plans (storage- vs throughput-bound, the paper-style 12×/⅓ ⇒ 4×); the world's relevance structure; every lifecycle rule (week-one drops, engagement-pruning bypass for high search value, old-content cap, news decays fastest) and that the search-value pool beats the engagement pool; INT8 round-trip error, exact integer accumulation, zero-vector safety, ANN quality and probe/scan behaviour; the InfoNCE variants (false-negative masking, engaged-vs-non-engaged, degenerate sessions), value-model operating points, pre-ranker; GPU pathway contract and matched-snapshot enforcement; the IVF index (each doc in one list, full-probe equals exact, **eager ≡ DAAT given enough prefetch**, prefetch-multiplier trade-off, sequential vs random access, filtering, live add/remove without re-clustering); the recall–candidate frontier (more centroids reach the same recall scanning fewer candidates); the lightweight model; aggregator dedup/attribution/deadlines; independent enable/disable; and end-to-end hybrid > either pathway.
