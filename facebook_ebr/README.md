# Embedding-based Retrieval in Facebook Search

Reference implementation of **Embedding-based Retrieval in Facebook Search** (Huang, Sharma, Sun, Xia, Zhang, Pronin, Padmanabhan, Ottaviano, Yang; Facebook; KDD 2020).

Paper: https://arxiv.org/abs/2006.11632

> **Read this first — what this is and isn't.** The paper is a full-stack experience report on a production system (Unicorn inverted index, Faiss, Facebook's social graph and search logs). There is no public code or data. This folder implements **each technique described** — unified embedding model, training-data and hard-negative mining, embedding ensembles, IVF/IMI/PQ/OPQ ANN, an `nn` operator inside a Boolean retrieval engine, query/index selection, ranking features and the human-rating feedback filter — against a **synthetic personalized people-search log**. Several of the paper's empirical findings reproduce in direction; **some do not** (listed below), which is expected from a toy world and is reported as found.

---

## Summary

### Why Facebook search is different

Search intent depends on the **query text and the searcher's context**: among thousands of people called "John Smith", the one you mean is probably a friend or lives near you. So embedding retrieval (EBR) here is not a text-embedding problem. It is also a retrieval-stage problem (billions of documents, recall-oriented), and it must live **inside a term-matching inverted-index engine** that cannot be replaced.

### Modeling

* **Unified embedding** (Sec. 2.3): a query tower over *(text, searcher location, searcher social features)* and a document tower over *(text, document location, social cluster)*. Cosine similarity; **triplet loss** with margin (Eq. 2): `L = Σ max(0, D(q,d⁺) − D(q,d⁻) + m)` with `D = 1 − cos`. Margin matters a lot (5–10% recall variance across tasks). The authors argue a triplet loss with `n` random negatives per positive approximately optimises recall@(N/n).
* **Features** (Sec. 3): character n-grams beat word n-grams (small OOV problem, small table), and hashed word n-grams add a further +1.5%; location features and social-graph embeddings add +2.20% and +1.77% absolute recall on group search; unified embedding vs. text-only: **+18% recall (events), +16% (groups)**. Embeddings learn fuzzy match ("kacis creations" ↔ "Kasie's creations") and *optionalization* (dropping "nw" in "mini cooper nw").
* **Training data** (Sec. 2.4): *random* negatives beat *non-click impressions* by a wide margin (impressions are biased toward hard cases while most index documents are trivially irrelevant; **−55% absolute recall** on people search). Click and impression positives perform the same, and mixing them adds nothing.
* **Hard negative mining** (Sec. 6.1): *online* (the most similar other positives in the batch, ≤2 per positive best) gave **+8.38% (people), +7% (groups), +5.33% (events)** recall. *Offline* (retrieve top-K per query, retrain on negatives picked by a strategy) initially looked worse than random negatives; fixes were: don't take the hardest — sample from a deeper band (rank 101–500 was best); **blend easy and hard** (recall saturates near easy:hard = 100:1); and transfer hard → easy. Mining uses ANN on a random shard.
* **Hard positive mining**: sessions where production retrieval failed but the searcher engaged anyway — training on these alone matches the click-trained model's recall with only **4%** of the data; combining adds more.
* **Embedding ensemble** (Sec. 6.2): different hardness levels trade recall against precision. *Weighted concatenation* `E_Q = (α₁V_Q1/‖V_Q1‖, …)`, `E_D = (U_D1/‖U_D1‖, …)` makes `cos(E_Q,E_D) ∝ Σ αᵢ cos(V_Qi,U_Di)` (Eq. 6), so one index serves several models (weights on one side only). *Cascade*: stage two re-ranks stage one's output; the offline-HNM model was the right second stage (+3.4% recall), and a text model → unified model cascade restores text matching precision.

### Serving

* **ANN** (Sec. 4.1): Faiss coarse quantization (IVF or IMI) + product quantization, integrated into the inverted index. Lessons: compare algorithms at equal **% of index scanned** (IMI clusters are heavily imbalanced — about half had few points); **retune ANN whenever the model changes** (a model better before quantization can be worse after); **always try OPQ** over PCA (Table 3: PQ16 67.5% → OPQ16,PQ16 70.5/74.3% 1-recall@10; flat 74.8%); **`pq_bytes = d/4`** (little gain beyond); tune `nprobe`/`num_clusters`/`pq_bytes` online as well as offline.
* **Unicorn integration** (Sec. 4.2): documents get an embedding payload; at index time the embedding becomes (coarse-cluster **term**, PQ-residual **payload**); `(nn <key> :radius r :nprobe p)` is rewritten into an OR over the nearest clusters' terms, with the payload verifying the radius. Because it reuses existing primitives it inherits real-time updates and query planning, and enables **hybrid retrieval** — e.g. `(… (or (and (term text:john) (term text:smithe)) (nn model :radius 0.24 :nprobe 16)))` rescues a misspelling. **Radius mode** (constrained NN) gave a better performance/quality trade-off than top-K mode, which picks K nearest first and then applies the rest of the query. The query tower runs online; the document tower runs offline in Spark and is published, quantized, with the forward index.
* **Query and index selection** (Sec. 4.3): don't trigger EBR when searchers re-find a known target or the intent isn't what the model learned; index only monthly-active entities, recent events and popular pages/groups.

### Later-stage optimisation (Sec. 5)

Retrieval results must also be ranked well by rankers designed for the old retrieval. (1) **Embedding similarity as a ranking feature** — cosine similarity beat Hadamard product and raw embeddings consistently. (2) **A training-data feedback loop** — log EBR results, have humans rate them, retrain the relevance model to filter irrelevant EBR results while keeping relevant ones, recovering precision.

---

## What is implemented

| Component | Paper | Code |
|---|---|---|
| Char / word n-gram features, hashing | Sec. 3 | `char_ngrams`, `word_ngrams`, `hash_id`, `text_feature_ids` |
| Unified two-tower model (text + location + weighted multi-hot social), shared-tower option | Sec. 2.3, Fig. 2 | `SideEncoder`, `UnifiedEmbeddingModel`, `FeatureConfig` |
| Triplet loss (Eq. 2), cosine distance | Sec. 2.2 | `triplet_loss`, `cosine_distance` |
| recall@K over the full index (Eq. 1) | Sec. 2.1 | `recall_at_k`, `evaluate` |
| Random / non-click negatives; hard positives | Sec. 2.4, 6.1.2 | `TrainConfig(neg=...)`, `mine_hard_positives` |
| Online and offline hard negative mining, mixed easy/hard, hard→easy init | Sec. 6.1.1 | `mine_online_hard`, `mine_offline_negatives`, `train_model(model=..., n_online_hard=...)` |
| Weighted-concatenation ensemble (Eqs. 4–6), cascade re-ranking | Sec. 6.2 | `weighted_ensemble_vectors`, `ensemble_similarity`, `cascade_rerank` |
| k-means, PQ + ADC, PCA, OPQ, IVF(Flat/PQ, residual), IMI | Sec. 4.1 | `kmeans`, `ProductQuantizer`, `PCATransform`, `OPQTransform`, `IVFIndex`, `IMIIndex` |
| 1-recall@10 vs. % scanned | Fig. 3, Table 3 | `one_recall_at_k` |
| Unicorn-style `nn` operator: cluster = term, PQ residual = payload, radius & top-K modes, Boolean s-expressions | Sec. 4.2 | `UnicornLite`, `parse_sexpr` |
| Query / index selection | Sec. 4.3 | `should_trigger_ebr`, `select_index_docs` |
| Ranking features (cosine / Hadamard / raw); human-rating relevance filter | Sec. 5 | `embedding_features`, `RelevanceFilter` |
| *(ours)* synthetic personalized people-search log | — | `SocialSearchWorld`, `Sessions` |

### Choices the paper does not specify (ours)

* The synthetic log: ~5 entities share each name; the intended target among same-name entities is the one with the highest affinity to the searcher (same primary cluster 2.0, secondary 1.0, same location 1.5, plus noise). Queries carry typos (25%) and an optional extra token (10%); impressions are the target plus same-name and similar-name entities. 32% of sessions are ones exact term matching fails on.
* **Tower design.** Feature groups are embedded, projected and L2-normalised *separately* and concatenated, so the tower's cosine is an average of per-group matches. A single MLP over the concatenated features (my first attempt) could not learn the location/social matching next to a strong text signal at this scale — the model stayed at the text-only recall@1 and degraded when given hard negatives. The paper does not say how its encoders combine feature groups.
* Triplet loss is averaged over triplets (the paper sums); online hard negatives come from the batch; margin 0.3.
* `UnicornLite` keeps postings in Python sets; the PQ residual payload verifies the radius by decoding `centroid + residual`.
* Not implemented: the social-graph embedding model, Spark inference/publishing, Unicorn's real-time updates and planner, human rating pipelines (the relevance filter is trained on synthetic labels in the tests).

---

## Demo

```bash
python3 facebook_ebr.py                 # ~30 s on CPU
python3 -m pytest test_facebook_ebr.py -v
```

Text alone can only guess among same-name entities (recall@1 ≈ 0.20 here), so **recall@1 is the personalization metric**; recall@10 saturates because the ~5 same-name entities all fit in the top 10.

```
3000 entities, 600 names (~5.0 entities per name -> text alone gives recall@1 of about 0.20); 8000 training sessions
== unified embedding vs text-only (Table 1 analogue; trained with online hard negatives) ==
  text only              recall@1 0.235  recall@10 0.962
  + location             recall@1 0.413  recall@10 0.949
  + location + social    recall@1 0.467  recall@10 0.959
== training-data mining (random-negative baseline, 600 steps) ==
  random negatives                     recall@1 0.327  recall@10 0.833
  non-click impressions as negatives   recall@1 0.017  recall@10 0.087
  hard positives only (32% of data)    recall@1 0.245  recall@10 0.689
== hard negative mining ==
  online HNM, 2 hardest in batch       recall@1 0.467  recall@10 0.959
  online HNM, 8 hardest in batch       recall@1 0.475  recall@10 0.963
  offline HNM ranks 2-6, mixed 3:1     recall@1 0.271  recall@10 0.751   (continued from the baseline)
  offline HNM ranks 7-30, mixed 3:1    recall@1 0.247  recall@10 0.671   (continued from the baseline)
== ANN tuning on the learned embeddings (1-recall@10 vs % index scanned) ==
  IVF flat               1-recall@10 0.905  scanned 14.1%
  IVF + PQ (d/4 bytes)   1-recall@10 0.890  scanned 14.1%
  IVF + PQ (2 bytes)     1-recall@10 0.570  scanned 14.1%
```

**What reproduces (direction):** searcher context in the embedding is what resolves ambiguity (recall@1 0.235 → 0.413 → 0.467 as location and social features are added); non-click impressions as negatives are catastrophic compared with random ones (the paper's −55% shows up as −95% here); online hard negative mining is the largest single modeling gain (random 0.327 → 0.467 recall@1) and more than 2 hard negatives per positive does not help (0.467 vs 0.475); `pq_bytes = d/4` loses little against flat (0.890 vs 0.905) while far fewer bytes lose a lot.

**What does not reproduce:**
* **Offline HNM** (continuing the random-negative model on negatives mined from a rank band, mixed 3:1) *hurt* here (0.327 → 0.27), and the sweep over bands never beat the baseline. The paper's finding that a deep band + easy/hard blending is best could not be shown at this scale; in my sweeps hard-only same-name negatives collapsed text matching entirely, consistent with the paper's "hard-only is worse than random" observation, but the blended recipe never recovered the gain.
* **Hard positives alone** do not match the full-data model (0.245 vs 0.327 recall@1; the paper reports similar recall with 4% of the data). Here the "failed" sessions are a biased 32% slice (all typos/extra terms), so they under-represent clean queries.
* Ensemble, cascade, Unicorn hybrid retrieval, OPQ vs PCA, IMI imbalance, query/index selection and the feedback filter are implemented and unit-tested for their stated properties (e.g. Eq. 6 holds numerically; radius mode keeps constrained matches that top-K mode loses; OPQ lowers PQ error on anisotropic data; IMI cells are imbalanced) but are **not** part of the demo numbers, so I make no end-to-end claim about their recall gains.
* Single seed; recall differences under ~0.02 are noise.

```python
from facebook_ebr import *

world = SocialSearchWorld(seed=0)
train_s, test_s = world.sample_sessions(8000, seed=1), world.sample_sessions(1500, seed=2)
store = FeatureStore(world, FeatureConfig())                       # text + location + social
model = train_model(world, store, train_s, TrainConfig(steps=600, n_online_hard=2))
print(evaluate(model, store, test_s))                              # {1: recall@1, 10: ..., 50: ...}

index = embed_index(model, store)
u = UnicornLite("m1", index, {i: {f"location:{world.entity_loc[i]}"} for i in range(world.n_entities)})
q = embed_queries(model, store, test_s.subset([0]))[0]
hits = u.search("(and (term location:3) (nn m1 :radius 0.3 :nprobe 8))", {"m1": q})
```

---

## Tests

`test_facebook_ebr.py` (67 tests, ~7 s): n-gram/hashing/padding; the world (name sharing, context-driven targets, impressions, exact matching fails on typos/extra terms, seeded reproducibility); encoders (unit norm, feature flags truly remove context dependence, social weights matter, shared towers); triplet loss; recall@K definition; that the unified model beats text-only on recall@1, and that non-click negatives are far worse than random; online/offline hard-negative selection (excludes same entity, honours the requested rank band) and hard-positive selection; Eqs. 4–6 numerically and the cascade; PQ/ADC correctness, OPQ orthogonality and error reduction, PCA, IVF (exactness with all probes, recall/scan monotone in nprobe, transform consistency), IMI cell imbalance; the s-expression parser, `UnicornLite` term/and/or semantics, cluster terms, radius NN (superset in radius and probes, constrained vs top-K) and the fuzzy hybrid rescue of a misspelled term; query/index selection; ranking features; and the relevance filter's precision gain.
