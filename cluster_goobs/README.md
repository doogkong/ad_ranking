# Cluster GOOBS — Real-Time Cluster-Based Hard Negative Sampling

Reference implementation of **Real-Time Hard Negative Sampling via LLM-based Clustering for Large-Scale Two-Tower Retrieval** (Ji, Hu, Zhao, Huang, Zhang, Fan, Singh; Meta; OARS @ RecSys 2026; arXiv 2607.00448).

Paper: https://arxiv.org/abs/2607.00448

> **Read this first — what this is and isn't.** The paper's LLM content encoder, production pool, and Meta data are not public. This folder implements the **technique and its system** — k-means clustering with online assignment, the cluster-segmented hash pool with its update/sample engines, the loss, and all six samplers from the paper's comparison — and runs them on a **synthetic interaction dataset**. **The synthetic benchmark does not reproduce the paper's headline result**: see "What the demo does and doesn't show" below before reading anything into the numbers.

---

## Summary

### Problem

Two-tower retrieval is trained as an extreme classification problem over millions-to-billions of items, so negatives are *sampled*. The common choices are weak:

* **In-batch negatives** (+ LogQ correction) are limited by batch size and, being other users' positives, are mostly semantically unrelated to the positive — easy negatives with near-zero gradient.
* **Random out-of-batch (OOB) negatives** widen exposure but are just as easy.
* Existing hard-negative methods (DNS, ANCE, cross-batch) either need extra scoring passes, a global ANN index, or pick by recency rather than meaning, so they are rarely used at industrial scale.
* Retrieval is trained on clicks, which depend on what the multi-stage system showed, creating a **popularity feedback loop**.

### Idea

Draw extra negatives from **the same semantic cluster as the positive item**. Items in a cluster share latent characteristics, so their score under the model is already close to the positive's: the softmax mass (and gradient) on them is large, which pushes the model to learn *fine* intra-cluster distinctions rather than trivial inter-cluster ones. This matches the ANCE theory that negatives near the query give larger gradients; but cluster membership is known statically, with no global index.

```
L = −r_i · log( e^{s(x_i,y_i)} / ( e^{s(x_i,y_i)} + Σ_k e^{s(x_i, y⁻_ik)} ) ),    s = v_iᵀ u_i
∂L/∂s(x, y⁻) = softmax prob of y⁻  ⇒ grows with s(x, y⁻), which same-cluster negatives raise
```

The paper also reports an important failure mode: too many in-cluster negatives per positive degrades training (they over-weight the contrastive term and destabilise it). Sweeping 1 → 64 per positive, a **small number is best**.

### Cluster generation (Sec. 3.4)

Item clusters come from **k-means over embeddings from a fine-tuned multimodal LLM encoder** (text + image + audio + video → fusion → LLM fine-tuned for concept understanding). New items get the nearest centroid online. Granularity k is the key knob: too fine → in-cluster items are effectively relevant (false negatives); too coarse → negatives aren't hard. A false-negative argument at scale: a user likes at most thousands of items out of billions, so in a cluster of ~10⁴ items the relevant fraction is well under 1%. Production uses ~300 clusters (swept 100–10,000; 250–500 best), 98% with ≥10⁴ items.

### Real-time system (Sec. 4) — Global Out-of-Batch Sampling (GOOBS)

A preallocated **item pool** of feature tensors, split into **per-cluster segments of S slots**; each slot holds one item's features.

* **Update engine** (Alg. 1, Fig. 3): for each in-batch item, `slot = cluster_id · S + item_id mod S`. Colliding items overwrite earlier ones *by design* — the pool stays fresh and its memory is bounded by `n_clusters · S` however large the corpus.
* **Sample engine** (Alg. 2, Fig. 4): for each in-batch example, look up its positive's cluster segment and take a random slot within it. Features come straight from the pool, so the negatives are encoded by the current item tower like any other item.
* Pool is **pre-loaded** from an item table so the sampling hit rate is high from step 0, then continuously refreshed by training batches.
* In training, cluster negatives are **mixed with random OOB samples** (cluster:random = 1:15 on MovieLens, 1:31 on Amazon). No extra index, no extra scoring pass, plugged into the existing training/serving path.

### Reported results

| Setting | Result |
|---|---|
| Public (MovieLens-1M, Amazon Grocery/Electronics/Home; global HR@50/100, no k-core, no sampled eval) | Cluster GOOBS best everywhere: HR@50 gains over in-batch **+7.2%** (ML-1M) to **+55.6%** (Electronics); e.g. Home +47.3% HR@50 vs ANCE +30.0%, GOOBS +34.2%. Random-OOB GOOBS alone already +4.2% to +34.2%. Gains larger on HR@50 than HR@100. Public runs use genre/category labels as clusters — the LLM clustering is only evaluated in production. |
| Production A/B (14 days, 3% of users per arm, ~100M MAU, tens of millions of posts) vs GOOBS | **+53% CTR on the retrieval source**, **+6.5% overall system CTR**, **−1.4% training QPS** (no inference QPS regression) |
| Popularity bias | items with ≥1K impressions in a day **+50%**; top-100 items' impression share **50% → 32%** |

### Takeaways

1. Out-of-batch exposure alone helps; *semantic* hard negatives help more, and beat recency (CBNS), model-score (DNS) and ANN-geometry (ANCE) selection **without** a global index.
2. The system design (hashed per-cluster segments, overwrite-on-collision, pre-load) is what makes it deployable: O(1) update and sample, fixed memory, negligible training overhead.
3. Cluster granularity is a precision/false-negative trade-off; keep it relatively coarse.
4. A better training signal for tail items also reduces the feedback-loop popularity bias.

---

## What is implemented

| Component | Paper | Code |
|---|---|---|
| k-means + online assignment of new items | Sec. 3.4 | `kmeans`, `ContentClusterer` |
| Cluster-segmented pool; hash `c·S + id mod S`; overwrite on collision; preload | Sec. 4, Alg. 1, Fig. 3 | `ClusterItemPool.slot / update / preload` |
| In-cluster sampling (random slot inside the positive's segment) | Alg. 2, Fig. 4 | `ClusterItemPool.sample_in_cluster` |
| Random OOB sampling over filled slots (GOOBS) | Sec. 4 | `ClusterItemPool.sample_random`, `GOOBSampler` |
| Cluster GOOBS with cluster:random ratio | Sec. 5.1 | `ClusterGOOBSampler(n_cluster, random_per_cluster)` |
| Loss with labels `r_i`, in-batch LogQ, false-negative mask (known positives and equal ids) | Sec. 3.1–3.2, 5.1.2 | `contrastive_loss` |
| Baselines: in-batch+LogQ, DNS, CBNS, ANCE | Sec. 5.1.2 | `NegativeSampler`, `DNSSampler`, `CBNSSampler`, `ANCESampler` |
| Global HR@K over the whole corpus (no sampled metrics) | Sec. 5.1.1 | `evaluate`, `retrieval_scores` |
| False-negative rate of a same-cluster draw (granularity risk) | Sec. 3.4 | `false_negative_rate` |
| Popularity-bias report (head share, coverage, items over a threshold) | Table 3 | `popularity_report` |
| *(ours)* synthetic interactions, two-tower model | — | `SyntheticInteractions`, `TwoTower` |

### Choices the paper does not specify (ours)

* Towers are L2-normalised with a temperature (τ = 0.1); the paper writes `s = vᵀu`.
* The item tower takes dense features + cluster embedding + item-id embedding; user tower takes a user-id embedding + a profile vector (mean of the user's training-item features).
* In-cluster and random OOB negatives are drawn **per example**; the paper's description of how OOB batches are shared across examples is not detailed. LogQ is applied to in-batch negatives only.
* LogQ uses empirical training frequencies rather than a streaming estimate.
* DNS scores a random candidate set and keeps the top-k; ANCE refreshes an exact full-corpus index every 50 steps (stand-in for the asynchronous ANN refresh).
* Empty pool slots and any sampled item equal to the positive are masked out of the softmax.

---

## What the demo does and doesn't show

`SyntheticInteractions`: 6,000 items in latent clusters, users with latent tastes, popularity skew; clusters used by the sampler are recovered by k-means on a *noisy* content embedding (stand-in for the LLM encoder), not ground truth. Evaluation is global HR@K over the full corpus, train items excluded.

```bash
python3 cluster_goobs.py                      # ~20 s on CPU
python3 -m pytest test_cluster_goobs.py -v
```

```
2000 users, 6000 items, 150 k-means clusters (size min/median/max = 1/37/81)
false-negative rate of a same-cluster draw: 0.0207
  in-batch       HR@50 0.3723  HR@100 0.5155   vs in-batch:   +0.0%   +0.0%
  dns            HR@50 0.5542  HR@100 0.6762   vs in-batch:  +48.9%  +31.2%
  cbns           HR@50 0.3217  HR@100 0.4651   vs in-batch:  -13.6%   -9.8%
  ance           HR@50 0.4431  HR@100 0.5612   vs in-batch:  +19.0%   +8.9%
  goobs          HR@50 0.3932  HR@100 0.5328   vs in-batch:   +5.6%   +3.3%
  cluster_goobs  HR@50 0.3997  HR@100 0.5396   vs in-batch:   +7.4%   +4.7%
```

**Honest reading.**

* Reproduced: out-of-batch exposure helps (GOOBS > in-batch); CBNS (recency) is the weakest; cluster GOOBS is at least as good as GOOBS in this run.
* **Not reproduced:** the paper's claim that cluster negatives clearly beat the alternatives. Here Cluster GOOBS vs. plain GOOBS (+7.4% vs +5.6% HR@50) is within run-to-run noise — across the other synthetic configurations I tried (different cluster counts, cluster/fine scales, 1–8 in-cluster negatives, with/without ID embeddings) the two were indistinguishable, and **DNS (+49%) and ANCE beat both**. In this toy world the model-score-based hard negatives are far more informative than same-cluster draws, and when item features alone determine relevance every sampler saturates at the same HR.
* Plausible reasons (untested): synthetic users' tastes vary smoothly in a 6-d latent space so a random same-cluster item is only moderately hard; the paper's gains come from billions of items and ~10⁴-item clusters where in-batch contains almost no same-cluster items (here, with 150 clusters and batch 128, in-batch already holds ~1 per example); and its production system also benefits from LLM-derived clusters that capture semantics our noisy content embedding only approximates.
* Seed-to-seed/thread nondeterminism changes the third decimal; a single seed is shown. Treat differences below ~1–2% as noise.

What the tests *do* verify is the mechanism the paper argues from: same-cluster negatives are nearer the positive than random ones, a hard negative gets a much larger gradient than an easy one, the false-negative rate rises with cluster fineness, and the pool implements exactly the hashing/overwrite/segment behaviour of Algorithms 1–2 (including the Fig. 3 worked example).

```python
from cluster_goobs import (SyntheticInteractions, ContentClusterer, TwoTower, ItemCatalog,
                           build_pool, make_sampler, train, evaluate)

data = SyntheticInteractions(seed=0)
clusters = ContentClusterer(150).fit(data.content_emb).assign(data.content_emb)
model = TwoTower(data.n_users, data.n_items, 150, 6, 6)
pool = build_pool(data, clusters, 150, slots=20)               # preloaded from the item table
sampler = make_sampler("cluster_goobs", ItemCatalog(data.item_feats, clusters), pool,
                       n_cluster=1, random_per_cluster=15)      # 1:15 as on MovieLens
train(model, data, clusters, sampler, steps=1000)
print(evaluate(model, data, clusters))                          # {50: HR@50, 100: HR@100}
```

---

## Tests

`test_cluster_goobs.py` (45 tests, ~3 s): k-means recovery and online assignment; the pool (the paper's `S + 12 % S` example, slots always inside the cluster's segment, fixed memory, overwrite-on-collision, in-cluster sampling stays in cluster, unfilled slots flagged, random sampling spans clusters, preload); the loss (matches a hand-written cross-entropy, false-negative and equal-id masking, invalid-slot masking, LogQ discount, label-0 rows, gradients to negatives, hard-vs-easy gradient magnitude); every sampler (train step, cluster membership, DNS picks high scorers, ANCE index stays stale until refresh and excludes the positive, CBNS bounded and detached); data splits and logQ; false-negative rate rising with cluster fineness; train-item masking, HR monotone in K, popularity report; and short end-to-end trainings.
