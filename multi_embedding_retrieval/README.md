# Multi-Embedding Retrieval (Implicit + Explicit User Interests)

Reference implementation of **Synergizing Implicit and Explicit User Interests: A Multi-Embedding Retrieval Framework at Pinterest** (Fan, Lin, Chen, Deng, Xia, Yan, Li; Pinterest; KDD 2025).

Paper: https://arxiv.org/abs/2506.23060

> **Read this first — what this is and isn't.** Pinterest's data, PinSage/PinnerFormer features, ANN service and production-scale DHEN are not public. This folder implements the **method** (DCM with VA-FPI and single-assignment routing, Conditional Retrieval, the association/loss, budgeted round-robin serving, and the paper's baselines) at small scale, and exercises it on a **synthetic multi-interest world**. The numbers below show the mechanism behaves as the paper argues; they do not reproduce Pinterest's online results.

---

## Summary

### Problem

A two-tower retrieval model compresses a user into **one** embedding. Dominant interests overwhelm it, so candidates for torso and long-tail interests are rarely retrieved (the paper's Fig. 1: a user who saved food, education and fashion Pins gets only fashion). Retrieval sets the ceiling on recommendation quality, so poor interest *coverage* here cannot be repaired by ranking.

### Idea: condition-based multi-embedding retrieval

Generate K user embeddings, each conditioned on one interest: `f(i|u) → f(i|u,c) ∝ exp(φ(u,c)ᵀψ(i))`. Both of the paper's models are "conditional representation learning" and differ in two things only — **condition construction** (where interests come from) and **condition association** (which positive item trains which condition):

| | Implicit (DCM) | Explicit (Conditional Retrieval) |
|---|---|---|
| Condition source | clustering of the engagement sequence | followed topics (sign-up / user-edited) |
| Construction | Differentiable Clustering Module | topic-embedding table, fed into the user tower's embedding layer |
| Association | inside the network: `j* = argmax_j o_uʲ · o_y` | at **logging time**: the topic that sourced the engagement |
| Strength | quickly tracks live behaviour; helps active (core) users | recovers forgotten / long-term / new interests; helps non-core users; controllable |

Average overlap between candidates retrieved by the two is only **3.2%** — they are genuinely complementary.

### Differentiable Clustering Module (Sec. 3.2)

Starts from Capsule Networks (MIND): item embeddings `e_i` (Eq. 4: 2-layer MLP over item features) are routed to K centroids `c_j` with a shared bilinear map `S`:

```
b_ij ← softmax_j( c_jᵀ S e_i )          (Eq. 1)       c_j ← squash( Σ_i b_ij S e_i )   (Eq. 2)
squash(v) = ||v||²/(1+||v||²) · v/||v||  (Eq. 3)
```

DCM changes initialization and routing, both of which matter for centroid quality:

1. **Validity-Aware Farthest Point Initialization (VA-FPI)** — instead of Gaussian init, pick the first centroid at random, then repeatedly the item *farthest* from the chosen ones (Eq. 5: `i* = argmin_i max_j c_jᵀ e_i`), **restricted to valid items** (Eq. 6) — items with missing/negative features produce out-of-distribution embeddings that would otherwise be picked as "farthest" and collapse the clustering.
2. **Single-Assignment Routing (SAR)** (Eq. 7) — zero all but each item's max-affinity centroid, so nearby centroids stop contributing to each other and diverge instead of converging.

Routing mass `Σ_i b_ij` is kept as each interest's **importance**, used to size its retrieval budget at serving time.

### Association and loss (Eqs. 8–9)

Given the target item `y`, pick `j* = argmax_j o_uʲ·o_y` and apply a sampled softmax with logQ correction **on that embedding only** (negatives use only `o_u^{j*}`, so the other K−1 embeddings don't cost memory):

```
L = − log  exp(o_u^{j*}·o_y − log p_y) / ( exp(o_u^{j*}·o_y − log p_y) + Σ_{k∈B} exp(o_u^{j*}·o_k − log p_k) )
```

`p` is the item sampling probability from a streaming estimator [Yi et al. 2019].

For the self-attention and interest-token baselines, a hard argmax causes a "winner-takes-all" collapse (random-init conditions → only one gets gradient → shared params make all embeddings identical). The paper uses a **Straight-Through Gumbel-Softmax** there so every embedding receives gradient; DCM, whose conditions are diverse by construction, does not need it.

### Explicit interests (Sec. 3.3)

* **Construction** — the followed-topic embedding goes into the user tower's embedding layer, then through the feature-crossing layers (DHEN: Transformer + MLP, then MaskNet + MLP).
* **Association** — prior work (Lin et al.) sampled a topic from item→topic signals at *training* time, which can disagree with the user's real followed topics. Here association is recorded at *logging* time from the inverted-index retriever that surfaced the item, so training matches serving.
* **Explicit relevance filter** — post-filter retrieved items by item-to-topic signals; the biggest online win among the explicit variants.

### Serving (Sec. 3.5)

K_im = 7 implicit and K_ex = 5 explicit embeddings, one ANN search each. Implicit budgets ∝ routing mass `Σ_i b_ij` (don't over-fetch torso/tail); explicit budgets are equal over randomly sampled followed topics. Candidates are merged **round-robin with dedup** so no interest dominates and ranking is delayed to later stages. p90 embedding-retrieval latency rose 150 → 205 ms (user tower runs once per request).

### Reported results

| Experiment | Result |
|---|---|
| Offline implicit (HR@100 / HR@1000) | self-attention .167/.470 · interest token .180/.474 · MIND .175/.464 · **DCM .185/.476** |
| Offline explicit, filtered HR@100 | CR w/ item-interest .164 → CR w/ **source interest** .191 (HR@1000 .541 → .565) |
| Online implicit (HF Repins / A-Pincepts, all users) | DCM **+0.86% / +0.46%** (core +1.23% / +0.87%); self-attention +0.83/+0.44; MIND +0.43/+0.04 |
| Online explicit vs inverted index | CR w/ filter: HF Repins +0.56% (non-core +1.13%); w/ source interest +0.98% (non-core **+3.04%**) |
| **Full framework** | sitewide Repins **+0.48%**, HF Repins **+1.09%**, A-Pincepts **+0.81%** |
| Ablation (DCM, K=7) | VA-FPI and SAR together best (HR@100 .185); K=2 hurts online (−0.53% HF Repins), K=4 ≈ neutral, K=7 best |

Setup: 6B engagement records / 160M users, 14 days train + 1 day eval; sampled softmax with logQ; HR on a 1M-item corpus, an item's score = max over the user's embeddings.

### Things worth remembering

* Validity filtering is not cosmetic — without it centroids collapse onto one point (their Fig. 5c).
* Offline gains are small (+0.01 HR@100) but online gains are meaningful: interest *coverage* matters more than top-k accuracy for the metric they care about (diversity tracks Repins).
* Making a fixed number of clusters per user adaptive hurt (−0.12% HF Repins) vs. fixed K.
* PinnerFormer subsequences (PFS; fixed offline concepts, not end-to-end) underperform DCM: clusters can't adapt to the objective, the signal isn't consumed in real time, concept granularity is fixed.

---

## What is implemented

| Component | Paper | Code |
|---|---|---|
| Feature crossing: Transformer ∥ MLP → MaskNet ∥ MLP | Appendix A | `FeatureCrossing`, `MaskNetBlock` |
| User tower φ(u,c) (condition as a field token), item tower ψ(i) | Sec. 3.1 | `UserTower`, `ItemTower` |
| Item summarisation `e_i = W₂ GELU(W₁ concat f)` | Eq. 4 | `ItemSummarizer` |
| Squash | Eq. 3 | `squash` |
| **VA-FPI** | Eqs. 5–6 | `validity_aware_fpi` |
| **DCM** (routing, SAR, validity masking, importance) | Eqs. 1, 2, 7 | `DifferentiableClusteringModule` |
| Vanilla Capsule / MIND (Gaussian init, multi-assignment) | Sec. 3.2.1 | same class, `init="gaussian", single_assignment=False` |
| Self-attention, interest tokens, PFS-style concept pooling | Sec. 3.2.4, App. B | `SelfAttentionConditions`, `InterestTokenConditions`, `ConceptPoolConditions` (+ `kmeans`) |
| argmax association; ST-Gumbel-Softmax association | Eq. 8, Sec. 3.2.4 | `select_embedding` |
| Sampled-softmax + logQ loss | Eq. 9 | `sampled_softmax_logq_loss` |
| Streaming item-frequency estimator | ref. [36] | `StreamingFrequencyEstimator` |
| Implicit model / Explicit (CR) model | Secs. 3.2, 3.3 | `ImplicitInterestModel`, `ExplicitInterestModel` |
| Budget allocation by routing mass, round-robin merge w/ dedup | Sec. 3.5 | `allocate_budget`, `round_robin_merge`, `retrieve_multi_embedding` |
| Explicit relevance filter | Sec. 3.3.3 | `filter_by_topic` |
| HR@k with max-over-embeddings scoring; overlap; embedding diversity | Sec. 4.1 | `hit_rate`, `multi_embedding_scores`, `candidate_overlap`, `embedding_diversity` |
| *(ours)* synthetic multi-interest world | — | `SyntheticWorld` |

### Choices the paper does not specify (ours)

* `S` starts as the identity so `S e_i ≈ e_i` (Eq. 5 uses `e_i`, Eq. 1 uses `S e_i`). Three routing iterations.
* The softmax in the loss uses a temperature (0.07); with unit-norm embeddings it is needed in practice and Eq. 9 omits it.
* Hash buckets in the frequency estimator are `id % size`.
* Empty centroids (no assigned items under SAR) and VA-FPI repeats (fewer valid items than K) are masked out of association, scoring and serving rather than used.
* Feature crossing is far smaller than the production DHEN (2-layer Transformer, one MaskNet with 4 blocks).
* "both" in the demo = a hit if the target is in the implicit **or** the explicit retriever's top-k/2, so it spends the same total budget k as a single retriever (the two retrievers use separate item towers, as in the paper's separately trained models).

---

## Synthetic world and demo

`SyntheticWorld`: 12 topics × 40 items (feature = topic prototype + noise, Zipf popularity inside a topic, ~5% of items have out-of-distribution "invalid" features). Each user has 3 history interests with weights 0.6 / 0.3 / 0.1, and follows 2 topics — one from their history and one **not** in it. Targets are drawn **uniformly** over a user's interests, so the tail matters as much as the head.

```bash
python3 multi_embedding_retrieval.py            # ~40 s on CPU
python3 -m pytest test_multi_embedding_retrieval.py -v
```

Output (seed 0, 1000 test users, 480-item corpus; chance HR@50 ≈ 0.10):

```
== implicit interest modeling (targets uniform over 3 history interests) ==
  single embedding (K=1)   HR@10 0.348   HR@50 0.592
  DCM (K=3)                HR@10 0.398   HR@50 0.728
  DCM (K=4)                HR@10 0.391   HR@50 0.710
  MIND capsule (K=4)       HR@10 0.369   HR@50 0.657
  self-attention (K=4)     HR@10 0.355   HR@50 0.615
== synergy: targets uniform over history interests + followed topics ==
  implicit   HR@10 0.291   HR@50 0.544
  explicit   HR@10 0.307   HR@50 0.524
  both       HR@10 0.328   HR@50 0.629
```

The ordering (DCM > MIND > self-attention > single embedding; implicit + explicit > either alone) matches the paper's direction. These are single-seed toy numbers; K=3 vs K=4 differences are within noise (the true number of interests here is 3).

```python
import torch
from multi_embedding_retrieval import (SyntheticWorld, ImplicitInterestModel, ExplicitInterestModel,
                                       train_implicit, train_explicit, eval_joint)

w = SyntheticWorld(seed=0)
imp = ImplicitInterestModel(w.Fi, w.Fu, d=32, n_interests=4, kind="dcm")   # dcm | mind | self_attention | interest_token
exp = ExplicitInterestModel(w.Fi, w.Fu, w.T, d=32)
train_implicit(imp, w, steps=300)
train_explicit(exp, w, steps=300)
users = w.sample_users(1000)
print(eval_joint(imp, exp, w, users, w.joint_targets(users)))
```

Serving-side pieces compose as: `allocate_budget(cond.importance[u], total)` → `retrieve_multi_embedding(user_embs[u], budgets, ItemIndex(item_embs), total)`; for explicit interests use equal budgets and `filter_by_topic` on the result.

---

## Tests

`test_multi_embedding_retrieval.py` (80 tests, ~20 s): feature crossing and tower shapes/normalisation, K-folding equivalence; squash properties; **VA-FPI** (one centroid per cluster, diverse, never picks invalid items, picks the outlier without the validity filter, duplicates flagged when items < K); **DCM** (shapes, gradients to features and `S`, SAR mass bounds vs multi-assignment mass = sequence length, padding/invalid items provably ignored, empty history, and that DCM yields more diverse conditions than vanilla capsules); all baseline builders; association (argmax gradient only to the winner, Gumbel forward hard / backward to all, invalid embeddings never selected); loss (alignment, logQ discount, duplicate-id masking); frequency estimator; budget allocation (exact totals, proportionality, degenerate cases); round-robin merge (dedup, cap, tail not crowded out); retrieval, filter, overlap; HR / max-over-embeddings scoring; synthetic-world structure; and end-to-end checks that multi-embedding DCM beats a single embedding on tail interests, that explicit retrieval recovers an interest absent from the history, and that implicit + explicit beats either alone.
