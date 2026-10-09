# CoGR

Python reference implementation of **It Takes Two to Match: Co-Evolving Generative Retriever with Reinforcement Learning** (Dai, Huang, Kang, Liao — UNC Chapel Hill / Apple, 2026).

Paper: https://arxiv.org/abs/2609.00638

On an internal APP Marketplace dataset and the public WANDS product-search benchmark, CoGR improves F1 over the strongest of 10 sparse/dense/generative baselines by **+10.9%** (0.396 vs 0.358, ANCE-Qwen4B) and **+36.1%** (0.682 vs 0.501) respectively.

---

## Summary

Retrieval is the first stage of search and ads: an item that is not retrieved cannot be recovered later. LLM-based retrieval work mostly uses the LLM on the *query side only* (expansion, rewriting) and still hands final matching to a separate retriever. CoGR trains LLMs to build the retrieval representation on **both sides**. A query generator and an item generator each emit a compact keyword set (≤ 30), and matching is a plain **inverted-index keyword overlap**, so it drops into existing keyword-based ad infrastructure.

The hard part is aligning two independently generated keyword spaces. CoGR does this in two phases:

1. **SFT** builds an aligned initialization: the base LLM generates item keywords; each query's target is the top-N most frequent keywords pooled across its relevant items (Alg. 1), which guarantees overlap between relevant query–item pairs.
2. **Co-evolving RL** alternates GRPO updates between the sides, each against the *frozen* index of the other (Alg. 2), so each generator sees a fixed environment rather than a moving target.

---

## Key Ideas

### Query-side reward (Eq. 2.1–2.2)
Retrieval F1 of the set matched by the generated keywords, forced to 0 if `|S_q| > K_max`. F1 balances precision (user experience) against recall (coverage passed to ranking/bidding).

### Item-side counterfactual marginal reward (Eq. 2.3)
An item has no query of its own to be scored on, so its keywords are judged by their effect on the *whole* query-side objective: replace only item `i`'s keywords in the reference index with the sampled `S_i`, and reward
`R_i = Σ_q F1_cand(q) − Σ_q F1_ref(q)` (−1 if over budget). Ablation: this beats a symmetric "transposed F1" item reward (0.3963 vs 0.3743 F1).

### Efficient reward computation (Appendix B.3)
Replacing one item's keywords only adds/removes that item from the results of queries in `Q_cand △ Q_ref`. Cache per-query `(n_ret, n_tp, n_rel)` once per round, then every rollout costs one query-index lookup plus an O(1) update `F1 = 2·tp / (n_ret + n_rel)` per affected query. Unaffected queries cancel. `ItemRewardCache` implements this and is tested against a literal rebuild-everything implementation.

### Findings
Keywords become more specific over training (unigrams 37% → 13%, 3+-word phrases 12% → 31%); query/item vocab sizes converge after ~3–4 rounds. Ablations: separate generators > shared (0.3963 vs 0.3798); SFT init > none (0.3963 vs 0.3751).

---

## Key Components

| Component | Paper section |
|---|---|
| `InvertedIndex`, `representation` | Sec. 2.1 — `I_ret(q) = {i : (S_q ∪ {q}) ∩ (S_i ∪ {i}) ≠ ∅}` |
| `bm25_rank` | Sec. 2.1 — BM25 over generated keyword bags |
| `precision_recall_f1`, `f1_from_counts`, `evaluate`, `mrr_at_k`, `ndcg_at_k` | Eq. 2.1, Table 2 metrics |
| `query_reward` | Eq. 2.2 |
| `item_reward_naive` | Eq. 2.3 (literal, oracle) |
| `ItemRewardCache` | Eq. 2.3 via Appendix B.3 |
| `build_sft_query_targets` | Alg. 1 |
| `grpo_advantages`, `grpo_loss`, `grpo_train` | Sec. 2.3 — group-normalised advantages, clipped surrogate, no KL (Table 6), n=8 rollouts |
| `co_evolve` | Alg. 2 |
| `KeywordGenerator`, `ToyKeywordGenerator` | Stand-in for the Qwen3 generators |

**Scope note.** The paper's generators are Qwen3-4B/1.7B trained with verl on 8 GPUs; none of the datasets are included. This implementation reproduces everything except the LLM: the generator is a pluggable interface, and `ToyKeywordGenerator` (hashed bag-of-tokens → per-keyword Bernoulli over a closed vocabulary) lets the full SFT → co-evolving RL pipeline run on CPU. It demonstrates the algorithm, not the paper's numbers. To use a real LLM, implement `sample`, `log_prob` and `generate` on `KeywordGenerator`.

---

## Usage

```python
from cogr import (InvertedIndex, ItemRewardCache, ToyKeywordGenerator,
                  build_sft_query_targets, co_evolve, evaluate, sft_train)

targets = build_sft_query_targets(train_q, init_item_kws, rel, top_n=15)   # Alg. 1
sft_train(q_gen, train_q, [targets[q] for q in train_q])
sft_train(i_gen, items, [init_item_kws[i] for i in items])
co_evolve(q_gen, i_gen, train_q, items, rel, rounds=5, k_max=30)           # Alg. 2

idx = InvertedIndex(i_gen.generate(items, 30))
print(evaluate(val_q, q_gen.generate(val_q, 30), idx, rel))
```

## Files

```
CoGR/
├── cogr.py        # implementation
├── demo.py        # synthetic many-to-many end-to-end run
├── test_cogr.py   # pytest suite (49 tests)
└── README.md
```

## Running

```bash
python3 demo.py
```

Example output (synthetic data, CPU, ~2s):

```
     after SFT: P=0.154 R=0.486 F1=0.231 MRR@20=0.222
  round 1: R_q=0.406 R_i=-0.001
  round 2: R_q=0.434 R_i=0.003
  round 3: R_q=0.467 R_i=0.020
after co-evolve: P=0.190 R=0.403 F1=0.257 MRR@20=0.284
```

```bash
python3 -m pytest test_cogr.py -v     # 49 passed
```

Key tests: `test_incremental_matches_naive` (B.3 cache equals full recompute across random worlds), `test_clipping_blocks_gradient_beyond_trust_region`, `test_target_creates_overlap_with_relevant_items`, `test_co_evolving_improves_training_reward`.
