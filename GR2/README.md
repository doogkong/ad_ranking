# GR2

PyTorch reference implementation of the algorithmic components of **GR2: Generative Reasoning Re-Ranker**, from the *GR2 Technical Report* (Meta AI, Jul 2026).

Paper: https://arxiv.org/abs/2606.31984

Delivers **+18.7% R@1**, **+7.1% R@3**, **+9.6% N@3** over a heavily-tuned legacy re-ranking baseline on industrial-scale traffic, with gains that hold across test-set scale, 9 days of traffic staleness, and model size up to 32B — while a 1.7B distilled student recovers ~82% of a 32B teacher's gain at ~15x lower serving cost.

---

## Summary

GR2 is not a new encoder architecture like RankMixer or HSTU — it's a **four-stage training recipe** for turning a pretrained LLM into the final re-ranking stage of an industrial recommender, the stage closest to what a user actually sees (carousel/grid position dominates engagement). The recipe: (1) mid-train on tokenized Semantic IDs so the model can reason over a billion-item, non-semantic catalog; (2) activate chain-of-thought reasoning by distilling from a stronger teacher; (3) sharpen ranking behavior with RL on **verifiable** rewards (AUC/NDCG, not human preference); (4) shrink serving cost via context compression and reasoning internalization.

Because most of that recipe's real content is *objective design* — losses and rewards — rather than a forward-pass architecture, this repo implements the paper's equations as a standalone, LLM-agnostic **training-objective toolkit**: the Semantic-ID tokenizer, the SFT/OPD losses, the ranking rewards, the DAPO RL objective, and the context-compressor reward. Each is testable with synthetic tensors, without needing an actual multi-billion-parameter model.

---

## Key Ideas

### Three industrial gaps the paper identifies

1. **(G1) Reasoning left on the table.** LLM rerankers are typically zero-shot or SFT'd on plain labels, never given RL on verifiable rewards — the mechanism that most reliably elicits chain-of-thought.
2. **(G2) Vocabulary mismatch.** Billions of catalog items are non-semantic IDs outside any base-LLM vocabulary, so the model can't reason about candidates directly without a tokenizer that maps items into it.
3. **(G3) Industrial-scale tax.** Naive recipes are too expensive to train and serve, and — critically — **RL reward hacking inflates offline metrics without improving real ranking quality** unless the reward function is designed against it.

### GR2's four stages, and what's implemented here

| Stage | Paper mechanism | This repo |
|---|---|---|
| 1. Mid-training | Tokenize items into Semantic IDs (SIDs) via a residual quantizer, mid-train the LLM to recognize them | `RQKMeansTokenizer` |
| 2. Reasoning enhancement | Distill CoT traces from a stronger teacher via **targeted sampling** (ground-truth revealed) and **rejection sampling** (iterative verification); train with a reasoning/ranking-weighted SFT loss; or use **On-Policy Distillation (OPD)** as a scalable alternative to SFT | `build_targeted_prompt`, `build_rejection_prompt`, `rejection_sample`, `sft_loss`, `opd_loss` |
| 3. RL post-training | Sharpen ranking with **DAPO** on verifiable rewards: multi-positive AUC/NDCG, gated against two reward-hacking paths | `auc_reward`, `ndcg_reward`, `conditional_reward`, `group_relative_advantage`, `dynamic_sampling_filter`, `dapo_loss` |
| 4. Serving ROI | A learned context compressor cuts input tokens >80% at iso-quality; CoT is internalized into a reasoning-free policy for cheap serving | `compressor_reward` |

**The headline finding**: OPD supplies the reasoning prior, RL sharpens the ranking objective on top of it — *neither is sufficient alone* (an RL-only policy learns to rank well without "thinking well"; SFT alone collapses at industrial scale). And **reward design is not optional**: naive rewards let the policy earn credit by parroting the input order or exploiting position bias instead of actually reasoning — the whole point of `conditional_reward`.

---

## Key Components

### 1. Semantic-ID Tokenization (`RQKMeansTokenizer`, Sec 2.1)

`Tokenizer(x) = (z_1,...,z_K)`, a K-level residual quantizer: quantize against a codebook, pass the residual to the next level, repeat. The paper's production core is an RQ-VAE; this is the simpler gradient-free **RQ-KMeans** alternative (Lloyd's algorithm per level), which reproduces the paper's key tokenizer-quality metric directly: **SID uniqueness across the catalog** (paper targets ≥99%).

### 2. Reasoning-trace generation (Sec 3.2)

- **Targeted sampling** (Eq. 1): reveal the ground-truth target to the teacher and ask it to explain why the user would favor it — always "correct," but can encode post-hoc, label-aware shortcuts rather than causal reasoning.
- **Rejection sampling** (Eq. 2): don't reveal the target; keep resampling until the teacher's own prediction matches it. `rejection_sample` implements the accept/discard loop against a caller-supplied `teacher_fn`.

### 3. SFT loss (Eq. 3)

`L_SFT = -λ_r · Σ log P(reasoning tokens) - λ_o · Σ log P(ranking tokens)` — reasoning and ranking segments are weighted separately (λ_r < λ_o) so the model is pushed harder on getting the ranked list right than on any particular phrasing of the rationale.

### 4. On-Policy Distillation — OPD (Eq. 4)

A GRPO-style clipped surrogate computed on the **student's own rollouts**, plus a per-token reverse-KL anchor to a frozen teacher (estimated from sampled-token log-probabilities via the standard unbiased "k3" estimator). The teacher only ever contributes log-probabilities, never labels. This is what lets OPD avoid both SFT failure modes: it never trains on states the student won't visit at deployment, and every prompt contributes signal regardless of whether the teacher could solve it (unlike rejection sampling, which silently discards the hardest examples).

### 5. Ranking rewards (Eq. 5-6)

- `auc_reward`: per-impression AUC of the predicted permutation against **multi-positive** binary labels — handles slates with several positives natively, unlike a single-target rank-delta.
- `ndcg_reward`: graded-relevance NDCG@K (e.g. {none, click, click+conversion}).

### 6. De-hacked conditional reward (Eq. 7)

Zeros the ranking reward in two reward-hacking scenarios the paper explicitly identifies: (1) an unparseable output that could otherwise still earn partial-parse credit, and (2) the model re-emitting the **identity permutation** (the input order, unchanged) when that input order is *not* already optimal — i.e., the policy dodging the re-ranking task entirely while still collecting format reward.

### 7. DAPO objective (Eq. 8-9)

`group_relative_advantage` z-scores a group of G rollouts' rewards for the same prompt; `dynamic_sampling_filter` drops prompts whose group is all-correct or all-incorrect (zero gradient signal either way); `dapo_loss` is the clipped surrogate normalized by the **total token count across the group** — DAPO's "decoupled" token-level normalization, which fixes GRPO's bias toward under-weighting longer rollouts.

### 8. Context-compressor reward (Eq. 10-11)

An LLM-judge-scored reward blending a compression-ratio term with a solvability-conditioned judge term. When the compressed input is still solvable, ranking quality is weighted far more than raw information preservation (0.8 vs 0.2) — the paper finds preservation correlates only weakly with downstream ranking quality.

---

## Usage

```python
from gr2 import (
    RQKMeansTokenizer, rejection_sample, sft_loss, opd_loss,
    auc_reward, ndcg_reward, conditional_reward,
    group_relative_advantage, dynamic_sampling_filter, dapo_loss,
    compressor_reward,
)

# Sec 2.1 — tokenize a catalog into Semantic IDs
tokenizer = RQKMeansTokenizer(num_levels=3, codebook_size=256, dim=64).fit(catalog_embeddings)
sids = tokenizer.encode(catalog_embeddings)
print(tokenizer.uniqueness(sids))  # target: >= 0.99

# Sec 4 — RL reward + DAPO update for one training step
rewards = torch.tensor([auc_reward(r, y) for r, y in zip(pred_ranks, labels)])
keep = dynamic_sampling_filter(rewards.view(num_prompts, G))
advantages = group_relative_advantage(rewards)
loss = dapo_loss(importance_ratios, advantages, token_mask)
loss.backward()
```

---

## Files

```
GR2/
├── gr2.py         # full implementation + smoke test
│   ├── RQKMeansTokenizer       # Sec 2.1: residual K-means Semantic-ID tokenizer
│   ├── build_targeted_prompt   # Eq. 1
│   ├── build_rejection_prompt  # Eq. 2
│   ├── rejection_sample        # Eq. 2
│   ├── sft_loss                # Eq. 3
│   ├── opd_loss                # Eq. 4
│   ├── auc_reward               # Eq. 5
│   ├── ndcg_reward              # Eq. 6
│   ├── conditional_reward       # Eq. 7 (de-hacked reward)
│   ├── group_relative_advantage # Eq. 9
│   ├── dynamic_sampling_filter  # DAPO dynamic sampling
│   ├── dapo_loss                # Eq. 8
│   └── compressor_reward        # Eq. 10-11
├── test_gr2.py    # pytest test suite (46 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 gr2.py
```

Expected output (abridged):

```
--- Semantic-ID tokenizer (Sec 2.1) ---
codes shape: torch.Size([500, 3]), uniqueness: 0.9900

--- Rejection sampling (Sec 3.2) ---
accepted=True after 3 attempts, predicted_index=1

--- SFT / OPD losses (Sec 3.3-3.4) ---
sft_loss: -0.9000, backward OK
opd_loss:  0.3096, backward OK

--- Ranking rewards + de-hacked reward (Sec 4.1-4.2) ---
auc_reward:  0.8333
ndcg_reward: 0.9639
conditional_reward (identity, suboptimal input order): 0.1000  (should equal alpha*r_fmt = 0.1)

--- DAPO objective (Sec 4.3) ---
dynamic_sampling_filter keep-mask: [False, True, False]
dapo_loss: -0.0143

--- Context-compressor reward (Sec 5.1) ---
compressor_reward: 0.8300
```

### Test suite

```bash
python3 -m pytest test_gr2.py -v
```

Expected output:

```
collected 46 items
...
46 passed in <1s
```

Useful variants:

```bash
python3 -m pytest test_gr2.py::TestConditionalReward -v
python3 -m pytest test_gr2.py -x
```
