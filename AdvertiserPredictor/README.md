# AdvertiserPredictor

Reference implementation of the algorithmic components described in *"Fine-Tuned LLM as a Complementary Predictor Improving Ads System"* (Pinterest, May 2026).

Paper: https://arxiv.org/abs/2605.27856

In production at Pinterest: the LLM-based candidate generator lifts U.S. Shopping Return-on-Ad-Spend by **+4.94%** overall and **+6.69%** on the opt-in slice; offline, Semantic-ID-enhanced GRPO training improves advertiser-prediction Recall@1/5/20 by **>10%** over text-only SFT.

---

## Summary

Like GR2, LoopFM and ExFM, this paper's contribution is a **training and serving recipe**, not a single encoder architecture. Its core idea is a pragmatic middle path for getting LLM value into an industrial ads system: rather than replacing the retrieval/ranking stack with an LLM (expensive, hard to align with ID-centric features) or restricting LLMs to reranking a handful of already-retrieved candidates, a fine-tuned open-source LLM is used purely as an **ads-specific ancillary predictor** — given a structured summary of a user's profile and behavior, it forecasts the advertiser(s) the user is most likely to convert with next, plus a short list of user interests. Those predictions then feed two existing systems as additional signal, not as a replacement for either:

- **Retrieval**: the predicted advertisers become targeting filters for a new LLM-based Candidate Generator, blended into the main retrieval candidate pool.
- **Ranking**: the predicted advertisers/interests are featurized as extra inputs to downstream conversion models (ctcvr, vtcvr).

This module implements the pipeline's concretely algorithmic pieces — reward design, data selection, output parsing, evaluation, and the Semantic-ID pretraining mechanism — as LLM-agnostic scaffolding. The prompt templates here are an **original, functionally-equivalent re-implementation** of the paper's stage-specific prompt *structure*, not a reproduction of its exact wording (see the paper's Appendix A for that).

---

## Key Ideas

### Why not a direct end-to-end LLM ranker?

Three practical mismatches motivate treating the LLM as a complementary predictor instead: (1) industrial recommenders are built on sparse, ID-centric features (user/item/advertiser IDs) that don't naturally map onto an LLM's token vocabulary; (2) production ranking stacks model deep, calibration-sensitive cross-feature interactions that are non-trivial to reproduce with an LLM's inductive biases; (3) LLM parameter counts and decoding costs strain real-time serving budgets under tail-latency SLOs. Positioning the LLM as an upstream *predictor of priors* — not the ranker itself — sidesteps all three.

### Stage-specific prompting (Table 1)

| Stage | Target | # Advertisers | Output |
|---|---|---|---|
| SFT | single next advertiser | 1 | free text |
| GRPO | ranked advertisers + interests | 20 + 5 | structured block |
| Inference | same as GRPO | 20 + 5 | structured block |

Predicting 20 advertisers instead of 5 during GRPO (even though the ground truth is always a single advertiser) is deliberate: a longer ranked list produces more reward variance per optimization step, giving GRPO a richer training signal. Keeping the inference format identical to GRPO's removes any train/inference format mismatch.

### The GRPO reward (Eq. 1-5)

```
R_total = R_match - P_adv_len - P_interest_len

R_match(i)  = R_base(i) + R_bonus(i)          i = rank of the ground-truth advertiser (1 = top)
R_base(i)   = 0.1 * (20 - i)
R_bonus(i)  = 2.0 if i <= 4 else 0.0

P_len(n, n*) = 0                                     if n == n*
             = min(0.1 * |n - n*|, 1.0) + 1.0          otherwise
```

`R_match` rewards ranking the true advertiser both *correctly* (via the linear base term) and *near the top* (via a step bonus for the top 4 positions); `P_len` is applied to both the predicted advertiser count and interest count, penalizing any deviation from the requested quantities — discouraging the policy from padding or truncating its output to game the reward.

### Semantic-ID two-phase pretraining (Sec 3.5)

Adding Semantic IDs (RQ-VAE codes over multimodal item embeddings) to the prompt gives >10% Recall gains over text alone — but naively continuing pretraining on SID-augmented data risks catastrophic forgetting of the base model's general knowledge. The paper's fix is staged:

1. **Phase 1** — freeze the entire model except the embedding rows for the new SID tokens, roughly aligning them with the existing text-token space.
2. **Phase 2** — unfreeze everything for full-parameter pretraining, mixing SID-augmented recommendation data with general-domain data to guard against forgetting.
3. **Phase 3** — the same task-specific SFT/GRPO fine-tuning as before, now with SIDs in the prompt.

### Downstream integration and its failure mode (Sec 3.6, 4.5.2)

Blending the LLM-based Candidate Generator's output into the main retrieval pool is sensitive to its quota: because the LLM-CG specifically targets high-conversion-intent advertisers, too large a quota lets a handful of advertisers dominate the blended, de-duplicated candidate list — hurting downstream advertiser diversity even as individual conversion metrics look good.

---

## Key Components

| Function | Paper section |
|---|---|
| `select_active_users`, `construct_next_advertiser_label`, `select_users_for_incremental_inference` | Sec 3.2 — user selection, label construction, incremental daily inference |
| `build_advertiser_prediction_prompt`, `parse_structured_output` | Sec 3.3 — stage-specific prompt design + deterministic output parsing |
| `rank_match_reward`, `length_penalty`, `grpo_reward` | Sec 3.4, Eq. 1-5 — the GRPO reward |
| `freeze_all_but_sid_embeddings`, `unfreeze_all` | Sec 3.5 — Semantic-ID two-phase pretraining (via gradient masking on the embedding table) |
| `blend_candidates`, `advertiser_concentration` | Sec 3.6.1, 4.5.2 — LLM-CG blending and its over-concentration risk |
| `rank_of_advertiser`, `recall_at_k`, `mean_recall_at_k` | Sec 4 — Recall@K evaluation |

### What's not implemented

The actual LLM fine-tuning (SFT/GRPO training loops over a real open-source model), the vLLM+Ray serving stack (prefix caching, paged-attention KV cache, continuous batching, virtual epochs), and the RQ-VAE Semantic ID tokenizer itself (see [`../TIGER`](../TIGER) and [`../semantic_id`](../semantic_id) for that) are all out of scope here — this module is the LLM-agnostic scaffolding around them: reward, parsing, data selection, and evaluation.

---

## Usage

```python
from advertiser_predictor import (
    select_active_users, construct_next_advertiser_label,
    build_advertiser_prediction_prompt, parse_structured_output,
    grpo_reward, rank_of_advertiser,
    freeze_all_but_sid_embeddings, unfreeze_all,
    blend_candidates, mean_recall_at_k,
)

# Sec 3.2-3.3: build a GRPO-stage prompt for an active user
active = select_active_users(users, as_of_date=today, lookback_days=90)
prompt = build_advertiser_prediction_prompt(
    profile=user_profile, behavior=user_behavior,
    active_advertisers=active_advertisers, preset_advertiser_pool=preset_pool,
    stage="grpo", num_advertisers=20, num_interests=5,
)

# Sec 3.4: score a rollout against the GRPO reward
parsed = parse_structured_output(llm_output_text)
rank = rank_of_advertiser(parsed["advertisers"], ground_truth_advertiser)
reward = grpo_reward(rank, len(parsed["advertisers"]), len(parsed["interests"]))

# Sec 3.5: Phase 1 -> Phase 2 of SID pretraining
hook = freeze_all_but_sid_embeddings(model, model.embed_tokens, sid_token_ids)
train(model, sid_alignment_data)          # Phase 1: only SID embedding rows move
hook.remove(); unfreeze_all(model)
train(model, mixed_recommendation_and_general_data)  # Phase 2: full-parameter

# Sec 3.6: blend the LLM-CG into the main retrieval pool
final_candidates = blend_candidates(main_candidates, llm_candidates, llm_quota=200)
```

---

## Files

```
AdvertiserPredictor/
├── advertiser_predictor.py       # full implementation + smoke test
├── test_advertiser_predictor.py  # pytest test suite (46 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 advertiser_predictor.py
```

Expected output (abridged):

```
--- Sec 3.2: Data pipeline ---
active users (of 2): ['u1']
next-advertiser label: AdvA
users needing re-inference: ['u2']

--- Sec 3.4: GRPO reward (Eq. 1-5) ---
rank=2, reward=0.5000

--- Sec 3.5: SID two-phase pretraining ---
phase 1 grad on non-SID row: 0.000000, on SID row: 8.766323
phase 2: all params trainable = True

--- Sec 3.6: Candidate blending ---
blended: ['A1', 'A2', 'A3', 'A4', 'A5', 'A9'], concentration: 0.17

--- Sec 4: Recall@K ---
Recall@1: 0.00, Recall@3: 0.50
```

### Test suite

```bash
python3 -m pytest test_advertiser_predictor.py -v
```

Expected output:

```
collected 46 items
...
46 passed in <1s
```

Useful variants:

```bash
python3 -m pytest test_advertiser_predictor.py::TestGrpoReward -v
python3 -m pytest test_advertiser_predictor.py -x
```
