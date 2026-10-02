# LIGE-GR

PyTorch reference implementation of **LIGE-GR: A Smooth Leap from Ranking to Generative Recommendation in the LLM Era** (Srinivas, He, Woodmansee, Lian, Hu, Jiang, … Liu; Meta).

Paper: https://arxiv.org/abs/2609.18148

Validated on short-video recommendation: **+1.14% time spent on Instagram Reels** and **+0.72% on Facebook Video** against strong production baselines, at ~10% extra inference resources relative to the context-free ranker (~7% end-to-end latency on Reels for the b=1 configuration; ~20% resources for b=6).

---

## Summary

Industrial recommenders are *itemwise*: a ranking model predicts engagement signals (p_like, p_follow, …) for each candidate independently, an item value model (itemVM) collapses them into a scalar, and a greedy decoder takes the top-T, with rule-based "control layer" tweaks (gap demotion, DPP, business constraints) patched on top. LLMs solve the same problem — produce the best *sequence* — by generating each token conditioned on what was already generated.

Replacing a mature stack wholesale is risky (system and organizational cost), so LIGE-GR is an **additive, reversible, low-resource upgrade** that strictly generalizes the incumbent. It upgrades three components:

| Component | Incumbent (itemwise) | LIGE-GR (listwise) |
|---|---|---|
| Ranking model | Context-free predictor CF(u, v_t) | + Context-aware predictor CA(u, v_t \| V_{t-1}), a 4-layer / 4-head causal Transformer over CF's intermediate representations v' |
| Value model | Σ_t itemVM + CL | ListVM_golden: each term weighted by the probability the user is still watching |
| Decoder | Greedy, top score at each position | **Palette**: RL-style beam search with a closed-form future-value estimate |

Setting CA→CF, ListVM→itemVM+CL, beam b=1, p_continue≡1, F̂=0 **recovers the incumbent exactly** (tested), so every piece can be reverted by config, with no retraining.

---

## Key Ideas

### Listwise objective (Eq. 6–9)

```
ListVM_vanilla(V_T) = Σ_t [ itemVM(CA(u, v_t | V_{t-1})) + CL(v_t | V_{t-1}) ]
ListVM_golden(V_T)  = Σ_t p_continue(V_{t-1}) · [ itemVM(CA(·)) + CL(·) ]
p_continue(V_t)     = p_continue(V_{t-1}) · CA_continue(u, v_t | V_{t-1})
```

Later items only count to the extent the user reaches them.

### Context-aware module (Sec 3.1, Fig 3)

CF runs once per request and yields v'_c for every candidate. CA is a small GPT-style causal Transformer fed `(v'_1 … v'_t)`; output at position t predicts all task heads (like, follow, share, watch-time, **continue**, …) for item t given the preceding items. CF is untouched, so CA can be retrained/published independently (`LigeGR.ca_training_step` runs CF under `no_grad`).

### Palette decoder (Alg. 1, Eq. 10)

Maintain top-b prefixes. Each step expands every prefix with every admissible candidate (CL > −∞), computes the accumulated return `ListVM_golden(V+c)` plus a future-value estimate `F̂(V+c)`, and keeps the top-b by `Q(V+c) = ListVM_golden(V+c) + F̂(V+c)`.

### Future-value estimators (Eq. 11–14)

With `s̄` = mean VM+CL score of the prefix and `q` = continuation probability of the last item:

```
F_step(V_t) = s̄ · Σ_{j=1}^{T-t} q^j
F_dur (V_t) = s̄ · Σ_{j=1}^{T-t} (q^{d̄/d_t})^j
```

`F_step` compounds the last item's whole-item continuation probability, biasing against prefixes that end in long videos; `F_dur` rescales from the item's duration d_t to the prefix-average duration d̄. (The paper's online win uses b=6 + ListVM_golden + F̂_dur: +0.69% time spent over the b=1 base on Reels.)

### Serving (Sec 4, Alg. 2)

* Phase 1: CF forward once; cache v'. Phase 2: T batched lightweight CA passes over cached v' (beam width enlarges the batch, not the number of sequential passes).
* Only the top ~⅓ of candidates by CF score are re-scored by CA (60–80% throughput gain).
* Per-request fallback to itemwise greedy on the cached CF scores if phase 2 exceeds the latency budget or fails.

---

## Key Components

| Component | Paper section |
|---|---|
| `ItemValueModel` | Eq. 3 |
| `ControlLayer` (hard rule, gap demotion, DPP, distinctness) | Sec 2.2 |
| `ContextFreeModel`, `ContextAwareModel` | Sec 3.1, Fig 3 |
| `list_value(golden=…)` | Eq. 8–9 |
| `future_value_step`, `future_value_duration` | Eq. 11–14 |
| `palette_decode` | Alg. 1, Eq. 10 |
| `make_cf_scorer`, `make_ca_scorer` | Sec 4.1 (cached CF representations) |
| `LigeGR.generate` | Alg. 2 (pool restriction, latency fallback, revert switch) |
| `context_aware_loss`, `normalized_entropy`, `relative_ne_improvement` | Sec 5.1, Table 1 |
| `list_composition_metrics` | Table 4 style list diagnostics |

**Implementation notes.** The paper does not specify the CF architecture, so `ContextFreeModel` is a small stand-in MLP (plug in your production ranker: all that is needed is v' and task logits). The CA scorer re-runs the small Transformer over `prefix + candidate` sequences (no KV cache), which is fine at reference scale. The paper's own 17-task heads, feature pipelines, and training data are not reproduced.

---

## Usage

```python
from lige_gr import *

cf = ContextFreeModel(user_dim=12, item_dim=10, d_model=32, num_tasks=3)
ca = ContextAwareModel(d_model=32, num_tasks=3)           # 4 layers, 4 heads
model = LigeGR(cf, ca, ItemValueModel([1.0, 2.0, 1.5]))

# train CA on logged lists (labels [B, T, K+1]; last channel = continue)
loss = model.ca_training_step(user, list_items, labels)

# serve
control = ControlLayer(C, categories=cats, gap_coef=0.5)
res = model.generate(u, items, list_len=10, control_fn=control,
                     beam_width=6, golden=True, future="duration",
                     durations=durs, pool_frac=1/3, latency_budget_ms=50)
res.order        # chosen candidate indices
# incumbent behavior: model.generate(..., use_context_aware=False)
```

---

## Files

```
LIGE-GR/
├── lige_gr.py        # implementation + smoke test
├── test_lige_gr.py   # pytest suite (50 tests)
└── README.md
```

## Running

```bash
python3 lige_gr.py                      # smoke test
python3 -m pytest test_lige_gr.py -v    # 50 passed
```

Notable tests: strict generalization to the itemwise greedy decoder; full-width beam equals brute force over all permutations for both ListVM variants; CA causality; CA learns a context-only signal CF cannot see; pool restriction and latency fallback.
