# UniPinRec

PyTorch reference implementation of **UniPinRec: Unifying Generative Retrieval and Ranking at Pinterest Scale** (Pinterest, 2026).

Paper: https://arxiv.org/abs/2606.00422

Deployed in Pinterest's core surfaces: **+14.8%** ranking Hit@3 over the production ranker while matching production retrieval recall, a **>3x** total ranking forward-pass speedup from cross-stage KV-cache reuse, and **-11.1%** end-to-end serving latency / **+63.6%** QPS versus deploying retrieval and ranking as two independent models.

---

## Summary

Retrieval and ranking are conventionally trained as two separate transformers over the *same* user action history — duplicating parameters, training compute, and the O(n²) cost of self-attention over that history at serving time. UniPinRec, which builds directly on [`../PinRec`](../PinRec), unifies both stages into **one model, one input format, one training stage, deployed within one serving pipeline**, via three ideas:

1. **Masked Action Modeling (MAM)** (Sec 3.1.1) — adds ranking supervision to the exact same non-interleaved sequence retrieval already uses, without doubling context length.
2. **A single attention pattern for both stages** (Fig. 2) — one concatenated past+future sequence, with past causal and future target-aware-but-mutually-blocked (the same "M-FALCON pattern" as `../HSTU`).
3. **A joint loss** (Eq. 3-5) — retrieval's next-item sampled-softmax plus per-action-type ranking BCE, trained together in a single forward pass.

Because both stages now share one backbone over one non-inflated sequence, the expensive O(n²) history encoding computed during retrieval can be **cached and reused as-is by ranking**, which then only pays the O(nk) cost of scoring k candidates against that cache — this repo's `encode_history` / `score_candidates` split, verified to produce numerically identical results to a single dense joint forward pass.

---

## Key Ideas

### Masked Action Modeling, MAM (Sec 3.1.1)

Prior unified approaches (HSTU-for-ranking) interleave item and action as two separate sequence tokens, doubling context length. UniPinRec instead **concatenates the action embedding onto the item embedding along the feature dimension** at the same position, then randomly masks it:

```
x_i = Linear([item_emb_i ; action_emb(a_i or [MASK])])
```

- Past (history) positions: each action independently masked with probability `p_mask` (`sample_action_mask`).
- Future (candidate) positions: action is **always** masked — at inference, the action on an unscored candidate is unknown by definition, which is exactly what makes "predict the masked action" a well-posed ranking task on candidates.
- A dedicated `[MASK]` class (not zero, not any real action id) is appended to the action vocabulary, so the model can distinguish "action truly unknown" from "no action taken" (`MaskedActionInputEncoder`).

### One attention pattern, two objectives (Fig. 2, `build_unipinrec_attention_mask`)

```
            past          future (candidates)
past    [  causal      |     blocked          ]
future  [  full (TAE)  |  diagonal-only (self) ]
```

Past keeps standard causal attention (used for retrieval's next-item objective). Each candidate attends to the *entire* past (target-aware, same idea as [`../foundation_expert`](../foundation_expert)'s TAE mask) but is blocked from attending to any other candidate — the same "M-FALCON pattern" HSTU uses for candidate scoring — so a candidate's ranking score never depends on which other candidates happen to share its batch.

### Joint loss (Eq. 3-5)

| Loss | Applies to | What it predicts |
|---|---|---|
| `item_retrieval_loss` (Eq. 3) | every past position | next item, sampled softmax with Count-Min-Sketch popularity bias correction (identical mechanism to `../PinRec`'s Eq. 2) |
| `action_prediction_loss` (Eq. 4) | masked past positions **+** all candidate positions | per-action-type binary label, via one dedicated scalar-logit head per action type, weighted by `w_c` |

`L = L_item + L_action` (Eq. 5) is optimized jointly in a single forward pass. The paper's ablation (Table 1) shows the jointly-trained model *exceeds* both a ranking-only variant (no retrieval loss) and a sequential pretrain-then-finetune variant — retrieval supervision measurably improves ranking quality, not just vice versa.

### Cross-stage KV-cache reuse (Sec 3.3.3)

The dominant cost in either stage's forward pass is self-attention over the user's O(n)-length history, which is O(n²). Because UniPinRec's past representations depend only on the past (never on candidates, by construction of the attention mask above), retrieval's history encoding can be computed **once** and its per-layer K/V cache handed directly to ranking:

```python
item_repr, kv_cache = model.encode_history(item_emb_past, action_ids_past, past_masked)   # O(n^2), once
action_logits = model.score_candidates(kv_cache, item_emb_future)                          # O(n*k), reused
```

turning ranking into an O(nk) *decode* step instead of a second O(n²) *prefill*. `test_unipinrec.py` verifies this cached path is numerically identical (not just asymptotically cheaper) to running the dense joint pass over `[past; future]` directly — the same correctness property HSTU's M-FALCON and `../PinFM`'s DCAT rely on.

---

## Architecture

```
Past: (item_i, action_i) for i=1..n           Future: candidate items j=1..k (action unknown)
        │                                              │
MaskedActionInputEncoder                     MaskedActionInputEncoder (action always [MASK])
   (random mask, p_mask)                              │
        │                                              │
        └──────────────── concat [x_past ; x_future] ──┘
                              │
                    UniPinRecTransformer
              (past: causal | future: full-past + self-only)
                              │
                z_past (B,n,d)         z_future (B,k,d)
                    │                        │
          ItemOutputHead              ActionPredictionHeads
                    │                        │
        item_retrieval_loss (Eq.3)   action_prediction_loss (Eq.4, + masked past)
                    └──────────┬─────────────┘
                          L = L_item + L_action  (Eq. 5)

Serving (cross-stage reuse):
  encode_history  -->  item_repr (retrieval) + per-layer KV cache
  score_candidates(KV cache, candidates)  -->  action_logits (ranking), O(n*k) only
```

---

## Usage

```python
from unipinrec import (
    UniPinRecModel, TargetItemEncoder, CountMinSketch,
    item_retrieval_loss, action_prediction_loss,
)

model = UniPinRecModel(d_item=256, num_action_types=6, d_model=512, n_layers=12, n_heads=8)
target_encoder = TargetItemEncoder(d_item=256, d_model=512)

# --- Joint training: one forward pass, two losses ---
item_repr_past, action_logits_past, action_logits_future, past_masked = model.forward_train(
    item_emb_past, action_ids_past, item_emb_future, p_mask=0.2,
)

cms = CountMinSketch()
cms.update(item_ids)
targets = target_encoder(item_emb_past)
l_item = item_retrieval_loss(item_repr_past, targets, item_ids, cms)
l_action = action_prediction_loss(
    action_logits_past, action_labels_past, past_masked,
    action_logits_future, action_labels_future, action_weights,
)
(l_item + l_action).backward()

# --- Serving: retrieval prefill, then ranking reuses its KV cache ---
item_repr, kv_cache = model.encode_history(item_emb_past, action_ids_past, past_masked)
# ... use item_repr for ANN retrieval to get candidate items ...
action_logits = model.score_candidates(kv_cache, candidate_item_embeddings)  # ranking, O(n*k)
```

---

## Files

```
UniPinRec/
├── unipinrec.py         # full implementation + smoke test
│   ├── sample_action_mask            # MAM masking schedule for past positions
│   ├── MaskedActionInputEncoder      # feature-dim item+action concat, [MASK] handling
│   ├── build_unipinrec_attention_mask  # past-causal + M-FALCON-future mask (Fig. 2)
│   ├── UniPinRecLayer                # pre-LN block: forward_masked + forward_cross
│   ├── UniPinRecTransformer          # forward_combined (dense) + forward_self/forward_cross (cached)
│   ├── ItemOutputHead / TargetItemEncoder  # retrieval-side projections
│   ├── ActionPredictionHeads         # per-action-type ranking heads h_{psi_c}
│   ├── CountMinSketch                # popularity bias estimator (shared design w/ PinRec)
│   ├── item_retrieval_loss           # Eq. 1/3
│   ├── action_prediction_loss        # Eq. 4
│   └── UniPinRecModel                # forward_train / encode_history / score_candidates
├── test_unipinrec.py    # pytest test suite (35 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 unipinrec.py
```

Expected output (values vary slightly by seed):

```
--- Joint training forward pass ---
item_repr_past: torch.Size([3, 16, 32]), action_logits_past: torch.Size([3, 16, 4]), action_logits_future: torch.Size([3, 5, 4])
fraction of past positions masked: 0.17 (target ~0.2)
L_item=3.8807  L_action=5.9624  L_total=9.8431  backward: OK

--- Cross-stage KV-cache reuse (retrieval prefill -> ranking decode) ---
cached candidate logits:  torch.Size([3, 5, 4])
matches dense joint forward pass (same masking): True

params: 29,404
```

### Test suite

```bash
python3 -m pytest test_unipinrec.py -v
```

Expected output:

```
collected 35 items
...
35 passed in ~1s
```

Useful variants:

```bash
python3 -m pytest test_unipinrec.py::TestUniPinRecTransformer -v
python3 -m pytest test_unipinrec.py -x
```
