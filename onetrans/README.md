# OneTrans

PyTorch reference implementation of **OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer** (ByteDance / NTU, WWW 2026).

Paper: https://arxiv.org/abs/2510.26104

---

## Key Ideas & Application to Ad Ranking

### What problem it solves

Traditional ad ranking stacks **sequence modeling** (DIN/SIM/HSTU) and **feature interaction** (DCN/DIEN) as separate modules with separate training objectives. OneTrans fuses both into a single Transformer pass: one model, one training objective, one serving graph.

### Core design choices

| Idea | What it does |
|---|---|
| **Unified tokenization** | Sequential features (click/purchase/view histories) → S-tokens; all non-sequential features (user profile, item, context) → NS-tokens. Everything runs through one model. |
| **Mixed parameterization** | S-tokens share one W_Q/K/V + FFN (behavioral events are homogeneous). Each NS-token i has its own W_{NS,i}^{Q,K,V} + FFN (user/item/context features are heterogeneous). |
| **Causal attention** | S-tokens attend only to preceding S-tokens (autoregressive over behavior history). NS-tokens attend to the full S-history + preceding NS-tokens (cross-feature interaction). |
| **Pyramid stack** | At each layer, only the last L' S-token queries are computed; K/V still span the full history. L' shrinks linearly L_S → L_NS across layers, concentrating behavioral signals progressively into NS-tokens. |
| **Cross-request KV caching** | The user's S-token K/V is computed once and reused across all candidate ads in a serving batch. |

### Application to ad ranking

```
User behavior sequences          →  S-tokens  (click history, purchase history, view history)
                                      [shared W_Q/K/V, causal attention]
User profile features            →  NS-token 1  (token-specific weights)
Item / ad features               →  NS-token 2  (token-specific weights)
Context (time, device, query)    →  NS-token 3  (token-specific weights)
                                         │
                                   Pyramid Stack
                                         │
                              mean-pool NS-tokens
                                         │
                              ┌──────────┴──────────┐
                           pCTR head            pVTR head    (+ pEV etc.)
```

**Biggest serving win**: cross-request KV cache. In standard serving you re-encode the user's 500-item click history for every one of 1000 candidates. OneTrans computes S-token K/V once (user-level), then each candidate only needs to project its own NS-tokens and run attention against the cached K/V. Per-candidate FLOPs drop from O(L_S · d) to O(L_NS · d).

**Where it fits**: replaces the DIN + DCN pair that most ad stacks use today. S-tokens handle sequence modeling (formerly DIN/SIM); NS-token cross-attention handles feature interaction (formerly DCN/DIEN).

**Reported results (paper Table 2)**: +1.53% CTR AUC, +2.79% CVR AUC on industrial dataset vs. HSTU+DCN baseline.

---

## Architecture

```
Sequential features  (behavior histories)  →  SequentialTokenizer  →  S-tokens  ]
                                                                                  ├─ concat ─→ Pyramid Stack ─→ Task heads
Non-sequential features (user/item/ctx)    →  AutoSplitNSTokenizer →  NS-tokens ]
```

### SequentialTokenizer

Projects each behavior sequence to `d_model` and merges with `[SEP]` delimiter tokens between sequence types. Optional per-sequence type embeddings for heterogeneous behavior types (clicks vs. purchases vs. views).

```
Output L_S = Σ L_i + (n_seqs − 1) SEP tokens
```

### AutoSplitNSTokenizer

Preferred over group-wise tokenizer (ablation Table 3 in paper). Concatenates all NS features → two-layer MLP → reshape into `L_NS` tokens. The model learns feature groupings rather than relying on manual semantics.

```
ns_features: (B, ns_input_dim)
→ Linear → SiLU → Linear  →  (B, d_model * L_NS)
→ view                     →  (B, L_NS, d_model)
```

### MixedCausalAttention

S-tokens share `W_S^{Q,K,V}` (one projection for all S-positions).
Each NS-token i has its own `W_{NS,i}^{Q,K,V}` stored as `(L_NS, d_model, 3*d_model)` parameter, applied via `einsum`.

Causal mask:
```
q_pos = [query_start .. L_S-1,  L_S .. L_S+L_NS-1]
k_pos = [0 .. L_total-1]
mask[q, k] = True (masked) if q_pos[q] < k_pos[k]
```

Pyramid: only `x[:, query_start:L_S]` S-tokens are used as queries. Keys and values span the full sequence.

### MixedFFN

S-tokens: one shared `(W_S1, W_S2)`.
NS-tokens: per-token `(W_NS1_i, W_NS2_i)` via `einsum("bnd,ndo->bno", x_NS, W_NS1)`.

### Pyramid Stack

Query counts per layer, linearly decreasing from `L_S` to `L_NS`:

```
layer 0:  L_q = L_S        (all S-tokens query)
layer l:  L_q = round(L_S + l/(n_layers-1) * (L_NS - L_S))
last:     L_q = L_NS       (only NS-size slice queries)
```

---

## Usage

```python
from onetrans import OneTrans
import torch

model = OneTrans(
    seq_dims=[32, 32],          # dims of each behavior sequence embedding
    ns_input_dim=128,           # total dim of concatenated non-sequential features
    d_model=256,
    n_heads=4,
    n_layers=6,
    L_NS=32,                    # number of NS-tokens
    max_seq_len=512,            # max total S-token length
    num_tasks=2,                # e.g. pCTR + pVTR
)

# Forward pass
sequences   = [clicks, purchases]     # each (B, L_i, seq_dim_i)
ns_features = user_item_context_feats # (B, ns_input_dim)
logits = model(sequences, ns_features, type_ids=[0, 1])  # (B, num_tasks)

# Training
loss = F.binary_cross_entropy_with_logits(logits, targets)
loss.backward()
```

### Mapping ad ranking features

```python
# Typical ad ranking setup
model = OneTrans(
    seq_dims=[item_emb_dim, item_emb_dim, item_emb_dim],  # clicks, purchases, views
    ns_input_dim=user_dim + item_dim + context_dim,
    d_model=256,
    n_heads=8,
    n_layers=6,
    L_NS=16,          # one NS-token per ~2 feature groups; tune with ablation
    max_seq_len=1024, # accommodate long click histories
    num_tasks=3,      # pCTR, pVTR, pEV
)

sequences = [click_seq, purchase_seq, view_seq]  # each (B, L_i, item_emb_dim)
ns_feats  = torch.cat([user_feats, item_feats, context_feats], dim=-1)  # (B, ns_input_dim)
logits    = model(sequences, ns_feats, type_ids=[0, 1, 2])              # (B, 3)
```

### Hyperparameter guidance

| Parameter | Guidance |
|---|---|
| `L_NS` | 8–32; larger captures more NS heterogeneity; match to number of semantic feature groups |
| `n_layers` | 4–8; pyramid compresses well with 6 |
| `d_model` | 128–512; scale with data size |
| `n_heads` | 4–8; `d_model` must be divisible by `n_heads` |
| `num_tasks` | Match your prediction heads (pCTR + pVTR + pEV = 3) |
| `max_seq_len` | Set ≥ sum of all sequence lengths + (n_seqs − 1) SEP tokens |

---

## Files

```
onetrans/
├── onetrans.py           # full implementation + smoke test
│   ├── RMSNorm
│   ├── AutoSplitNSTokenizer
│   ├── SequentialTokenizer
│   ├── MixedCausalAttention
│   ├── MixedFFN
│   ├── OneTransBlock
│   ├── _pyramid_query_counts
│   └── OneTrans
├── test_onetrans.py      # pytest test suite
└── README.md
```

---

## Running

### Smoke test

```bash
python3 onetrans.py
```

Expected output:

```
logits:   torch.Size([4, 2])   tensor([[-0.1648,  0.3603],
        [-0.1046,  0.2904],
        [-0.1844,  0.3016],
        [-0.1765,  0.3242]], grad_fn=<CatBackward0>)
params:   1,393,154
pyramid:  [24, 16, 16, 8]  (S-token query counts per layer)
backward: OK
```

### Test suite

```bash
python3 -m pytest test_onetrans.py -v
```

Expected output:

```
collected 40 items

test_onetrans.py::TestRMSNorm::test_output_shape PASSED
test_onetrans.py::TestRMSNorm::test_unit_rms_after_norm PASSED
test_onetrans.py::TestRMSNorm::test_weight_scales_output PASSED
test_onetrans.py::TestRMSNorm::test_gradient_flows PASSED
test_onetrans.py::TestAutoSplitNSTokenizer::test_output_shape PASSED
test_onetrans.py::TestAutoSplitNSTokenizer::test_different_L_NS PASSED
test_onetrans.py::TestAutoSplitNSTokenizer::test_gradient_flows PASSED
test_onetrans.py::TestSequentialTokenizer::test_output_shape_single_seq PASSED
test_onetrans.py::TestSequentialTokenizer::test_output_shape_multi_seq PASSED
test_onetrans.py::TestSequentialTokenizer::test_sep_token_count PASSED
test_onetrans.py::TestSequentialTokenizer::test_type_ids PASSED
test_onetrans.py::TestSequentialTokenizer::test_gradient_flows PASSED
test_onetrans.py::TestMixedCausalAttention::test_output_shape_no_pyramid PASSED
test_onetrans.py::TestMixedCausalAttention::test_output_shape_pyramid PASSED
test_onetrans.py::TestMixedCausalAttention::test_causal_no_future_leakage PASSED
test_onetrans.py::TestMixedCausalAttention::test_gradient_flows PASSED
test_onetrans.py::TestMixedFFN::test_output_shape PASSED
test_onetrans.py::TestMixedFFN::test_output_shape_pyramid PASSED
test_onetrans.py::TestMixedFFN::test_s_and_ns_use_different_weights PASSED
test_onetrans.py::TestMixedFFN::test_gradient_flows PASSED
test_onetrans.py::TestPyramidQueryCounts::test_length PASSED
test_onetrans.py::TestPyramidQueryCounts::test_first_is_L_S PASSED
test_onetrans.py::TestPyramidQueryCounts::test_last_is_L_NS PASSED
test_onetrans.py::TestPyramidQueryCounts::test_monotone_decreasing PASSED
test_onetrans.py::TestPyramidQueryCounts::test_all_at_least_L_NS PASSED
test_onetrans.py::TestPyramidQueryCounts::test_single_layer PASSED
test_onetrans.py::TestOneTransBlock::test_output_shape_no_pyramid PASSED
test_onetrans.py::TestOneTransBlock::test_output_shape_with_pyramid PASSED
test_onetrans.py::TestOneTransBlock::test_gradient_flows PASSED
test_onetrans.py::TestOneTrans::test_output_shape PASSED
test_onetrans.py::TestOneTrans::test_single_task PASSED
test_onetrans.py::TestOneTrans::test_three_tasks PASSED
test_onetrans.py::TestOneTrans::test_single_sequence PASSED
test_onetrans.py::TestOneTrans::test_type_ids PASSED
test_onetrans.py::TestOneTrans::test_backward PASSED
test_onetrans.py::TestOneTrans::test_grad_exists_for_all_parameters PASSED
test_onetrans.py::TestOneTrans::test_deterministic_with_seed PASSED
test_onetrans.py::TestOneTrans::test_different_inputs_different_outputs PASSED
test_onetrans.py::TestOneTrans::test_variable_sequence_length PASSED
test_onetrans.py::TestOneTrans::test_many_sequences PASSED

40 passed in 0.73s
```

Useful variants:

```bash
# Single test class
python3 -m pytest test_onetrans.py::TestMixedCausalAttention -v

# Stop on first failure
python3 -m pytest test_onetrans.py -x
```
