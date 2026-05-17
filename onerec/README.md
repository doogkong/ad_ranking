# OneRec

PyTorch reference implementation of **OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment** (KuaiShou Inc., Feb 2025).

Paper: https://arxiv.org/abs/2502.18965

---

## Key Ideas

### summary

  Summary

  OneRec (KuaiShou, Feb 2025) replaces the traditional 4-stage cascade (Retrieval → Pre-ranking → Ranking → Re-ranking) with a single unified generative model. Three key ideas:

  | Component | What it does |
  |---|---|
  | **Encoder-Decoder (T5-style + MoE) **    │ Encoder encodes user history H_u; Decoder autoregressively generates a full session S = {v₁,...,v_m} as semantic token sequences   |    
  │ **Session-wise generation   **           │ Generates an entire session (5 items) at once instead of next-item point-by-point — captures intra-session coherence and diversity              │
  │ ** Iterative Preference Alignment (IPA) **│ Trains a reward model (swt/vtr/ltr towers), generates N=128 candidates via beam search per user, picks winner/loser pair, applies DPO; iterates │
  
  Semantic tokenization: item embeddings → Balanced K-means residual quantization (L=3 levels, K=8192 codebook) → each item gets L integer codes. Balanced assignment avoids the hourglass phenomenon of standard
  RQ-VAE.

  Online results at Kuaishou: +1.68% total watch time, +6.56% average view duration vs. full multi-stage system.

### Problem: cascade ranking has a ceiling

Traditional recommenders chain Retrieval → Pre-ranking → Ranking → Re-ranking. Each stage acts as an upper bound for the next; no stage can recover items discarded upstream. Each ranker is trained independently, so the overall system is suboptimal.

### OneRec: one model does it all

A single encoder-decoder model directly generates a **session** (ordered list of m recommended items) from the user's history, replacing the entire cascade.

| Component | What it does |
|---|---|
| **Balanced K-means + Residual Quantization** | Encodes item embeddings into L integer codes per item (L=3, K=8192). Balanced assignment prevents the hourglass problem of standard RQ-VAE. |
| **T5-like Encoder** | Encodes user history H_u (as semantic token IDs) with fully-visible self-attention → H. |
| **MoE Decoder** | Autoregressively generates session token IDs via causal self-attention + cross-attention over H + Sparse MoE FFN (N_MoE=24, K_MoE=2 active). |
| **Session-wise generation** | Generates an entire session S = {v₁,...,v_m} at once (not next-item). Captures intra-session coherence and diversity; no hand-crafted rules needed. |
| **Iterative Preference Alignment (IPA)** | Post-trains with DPO. Reward model scores beam-search candidates; best/worst pair becomes DPO preference data. Only 1% of training samples use DPO. |

### Results at Kuaishou (hundreds of millions of DAU)

| Model | Total Watch Time | Avg View Duration |
|---|---|---|
| OneRec-0.1B | +0.57% | +4.26% |
| OneRec-1B | +1.21% | +5.01% |
| OneRec-1B+IPA | **+1.68%** | **+6.56%** |

---

## Architecture

```
User history H_u = {v_1, ..., v_n}          (n=256 items, each with L semantic codes)
        │
        ▼
  [SEP] s^1_1 s^2_1 s^3_1 [SEP] s^1_2 ...  (flat token sequence, vocab = K*L+2)
        │
        ▼
┌──────────────────────────────┐
│   OneRecEncoder  × N/2       │
│   (fully visible self-attn   │
│    + SwiGLU FFN)             │
└──────────┬───────────────────┘
           │  H  (B, T_enc, d)
           ▼
┌──────────────────────────────────────────────────────┐
│   OneRecDecoder  × N/2                               │
│                                                      │
│   [BOS] s^1_1 s^2_1 s^3_1 [BOS] s^1_2 ...           │
│      │                                               │
│   Causal Self-Attention  (autoregressive)            │
│      │                                               │
│   Cross-Attention  ← H  (full encoder context)       │
│      │                                               │
│   Sparse MoE FFN  (top-K_MoE of N_MoE experts)       │
└──────────────────────────────────────────────────────┘
        │
        ▼
   NTP loss over session codes  (eq. 4 in paper)
```

### Session token layout

Each item contributes `1 + L` decoder tokens:

```
[BOS]  s^1_i  s^2_i  s^3_i  [BOS]  s^1_{i+1} ...
  ↑     ↑      ↑      ↑
 item  level  level  level
 start   1      2      3
```

Total decoder length = m × (1 + L) tokens.

---

## Key Components

### 1. Balanced K-means (Algorithm 1)

Standard K-means leaves some clusters very large (hourglass phenomenon), creating bottlenecks in the codebook. Balanced K-means forces every cluster to hold exactly w = |V|/K items by sorting assignment candidates by distance and filling clusters sequentially until all reach capacity.

### 2. Residual Quantization

```
r^1_i = e_i
s^1_i = argmin_k ||r^1_i - c^1_k||²
r^2_i = r^1_i - c^1_{s^1_i}
s^2_i = argmin_k ||r^2_i - c^2_k||²
...
```

L=3 levels, K=8192 per level. Codebooks trained offline with `BalancedKMeans`, loaded as frozen buffers. Straight-through estimator preserves gradients for embedding fine-tuning.

### 3. Sparse MoE in Decoder (eq. 2)

```
H^{l+1}_t = Σ_{i=1}^{N_MoE} g_{i,t} FFN_i(H^l_t) + H^l_t

g_{i,t} = s_{i,t}  if s_{i,t} ∈ TopK({s_{j,t} | 1≤j≤N}, K_MoE)
         = 0        otherwise

s_{i,t} = Softmax_i(H^l_t^T e^l_i)
```

During inference, only K_MoE/N_MoE ≈ 8% of parameters are active, enabling a 1B parameter model at efficient inference cost.

### 4. Iterative Preference Alignment (Algorithm 2)

```
for t in 1..T:
    for each sample:
        if rand() < r_DPO:                      # 1% of samples
            Generate N=128 sessions via beam search
            Score each with RewardModel R(u, S)
            winner = argmax score
            loser  = argmin score
            L = L_NTP + λ * L_DPO(winner, loser)
        else:
            L = L_NTP
    M_{t+1} ← M_t updated with L
```

**DPO loss** (eq. 10):

```
L_DPO = -log σ(β · [log(M_t+1(S^w|H_u) / M_t(S^w|H_u))
                   - log(M_t+1(S^l|H_u) / M_t(S^l|H_u))])
```

**Reward Model** R(u, S): target-aware item representations e_i = v_i ⊙ u, fused via self-attention, then separate sigmoid towers for swt, vtr, ltr (session watch time, view-through rate, like-through rate).

---

## Usage

```python
from onerec import OneRec, RewardModel, IterativePreferenceAlignment

# Build model
model = OneRec(
    d_model=1024,
    n_heads=16,
    n_enc_layers=6,      # N/2 for the 1B variant
    n_dec_layers=6,
    codebook_size=8192,
    num_levels=3,
    num_experts=24,
    top_k=2,
    ffn_expand=4,
)

# Stage 1: NTP pre-training
loss = model(src_ids, tgt_codes)   # src_ids: (B, T_enc), tgt_codes: (B, m, L)
loss.backward()

# Stage 2: IPA fine-tuning
rm  = RewardModel(d_model=1024)
ipa = IterativePreferenceAlignment(model, rm, r_dpo=0.01, n_candidates=128)

def item_emb_fn(codes):   # codes: (B, m, L) → item embeddings: (B, m, d)
    return rq.decode(codes)

loss = ipa.step(src_ids, tgt_codes, user_embs, item_emb_fn)
loss.backward()

# Inference: greedy session generation
model.eval()
with torch.no_grad():
    codes = model.generate(src_ids, m=5)  # (B, 5, L)
```

### Hyperparameter guidance

| Parameter | Paper value | Notes |
|---|---|---|
| `codebook_size` K | 8192 | Entries per codebook level; trade-off between expressiveness and memory |
| `num_levels` L | 3 | More levels = better reconstruction at cost of longer decoder sequences |
| `n_enc_layers` | N/2 = 6 | Scales with model size; encoder sees full history bidirectionally |
| `num_experts` | 24 | Total experts in MoE; only `top_k` active per token at inference |
| `top_k` | 2 | 8% sparsity ratio; yields 13% active params in full 1B model |
| `r_dpo` | 0.01 | 1% DPO ratio gives 95% of max performance at 20% compute cost |
| `n_candidates` | 128 | Beam search width for DPO candidate generation |
| session size `m` | 5 | Items per generated session |
| history length `n` | 256 | User behavior sequence length |

---

## Files

```
onerec/
├── onerec.py              # full implementation + smoke test
│   ├── BalancedKMeans     # offline balanced K-means clustering (numpy)
│   ├── ResidualQuantizer  # multi-level RQ with straight-through gradient
│   ├── MultiHeadAttention
│   ├── FFN / MoELayer     # sparse top-k MoE
│   ├── OneRecEncoder      # T5-like bidirectional encoder
│   ├── OneRecDecoder      # causal + cross-attn + MoE decoder
│   ├── RewardModel        # multi-tower reward model (swt/vtr/ltr)
│   ├── OneRec             # full model: NTP training + greedy generation
│   ├── ntp_loss           # next-token prediction loss over session codes
│   ├── dpo_loss           # DPO preference alignment loss
│   └── IterativePreferenceAlignment  # IPA training loop
├── test_onerec.py         # pytest test suite (41 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 onerec.py
```

Expected output:

```
NTP loss:  4.3434
backward:  OK
generated: torch.Size([2, 3, 3])  (B, m, L) = [2, 3, 3]
params:    382,464
RM swt:    torch.Size([2])   [0.4731..., 0.4912...]
RM score:  [1.3743..., 1.4214...]
```

### Test suite

```bash
python3 -m pytest test_onerec.py -v
```

Expected output:

```
collected 41 items

test_onerec.py::TestRMSNorm::test_shape PASSED
test_onerec.py::TestRMSNorm::test_unit_rms PASSED
test_onerec.py::TestRMSNorm::test_gradient PASSED
test_onerec.py::TestBalancedKMeans::test_fit_encode PASSED
test_onerec.py::TestBalancedKMeans::test_balanced_clusters PASSED
test_onerec.py::TestResidualQuantizer::test_encode_shape PASSED
test_onerec.py::TestResidualQuantizer::test_decode_shape PASSED
test_onerec.py::TestResidualQuantizer::test_forward_straight_through PASSED
test_onerec.py::TestResidualQuantizer::test_round_trip_shape PASSED
test_onerec.py::TestMultiHeadAttention::test_self_attn_shape PASSED
test_onerec.py::TestMultiHeadAttention::test_cross_attn_shape PASSED
test_onerec.py::TestMultiHeadAttention::test_causal_mask PASSED
test_onerec.py::TestMultiHeadAttention::test_gradient PASSED
test_onerec.py::TestMoELayer::test_shape PASSED
test_onerec.py::TestMoELayer::test_top_k_less_than_experts PASSED
test_onerec.py::TestMoELayer::test_gradient PASSED
test_onerec.py::TestOneRecEncoder::test_output_shape PASSED
test_onerec.py::TestOneRecEncoder::test_gradient PASSED
test_onerec.py::TestOneRecDecoder::test_logits_shape PASSED
test_onerec.py::TestOneRecDecoder::test_greedy_decode_shape PASSED
test_onerec.py::TestOneRecDecoder::test_gradient PASSED
test_onerec.py::TestBuildDecoderInput::test_shape PASSED
test_onerec.py::TestBuildDecoderInput::test_bos_positions PASSED
test_onerec.py::TestNTPLoss::test_positive PASSED
test_onerec.py::TestNTPLoss::test_perfect_prediction PASSED
test_onerec.py::TestNTPLoss::test_gradient PASSED
test_onerec.py::TestRewardModel::test_output_shapes PASSED
test_onerec.py::TestRewardModel::test_score_shape PASSED
test_onerec.py::TestRewardModel::test_gradient PASSED
test_onerec.py::TestRewardModel::test_different_sessions_different_scores PASSED
test_onerec.py::TestOneRec::test_ntp_loss_positive PASSED
test_onerec.py::TestOneRec::test_backward PASSED
test_onerec.py::TestOneRec::test_grad_all_parameters PASSED
test_onerec.py::TestOneRec::test_generate_shape PASSED
test_onerec.py::TestOneRec::test_generate_valid_codes PASSED
test_onerec.py::TestOneRec::test_deterministic_with_seed PASSED
test_onerec.py::TestOneRec::test_different_history_different_output PASSED
test_onerec.py::TestIPA::test_step_ntp_only PASSED
test_onerec.py::TestIPA::test_step_with_dpo PASSED
test_onerec.py::TestIPA::test_update_reference PASSED
test_onerec.py::TestIPA::test_backward_through_dpo_step PASSED

41 passed in 1.62s
```

Useful variants:

```bash
python3 -m pytest test_onerec.py::TestOneRec -v
python3 -m pytest test_onerec.py -x
```
