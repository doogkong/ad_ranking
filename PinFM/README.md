# PinFM

PyTorch reference implementation of **PinFM: Foundation Model for User Activity Sequences at a Billion-scale Visual Discovery Platform** (Pinterest, Jul 2025).

Paper: https://arxiv.org/abs/2507.12704

Pretrained at **20B+ parameters** on 2 years of user activity, deployed to Home Feed and Related Items ranking at Pinterest with **cost-neutral** integration: **+600%** serving throughput / **+200%** training throughput from the Deduplicated Cross-Attention Transformer (DCAT), **+2.6%** Surface Saves / **+1.2%** Sitewide Saves online, and a **+20%** engagement lift on fresh items after cold-start remediation.

---

## Summary

Large sequence models for recommendation (HSTU, TIGER, TWIN-V2) are usually trained standalone, one per application — prohibitively expensive to repeat across every surface of a large platform. PinFM instead **pretrains once, fine-tunes everywhere**: a single ~20B-parameter causal transformer is pretrained on raw, compact (item id, action, surface) sequences across *all* of Pinterest's applications, then spliced as a reusable sequence-encoding module into each application's existing ranking model (which keeps its own DLRM/DCN-style feature-crossing and other features).

Three engineering problems had to be solved to make this practical at industrial scale:

1. **Latency/throughput** — a 20B-parameter model must still score millions of items/second. Solved by the **Deduplicated Cross-Attention Transformer (DCAT)**: run the expensive self-attention pass once per *unique* user sequence in a batch (not once per candidate), then cheaply cross-attend each candidate against the cached result.
2. **User-candidate interaction** — the model should contextualize its output on which specific item is being scored, not just summarize the user in isolation. Solved by **early fusion**: appending the candidate to the input sequence so cross-attention (not concatenation) captures the interaction.
3. **Cold start** — new items (unseen during pretraining) get an untrained id embedding. Solved by **Candidate Item Randomization** and **Item-age Dependent Dropout** during fine-tuning.

---

## Key Ideas

### Pretraining (Sec 3.1): compact IDs + multi-horizon InfoNCE objectives

```
H = phi_out(M(phi_in(E_item + E_action + E_surface)))                    [Eq. 1]
```

Unlike LLM-backbone recommendation foundation models (M6Rec, 360Brew) that describe items as text prompts, PinFM encodes users as a sequence of **learned categorical embeddings** (item id, action type, surface type — summed, not concatenated) fed through a GPT2-style, pre-LN, **causal** decoder-only transformer. This keeps pretraining cheap enough to run over 2 years of history at billions of items of vocabulary.

Next-token prediction alone under-captures that users have multiple, time-varying interests, so PinFM combines three InfoNCE-based (Eq. 2) objectives, all implemented via the same contrastive core (`info_nce_loss`):

| Loss | What it predicts | Why |
|---|---|---|
| `next_token_loss` (L_ntl) | the very next positively-engaged item | standard next-item objective |
| `multi_token_loss` (L_mtl) | every positive item in a short future window | interests are locally consistent — predicting only 1 step forward under-uses that signal |
| `future_token_loss` (L_ftl) | a future window, but *only* from the fixed downstream sequence length `L_d` | sharpens the representation specifically at the length the fine-tuned ranking model will actually use — analogous to instruction fine-tuning's fixed-prompt setup |

### Fine-tuning (Sec 3.2): early vs. late fusion

| Strategy | Mechanism | Trade-off |
|---|---|---|
| **Late fusion** (`late_fusion_representation`) | encode the user sequence alone; reuse the same vector for every candidate | fully cacheable per request, but not target-aware |
| **Early fusion** (`PinFMDCAT`) | append the candidate to the sequence, cross-attend | contextualizes on the specific candidate (stronger — paper's Table 1: +2.9-3.8% Save HIT@3 vs. +1.9% for late fusion), at the cost of one forward pass per candidate — which DCAT then makes cheap |

### Deduplicated Cross-Attention Transformer, DCAT (Sec 4.1, Eq. 3-4)

```
Context component (once per UNIQUE user sequence):
    Q_u, K_u, V_u = W_q X_u, W_k X_u, W_v X_u
    X_u' = Attention(Q_u, K_u, V_u)                      # cache K_u, V_u

Crossing component (once per candidate, incl. repeats):
    K, V = Psi^-1(K_u) concat K_cand,  Psi^-1(V_u) concat V_cand
    X_c' = Attention(Q_cand, K, V)                        # cross-attention
```

The same user's raw activity sequence is resent once per candidate scored (ratio **1:1000** at serving, **1:10** at training). DCAT exploits this by deduplicating identical raw id sequences (`deduplicate_sequences`, `Psi`), running the transformer's self-attention only on the unique set, then broadcasting the resulting K/V cache back to every (duplicate) row (`Psi^-1`) for cheap cross-attention against each candidate. This is structurally the same caching idea as HSTU's M-FALCON (see [`../HSTU/hstu.py`](../HSTU)), but deduplicating on identical *raw inputs within a batch* rather than caching one user's history across a whole scoring pass.

### Cold-start handling (Sec 3.2)

Early fusion makes the model directly dependent on the candidate's id embedding — bad news for fresh items with no interaction history to have learned a good embedding from. Two remedies, both applied during fine-tuning only:

- **Candidate Item Randomization** (`candidate_id_randomization`): replace 10% of candidate ids with a random id, simulating cold start so the model learns not to over-rely on the id embedding.
- **Item-age Dependent Dropout** (`item_age_dependent_dropout`): dropout `p=0.7` on PinFM's output for candidates <7 days old, `p=0.5` for 7-28 days, none otherwise.

Paper's Table 2: combining both turns HF-28-day-fresh-item Save HIT@3 from **-4.4%** to **+17.7%**.

### Embedding quantization (Sec 4.2, out of scope for modeling — simulated here)

The 20B parameters are almost entirely the item/action/surface embedding tables. `quantize_dequantize_embedding` simulates the paper's FBGEMM-style per-row min-max int8/int4 post-training quantization (fake-quantize + immediately dequantize) so its accuracy impact can be measured in pure PyTorch; `quantization_error` reproduces the paper's reported relative L2 error (0.45% @ int8, 7.8% @ int4). The actual bit-packed storage format and Triton dequantization kernels (Sec 4.2-4.3) are infrastructure, not modeling, and are out of scope here.

---

## Architecture

```
Pretraining:
  item_ids, action_ids, surface_ids  ──▶ UserActivityEmbedding (Eq. 1, phi_in)
                                              │
                                    CausalTransformer (GPT2-style, pre-LN, causal)
                                              │
                                    OutputProjection (phi_out)  ──▶  H  (B, m, d)
                                              │
                       TargetItemEncoder(item_emb(id))  ──▶  Z  (B, m, d)
                                              │
              next_token_loss / multi_token_loss / future_token_loss  (Eq. 2)

Fine-tuning (early fusion, DCAT):
  user context sequences (many duplicate rows across candidates in a request)
            │
    deduplicate_sequences  ──▶  unique sequences only
            │
    UserActivityEmbedding → CausalTransformer.forward(return_kv_cache=True)
            │                                        (context component, Eq. 3)
    per-layer K, V cache (B_u, H, L, hd)
            │
    broadcast via inverse indices (Psi^-1)  ──▶  (B, H, L, hd)
            │
    candidate_id ─▶ UserActivityEmbedding (no action/surface) ─▶ CausalTransformer.forward_cross
            │                                        (crossing component, Eq. 4)
    OutputProjection  ──▶  candidate_repr (B, d)  ──▶  downstream ranking model feature crossing
```

---

## Usage

```python
from pinfm import (
    UserActivityEmbedding, CausalTransformer, OutputProjection, TargetItemEncoder,
    next_token_loss, multi_token_loss, future_token_loss,
    PinFMDCAT, compute_savings, late_fusion_representation,
    candidate_id_randomization, item_age_dependent_dropout,
    quantize_dequantize_embedding, quantization_error,
)

# --- Pretraining ---
user_emb = UserActivityEmbedding(vocab_size=10_000_000, num_actions=6, num_surfaces=3, d_model=512)
backbone = CausalTransformer(d_model=512, n_layers=12, n_heads=8)
out_proj = OutputProjection(d_model=512)
target_encoder = TargetItemEncoder(d_model=512)

x = user_emb(item_ids, action_ids, surface_ids)   # (B, m, d)
H = out_proj(backbone(x))                         # user representation sequence
Z = target_encoder(user_emb.item_emb(item_ids))   # contrastive targets

loss = (
    next_token_loss(H, Z, is_positive, temperature=0.1)
    + multi_token_loss(H, Z, is_positive, window=8, temperature=0.1)
    + future_token_loss(H, Z, is_positive, l_d=256, window=8, temperature=0.1)
)
loss.backward()

# --- Fine-tuning: early fusion via DCAT ---
dcat = PinFMDCAT(vocab_size=10_000_000, num_actions=6, num_surfaces=3,
                  d_model=512, n_layers=12, n_heads=8)

candidate_ids = candidate_id_randomization(candidate_ids, vocab_size=10_000_000, p=0.1)  # cold-start aug
candidate_repr, num_unique = dcat(user_item_ids, user_action_ids, user_surface_ids, candidate_ids)
print(compute_savings(user_item_ids.shape[0], num_unique))  # fraction of compute avoided by dedup

candidate_repr = item_age_dependent_dropout(candidate_repr, item_age_days, training=True)
# candidate_repr now feeds the downstream ranking model's feature-crossing layer

# --- Late fusion alternative (cheaper, not target-aware) ---
user_only_repr = late_fusion_representation(H, mode="mean")

# --- Embedding quantization impact ---
print(quantization_error(user_emb.item_emb.weight, num_bits=4))  # ~0.078, matches paper
```

---

## Files

```
PinFM/
├── pinfm.py         # full implementation + smoke test
│   ├── UserActivityEmbedding        # phi_in: item+action+surface embedding (Eq. 1)
│   ├── TargetItemEncoder            # psi: contrastive target projection
│   ├── OutputProjection             # phi_out: final user representation projection
│   ├── CausalSelfAttentionLayer     # pre-LN block; forward (self-attn) + forward_cross
│   ├── CausalTransformer            # GPT2-style backbone M; KV-cache aware
│   ├── info_nce_loss                # Eq. 2 contrastive core
│   ├── next_token_loss              # L_ntl
│   ├── multi_token_loss             # L_mtl
│   ├── future_token_loss            # L_ftl
│   ├── deduplicate_sequences        # Psi: dedup identical raw sequences
│   ├── compute_savings              # reports compute avoided by dedup
│   ├── PinFMDCAT                    # Deduplicated Cross-Attention Transformer (Eq. 3-4)
│   ├── late_fusion_representation   # PinFM-lite-mean / PinFM-lite-last (Table 1)
│   ├── candidate_id_randomization   # CIR cold-start augmentation
│   ├── item_age_dependent_dropout   # IDD cold-start regularization
│   ├── quantize_dequantize_embedding  # simulated int8/int4 PTQ (Sec 4.2)
│   └── quantization_error           # relative L2 quantization error
├── test_pinfm.py    # pytest test suite (47 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 pinfm.py
```

Expected output (values vary slightly by seed):

```
--- Pretraining forward + losses ---
H: torch.Size([4, 24, 32]), Z: torch.Size([4, 24, 32])
L_ntl=3.5020  L_mtl=3.3728  L_ftl=3.5321  backward: OK

--- Deduplicated Cross-Attention Transformer (DCAT) ---
candidate_repr: torch.Size([12, 32]), unique contexts: 2/12 (compute savings: 83.3%)
backward: OK

--- Fusion strategies ---
late fusion (mean): torch.Size([4, 32]), (last): torch.Size([4, 32])

--- Cold-start handling ---
CIR changed 10.5% of candidate ids (target ~10%)
IDD output: torch.Size([4, 32])

--- Embedding quantization simulation ---
relative L2 error: int8=0.0046, int4=0.0788  (paper: 0.0045, 0.078)
```

### Test suite

```bash
python3 -m pytest test_pinfm.py -v
```

Expected output:

```
collected 47 items
...
47 passed in ~1s
```

Useful variants:

```bash
python3 -m pytest test_pinfm.py::TestPinFMDCAT -v
python3 -m pytest test_pinfm.py -x
```
