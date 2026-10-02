# OneTrans-V2

PyTorch reference implementation of **OneTrans-V2: Unifying Retrieval, Pre-rank, and Fine-rank with One Transformer in Industrial Recommender** (ByteDance Global E-Commerce Recommendation Foundation Team).

Paper: https://arxiv.org/abs/2609.28589

Deployed across all three stages of a large-scale e-commerce recommender: **+9.74% GMV per user** and **3.2x QPS** versus the cascade it replaces (OneTrans-GR + OneTrans-Lite + OneTrans), under the same hardware. Offline, OneTrans-V2_S (239 GFLOPs, below the 275 GFLOPs cascade) beats each single-stage baseline, e.g. +21.10% retrieval HR@1; OneTrans-V2_L adds +69.36% HR@1 over OneTrans-GR. Successor to [`onetrans`](../onetrans).

---

## Summary

Industrial recommenders are cascades (retrieval → pre-rank → fine-rank), usually trained and served as separate models. That duplicates engineering, re-encodes the same user behavior sequence in every stage, prevents stages from sharing supervision (pre-rank and fine-rank disagree on ordering), and fragments capacity. Retrieval is split further into objective-specific channels (clicks/orders, discovery, advertising), each with its own model and decoder.

OneTrans-V2 keeps the cascade (each stage retains its native candidate features and latency budget) but makes it **one jointly trained causal Transformer**:

1. **Shared user context.** The candidate-independent behavior sequence is encoded once (per-layer K/V cache); each stage appends its own tokens with *token-specific* parameters (mixed parameterization) and reads the cache through a **stage visibility mask**. Stage tokens are isolated from other stages/exposures and cannot modify the user context.
2. **Joint training with in-model distillation.** Fine-rank (teacher, logits detached) supervises pre-rank (student) inside the same backward pass — no separate teacher. Joint training improves every stage over separately trained ones; distillation removes the pre-rank/fine-rank ordering gap (online inversion rate −6.7%, top-10 coverage +31.4%).
3. **DCGR — Decision-Conditioned Generative Retrieval.** Before generating a semantic ID, the model predicts a short *decision prefix* (purchase level, discovery level, supply type, spending level) — a non-linguistic chain-of-thought. A **business offset** β·φ(z) added at decode time steers the decision distribution, so one generative model serves several business objectives without retraining or extra retrieval channels.
4. **Scaling & stabilization.** Sparse MoE backbone (shared expert + sigmoid router), GQA, gated attention, QKNorm, µP / Depth-µP, AdamW.
5. **Sequence-Native Training (SNT).** Train per user window: encode the behavior sequence once, let every exposure read its own causal prefix (4.4x training speedup).

---

## Key Ideas

### Stage visibility mask (Sec 4.1)
Behavior tokens attend causally within S and never to stage tokens. A stage token of exposure *e* attends to the behavior prefix up to its **anchor** `a_e` (last behavior that had arrived at request time), itself, and preceding tokens of its own (exposure, stage). Hence an exposure inside a multi-exposure sequence produces exactly what it would on the truncated prefix — the SNT invariant, tested in `TestSNT.test_exposure_sees_only_its_causal_prefix`.

### DCGR (Eq. 1–4)
Sequence after the behavior history: `[CTX, DT^p, DT^d, SID_0, SID_1] → SID_2`.
* **Parallel decisions** `z_oc, z_disc, z_ad` are predicted from the CTX state and *summed* into one token `DT^p` (chaining would impose an arbitrary decoding order; Table 7).
* **Dependent decision** `z_aov` (spending level) is predicted at `DT^p` and embedded as `DT^d`; it is only meaningful given the purchase level (`z_oc = 0 ⇔ z_aov = 0`).
* Objective: `P(z, Item | H) = P(z|H) · P(Item | H, z)`, all cross-entropy with teacher forcing.
* **Decoding:** enumerate all valid `z` in two batched passes over the cached context (not sequentially), score `log P(z|H) + β φ(z)`, then beam search over the SID codes with two-stage top-k:

```
 steered decision score   SID score
 log P(z|H) + β φ(z)   +  Σ_ℓ log P(SID_ℓ | H, z, SID_<ℓ)                      (Eq. 4)
```
`β=0` is the model's own ranking; the offset is soft and only moves which decision wins, leaving the item term untouched.
* **Classifier-free guidance** (Eq. 5, not deployed in the paper): with the decision prefix randomly replaced by null tokens in training, SID codes can be scored `log P(·|∅) + w [log P(·|z) − log P(·|∅)]`.

### Ranking tasks and distillation (Eq. 6–9)
Pre-rank: one lightweight token over a small feature subset → CTR/CVR. Fine-rank: several tokens over the full feature set → CTR/CVR. Mean-centered KD with fixed temperature:
```
p^T = σ((o^T − μ^T)/τ),  p^S = σ((o^S − μ^S)/τ),  L_kd = −Σ_t [p^T log p^S + (1−p^T) log(1−p^S)]
L = λ_r L_r + λ_p L_p + λ_f L_f + λ_kd L_kd      (0.1, 1, 1, 1; τ=0.5; μ as EMA, momentum 0.9999)
```

### TransBlock (Alg. 1) and stabilization
Pre-RMSNorm → fused QKVG → per-head QKNorm → GQA → gated attention (`A ⊙ σ(G)`) → residual scaled by `γ = 1/√(2N)` → sparse MoE (sigmoid router, top-k renormalized, routed scale, always-on shared expert). µP / Depth-µP (Table 1): hidden-weight init variance `1/fan_in`, Adam lr `η₀/(m_d·√m_N)`; AdamW (β₁=0, β₂=0.99999).

### Request-relative time rotary encoding (Eq. 11)
Stage-token queries are rotated by the request time, behavior keys by their event time, so attention depends only on `t_i − t_e` (shared by all stages/requests reading one encoding). Behavior self-attention is unchanged.

---

## Key Components

| Component | Paper section |
|---|---|
| `build_stage_mask`, `compute_anchors`, `StageBatch` | 4.1, Fig 2c |
| `MixedLinear`, `TransBlock`, `OneTransV2Backbone` (`encode_context` / `run_stage`) | 4.1, 4.3, Alg. 1 |
| `SparseMoE` | 4.3 (shared expert, sigmoid router) |
| `rotate_time` | 4.4, Eq. 11 |
| `OneTransV2.forward` / `.loss` (multi-exposure SNT forward, DCGR + BCE + KD) | 4.2, 4.4, Eq. 2–9 |
| `distillation_loss` | Eq. 8 |
| `OneTransV2.retrieve` (decision enumeration, business offset β·φ, CFG, beam search) | 4.2.1, 5.2, Eq. 4–5 |
| `two_stage_topk`, `hit_rate_at_m`, `sid_to_items` | 5.2, 6.1 |
| `OneTransV2.encode_user`, `.prerank`, `.finerank` | 5.1 (encode once, three stages) |
| `mup_hidden_lr`, `mup_param_groups`, `build_optimizers` | 4.3 Table 1, 6.1 |

**Implementation notes.** Sizes are toy-scale (the paper: 8192-entry SID codebooks, 662 / 239 GFLOPs). The decision space is the paper's four dimensions. Not reproduced: kernel fusion, FP16/FP8 serving, sparse embedding tables (AdaGrad is applied densely), RQ-KMeans SID training (SIDs are inputs), production feature pipelines. The tokenizers for stage features are simple linear maps, since the paper does not detail them. Behavior sequences are assumed unpadded. `retrieve` serves one request at a time.

---

## Usage

```python
from onetrans_v2 import *

cfg = Config()
model = OneTransV2(cfg)
sparse_opt, dense_opt = build_optimizers(model)        # AdaGrad (embeddings) + AdamW with muP lrs

# training: one behavior encoding, E exposures per user window (see make_batch for the batch layout)
out = model(batch, null_prob=0.1)                       # null_prob enables CFG training
losses = model.loss(batch, out)                         # retrieval / pre / fine / kd / total
losses["loss"].backward(); sparse_opt.step(); dense_opt.step()

# serving: encode the user ONCE, reuse for every stage
model.eval()
anchor = int(compute_anchors(seq_times, t_req)[0, -1])
caches = model.encode_user(seq_items, seq_actions, seq_times)
phi = {"ad": torch.tensor([0.0, 1.0])}                  # prefer sponsored supply
hits = model.retrieve(caches, ctx, anchor, t_req, k=100, beta=1.0, phi=phi)
items = sid_to_items(hits, sid_index)
pre = model.prerank(caches, pre_feats, anchor, t_req)   # thousands of candidates -> hundreds
fine = model.finerank(caches, fine_feats[keep], anchor, t_req)
```

---

## Files

```
OneTrans-V2/
├── onetrans_v2.py        # implementation + smoke test
├── test_onetrans_v2.py   # pytest suite (70 tests)
└── README.md
```

## Running

```bash
python3 onetrans_v2.py                        # smoke test
python3 -m pytest test_onetrans_v2.py -v      # 70 passed (~1 min; one brute-force DCGR check dominates)
```

Notable tests: an exposure in a multi-exposure sequence equals the same exposure alone on its truncated prefix; stage/exposure isolation; time-shift invariance; cached serving equals the joint forward; DCGR scores equal brute-force joint log-probs and the top-1 is the global argmax; the business offset shifts scores by exactly β·φ(z); two-stage top-k equals global top-k; KD only trains the student; µP rules.
