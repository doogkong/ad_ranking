# SUM: Scaling User Modeling for Ads Personalization

PyTorch reference implementation of:

- **`sum_model.py`** — "Scaling User Modeling: Large-scale Online User Representations for Ads Personalization in Meta" (Meta Platforms,https://arxiv.org/pdf/2311.09544,  arXiv 2311.09544v2)

---

## Problem

Large-scale ads systems run hundreds of downstream ranking models per surface (Feed, Reels, Stories, Marketplace, …).  Each model independently learns its own user representation from scratch, which causes:

| Problem | Description |
|---|---|
| **Capacity waste** | Each downstream model allocates embedding tables for the same user signals |
| **Cold start** | Low-volume surfaces have sparse training data; user representations are underfit |
| **Training cost** | Every model trains independently; no knowledge transfer |
| **Consistency** | Downstream models may develop contradictory user representations |

Root cause: there is no shared, high-quality upstream user representation.

**Solution:** Train one large-capacity **UserTower** jointly on all downstream tasks. Downstream models receive cached user embeddings and only need to learn the ad-side and fusion components — a much smaller problem.

---

## Architecture

```
User behaviour sequence (sparse IDs, dense features)
  item_emb_1, item_emb_2, …, item_emb_L        ∈ ℝ^(B, L, input_dim)
  dense context features                         ∈ ℝ^(B, dense_dim)
            │
            ▼
  UserTower — pyramid of N Interaction Modules (§3)
  ┌──────────────────────────────────────────────────────────────────────┐
  │  IM_1: X_1 = Concat(Interaction(X_0), X_dense) + Residual(X_0)     │
  │  IM_2: X_2 = Concat(Interaction(X_1), X_dense) + Residual(X_1)     │
  │  …                                                                  │
  │  IM_N: X_N                                                          │
  │                                                                      │
  │  Interaction(X) = MLP(X) + DotCompr(X) + MLPMixer(X) + DCN(X)     │
  │    Four parallel extractors summed → shared information             │
  │                                                                      │
  │  Tap K=2 embeddings from top K levels → pool → project to D=96     │
  └──────────────────────────────────────────────────────────────────────┘
            │  user_embs: K × (B, 96)
            │
            ▼  (cached in SOAP feature store, served async)
  ┌──────────────────────────────────────────────────────────────────────┐
  │  MixTower (§3.2) — per-request, per-surface                         │
  │    fused = Concat(user_embs, ad_feats)       (B, K*D + ad_dim)     │
  │    cross = DCNv2(fused)                      (B, hidden)           │
  │    deep  = MLP(fused)                        (B, hidden)           │
  │    logits = TaskHead(Concat(cross, deep))    (B, n_tasks)          │
  └──────────────────────────────────────────────────────────────────────┘
            │  (B, n_tasks) per-task logits
            ▼
  Training loss (Eq. 8):
    L = -1/N Σ_i Σ_t w_t [y_ti log(ŷ_ti) + (1-y_ti) log(1-ŷ_ti)]
```

---

## Interaction Module (§3.1, Eq. 1)

The core building block of the pyramid:

```
X_n = Concat(Interaction(X_{n-1}), X_dense) + Residual(X_{n-1})   (Eq. 1)

Interaction(X) = sum of four parallel extractors:

  1. MLPExtractor:
       h = ReLU(W_2 · ReLU(W_1 · X))                                (baseline)

  2. DotCompressionWithAttention:
       Q  = W_q · X_dense                           (B, H, d_k)    (Eq. 2)
       K  = W_k · X_seq                             (B, L, d_k)    (Eq. 3)
       Z  = softmax(Q K^T / √d_k) · X_seq → flatten               (Eq. 4)

  3. MLPMixerExtractor:
       Y_token   = X + W_2 · σ(W_1 · X^T)^T        token-mix      (Eq. 6)
       Y_channel = Y_token + W_4 · σ(W_3 · Y_token) channel-mix   (Eq. 7)

  4. DeepCrossExtractor (DCN-v2):
       x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l    explicit cross (Eq. 5)
```

The pyramid taps K embeddings from the top K levels, projecting each to dimension D.

---

## SOAP — SUM Online Asynchronous Platform (§4)

Decouples embedding freshness from serving latency:

```
Read path (latency-critical, ~1ms):
  1. Downstream model requests user embedding for user_id
  2. Return average of K most recent cached embeddings immediately
  3. Schedule async write task

Write path (off critical path):
  4. Compute new embedding with UserTower on latest features
  5. Write to feature store (ZippyDB in production)
  6. Rolling window of K=3 embeddings:
       Emb_served = mean(Emb_k, Emb_{k-1}, Emb_{k-2})
       Mitigates distribution shift from the async lag
```

This allows UserTower to run on large, expensive hardware without blocking
auction serving.

---

## Limitations of User Sequence Modeling and Solutions

This is a core research area with several known challenges. Each limitation has well-studied mitigations.

### 1. Sequence Length — O(L²) Attention Bottleneck

**Problem:** Full self-attention over thousands of events is prohibitively expensive.

**Solutions:**

| Approach | Idea | Paper |
|---|---|---|
| **SIM (Search-based Interest Model)** | Two-stage: hard search retrieves top-K relevant items from long history via inverted index; attention runs on K<<L items only | Alibaba, KDD 2020 |
| **ETA (Efficient Target Attention)** | LSH-based approximate nearest-neighbour lookup to filter history to target-relevant items in O(L) | Alibaba, RecSys 2021 |
| **Linear attention** | Kernel approximation (Performer, Linformer) reduces attention to O(L) without index structures | Choromanski et al. 2020 |
| **Hierarchical aggregation** | Compress each session to a fixed-size vector; attend over sessions, not items | reduces L by session count |

```
SIM two-stage retrieval:
  Long history (L=10000)
        │ hard search: item category / keyword match
        ▼
  Retrieved candidates (K=200)
        │ soft-attention with target item
        ▼
  Interest representation
```

### 2. Staleness — Async Embedding Lag

**Problem:** SOAP serves the *previous* embedding while the new one is computed. High-velocity users experience stale personalization.

**Solutions:**

| Approach | Idea |
|---|---|
| **Delta / incremental update** | Maintain a lightweight online updater (e.g. GRU) that applies new events on top of the cached embedding without rerunning the full UserTower |
| **Event-driven invalidation** | Trigger immediate re-computation for high-signal events (purchase, explicit dislike) rather than waiting for the periodic refresh cycle |
| **Hybrid: cached + real-time** | Concatenate the cached long-term embedding with a cheap real-time short-term embedding computed on the last 5 events at serving time |
| **Reduce async window** | Decrease the SOAP refresh period; trade compute cost for freshness; use dedicated serving hardware |

```
Hybrid long-term + short-term:
  SOAP cache (stale by ~1 cycle)   Last-N events (real-time)
         │                                   │
     LT-emb (96d)                     GRU/MLP (32d)
         └──────────────┬──────────────────┘
                     Concat → fusion MLP → final user emb
```

### 3. Distribution Shift on Model Update

**Problem:** Retraining the shared UserTower invalidates all cached embeddings simultaneously. Downstream models see a sudden input shift.

**Solutions:**

| Approach | Idea |
|---|---|
| **Shadow rollout** | Run old and new UserTower in parallel for N% of traffic; blend embeddings during transition period |
| **Embedding versioning** | Downstream models carry a version tag; feature store keeps two generations; traffic migrates gradually |
| **Consistency regularization** | During retraining, add a KL penalty between new embeddings and stored old ones to limit representation drift (analogous to knowledge distillation) |
| **Continual / incremental training** | Fine-tune from the previous checkpoint with a small LR rather than full retraining; limits the magnitude of the shift |

### 4. Negative Transfer in Multi-task Training

**Problem:** Tasks with high data volume (click) dominate the shared representation, degrading low-volume tasks (conversion).

**Solutions:**

| Approach | Idea | Paper |
|---|---|---|
| **Per-task loss weights** | Upweight low-volume tasks in Eq. 8 (already in SUM) | — |
| **Mixture-of-Experts (MoE)** | Each task routes to a different subset of experts in the tower; shared + task-specific capacity | MoE, Switch Transformer |
| **Gradient surgery / PCGrad** | Project task gradients to remove conflicting components before the shared parameter update | Yu et al. NeurIPS 2020 |
| **Task-specific adapter layers** | Shared backbone + small per-task adapter; adapter absorbs task-specific signal without polluting shared weights | Houlsby et al. 2019 |

```
MoE approach:
  Shared bottom layers
        │
  Expert router (softmax gating)
  ┌─────┼─────┐
  E1    E2   E3    (task-specific expert subsets)
  └─────┼─────┘
        │
  Task head
```

### 5. Cold Start for New Users

**Problem:** No behavioural history → the UserTower receives a near-zero sequence; the embedding is dominated by uninformative padding.

**Solutions:**

| Approach | Idea |
|---|---|
| **Side-information embedding** | Enrich sparse users with non-behavioural signals: device type, locale, install date, demographics (where permitted) |
| **Cross-surface transfer** | Aggregate events from *all* surfaces (not just the serving surface) to build a richer sequence for sparse users |
| **Popularity-based initialisation** | For users with zero history, substitute the average embedding of users with similar demographics or the global mean |
| **Meta-learning (MAML)** | Train the UserTower to adapt quickly to new users with few gradient steps; improves few-shot personalisation |

### 6. Session Structure Ignored — Recency Blindness

**Problem:** Pooling over the full lifetime history treats a 6-month-old event equally to one from 5 minutes ago.

**Solutions:**

| Approach | Idea | Paper |
|---|---|---|
| **Recency decay weighting** | Weight each item's contribution by exp(−λ·age) before pooling | DIN (Zhou et al., KDD 2018) |
| **Target-aware attention (DIN)** | Attention weight of each history item is a function of its similarity to the *target* ad — automatically surfaces relevant context | Alibaba DIN |
| **Hierarchical session model** | Encode each session independently (GRU/Transformer), then attend over session summaries; captures intra-session continuity | HSTU, SIM-b |
| **Dual-channel architecture** | Separate towers for short-term (last session) and long-term (lifetime) history; concatenate before MixTower | SIM, MIMN |

```
Hierarchical session model:
  Session_1  Session_2  …  Session_T
  [GRU/MHA]  [GRU/MHA]    [GRU/MHA]
      │           │              │
      s_1         s_2           s_T     (session summaries)
      └───────────┴──────────────┘
               MHA over sessions
                     │
               user embedding
```

### 7. Sparse Targets — Conversion Imbalance

**Problem:** Conversion labels are ~100× sparser than click labels; the shared tower underfits conversion.

**Solutions:**

| Approach | Idea |
|---|---|
| **Label smoothing + task weights** | Strongly upweight conversion loss; label smoothing reduces overconfident click predictions |
| **Auxiliary tasks as proxies** | Add add-to-cart, product page view, wishlist as intermediate proxy tasks; richer conversion signal |
| **Positive sample reweighting** | Oversample or upweight conversion-positive examples during training |
| **Two-stage tower** | Train click tower first; fine-tune a separate conversion head on top of the frozen click representation |

### 8. Privacy Constraints

**Problem:** Fine-grained behavioural sequences are among the most privacy-sensitive user data; regulatory constraints limit storage and cross-surface use.

**Solutions:**

| Approach | Idea |
|---|---|
| **On-device processing** | Compute user embeddings on-device using a distilled UserTower; only the embedding (not the raw sequence) leaves the device |
| **Differential privacy (DP-SGD)** | Add calibrated Gaussian noise to gradients during UserTower training; provides (ε, δ)-DP guarantees |
| **Federated learning** | Train UserTower across devices without raw data leaving the device; aggregate gradients via secure aggregation |
| **k-anonymity / feature hashing** | Hash user IDs and item IDs into buckets; prevents re-identification at the cost of some representation precision |
| **Data minimisation** | Cap history length; drop event metadata below a frequency threshold; store categories rather than raw item IDs |

---

## Key Results (§5)

| Metric | Result |
|---|---|
| Online RPM (downstream model) | **+1–3%** across surfaces |
| Offline AUC | Significant improvement on low-volume surfaces |
| Model size reduction (downstream) | Up to **40% fewer parameters** needed downstream |
| Training cost reduction | **~50%** reduction in downstream training FLOPs |
| Cold-start improvement | Low-volume surfaces benefit most from shared representation |

---

## Usage

```python
from sum_model import SUMModel, SOAPFeatureStore, SOAPClient

# Build and train the full model
model = SUMModel(
    input_dim    = 64,    # item embedding dimension
    dense_dim    = 32,    # dense context feature dimension
    seq_len      = 20,    # history sequence length
    ad_feat_dim  = 48,    # ad feature dimension (for MixTower)
    n_tasks      = 3,     # click, conversion, engagement
    n_layers     = 4,     # pyramid depth
    user_emb_dim = 96,    # D: user embedding dimension
    n_user_embs  = 2,     # K: number of user embedding outputs
)

user_embs, logits = model(x_seq, x_dense, ad_feats)
# user_embs: K × (B, 96)   — cached in SOAP after training
# logits:    (B, 3)         — per-task predictions

task_weights = torch.tensor([1.0, 2.0, 1.5])  # upweight conversion
loss = model.compute_loss(logits, labels, task_weights)
loss.backward()

# -------------------------------------------------------------------------
# SOAP async serving
# -------------------------------------------------------------------------
store  = SOAPFeatureStore(window_size=3)   # rolling average of K=3
client = SOAPClient(model.user_tower, store)

# Serve: returns cached embedding, schedules async update
emb = client.serve(user_id='user_001', x_seq=seq_tensor, x_dense=ctx_tensor)
# emb: (K*D,) concatenated user embedding

# Downstream model only needs to train ad-side + fusion (much cheaper)
from sum_model import MixTower
mix = MixTower(user_emb_dim=96, n_user_embs=2, ad_feat_dim=48, n_tasks=3)
logits = mix([emb[:96], emb[96:]], ad_feats)
```

### Hyperparameter guidance

| Parameter | Typical value | Notes |
|---|---|---|
| `n_layers` | 3–6 | Deeper pyramid = more expressive but slower; 4 is a good default |
| `user_emb_dim` (D) | 64–256 | 96 matches Meta's reported D; smaller for latency-constrained surfaces |
| `n_user_embs` (K) | 2–4 | K=2 captures short- and long-term user state |
| `extractor_out` | 64–256 | Output dim of each parallel extractor; larger = higher capacity |
| `hidden_dim` | 128–512 | Shared dimension inside pyramid and MixTower |
| `seq_len` | 10–200 | Longer history → better but more memory; use SIM retrieval for >200 |
| `task_weights` w_t | 1.0–5.0 | Upweight conversion vs click to counter data imbalance |
| SOAP window_size K | 3 | Rolling average over 3 cycles minimises distribution shift |

---

## Files

```
sum_user_modeling/
├── sum_model.py        # full SUM implementation
│   ├── MLPExtractor               # dense MLP extractor (baseline)
│   ├── DotCompressionWithAttention  # Eq. 2-4: attention-based sequence compression
│   ├── MLPMixerExtractor          # Eq. 6-7: token-mix + channel-mix (parameter-efficient)
│   ├── DeepCrossExtractor         # Eq. 5: DCN-v2 explicit feature crosses
│   ├── InteractionModule          # Eq. 1: pyramid building block (4 parallel extractors)
│   ├── UserTower                  # pyramid of N InteractionModules → K user embeddings
│   ├── MixTower                   # DHEN-style cross+deep fusion → per-task logits
│   ├── multi_task_loss            # Eq. 8: weighted multi-task binary cross-entropy
│   ├── SOAPFeatureStore           # rolling window K-embedding cache (production: ZippyDB)
│   ├── SOAPClient                 # async serve: read cached → compute async → write back
│   └── SUMModel                   # UserTower + MixTower end-to-end
└── README.md
```

---

## Running

```bash
python3 sum_model.py
```

Expected output:
```
UserTower embs: 2 × torch.Size([4, 96])
MixTower logits: torch.Size([4, 3])
Loss: 1.9751
Backward: OK
Total params:      985,607
  UserTower:       733,108
  MixTower:        252,499

SOAP serve  (cold):  torch.Size([192])
SOAP serve  (warm):  torch.Size([192])  (cached previous embedding)
```
