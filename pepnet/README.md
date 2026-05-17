# PEPNet

PyTorch reference implementation of **PEPNet: Parameter and Embedding Personalized Network for Infusing with Personalized Prior Information** (KDD 2023, Kuaishou Technology).

Paper: https://arxiv.org/abs/2302.01115

Deployed at Kuaishou serving **300M daily active users**: **+1.08–2.11% Like**, **+1.43–2.23% Follow**, **+1.31–1.55% Forward**, **+1.25–2.12% Watch Time** across three domains.

---

## Paper Summary

Online recommendation platforms serve users across multiple surfaces (Discovery Tab, Featured-Video Tab, Slide Tab) with multiple interaction targets (Like, Follow, Forward, Click, Watch Time). A naive joint model trained on all domains and tasks simultaneously faces the **imperfectly double seesaw phenomenon**: improving one domain tends to degrade another (domain seesaw), and improving one task tends to degrade another (task seesaw) — both at the same time.

The root cause is the mismatch between a shared, domain-agnostic representation and the fundamentally different feature importance profiles and target sparsity distributions across domains and tasks. Simply training harder or adding more capacity does not resolve this — the shared bottom layers learn average importance, washing out domain-specific and task-specific signals.

Two root challenges block effective multi-domain multi-task recommendation:

**Challenge 1 — Domain seesaw.** The shared embedding layer treats all domains equally. Features important for the Featured-Video Tab (long watch-time items) have different importance distributions than the Discovery Tab (diverse browsing). A single shared embedding necessarily compromises between them, degrading both.

**Challenge 2 — Task seesaw.** DNN task towers share the same bottom-layer representation. Click has 14.66% label sparsity in Domain A; Like has 3.68%. The model cannot simultaneously fit both sparse and dense targets without one degrading the other — different tasks require different DNN parameter magnitudes and activation patterns.

### How does PEPNet solve it?

PEPNet introduces two plug-and-play gating modules built on a shared **Gate Neural Unit (Gate NU)**, both of which inject personalized prior information into the model at different points:

| Module | Seesaw Targeted | Where It Acts | Prior Information Used |
|---|---|---|---|
| **Gate NU** | — | Shared primitive | Any personalized prior |
| **EPNet** | Domain seesaw | Embedding layer (bottom) | Domain ID + domain statistics |
| **PPNet** | Task seesaw | Every DNN hidden layer | User ID + item ID + author ID |

Both modules use **stop gradients** on the shared representations they read from, ensuring personalization does not interfere with the shared bottom layers learning common features.

### Measured results

| Component | Domain / Task | Metric | Online Gain |
|---|---|---|---|
| Full PEPNet | Discovery Tab | Like | +1.08% |
| Full PEPNet | Discovery Tab | Follow | +1.43% |
| Full PEPNet | Discovery Tab | Watch Time | +1.25% |
| Full PEPNet | Featured-Video Tab | Like | +1.36% |
| Full PEPNet | Featured-Video Tab | Follow | +1.81% |
| Full PEPNet | Featured-Video Tab | Watch Time | +1.93% |
| Full PEPNet | Slide Tab | Like | +2.11% |
| Full PEPNet | Slide Tab | Follow | +2.23% |
| Full PEPNet | Slide Tab | Watch Time | +2.12% |
| EPNet alone | All domains | GAUC avg | Eliminates domain seesaw |
| PPNet alone | All tasks | GAUC avg | Eliminates task seesaw |
| **Full PEPNet** | **All 3 domains × 6 tasks** | **AUC + GAUC** | **Best on all 18 metrics** |

---

## Key Innovations

### Gate Neural Unit (§2.2.1)

The Gate NU is the shared primitive underlying both EPNet and PPNet. Inspired by LHUC (Learning Hidden Unit Contributions) from speech recognition, it takes any personalized prior information **x** and produces a scaling gate **δ**:

```
x'  = ReLU(x W + b)          two-layer MLP
δ   = γ · Sigmoid(x' W' + b'),   δ ∈ [0, γ]
```

The scaling factor **γ = 2** is critical: the output range [0, 2] is centered at 1, so the gate can both **suppress** (δ < 1) and **amplify** (δ > 1) the signal it gates. A pure sigmoid would only attenuate; γ = 2 allows doubling important features while zeroing irrelevant ones. The paper evaluates γ ∈ {1, 2, 3} and finds γ = 2 consistently optimal.

---

### EPNet — Embedding Personalized Network (§2.2.2)

EPNet resolves the **domain seesaw** by injecting domain-specific personalization at the embedding layer. The shared embedding E (from sparse + dense features) is scaled element-wise by a domain-conditioned gate:

```
E        = concat(E(F_sparse), E(F_dense))     shared embedding, (B, emb_flat)

δ_domain = GateNU(E(F_d) ⊕ ∅(E))              ∅ = stop gradient on E
                                                 E(F_d) = domain ID emb + domain stats

O_ep     = δ_domain ⊗ E                         element-wise personalization
```

The stop gradient `∅(E)` is essential: the Gate NU reads the current embedding to decide which dimensions to amplify, but gradients do not flow back through `∅(E)` into the embedding table. This means EPNet *adjusts* the embedding output for each domain without *changing* what the embedding table learns — keeping the shared representation intact.

**Domain-side features** E(F_d) include the domain ID and domain-specific statistics (e.g., count of user behaviors and item exposures in each domain). These capture domain-level distributional differences that the shared embedding cannot represent.

---

### PPNet — Parameter Personalized Network (§2.2.3)

PPNet resolves the **task seesaw** by injecting user/item/author personalization into every DNN hidden layer of every task tower. Each layer receives a separate gate that scales its hidden activations before the next linear transformation:

```
O_prior       = concat(E(F_u), E(F_i), E(F_a))    user + item + author embeddings

δ_task^(l)    = GateNU_l(O_prior ⊕ ∅(O_ep))       one Gate NU per layer l
                δ_task^(l) ∈ ℝ^(h_l · T)           split into T task-specific gates

Per task t, per layer l:
  O_pp_t^(l)  = δ_task_t^(l) ⊗ H_t^(l)            personalize hidden state
  H_t^(l+1)   = ReLU(O_pp_t^(l) W_t^(l) + b_t^(l))   l < L (hidden layers)
  ŷ_t         = O_pp_t^(L) W_t^(L) + b_t^(L)          l = L (output layer)
```

Key design choices:
- **One Gate NU per layer**: PPNet uses L separate Gate NUs (one per DNN layer). Each gate independently reads the same `(O_prior ⊕ ∅(O_ep))` input but specializes for the dimension of that layer's hidden state.
- **Stop gradient on O_ep**: prevents PPNet from modifying the embedding-level personalization already provided by EPNet.
- **Split into T task gates**: δ_task^(l) ∈ ℝ^(h_l·T) is split into T per-task gates each of size h_l, so each task tower gets its own independent scaling of the shared hidden state.
- **Prior features**: user ID captures individual preferences; item ID captures content-level signals; author ID captures creator-level signals (critical for short-video recommendation where creator style drives engagement).

---

### Engineering Optimizations (§2.3)

Two large-scale engineering strategies enable PEPNet's deployment at Kuaishou scale:

**Global Shared Embedding Table (GSET):** Embedding tables for 300M users × millions of items exhaust server memory. GSET uses a feature score elimination strategy (not LFU/LRU) to evict low-frequency features, keeping memory bounded while retaining high-signal embeddings.

**Offline training strategy:** Embeddings and DNN parameters require different update frequencies. Embeddings are updated with AdaGrad (lr = 0.05) and DNN parameters with Adam (lr = 5×10⁻⁶). This separation prevents the fast-moving embeddings from destabilizing the slower-converging DNN towers.

---

## Summary

| Module | Seesaw Resolved | Mechanism |
|---|---|---|
| **Gate NU** | — | `δ = γ · Sigmoid(ReLU(xW) W')`, δ ∈ [0, γ], γ=2 |
| **EPNet** | Domain seesaw | `O_ep = GateNU(domain_emb ⊕ ∅(E)) ⊗ E` |
| **PPNet** | Task seesaw | `O_pp_t^(l) = GateNU_l(O_prior ⊕ ∅(O_ep))_t ⊗ H_t^(l)` per layer |
| Stop gradients | Interference prevention | ∅(E) in EPNet; ∅(O_ep) in PPNet |
| γ = 2 scaling | Amplification + suppression | Gate range [0, 2] centered at 1 |
| Per-layer PPNet gates | Task specialization | L separate Gate NUs, one per DNN layer |

---

## Architecture

```
General Input (sparse + dense)                Domain Input          Prior Input
  [user history, item features, context]      [domain_id, stats]    [user_id, item_id, author_id]
           │                                          │                       │
    Embedding Layer (shared)                   domain_emb (B, 2d)    O_prior (B, 3d)
           │                                          │                       │
    E ∈ (B, emb_flat)                                │                       │
           │                                          │                       │
  ┌─ EPNet (Gate NU 0) ──────────────────────────────┤                       │
  │  δ_domain = GateNU(domain_emb ⊕ ∅(E))           │                       │
  │  O_ep = δ_domain ⊗ E          ∈ (B, emb_flat)   │                       │
  └──────────────────────────────────────────────────┘                       │
           │                                                                  │
    O_ep → H_t^(1) for all tasks t                                           │
           │                                                                  │
  ┌─ PPNet (Gate NU 1..L) + Task Towers ─────────────────────────────────────┤
  │  For each layer l = 1..L:                                                 │
  │    δ_task^(l) = GateNU_l(O_prior ⊕ ∅(O_ep))     ∈ (B, h_l · T)        │
  │    split → [δ_task_1^(l), ..., δ_task_T^(l)]  each ∈ (B, h_l)          │
  │                                                                           │
  │    Task t:  O_pp_t^(l) = δ_task_t^(l) ⊗ H_t^(l)                        │
  │             H_t^(l+1)  = ReLU(O_pp_t^(l) W_t^(l) + b_t^(l))            │
  └───────────────────────────────────────────────────────────────────────────┘
           │
    ŷ_t = H_t^(L+1) ∈ (B, 1)   for each task t

  ŷ = concat(ŷ_1, ..., ŷ_T) ∈ (B, T)   binary cross-entropy per task
```

---

## Usage

```python
from pepnet import PEPNet

model = PEPNet(
    sparse_vocab_sizes = [1_000_000, 500_000, 200_000],  # general sparse features
    dense_input_dims   = [256, 64],                       # dense feature group dims
    d_embed            = 40,                              # embedding dimension
    domain_vocab_size  = 10,                              # number of domains
    n_domain_stats     = 16,                              # domain statistics features
    user_vocab_size    = 10_000_000,                      # user ID vocabulary
    item_vocab_size    = 5_000_000,                       # item ID vocabulary
    author_vocab_size  = 1_000_000,                       # author ID vocabulary
    dnn_hidden         = [256, 128],                      # task tower hidden dims
    n_tasks            = 6,                               # Like, Follow, Forward, Hate, Click, EffView
    gamma              = 2.0,                             # Gate NU scaling factor
)

logits = model(
    sparse_feats  = [torch.randint(0, v, (B,)) for v in [1_000_000, 500_000, 200_000]],
    dense_feats   = [torch.randn(B, 256), torch.randn(B, 64)],
    domain_id     = torch.randint(0, 10, (B,)),   # which domain/surface
    domain_stats  = torch.randn(B, 16),            # domain-level statistics
    user_id       = torch.randint(0, 10_000_000, (B,)),
    item_id       = torch.randint(0, 5_000_000, (B,)),
    author_id     = torch.randint(0, 1_000_000, (B,)),
)  # → (B, 6)

# Per-task binary cross-entropy loss
labels = torch.zeros_like(logits)
loss = sum(
    torch.nn.functional.binary_cross_entropy_with_logits(logits[:, t], labels[:, t])
    for t in range(6)
)
loss.backward()
```

### Hyperparameter guidance

| Parameter | Typical value | Notes |
|---|---|---|
| `d_embed` | 40–64 | Paper uses 40; larger = more capacity per feature |
| `dnn_hidden` | `[256, 128]` | 2 layers sufficient; more layers → more PPNet gates |
| `n_tasks` | 1–6 | One per interaction target; each gets separate tower |
| `gamma` | 2.0 | Paper evaluates 1/2/3; 2 best — center at 1, enables amplification |
| `n_domain_stats` | 4–32 | Domain behavior counts, item exposure rates, etc. |

---

## Implementation

### What is implemented

Every component described in the paper is implemented as a standalone `nn.Module`:

| Module | Paper Section | Description |
|---|---|---|
| `GateNU` | §2.2.1 | Two-layer MLP gate: `δ = γ·Sigmoid(ReLU(xW)W')`, δ ∈ [0, γ]. Shared primitive for EPNet and PPNet. |
| `EPNet` | §2.2.2 | Domain embedding gate. `δ_domain = GateNU(domain_emb ⊕ ∅(E))`, `O_ep = δ_domain ⊗ E`. Stop gradient on E. |
| `PPNet` | §2.2.3 | Per-layer DNN gates. One GateNU per tower layer; outputs `(B, n_tasks, h_l)` gates. Stop gradient on O_ep. |
| `PEPNet` | §2.2 | Full model: shared embedding → EPNet → PPNet → T personalized task towers. |

### Design decisions and simplifications

**Gate NU hidden dimension:** Both layers of Gate NU use `output_dim` as the hidden dimension (`fc1: input_dim → output_dim`, `fc2: output_dim → output_dim`). The paper does not specify a separate hidden size; this keeps the parameter count proportional to the gate's output dimension.

**PPNet gate dims:** The gate at each layer l scales H_t^(l), which has dimension `gate_dims[l]`. For the first layer, H^(1) = O_ep (dim = emb_flat). For subsequent layers, H^(l) = output of hidden layer l-1 (dim = dnn_hidden[l-1]). So `gate_dims = [emb_flat] + dnn_hidden`, one gate per linear layer in each task tower.

**Stop gradient implementation:** Both `EPNet` and `PPNet` use `E.detach()` / `O_ep.detach()` rather than `torch.no_grad()`, so the shared representations remain part of the computation graph for their own paths while being treated as constants by the gate networks.

**Task towers:** Each of the T tasks has its own separate `nn.Linear` layers. The towers share the same input (O_ep) and receive task-specific slices of the PPNet gates. There is no weight sharing between task tower parameters — this is intentional, as PPNet's per-task gating only makes sense when the underlying weights are also task-specific.

**Domain statistics:** The reference implementation accepts `domain_stats` as a pre-computed float tensor (B, n_domain_stats). In Kuaishou's deployment, these are real-time statistics (user behavior counts, item exposure rates per domain) computed during request handling. The `domain_stats_proj` linear layer projects them to `d_embed` before concatenation with the domain ID embedding.

**Author features:** In Kuaishou's short-video context, the author (creator) ID is a primary signal — creator style and audience overlap are strong predictors of engagement. In non-video settings, `author_id` can be replaced with any item-side feature (seller ID, category ID, etc.).

### Relation to other models in this directory

| Model | Multi-domain | Multi-task | Personalization | Plug-and-play |
|---|---|---|---|---|
| Wukong | No | No | No | No |
| InterFormer | No | No | No | No |
| Kunlun | No | Limited | No | No |
| Meta Lattice | Yes | Yes | No | No |
| **PEPNet** | **Yes** | **Yes** | **Yes (EPNet + PPNet)** | **Yes** |

PEPNet is unique in this directory for its **plug-and-play** design: EPNet and PPNet can be bolted onto any existing shared-bottom model (SharedBottom, MMoE, PLE) without changing the backbone architecture. The paper demonstrates this generality by adding PPNet to single-task and single-domain models, both of which also benefit.

---

## Files

```
pepnet/
├── pepnet.py           # full implementation + smoke test
│   ├── GateNU              # two-layer gating primitive: δ = γ·Sigmoid(ReLU(xW)W')
│   ├── EPNet               # domain embedding gate (resolves domain seesaw)
│   ├── PPNet               # per-layer DNN parameter gate (resolves task seesaw)
│   └── PEPNet              # full model: embedding → EPNet → PPNet → T task towers
├── test_pepnet.py      # pytest test suite (39 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 pepnet.py
```

Expected output:
```
logits:   torch.Size([4, 3])  [...]
loss:     0.6511
backward: OK
params:   264,739

Gate NU γ = 2.0 → δ ∈ [0, 2.0]
PPNet gate dims: [80, 64, 32]  (3 tasks each)
Task towers: 3 towers × 3 layers each

EPNet: personalizes embedding per domain (resolves domain seesaw)
PPNet: personalizes DNN hidden units per task (resolves task seesaw)
```

### Test suite

```bash
python3 -m pytest test_pepnet.py -v
```

Expected: `39 passed`.
