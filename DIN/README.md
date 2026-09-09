# DIN

PyTorch reference implementation of **DIN: Deep Interest Network for Click-Through Rate Prediction** (Zhou et al., Alibaba, KDD 2018).

Paper: https://arxiv.org/abs/1706.06978

Deployed on Alibaba's online display advertising system serving the main traffic: **+10.0% CTR** and **+3.8% RPM** over the prior production base model in a month-long online A/B test; offline, **+6.08% RelaImpr** over the base model on Alibaba's 2.14-billion-sample dataset.

---

## Summary

Prior Embedding&MLP CTR models compress a user's entire behavior history into one fixed-length vector — via sum or average pooling — before that vector ever interacts with the candidate ad. This is a real bottleneck: user interests are *diverse* (a young mother's history might span clothing, electronics, and toys), but only a small, ad-dependent slice of that history is actually relevant to any one candidate ad. Naively expanding the pooled vector's dimension to capture more of that diversity blows up parameter count and overfitting risk.

DIN's fix is architecturally small but effective: a **local activation unit** computes, for each historical behavior, a relevance weight conditioned on the *specific candidate ad* being scored, and pools the behavior sequence with those weights instead of pooling uniformly. The result is a user representation that **varies per candidate ad** — the same user gets a different "distillation" of their history depending on what's being shown to them — without needing to expand the embedding dimension at all. Two training techniques (mini-batch aware regularization, the Dice activation) make this practical on industrial-scale sparse features.

---

## Key Ideas

### The local activation unit (Eq. 3) — DIN's core contribution

```
v_U(A) = sum_{j=1}^{H} a(e_j, v_A) * e_j
```

For each historical behavior embedding `e_j`, a small feed-forward network `a(.)` scores its relevance to the candidate ad embedding `v_A`, taking as input the concatenation of `e_j`, `v_A`, and their element-wise product (an explicit interaction term the paper adds specifically to help relevance modeling). The pooled user representation is then the weighted sum of behaviors — not a plain average.

**Critically, these weights are *not* softmax-normalized to sum to 1** — this is a deliberate departure from standard attention. The paper treats the (unconstrained) sum of the weights as an approximation of the overall *intensity* of a user's activated interests: a user whose history is 90% clothing and 10% electronics should activate the clothing-relevant weights more strongly in absolute terms than a user with a 50/50 split, information a normalized attention distribution would throw away.

### Dice: a data-adaptive activation function (Eq. 8-9)

PReLU switches between two linear regimes at a *fixed* rectification point (0), which may not suit every layer's actual input distribution. Dice replaces that hard, fixed threshold with a smooth, data-adaptive one centered on the running mean of each layer's input — computed per mini-batch during training (exactly like BatchNorm's statistics) and via a moving average at inference:

```
f(s) = p(s)*s + (1 - p(s))*alpha*s,   p(s) = sigmoid((s - E[s]) / sqrt(Var[s] + eps))
```

When `E[s] = Var[s] = 0`, Dice degenerates exactly into PReLU — `Dice.forward` with the running stats manually zeroed reproduces PReLU's output bit-for-bit (see `test_degenerates_to_prelu_at_zero_stats`).

### Mini-batch aware regularization (Eq. 4-7)

With embedding tables scaling to hundreds of millions of rows, computing a standard L2 penalty over the *entire* table every mini-batch step is computationally prohibitive — most rows weren't even touched by that batch. The paper's fix: only regularize the rows for features that actually appear in the current mini-batch, each scaled by `1/n_j` (that feature's total occurrence count across the whole training set) rather than the mini-batch size — so infrequent features still get regularized proportionally to how rarely they're seen overall, while the per-step computation only ever touches a mini-batch's worth of rows.

---

## Key Components

| Component | Paper section |
|---|---|
| `Dice` | Eq. 8-9 — data-adaptive activation |
| `sum_pooling`, `average_pooling` | Eq. 1 — the base model's uniform, ad-independent pooling |
| `LocalActivationUnit`, `activation_weighted_pooling` | Eq. 3 — DIN's ad-conditioned, unnormalized weighted pooling |
| `mini_batch_aware_l2_penalty` | Eq. 4-7 |
| `DINModel` | Fig. 2 — the full network; `use_local_activation=False` gives the base model, `True` gives DIN, sharing every other structure |
| `pairwise_auc`, `user_weighted_auc`, `rel_impr` | Eq. 10-11 — the paper's own evaluation metrics |

`DINModel` is deliberately a single class with a toggle, mirroring how the paper itself frames DIN as a targeted replacement of *only* the base model's pooling layer. `test_din.py` verifies the resulting behavioral difference directly: `test_din_representation_varies_with_candidate_ad` shows DIN's pooled representation actually changes across candidate ads, while `test_base_model_representation_is_ad_independent` shows the base model's does not — the paper's central empirical claim, made mechanically checkable.

---

## Usage

```python
from din import DINModel
import torch.nn.functional as F

model = DINModel(
    profile_cardinalities=[2, 10],     # e.g. gender, age_level
    behavior_vocab_size=10_000_000,    # shared "goods" embedding table (behaviors + candidate ad)
    context_cardinalities=[10],        # e.g. pid
    embed_dim=18,
    mlp_hidden_dims=[200, 80],
    use_local_activation=True,         # DIN; set False for the base model
    activation="dice",                 # or "prelu"
)

logits, activation_weights = model(profile_ids, behavior_ids, behavior_mask, ad_id, context_ids)
loss = F.binary_cross_entropy_with_logits(logits, labels)
loss.backward()
```

Mini-batch aware regularization during training:

```python
from din import mini_batch_aware_l2_penalty

penalty = mini_batch_aware_l2_penalty(
    model.goods_embedding.weight, behavior_ids.reshape(-1), feature_occurrence_counts, lam=0.01
)
(loss + penalty).backward()
```

Evaluation:

```python
from din import user_weighted_auc, rel_impr

auc = user_weighted_auc(user_ids, scores, labels)
lift = rel_impr(auc, auc_base_model)  # e.g. 0.0608 -> "+6.08% RelaImpr"
```

---

## Files

```
DIN/
├── din.py         # full implementation + smoke test
│   ├── Dice                          # Eq. 8-9
│   ├── sum_pooling, average_pooling  # Eq. 1
│   ├── LocalActivationUnit           # Eq. 3
│   ├── activation_weighted_pooling   # Eq. 3
│   ├── mini_batch_aware_l2_penalty   # Eq. 4-7
│   ├── MLPTower                      # top FC tower (Fig. 2)
│   ├── DINModel                      # full network: base model or DIN via one flag
│   ├── pairwise_auc, user_weighted_auc  # Eq. 10
│   └── rel_impr                      # Eq. 11
├── test_din.py    # pytest test suite (38 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 din.py
```

Expected output (abridged):

```
DIN logits: torch.Size([8]), loss: 0.6992, backward OK
activation weights (unnormalized) sum per example: [-0.226, 2.455, 0.699, -0.301]

--- DIN varies its user representation per candidate ad ---
weights differ across candidate ads: True

--- Base model (uniform pooling) for comparison ---
base model logits: torch.Size([8]), weights sum to 1 per example: [1.0, 1.0, 1.0, 1.0]

--- Mini-batch aware regularization (Eq. 4-7) ---
mini-batch aware L2 penalty: 1.5792

--- Evaluation metrics (Eq. 10-11) ---
user-weighted AUC: 0.4375, RelaImpr vs base_model=0.6: -1.6250
```

### Test suite

```bash
python3 -m pytest test_din.py -v
```

Expected output:

```
collected 38 items
...
38 passed in ~1s
```

Useful variants:

```bash
python3 -m pytest test_din.py::TestDINModel -v
python3 -m pytest test_din.py -x
```
