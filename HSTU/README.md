# HSTU

PyTorch reference implementation of **HSTU: Hierarchical Sequential Transduction Units**, from *"Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"* (Meta AI, Feb 2024).

Paper: https://arxiv.org/abs/2402.17152

Deployed at 1.5 trillion parameters, improving online A/B metrics by **12.4%**, at **5.3x–15.2x** faster training/inference than a FlashAttention-2 Transformer at 8192-length sequences, and serving **285x** more complex models at the same inference budget via M-FALCON.

---

## Summary

The paper's central move is to stop treating ranking/retrieval as feature-crossing over a flat bag of DLRM features, and instead treat a user's history — content shown, actions taken — as a single, unified, chronologically ordered sequence: a **Generative Recommender (GR)**. Ranking and retrieval both become instances of the same sequential transduction problem (predict the next token in that stream), which lets the whole system be trained the way language models are: sequentially, autoregressively, with encoder cost amortized across every target in the sequence instead of recomputed per training example.

**HSTU** is the encoder built to make this cheap enough for production-scale, non-stationary, hundred-billion-item vocabularies: it keeps the Transformer's residual, highly-parallel block structure, but replaces softmax self-attention with a **pointwise aggregated attention** (a SiLU nonlinearity over biased logits, with no row-normalization) and fuses attention + the usual feed-forward block into a single SwiGLU-gated block — dropping the linear-layer count outside attention from six to two per layer.

---

## Key Ideas

### Problem: softmax attention is a poor fit for recommendation sequences

1. Softmax normalizes attention weights to sum to 1, which discards *how many* strong signals are present — but for recommendation, the sheer intensity/count of a user's prior engagement with similar items is itself a strong feature, not just their relative ordering.
2. Softmax's normalization is a poor fit for the non-stationary, unbounded, billion-item vocabularies streaming recommenders serve online — new items/vocab entries enter continuously, unlike a language model's ~100K static token vocabulary.
3. A standard Transformer block needs 6 linear layers outside attention itself (Q,K,V,O projections + a 2-layer FFN), which caps how deep the stack can go under a fixed activation-memory budget.

### HSTU's solution

```
U(X), V(X), Q(X), K(X) = Split(SiLU(f1(X)))                      [Eq. 1]
A(X)V(X) = SiLU(Q(X)K(X)^T + rab^(p,t)) . V(X)                    [Eq. 2]
Y(X) = f2(LayerNorm(A(X)V(X)) (elementwise*) U(X))                [Eq. 3]
X_out = X + Y(X)                                                  (residual)
```

| Piece | What it replaces | Why |
|---|---|---|
| **Pointwise aggregated attention** (Eq. 2) | Softmax attention | A pointwise SiLU over (biased, causally-masked) logits keeps engagement-intensity information that softmax normalizes away, and degrades more gracefully under a growing, non-stationary vocabulary |
| **rab^(p,t)** — relative attention bias | Absolute/rotary position embeddings | Learned bias over both relative *position* and relative *time* — recency of an action matters for recommendation the way it doesn't for a token's position in a sentence |
| **SwiGLU-style gate + single output proj** (Eq. 3) | Separate attention-output projection + 2-layer FFN | Fuses feature-interaction and transformation into one block, cutting linear layers outside attention from 6 to 2, which is what lets the encoder go far deeper under the same activation-memory budget |

Two further techniques from the paper are included as reference utilities:

- **Stochastic Length** (§3.2, Eq. 4): a training-time trick that mostly truncates long user histories to a short target length, but occasionally (with computed probability) keeps them full — increasing sparsity/reducing average training cost with negligible quality loss.
- **M-FALCON** (§3.4): a cost-amortized inference algorithm. Because attention is causal, a user's per-layer keys/values only need to be computed once; scoring `m` candidate next-actions against that same history then costs `O(m*n*d)` instead of `O(m*n^2*d)` for re-running the encoder per candidate. This repo implements the caching + microbatching idea in simplified form via `HSTU.score_candidates`.

---

## Architecture

```
Unified sequence of interleaved content/action embeddings: x_0, x_1, ..., x_(n-1)
    ▼
X^(0) ∈ R^(B, n, D)

for l = 1..L:                                                     [one HSTULayer]
    U,Q,K,V = Split(SiLU(f1(X^(l-1))))                            [Eq. 1]
    A = SiLU(QK^T + rab^(p,t)), causally masked (post-activation)  [Eq. 2]
    Y = f2(LayerNorm(A.V) * U)                                     [Eq. 3]
    X^(l) = X^(l-1) + Y

Encoded representations X^(L) feed:
  - Retrieval:  argmax over candidates of <X^(L)_i, candidate_embedding>   (Table 1)
  - Ranking:    RankingHead(X^(L) at the target-appended position)        (Table 1, target-aware)
```

Tokenization of raw heterogeneous DLRM features into this unified sequence (Fig. 2 of the paper) is treated as an input-pipeline concern orthogonal to HSTU itself and is out of scope here — `HSTU` consumes an already-unified `(B, n, D)` sequence, the same way `nn.TransformerEncoder` consumes pre-embedded tokens.

---

## Key Components

### 1. Relative Attention Bias — rab^(p,t)

A learned bias, bucketed logarithmically (à la T5) over relative *position*, and optionally over relative *time* between two events — added to raw attention logits before the pointwise nonlinearity. The time component is what lets HSTU distinguish "liked 2 items a minute apart" from "liked 2 items a year apart," something pure positional bias cannot express.

### 2. Pointwise Aggregated Attention (Eq. 2)

`A(X)V(X) = SiLU(QK^T + rab) . V(X)` — no softmax, no row-sum normalization. Causal masking is applied to the *post-activation* weights (not the logits): masking logits with `-inf` before `SiLU` produces `-inf * sigmoid(-inf) = NaN`, unlike softmax where `-inf` cleanly becomes 0 after normalization. This repo's `HSTULayer.forward` masks `A` directly to sidestep that.

### 3. Gated Pointwise Transformation (Eq. 3)

`Y(X) = f2(LayerNorm(A(X)V(X)) * U(X))` — a SwiGLU-style elementwise gate between the attention output and a projection `U(X)` computed from the same input, followed by one output projection. Combined with Eq. 1's single input projection, this is the entirety of an HSTU layer's linear layers (2, vs. a Transformer block's 6).

### 4. M-FALCON-style candidate scoring (`HSTU.score_candidates`)

Encodes a user's history once via `HSTU.forward(..., return_cache=True)`, caching each layer's keys/values. Any number of candidate "next action" tokens can then be scored against that cached history via `score_candidates`, in microbatches, without re-running the quadratic self-attention over history each time. `test_hstu.py` verifies this produces numerically identical results (to ~1e-5) to naively appending each candidate and doing a full recompute.

### 5. Stochastic Length (`stochastic_length_target`, `apply_stochastic_length`)

```
short_len = max_corpus_len^(alpha/2)
target = seq_len                    if seq_len <= short_len
       = seq_len   w.p. max_corpus_len^alpha / seq_len^2   (rare — keep full history)
       = short_len otherwise                                (common — truncate)
```

---

## Usage

```python
from hstu import HSTU, RankingHead, retrieval_scores

model = HSTU(d_model=64, num_layers=3, num_heads=4, dqk=16, dv=16, use_time_bias=True)
head = RankingHead(d_model=64, hidden_dims=[64], num_tasks=1)

x = torch.randn(B, n, 64)             # unified content/action sequence
timestamps = torch.rand(B, n) * 1e6   # optional, for the relative-time bias
encoded = model(x, timestamps=timestamps)     # (B, n, 64)

# Ranking: read off the representation at the target-appended position.
logits = head(encoded[:, -1, :])

# Retrieval: rank candidates by dot-product similarity to a user's state.
scores = retrieval_scores(encoded[:, -1, :], candidate_embeddings)  # (B, num_candidates)
```

M-FALCON-style candidate scoring:

```python
_, cache = model(history, return_cache=True)              # encode once
candidate_reprs = model.score_candidates(cache, candidate_embeddings, microbatch_size=256)
logits = head(candidate_reprs)
```

Stochastic Length during training:

```python
from hstu import apply_stochastic_length
x = apply_stochastic_length(x, alpha=1.6, max_corpus_len=8192)
```

---

## Files

```
HSTU/
├── hstu.py         # full implementation + smoke test
│   ├── RelativeAttentionBias      # rab^(p,t): relative position + optional relative time bias
│   ├── HSTULayer                  # one Eq. 1-3 block, + forward_incremental for M-FALCON caching
│   ├── HSTU                       # stack of L HSTULayers; forward / score_candidates
│   ├── RankingHead                # small MLP for target-aware multi-task ranking
│   ├── retrieval_scores           # dot-product retrieval scoring
│   ├── stochastic_length_target   # Eq. 4 target-length sampler
│   └── apply_stochastic_length    # subsamples a batch of sequences per Eq. 4
├── test_hstu.py    # pytest test suite (33 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 hstu.py
```

Expected output:

```
encoded:    torch.Size([2, 20, 32])
loss:       0.6921
backward:   OK
params:     16,288

--- M-FALCON-style candidate scoring ---
candidate reprs: torch.Size([6, 32])
backward:        OK

--- Stochastic Length ---
4096 -> 776 tokens (alpha=1.6)
```

### Test suite

```bash
python3 -m pytest test_hstu.py -v
```

Expected output:

```
collected 33 items
...
33 passed in ~1s
```

Useful variants:

```bash
python3 -m pytest test_hstu.py::TestHSTU -v
python3 -m pytest test_hstu.py -x
```
