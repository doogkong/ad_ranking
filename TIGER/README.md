# TIGER

PyTorch reference implementation of **TIGER: Transformer Index for GEnerative Recommenders**, from *"Recommender Systems with Generative Retrieval"* (Rajput, Mehta, Singh et al., Google DeepMind / UW-Madison, NeurIPS 2023).

Paper: https://arxiv.org/abs/2305.05065

Outperforms all sequential-recommendation baselines across three Amazon Reviews categories — up to **+29% NDCG@5** vs. SASRec on Beauty and **+21% NDCG@5** vs. S³-Rec on Toys and Games — and, unlike prior atomic/random-ID sequence models, natively handles cold-start (never-seen) items and tunable recommendation diversity.

---

## Summary

TIGER (and its follow-ups, referenced throughout this repo — HSTU, GR2, LoopFM all build on or cite it) reframes sequential recommendation as **generative retrieval**: instead of embedding a user and every candidate item into one vector space and running approximate nearest-neighbor search, the model directly *generates* the identifier of the next item, token by token, with a sequence-to-sequence Transformer.

The key enabler is the **Semantic ID**: rather than assigning each item an arbitrary atomic integer, TIGER encodes the item's content (title, description, category, brand — anything a text encoder can embed) and quantizes that embedding into a short tuple of discrete codewords via a **Residual-Quantized VAE (RQ-VAE)**. Because the quantizer is hierarchical (coarse-to-fine, one codebook per level), items with overlapping Semantic-ID prefixes are semantically related by construction — the model shares knowledge across similar items instead of memorizing one embedding per atomic ID, generalizes to items it never saw at training time, and can trade off precision for diversity just by choosing which hierarchy level to sample more loosely.

---

## Key Ideas

### Why not embed-and-search?

Standard dual-encoder retrieval learns a user tower and an item tower into one shared space and does MIPS/ANN search over a candidate index. This requires maintaining that index and scales the item-representation cost linearly with catalog size. TIGER instead uses the Transformer's own parameters as an implicit index (echoing the "Differentiable Search Index" line of work in document retrieval) — there's no separate ANN structure to build or refresh; the model *is* the retrieval mechanism.

### Semantic IDs via residual quantization (Fig. 3)

```
z = Encoder(x)              # x: pretrained content embedding (e.g. Sentence-T5, 768-dim)
r_0 = z
for level d in 0..m-1:
    c_d = argmin_k || r_d - codebook_d[k] ||     # nearest codeword at this level
    r_{d+1} = r_d - codebook_d[c_d]              # residual carried to the next, finer level
Semantic ID := (c_0, ..., c_{m-1})
```

Each level's codebook captures a different granularity: the paper's qualitative study (Fig. 4) shows the first codeword alone recovers coarse product category ("Hair" vs. "Makeup" vs. "Skin"), with later codewords refining within that category — a property that falls directly out of quantizing residuals rather than the raw embedding at every level.

### Generative retrieval (Fig. 2b)

A bidirectional Transformer encoder reads a user's interaction history as a flattened sequence of past items' Semantic ID tokens (plus a hashed user-ID token); an autoregressive decoder then generates the next item's Semantic ID tokens one at a time. Training is plain next-token cross-entropy over this shared, compact vocabulary (`num_levels * codebook_size` item tokens + a fixed block of hashed user tokens) — orders of magnitude smaller than one embedding row per item.

### Two new capabilities this framing unlocks (Sec 4.3)

- **Cold-start retrieval**: a never-seen item still has a well-defined Semantic ID (RQ-VAE only needs its content embedding, not interaction data), so the model can generate and match it even with zero training-time impressions.
- **Tunable diversity**: temperature-based sampling during decoding can be applied at any level of the ID hierarchy — sampling loosely at the *first* token diversifies across coarse categories, while sampling loosely only at later tokens keeps recommendations within one category but varies specifics.

### A consequence the paper studies directly: invalid IDs (Sec 4.5)

Because the decoder generates tokens autoregressively over a space of `codebook_size^num_levels` possible tuples while the real catalog only occupies a tiny fraction of it, the model can generate a well-formed but unassigned Semantic ID. The paper reports this happens for ~0.1–1.6% of top-10 predictions in practice, and suggests prefix matching (falling back to same-category items) as a future mitigation.

---

## Key Components

### 1. RQ-VAE (`RQVAE`, `ResidualQuantizer`, `rqvae_quant_loss`)

Implements Fig. 3's encoder → m-level residual quantizer → decoder exactly, including the straight-through gradient estimator (Van Den Oord et al., 2017) that makes discrete quantization end-to-end trainable, and K-means codebook initialization (`kmeans_init`) to avoid codebook collapse.

### 2. Collision handling (`resolve_collisions`)

Multiple items can land on the same Semantic ID (Sec 3.1). Appends a disambiguating extra token — the k-th colliding item gets token `k` — exactly as described: two items sharing `(12, 24, 52)` become `(12, 24, 52, 0)` and `(12, 24, 52, 1)`.

### 3. Vocabulary (`SemanticIDVocab`, `stable_hash`)

Builds the shared token space: special tokens, then one token per `(level, codeword)` pair, then a fixed-size block of hashed user-ID tokens (the Hashing Trick, Weinberger et al., 2009) — the paper uses 2000 user buckets to keep vocabulary size bounded regardless of how many actual users exist.

### 4. Seq2seq Transformer + decoding (`TigerTransformer`, `generate_semantic_id`)

A standard T5-style bidirectional-encoder / causal-decoder Transformer (the paper's contribution here is the *framing*, not the attention mechanism, so this reuses `nn.TransformerEncoder`/`nn.TransformerDecoder`). `generate_semantic_id` decodes greedily (`temperature=0`) or samples (`temperature>0`, restricted to the current level's sub-vocabulary) — directly exercising the diversity-via-decoding capability of Sec 4.3.

### 5. Retrieval + diversity metric (`SemanticIDLookup`, `category_entropy`)

A tuple → item lookup table (misses return `None`, i.e. an invalid ID), and the paper's Entropy@K diversity metric over retrieved items' ground-truth categories.

### What's not implemented

Beam search (used in the paper to guarantee K valid IDs by exploring multiple decoding paths) and prefix-matching fallback for invalid IDs are mentioned by the paper as extensions/future work and are not implemented here — `generate_semantic_id` decodes greedily or via temperature sampling only.

---

## Usage

```python
from tiger import RQVAE, resolve_collisions, SemanticIDVocab, TigerTransformer, generate_semantic_id, SemanticIDLookup

# Stage 1: learn Semantic IDs from item content embeddings
rqvae = RQVAE(input_dim=768, hidden_dims=[512, 256, 128], latent_dim=32, num_levels=3, codebook_size=256)
rqvae.init_codebooks_kmeans(content_embeddings)
for _ in range(num_steps):
    losses, _ = rqvae.loss(content_embeddings)
    losses["total"].backward(); ...

codes = resolve_collisions(rqvae.encode_codes(content_embeddings))  # (N, 4): unique per item

# Stage 2: train the seq2seq generative retriever
vocab = SemanticIDVocab(num_levels=4, codebook_size=256, num_user_buckets=2000)
model = TigerTransformer(vocab_size=vocab.vocab_size, d_model=128, n_head=6,
                         num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=1024)
encoder_ids = torch.tensor([vocab.build_encoder_input(user_id, item_history_codes)])
logits = model(encoder_ids, decoder_input_ids)
loss = F.cross_entropy(logits.reshape(-1, vocab.vocab_size), decoder_target_ids.reshape(-1))

# Retrieval
generated = generate_semantic_id(model, encoder_ids, vocab, temperature=0.0)
item_id = SemanticIDLookup(item_ids, codes).retrieve(generated)
```

---

## Files

```
TIGER/
├── tiger.py         # full implementation + smoke test
│   ├── ResidualQuantizer   # m-level residual VQ with straight-through gradient + k-means init
│   ├── rqvae_quant_loss    # codebook + commitment loss
│   ├── RQVAE               # encoder + ResidualQuantizer + decoder
│   ├── resolve_collisions  # disambiguates items sharing a Semantic ID
│   ├── stable_hash         # deterministic feature hashing for user-ID tokens
│   ├── SemanticIDVocab     # shared item/user token vocabulary
│   ├── TigerTransformer    # T5-style encoder-decoder over that vocabulary
│   ├── generate_semantic_id  # autoregressive (greedy/temperature) decoding
│   ├── SemanticIDLookup    # tuple -> item id, or None (invalid ID)
│   └── category_entropy   # Entropy@K diversity metric
├── test_tiger.py    # pytest test suite (39 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 tiger.py
```

Expected output (abridged):

```
--- Stage 1: RQ-VAE Semantic ID generation ---
final RQ-VAE loss: recon=0.6561, rqvae=0.1691
codes: torch.Size([200, 3]), collisions resolved: 9, unique after fix: 200/200

--- Stage 2: Generative retrieval ---
vocab_size: 117
seq2seq logits: torch.Size([1, 4, 117]), ce loss: 4.7727, backward OK
greedy-decoded Semantic ID: [[14, 8, 8, 0]]
retrieved item id: [None]

--- Diversity metric ---
category_entropy: 1.4591 bits
```

(An untrained model's greedy decode landing on `None` is expected — it's the exact "invalid ID" phenomenon Sec 4.5 studies.)

### Test suite

```bash
python3 -m pytest test_tiger.py -v
```

Expected output:

```
collected 39 items
...
39 passed in ~1s
```

Useful variants:

```bash
python3 -m pytest test_tiger.py::TestRQVAE -v
python3 -m pytest test_tiger.py -x
```
