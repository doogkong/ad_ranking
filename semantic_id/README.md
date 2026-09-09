# Semantic ID

Reference implementation of **Residual-Quantized Semantic IDs** for ads/recommendation: converting high-dimensional item embeddings into short, hierarchical, discrete token sequences that a generative model can predict directly, instead of relying on random atomic item IDs.

References:
- **RQ-KMeans**: https://arxiv.org/pdf/2512.24762v1
- **RQ-VAE**: https://arxiv.org/pdf/2203.01941

This technique underlies the tokenization stage of several generative-retrieval systems implemented elsewhere in this repo — see [`../TIGER`](../TIGER) (RQ-VAE Semantic IDs for generative retrieval) and [`../GR2`](../GR2) (an RQ-KMeans Semantic-ID tokenizer for LLM-based re-ranking).

---

## Summary

Instead of assigning each ad/item a random, semantically meaningless integer ID, an item's embedding `E` (e.g. from a vision or text encoder) is decomposed into a short sequence of discrete tokens `[c1, c2, ..., ck]` via **residual quantization**: cluster the embedding at level 1 to get a coarse token `c1`, then quantize the *residual* (`E` minus its assigned centroid) at level 2 to get a finer token `c2`, and so on. Because later tokens only need to describe what earlier, coarser tokens didn't, items with similar Semantic IDs end up being genuinely similar — sharing a prefix means sharing a coarse category, sharing the full sequence means being near-duplicates. This turns "predict the next ad" into an ordinary next-token-prediction problem over a small, structured vocabulary.

Two ways to learn the quantizer:

| Approach | Style | Tradeoff |
|---|---|---|
| **RQ-KMeans** (`semantic_id.py`) | Algorithm-driven (iterative clustering) | Faster to train, stable, decouples the tokenizer from the recommendation model |
| **RQ-VAE** | Model-driven (learned encoder/decoder + quantizer) | Often lower reconstruction error, but more complex to train and can be less stable |

`semantic_id.py` implements the RQ-KMeans variant: at each of `k` levels, `sklearn.cluster.KMeans` clusters the current residual, the cluster index becomes that level's token, and the centroid is subtracted out before moving to the next level.

---

## Using Semantic IDs in a Generative Model

Once every item has a Semantic ID, sequential recommendation becomes a language-modeling problem:

- **Input**: user history tokens + context tokens (the Semantic IDs of previously interacted items)
- **Output**: next-token prediction for the target item's Semantic ID sequence
- **Inference**: beam search over the token sequence to get the top-N most probable IDs, then map each back to a real item

Typical training loop:

1. **Warm-up** — fit the quantizer (KMeans, one level at a time) on the full item corpus once.
2. **Fine-tune** — train a Transformer (e.g. a small GPT/Llama-style decoder) whose vocabulary is the union of all levels' cluster indices.
3. **Optimize** — plain cross-entropy loss against the ground-truth Semantic ID of items users actually interacted with/converted on.

---

## Usage

```bash
python3 semantic_id.py
```

Requires `numpy` and `scikit-learn` (`pip install numpy scikit-learn`) — this is the one implementation in the repo built on classic clustering rather than PyTorch.

```python
from semantic_id import generate_semantic_ids
import numpy as np

ad_embeddings = np.random.randn(1000, 512)      # 1000 ads, 512-dim embeddings
tokens = generate_semantic_ids(ad_embeddings, codebook_sizes=[32, 32, 32])
# tokens: (1000, 3) — one 3-level Semantic ID per ad
```

### Expected output

```
Ad 0 Semantic ID: [ 7 23  3]
Ad 0 Semantic ID: [10 23  9]
Ad 0 Semantic ID: [10  3  3]
Ad 0 Semantic ID: [ 7 18  3]
Ad 0 Semantic ID: [ 4 23  9]
Ad 0 Semantic ID: [22 18  3]
Ad 0 Semantic ID: [ 5 18 26]
Ad 0 Semantic ID: [20 18  4]
Ad 0 Semantic ID: [22 24  9]
Ad 0 Semantic ID: [15 18  8]
```

---

## Files

```
semantic_id/
├── semantic_id.py   # RQ-KMeans Semantic ID generator
└── README.md
```

For a PyTorch, RQ-VAE-based (learned, gradient-trained) alternative with a straight-through quantizer, K-means codebook initialization, and collision handling, see [`../TIGER/tiger.py`](../TIGER/tiger.py)'s `RQVAE`/`ResidualQuantizer`.
