# ad_ranking
ad ranking related repo

## Generative ads recommendation with Semantic ID 

To implement the generative core of a system like GEM, one need to transform your ads into a "vocabulary" that a transformer model can predict. This is done via Residual Quantized Semantic IDs.

1. The Mathematical Framework (RQ-KMeans)
Instead of a random ID, an ad is represented as a sequence of discrete tokens

[c1, c2, ..., ck]

1.1 Hierarchical Decomposition: An ad embedding E is decomposed into a series of centroids.

1.2  Residual Learning: The first token c1 represents the coarse category. The second token c2 represents the detail within that category by quantizing the residual (the difference between  E and the first centroid).

Objective: This ensures that ads with similar IDs are semantically similar, allowing the model to "hallucinate" relevant ads by predicting the next ID token.

2. Python Implementation: Semantic ID Generator

This script demonstrates how to take high-dimensional ad embeddings (e.g., from a vision or text model) and convert them into a 3-level semantic sequence.


% python3 semantic_id.py

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

3. Using the IDs in a Generative Model

Once you have these sequences, the recommendation task becomes a Language Modeling problem.

Input: User history tokens + Context tokens.
Output: Next-token prediction for the Ad ID sequence.
Inference: Use Beam Search to find the top-𝑁 most probable sequences. These sequences are then mapped back to actual Ads in the database for delivery.


4  Key Implementation Steps

4.1 Warm-up: Train the quantization (KMeans) on your entire ad corpus once.

4.2 Fine-tuning: Train a Transformer (like a small Llama or GPT) where the "vocabulary" is the union of all 𝑘 cluster indices.

4.3 Cross-Entropy Loss: Optimize the model to predict the ground-truth semantic ID of ads that users previously converted on.

5 two representative algorithms:

5.1 RQ-KMeans (Residual Quantization K-Mean) is algorithm-driven (clustering), faster to train, stable, and decouples the tokenizer from the recommendation model.  Reference: https://arxiv.org/pdf/2512.24762v1

5.2 RQ-VAE (Residual-Quantized Variational Autoencoder) is model-driven (learning), often produces lower reconstruction error, but is more complex to train and can suffer from training instability. https://arxiv.org/pdf/2203.01941


## Interformer 

InterFormer (https://arxiv.org/pdf/2411.09852) is not a "generative" model in the sense of creating new ad content or generating ad IDs like an LLM. Instead, it is a discriminative ranking module designed to improve Click-Through Rate (CTR) prediction by better modeling how different types of data interact. 

1. The Core Problem InterFormer Solves

Traditional models often "summarize" user data too early, losing detail. InterFormer uses three specific "Arches" to keep information intact until the very end:

- Global Arch: Processes static "non-sequence" data like user age or location.

- Sequence Arch: Processes "dynamic" behavioral data like past clicks or searches.

- Bridging Arch: Acts as a highway between the two, allowing them to "talk" to each other in every layer without losing detail. 


Files created:
  - ranking/web/python/interformer/interformer.py — full model (~430 lines)
  - ranking/web/python/interformer/test_interformer.py — tests for each component

  Architecture implemented (Section 4 of the paper):

  ┌──────────────────┬──────────────────────────────────────────────────────────────────────────────────────┐
  │    Component     │                                     What it does                                     │
  ├──────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ MaskNet          │ Unifies k behavior sequences via self-masking (eq. 6)                                │
  ├──────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ CrossArch        │ Computes X_sum (non-seq → seq) and S_sum (seq → non-seq) via self-gating (eq. 10–11) │
  ├──────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ PoolingByMHA     │ PMA: summarizes sequence with learnable query tokens (eq. 4)                         │
  ├──────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ PFFN             │ Personalised FFN: modulates sequence by non-seq context f(X_sum) * S (eq. 8)         │
  ├──────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ SequenceArch     │ S^(l+1) = MHA(PFFN(X_sum, S)) (eq. 9)                                                │
  ├──────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ InteractionArch  │ X^(l+1) = MLP(DCNv2([X ‖ S_sum])) with residual (eq. 7)                              │
  ├──────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ InterFormerBlock │ One interleaving layer combining all three arches                                    │
  ├──────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ InterFormer      │ Full model: preprocessing → CLS prepend → L blocks → classifier head                 │
  └──────────────────┴──────────────────────────────────────────────────────────────────────────────────────┘

  Key design choices: dense/sparse non-seq features are embedded to d-dim tokens; the CLS token is initialized from X_sum^(1) (first layer's non-seq summary) before being fed to the Sequence Arch (Section 4.3); DCNv2 is used as the default interaction backbone.

