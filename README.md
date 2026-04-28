# AD Ranking

Ad ranking research repository implementing generative recommendation and discriminative ranking approaches.

---

## Generative Ads Recommendation with Semantic ID

To implement the generative core of a system like GEM, ads are transformed into a "vocabulary" that a transformer model can predict using Residual Quantized Semantic IDs.

### 1. Mathematical Framework (RQ-KMeans)

Instead of a random ID, an ad is represented as a sequence of discrete tokens:

```
[c1, c2, ..., ck]
```

#### 1.1 Hierarchical Decomposition
An ad embedding E is decomposed into a series of centroids.

#### 1.2 Residual Learning
- **c1**: Represents the coarse category
- **c2**: Represents the detail within that category by quantizing the residual (the difference between E and the first centroid)

**Objective**: Ensures that ads with similar IDs are semantically similar, allowing the model to predict relevant ads by predicting the next ID token.

### 2. Python Implementation: Semantic ID Generator

This script transforms high-dimensional ad embeddings (e.g., from vision or text models) into 3-level semantic sequences.

#### Usage

```bash
python3 semantic_id.py
```

#### Example Output

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

### 3. Using the IDs in a Generative Model

Once sequences are generated, the recommendation task becomes a Language Modeling problem.

- **Input**: User history tokens + Context tokens
- **Output**: Next-token prediction for the Ad ID sequence
- **Inference**: Use Beam Search to find the top-N most probable sequences, then map back to actual Ads in the database

### 4. Key Implementation Steps

#### 4.1 Warm-up
Train the quantization (KMeans) on your entire ad corpus once.

#### 4.2 Fine-tuning
Train a Transformer (like a small Llama or GPT) where the "vocabulary" is the union of all k cluster indices.

#### 4.3 Cross-Entropy Loss
Optimize the model to predict the ground-truth semantic ID of ads that users previously converted on.

### 5. Representative Algorithms

#### 5.1 RQ-KMeans
**Residual Quantization K-Means** — Algorithm-driven approach (clustering)
- Faster to train
- Stable and decouples tokenizer from recommendation model
- Reference: https://arxiv.org/pdf/2512.24762v1

#### 5.2 RQ-VAE
**Residual-Quantized Variational Autoencoder** — Model-driven approach (learning)
- Often produces lower reconstruction error
- More complex to train with potential training instability
- Reference: https://arxiv.org/pdf/2203.01941


---

## InterFormer

InterFormer is a discriminative ranking module designed to improve Click-Through Rate (CTR) prediction by better modeling how different types of data interact.

**Paper**: https://arxiv.org/pdf/2411.09852

### 1. The Core Problem InterFormer Solves

Traditional models often "summarize" user data too early, losing detail. InterFormer uses three specific "Arches" to keep information intact until the very end:

- **Global Arch**: Processes static "non-sequence" data (e.g., user age, location)
- **Sequence Arch**: Processes "dynamic" behavioral data (e.g., past clicks, searches)
- **Bridging Arch**: Acts as a highway between the two, allowing them to communicate in every layer without losing detail

### 2. Files & Implementation

**Files created:**
- `interformer/interformer.py` — Full model implementation (~430 lines)
- `interformer/test_interformer.py` — Unit tests for each component

### 3. Architecture (Section 4 of Paper)

| Component | Purpose |
|-----------|---------|
| **MaskNet** | Unifies k behavior sequences via self-masking (eq. 6) |
| **CrossArch** | Computes X_sum (non-seq → seq) and S_sum (seq → non-seq) via self-gating (eq. 10–11) |
| **PoolingByMHA** | PMA: summarizes sequence with learnable query tokens (eq. 4) |
| **PFFN** | Personalised FFN: modulates sequence by non-seq context f(X_sum) * S (eq. 8) |
| **SequenceArch** | S^(l+1) = MHA(PFFN(X_sum, S)) (eq. 9) |
| **InteractionArch** | X^(l+1) = MLP(DCNv2([X ‖ S_sum])) with residual (eq. 7) |
| **InterFormerBlock** | One interleaving layer combining all three arches |
| **InterFormer** | Full model: preprocessing → CLS prepend → L blocks → classifier head |

### 4. Key Design Choices

- Dense/sparse non-seq features are embedded to d-dimensional tokens
- CLS token is initialized from X_sum^(1) (first layer's non-seq summary) before feeding to Sequence Arch (Section 4.3)
- DCNv2 is used as the default interaction backbone




