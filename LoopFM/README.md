# LoopFM

PyTorch reference implementation of the **LoopFM** pipeline, from *"LoopFM: Learning frOm HistOrical RePresentations of Foundation Model for Recommendation"* (Meta AI, Jun 2026).

Paper: https://arxiv.org/abs/2605.29280

On industrial-scale systems (billions of examples, trillion-parameter FMs), LoopFM roughly **doubles** the FM→VM knowledge transfer ratio on top of scalar KD, delivering **+0.5%** conversion in the first deployment half and **+1.03%**/**+1.22%** from two subsequent launches. On public benchmarks it adds **6%+ AUC** on TaobaoAd on top of KD.

---

## Summary

Like GR2, LoopFM is not a single encoder architecture — it's a **modular transfer pipeline** for the standard industrial two-tier setup: a large foundation model (FM, up to trillions of parameters) trained offline on rich cross-domain data, and compact vertical models (VMs) that actually serve predictions under strict latency budgets. The existing bridge between the two is scalar knowledge distillation (KD): the FM's prediction becomes a soft label for the VM. LoopFM's core observation is that this is a **bandwidth bottleneck** — a single scalar can't convey the FM's rich cross-domain features, multi-level interaction patterns, and contextual signals, and the gap only widens as FMs scale.

LoopFM opens a second, high-bandwidth channel by **materializing the FM's own historical intermediate embeddings as structured input features** for the VM — concretely, a user-keyed temporal sequence of the FM's past representations for that user. Because only *historical* (not current-sample) embeddings are used, the VM can consume this at serving time **without any real-time FM inference or architectural coupling** to the FM.

---

## Key Ideas

### The bandwidth bottleneck behind declining KD transfer ratio

As FMs scale to multi-trillion parameters, the *transfer ratio* (TR = ΔNE_VM / ΔNE_FM: the fraction of an FM's own improvement that shows up as VM improvement) keeps deteriorating. The paper's hypothesis: a single scalar prediction is a fixed-bandwidth channel, and the gap between what a growing FM learns and what one number can convey only gets wider.

### LoopFM's four-stage pipeline (Figure 1)

| Stage | What it does | This repo |
|---|---|---|
| 1. Extraction | Concatenate a chosen subset of the FM's layer activations into one raw embedding per example | `extract_embedding` |
| 2. Compression | Compress the raw embedding with an autoencoder (stop-gradient into the FM), Tanh-bounded for INT4 quantization, Matryoshka-style so any prefix is a valid standalone representation | `MatryoshkaAutoencoder`, `quantize_int4` |
| 3. Structuring | Group compressed historical embeddings by a key (e.g. user ID) into a temporal sequence, **excluding the current example** | `build_user_sequence`, `group_by_key` |
| 4. VM serving | The VM pools the sequence (mean/sum/attention) and concatenates it with its other features; trained with task loss + scalar KD | `MeanPoolSequenceEncoder`, `SumPoolSequenceEncoder`, `AttentionPoolSequenceEncoder`, `loopfm_training_loss` |

Because the sequence is built entirely from **historical** embeddings (excluding the sample currently being served), the VM never needs to call the FM at inference time — the transfer channel is entirely offline.

### Why this repo doesn't implement Section 4's theory directly

Section 4 proves LoopFM's information gain decomposes into temporal-history, cross-feature, and compression-loss mutual-information terms, with a transfer-ratio lower bound that grows with the FM-VM feature gap. This is a **population-level bound** (Bayes risk, mutual information, under an NTK/benign-overfitting regime) that explains *why* the pipeline works — it isn't itself an algorithm. What the paper's own experiments actually compute and report is the *empirical* transfer ratio from measured NE values, which this repo implements directly: `normalized_entropy` and `transfer_ratio`.

---

## Key Components

### 1. Extraction (`extract_embedding`)

Given a subset of K selected FM layers, concatenate their per-example activations into one raw embedding `e ∈ R^D`. The paper notes shallower layers tend to transfer better than deep ones (an information-bottleneck effect: deeper layers discard more raw input detail), while the embedding layer alone (no learned interactions) transfers slightly worse than the first hidden layer — the sweet spot balances information richness against depth of learned interaction.

### 2. Compression — Matryoshka Autoencoder (`MatryoshkaAutoencoder`, Eq. 1)

```
z = f_enc(e),  e_hat = f_dec(z),  L_AE = ||e - e_hat||^2
```

A stop-gradient (`stop_gradient`) ensures the autoencoder's loss never updates the FM backbone. The encoder's last layer is `Tanh`, bounding `z` to `[-1, 1]` — exactly what makes `quantize_int4`'s 16-level scheme (`round(z*8).clamp(-8,7)/8`, a 4x storage reduction from FP32) low-loss. Trained Matryoshka-style (Kusupati et al., 2022): the shared decoder is also asked to reconstruct `e` from every zero-padded *prefix* `z[:, :d']`, so any prefix of the full embedding is itself directly usable — letting deployment trade off feature width against storage/latency without retraining.

### 3. Structuring (`build_user_sequence`, Eq. 2)

```
S_k = [z_{k,t_L}, ..., z_{k,t_2}, z_{k,t_1}],   t_L < ... < t_2 < t_1 < t_cur
```

Historical compressed embeddings for one key, truncated to the `L` most recent entries strictly before the current example's timestamp `t_cur` — the current sample is always excluded, which is what removes the need for real-time FM inference at serving.

### 4. VM-side sequence encoders (Table 3)

Three pooling strategies from the paper's ablation, all of which beat scalar-KD-only: `MeanPoolSequenceEncoder`, `SumPoolSequenceEncoder` (surprisingly strong — even a single sum-pooled feature captures much of LoopFM's gain at far lower cost than a full sequence architecture), and `AttentionPoolSequenceEncoder` (a DIN-style target-aware activation unit, the paper's strongest aggregator on TaobaoAd).

### 5. Combined training objective (`loopfm_training_loss`)

```
L = L_task(y_hat_V, y) + lambda * L_KD(y_hat_V, y_hat_F)
```

LoopFM's structured-embedding channel and the existing scalar-KD channel are shown to be largely complementary (Table 2) — production combines both.

### 6. Evaluation metrics (`normalized_entropy`, `transfer_ratio`)

`NE` (Normalized Entropy, He et al. 2014) is log-loss normalized by the entropy of the empirical label rate. `TR = ΔNE_VM / ΔNE_FM` is the fraction of an FM's own improvement that shows up in the VM once transferred — the paper's central metric for whether a transfer channel is bandwidth-limited.

---

## Usage

```python
from loopfm import (
    extract_embedding, stop_gradient, MatryoshkaAutoencoder, quantize_int4,
    build_user_sequence, group_by_key, AttentionPoolSequenceEncoder,
    loopfm_training_loss, normalized_entropy, transfer_ratio,
)

# Stage 1-2: extract + compress
e = extract_embedding([fm_layer1_act, fm_layer2_act])          # (B, D)
ae = MatryoshkaAutoencoder(raw_dim=e.shape[-1], latent_dim=32, prefix_dims=[8, 16, 32])
z, recons = ae(stop_gradient(e))
ae_loss = ae.mrae_loss(e.detach(), recons)
z_stored = quantize_int4(z.detach())                            # INT4 for storage

# Stage 3: structure into per-user sequences (done once, offline, per key)
grouped = group_by_key(historical_records)                      # {user_id: [(t, z), ...]}
seq = build_user_sequence(grouped[user_id], t_cur=now, max_len=200)

# Stage 4: VM-side pooling + combined loss
encoder = AttentionPoolSequenceEncoder(d_model=32)
pooled = encoder(seq_batch, mask, query=candidate_features)
loss = loopfm_training_loss(vm_logits, fm_logits, labels, kd_weight=5.0)
```

---

## Files

```
LoopFM/
├── loopfm.py         # full implementation + smoke test
│   ├── extract_embedding             # Stage 1
│   ├── stop_gradient                 # Stage 2 helper
│   ├── MatryoshkaAutoencoder         # Stage 2 (Eq. 1)
│   ├── quantize_int4                 # Stage 2
│   ├── build_user_sequence           # Stage 3 (Eq. 2)
│   ├── group_by_key                  # Stage 3 helper
│   ├── MeanPoolSequenceEncoder       # Stage 4
│   ├── SumPoolSequenceEncoder        # Stage 4
│   ├── AttentionPoolSequenceEncoder  # Stage 4
│   ├── loopfm_training_loss          # Stage 4 combined objective
│   ├── normalized_entropy            # Sec 5.1 metric
│   └── transfer_ratio                # Sec 5.1 metric
├── test_loopfm.py    # pytest test suite (41 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 loopfm.py
```

Expected output (abridged):

```
--- Stage 1: Extraction ---
raw embedding e: torch.Size([4, 56])

--- Stage 2: Matryoshka Autoencoder + INT4 quantization ---
z: torch.Size([4, 32]), prefixes reconstructed: [8, 16, 32], L_MRAE: 2.7846
quantized z range: [-0.875, 0.875], unique levels: 15

--- Stage 3: Structuring ---
user sequence S_u: torch.Size([10, 32])  (most recent 10 of 20 historical entries)
grouped keys: ['u1', 'u2'], u1 history length: 2

--- Stage 4: VM-side sequence encoders + combined loss ---
mean-pool: torch.Size([4, 32]), sum-pool: torch.Size([4, 32]), attn-pool: torch.Size([4, 32])
loopfm_training_loss: 4.1279, backward OK

--- Evaluation metrics ---
NE before: 1.0653, NE after: 0.7185
transfer_ratio: 6.9363
```

### Test suite

```bash
python3 -m pytest test_loopfm.py -v
```

Expected output:

```
collected 41 items
...
41 passed in ~1s
```

Useful variants:

```bash
python3 -m pytest test_loopfm.py::TestMatryoshkaAutoencoder -v
python3 -m pytest test_loopfm.py -x
```
