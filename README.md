# AD Ranking

Ad ranking research repository: PyTorch reference implementations of papers on large-scale ranking/retrieval architectures, generative recommenders, semantic-ID tokenization, and foundation-model-to-vertical-model (FM→VM) knowledge transfer for industrial ads/recommendation systems.

Every paper implementation lives in its own folder with the same three-part structure: `<name>.py` (the implementation + a runnable smoke test), `test_<name>.py` (a pytest suite), and `README.md` (the paper's key ideas, equations, and usage). See `PROPOSAL.md` for the broader technical proposal this repo supports.

`Date` below is each paper's original arXiv submission month (as encoded in its arXiv ID), not necessarily the month of its latest revision or conference publication.

---

## Contents

### Feature-interaction ranking architectures

Models whose core contribution is *how* to cross/combine heterogeneous ranking features efficiently at scale.

| Folder | Paper | Date | Summary |
|---|---|---|---|
| [`DIN`](DIN) | [Deep Interest Network](https://arxiv.org/abs/1706.06978) (Alibaba, KDD 2018) | Jun 2017 | Introduces a local activation unit that pools a user's behavior history into an ad-conditioned representation (unnormalized attention weights) instead of a fixed, ad-independent vector. Deployed at Alibaba: +10.0% CTR, +3.8% RPM online. |
| [`interformer`](interformer) | [InterFormer](https://arxiv.org/abs/2411.09852) (UIUC + Meta AI) | Nov 2024 | Bidirectional heterogeneous interaction between non-sequential features and behavior sequences via three mutually-reinforcing "arches," incl. a Personalized FFN. Deployed at Meta Ads: +0.15% NE, +24% QPS. |
| [`RankMixer`](RankMixer) | [RankMixer](https://arxiv.org/abs/2507.15551) (ByteDance) | Jul 2025 | Replaces self-attention with parameter-free multi-head token mixing + per-token FFNs (optionally Sparse-MoE) for a hardware-aligned, highly-parallel ranking backbone. Scaled to 1.1B params on Douyin at flat latency. |
| [`tokenmixer_large`](tokenmixer_large) | [TokenMixer-Large](https://arxiv.org/abs/2602.06563) (ByteDance) | Feb 2026 | RankMixer's successor: fixes sub-optimal residuals, adds inter-residual connections + auxiliary loss, and a sparser "enlarge-then-sparsify" MoE for stable scaling to billions of parameters. |
| [`wukong`](wukong) | [Wukong](https://arxiv.org/abs/2403.02545) (Meta AI) | Mar 2024 | Stacks Factorization-Machine and Linear-Compress blocks (following DHEN) to establish a scaling law for feature-interaction models. |
| [`kunlun`](kunlun) | [Kunlun](https://arxiv.org/abs/2602.10016) (Meta Platforms) | Feb 2026 | A unified architecture design establishing scaling laws for massive-scale recommenders; +1.2% NE and 2x scaling efficiency over InterFormer at Meta Ads. |
| [`meta_lattice`](meta_lattice) | [Meta Lattice](https://arxiv.org/abs/2512.09200) (Meta Platforms) | Dec 2025 | Redesigns the ranking model's space for cost-effective scaling; +10% topline NE, +20% capacity savings at Meta Ads. |
| [`foundation_expert`](foundation_expert) | [Foundation-Expert Paradigm](https://arxiv.org/abs/2508.02929) (Meta AI) | Aug 2025 | Splits a hyperscale model into a shared Foundation backbone plus lightweight per-surface Experts, transferring knowledge FM-to-Expert at latency parity. Deployed across Feed, Reels, Groups, etc. |
| [`pepnet`](pepnet) | [PEPNet](https://arxiv.org/abs/2302.01115) (Kuaishou, KDD 2023) | Feb 2023 | Parameter- and Embedding-Personalized gating network that infuses personalized prior information into a shared multi-domain/multi-task backbone. Deployed at Kuaishou (300M DAU). |
| [`onetrans`](onetrans) | [OneTrans](https://arxiv.org/abs/2510.26104) (ByteDance / NTU, WWW 2026) | Oct 2025 | Unifies feature interaction and user-sequence modeling into a single Transformer backbone instead of separate modules. |

### Generative & sequential recommenders

Models that reframe ranking/retrieval as sequence generation over discrete item tokens.

| Folder | Paper | Date | Summary |
|---|---|---|---|
| [`HSTU`](HSTU) | [Actions Speak Louder than Words](https://arxiv.org/abs/2402.17152) (Meta AI) | Feb 2024 | Hierarchical Sequential Transduction Units: replaces softmax attention with pointwise-aggregated attention + relative position/time bias for trillion-parameter Generative Recommenders. Includes Stochastic Length and M-FALCON-style cached candidate scoring. |
| [`TIGER`](TIGER) | [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) (Google DeepMind / UW-Madison, NeurIPS 2023) | May 2023 | Assigns items hierarchical Semantic IDs via RQ-VAE, then trains a seq2seq Transformer to autoregressively generate the next item's Semantic ID — enabling cold-start retrieval and tunable diversity. |
| [`semantic_id`](semantic_id) | [RQ-KMeans](https://arxiv.org/pdf/2512.24762v1) / [RQ-VAE](https://arxiv.org/pdf/2203.01941) | Dec 2025 / Mar 2022 | The residual-quantization tokenization technique underlying TIGER/GR2: turns item embeddings into short, hierarchical, discrete Semantic ID sequences. |
| [`onerec`](onerec) | [OneRec](https://arxiv.org/abs/2502.18965) (KuaiShou) | Feb 2025 | Unifies retrieval and ranking into one generative recommender with preference alignment (RLHF-style) on top. |

### LLM-based retrieval, ranking & re-ranking

Using large language models directly in the recommendation pipeline.

| Folder | Paper | Date | Summary |
|---|---|---|---|
| [`llm_ads`](llm_ads) | [LLM Retrieval](https://arxiv.org/pdf/2605.21969) (Meta, SIGIR Workshop AgentSearch 2026) + ranking extension | May 2026 | LLM semantic features for stable/predictable ad retrieval, extended to the ranking stage combined with the Foundation-Expert paradigm. |
| [`GR2`](GR2) | [GR2 Technical Report](https://arxiv.org/abs/2606.31984) (Meta AI) | Jun 2026 | Generative Reasoning Re-Ranker: Semantic-ID mid-training, teacher-distilled chain-of-thought reasoning (SFT / On-Policy Distillation), and DAPO-based RL with a de-hacked verifiable reward for LLM-based re-ranking. +18.7% R@1 on industrial traffic. |
| [`AdvertiserPredictor`](AdvertiserPredictor) | [Fine-Tuned LLM as a Complementary Predictor](https://arxiv.org/abs/2605.27856) (Pinterest) | May 2026 | Uses a fine-tuned LLM not as a ranker but as an ads-specific ancillary predictor of likely next advertisers/interests (GRPO-trained, Semantic-ID-enhanced), fed into both retrieval (candidate generation) and ranking (features). +4.94% RoAS online. |

### Foundation-model-to-VM transfer frameworks

Frameworks for transferring a large, separately-trained foundation model's knowledge into the small vertical models that actually serve traffic, under strict latency/streaming-data constraints.

| Folder | Paper | Date | Summary |
|---|---|---|---|
| [`ExFM`](ExFM) | [External Large Foundation Model](https://arxiv.org/abs/2502.17494) (Meta AI) | Feb 2025 | External distillation + a Data Augmentation Service that amortizes FM inference across VMs; an Auxiliary Head (with Gradient/Label Scaling) to reduce cross-domain bias transfer; a Student Adapter to close the FM-VM freshness gap. |
| [`LoopFM`](LoopFM) | [LoopFM](https://arxiv.org/abs/2605.29280) (Meta AI) | May 2026 | Opens a second, high-bandwidth transfer channel beyond scalar KD: materializes the FM's own historical embeddings as a user-keyed input sequence for the VM (Matryoshka-compressed, INT4-quantized), roughly doubling the FM→VM transfer ratio. |
| [`Rec-Distill`](Rec-Distill) | [Rec-Distill](https://arxiv.org/abs/2605.29755) (ByteDance AML) | May 2026 | A decoupled "1-to-N" teacher-student distillation pipeline: a black-box CE distillation loss, a fault-isolated decoupled-tower student, and a sampling-aware cross-debias correction for when teacher/student are sampled differently. Scales teachers to 24B params / 20K-length sequences with >60% transferability. |
| [`sum_user_modeling`](sum_user_modeling) | [Scaling User Modeling](https://arxiv.org/pdf/2311.09544) (Meta Platforms) | Nov 2023 | Large-scale, reusable online user representations shared across many downstream ads-personalization models. |

---

## Getting Started

```bash
git clone https://github.com/doogkong/ad_ranking.git
cd ad_ranking
```

Every implementation is self-contained (PyTorch only, except `semantic_id/` which uses `scikit-learn`). From any paper's folder:

```bash
cd HSTU                        # or any other folder
python3 hstu.py                # runs a smoke test end-to-end
python3 -m pytest test_hstu.py -v   # runs the full test suite
```

---

## References

Each folder's `README.md` links directly to its paper; see the tables above for the full list.
