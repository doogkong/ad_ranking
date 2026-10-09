# AD Ranking

Ad ranking research repository: PyTorch reference implementations of papers on large-scale ranking/retrieval architectures, generative recommenders, semantic-ID tokenization, and foundation-model-to-vertical-model (FM→VM) knowledge transfer for industrial ads/recommendation systems, plus reference designs of autonomous ML-engineering agents.

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
| [`ULTRA-HSTU`](ULTRA-HSTU) | [Bending the Scaling Law Curve](https://arxiv.org/abs/2602.16986) (Meta Recommendation Systems) | Feb 2026 | Bends HSTU's scaling curve via input-sequence merging, linear-complexity Semi-Local Attention (local + global windows), and Attention Truncation (dynamic depth over a truncated recent segment). 5.3x training / 21.4x inference scaling efficiency; +4-8% consumption/engagement online. |
| [`TIGER`](TIGER) | [Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) (Google DeepMind / UW-Madison, NeurIPS 2023) | May 2023 | Assigns items hierarchical Semantic IDs via RQ-VAE, then trains a seq2seq Transformer to autoregressively generate the next item's Semantic ID — enabling cold-start retrieval and tunable diversity. |
| [`semantic_id`](semantic_id) | [RQ-KMeans](https://arxiv.org/pdf/2512.24762v1) / [RQ-VAE](https://arxiv.org/pdf/2203.01941) | Dec 2025 / Mar 2022 | The residual-quantization tokenization technique underlying TIGER/GR2: turns item embeddings into short, hierarchical, discrete Semantic ID sequences. |
| [`onerec`](onerec) | [OneRec](https://arxiv.org/abs/2502.18965) (KuaiShou) | Feb 2025 | Unifies retrieval and ranking into one generative recommender with preference alignment (RLHF-style) on top. |
| [`PinRec`](PinRec) | [PinRec](https://arxiv.org/abs/2504.10507) (Pinterest, KDD 2026) | Apr 2025 | Unified generative retrieval model for all Pinterest surfaces via Outcome-Conditioned Generation (steer output toward a surface's target action) plus cross-surface pretrain/fine-tune with impression negatives. Multi-step outcome-conditioned generation needs budget allocation + embedding compression at retrieval time. +4% search saves, +71.4% recall from 16-step generation. |
| [`UniPinRec`](UniPinRec) | [UniPinRec](https://arxiv.org/abs/2606.00422) (Pinterest) | Jun 2026 | Unifies retrieval AND ranking into one PinRec-based model: Masked Action Modeling adds ranking supervision to the non-interleaved retrieval sequence without inflating context length, and cross-stage KV-cache reuse lets ranking decode from retrieval's cached history instead of re-encoding it. +14.8% ranking Hit@3, >3x forward-pass speedup, -11.1% e2e latency. |
| [`multi_embedding_retrieval`](multi_embedding_retrieval) | [Multi-Embedding Retrieval](https://arxiv.org/abs/2506.23060) (Pinterest, KDD 2025) | Jun 2025 | Not generative, but the retrieval-stage counterpart to PinRec: K user embeddings conditioned on interests instead of one two-tower vector. Implicit interests via a Differentiable Clustering Module (validity-aware farthest-point init + single-assignment capsule routing, argmax condition association, sampled-softmax + logQ); explicit followed-topic interests via Conditional Retrieval with logging-time association and a relevance filter; budgeted round-robin serving. Online: +1.09% home-feed repins, +0.81% adopted Pincepts. |
| [`LIGE-GR`](LIGE-GR) | [LIGE-GR](https://arxiv.org/abs/2609.18148) (Meta) | Sep 2026 | Additive, reversible upgrade from itemwise ranking to listwise generative recommendation: a lightweight causal-Transformer context-aware predictor on top of the existing ranker, a continuation-weighted listwise value model, and the Palette beam decoder with closed-form (duration-aware) future value. Recovers the incumbent greedy system exactly when switched off. +1.14% time spent on Instagram Reels, +0.72% on Facebook Video at ~10% extra inference resources. |
| [`OneTrans-V2`](OneTrans-V2) | [OneTrans-V2](https://arxiv.org/abs/2609.28589) (ByteDance) | Sep 2026 | Unifies retrieval, pre-rank and fine-rank into one jointly trained causal Transformer: the behavior sequence is encoded once and shared, each stage adds token-specific tokens under a stage visibility mask, and fine-rank distills into pre-rank in-model. Decision-Conditioned Generative Retrieval (DCGR) predicts a decision prefix before the semantic ID so a business offset steers one model across objectives; sparse MoE + µP scaling; Sequence-Native Training (4.4x). +9.74% GMV, 3.2x QPS. |
| [`cluster_goobs`](cluster_goobs) | [Real-Time Hard Negative Sampling via LLM-based Clustering](https://arxiv.org/abs/2607.00448) (Meta, OARS @ RecSys 2026) | Jul 2026 | Draws extra two-tower training negatives from the positive's own semantic cluster (k-means over LLM content embeddings) via a real-time, cluster-segmented hash pool (update/sample engines, preload, overwrite-on-collision) — no global ANN index. +7% to +56% HR@50 over in-batch on public data; in production +53% source CTR, +6.5% overall CTR, top-100 impression share 50%→32%. The folder includes DNS/CBNS/ANCE/GOOBS baselines; the synthetic demo does not reproduce the headline gain over plain GOOBS (documented in the folder README). |
| [`hybrid_gpu_cpu_retrieval`](hybrid_gpu_cpu_retrieval) | [Hybrid GPU–CPU Retrieval for Personalized Search](https://arxiv.org/abs/2609.21281) (Meta, KDD '27) | Sep 2026 | Resolves the personalization–scale paradox by co-serving two independently selected and versioned pathways behind a dedup/attribution aggregator: a GPU pathway (search-value pool, fused INT8 ANN + DeepFM-style interaction pre-ranking, joint InfoNCE/Smooth-L1/BCE training) and a ~20x larger CPU pathway (dedicated IVF embedding index, eager term-at-a-time evaluation, lightweight personalized two-tower). Full-system A/B: +4.51% DCG@20, +2.01% GSRR; ~4x CPU capacity cost for the accelerator plan. Folder reproduces each component plus the paper's metrics on a synthetic search world; some toy results diverge from the paper (documented). |
| [`facebook_ebr`](facebook_ebr) | [Embedding-based Retrieval in Facebook Search](https://arxiv.org/abs/2006.11632) (Facebook, KDD 2020) | Jun 2020 | Full-stack EBR for personalized social search: a unified two-tower embedding over text + searcher location/social context (triplet loss), online/offline hard negative mining and hard positives, weighted-concatenation and cascade ensembles, Faiss-style IVF/PQ/OPQ ANN tuning, and an `nn` operator inside a Unicorn-style Boolean engine for hybrid retrieval, plus query/index selection and ranking-stage feedback. Unified embedding +18% recall (events) / +16% (groups) over text; online HNM +5-8%. The folder reproduces each piece on a synthetic people-search log; the offline-HNM and 4%-data hard-positive findings do not reproduce at toy scale (documented). |

### LLM-based retrieval, ranking & re-ranking

Using large language models directly in the recommendation pipeline.

| Folder | Paper | Date | Summary |
|---|---|---|---|
| [`llm_ads`](llm_ads) | [LLM Retrieval](https://arxiv.org/pdf/2605.21969) (Meta, SIGIR Workshop AgentSearch 2026) + ranking extension | May 2026 | LLM semantic features for stable/predictable ad retrieval, extended to the ranking stage combined with the Foundation-Expert paradigm. |
| [`CoGR`](CoGR) | [It Takes Two to Match: Co-Evolving Generative Retriever with RL](https://arxiv.org/abs/2609.00638) (Apple + UNC) | Sep 2026 | LLM keyword generators on BOTH query and item sides, matched through an inverted index; SFT init then alternating GRPO against the other side's frozen index, with a counterfactual marginal F1 reward for the item side. +10.9% F1 (internal APP marketplace) and +36.1% (WANDS) over the best baseline. |
| [`deepretrieval`](deepretrieval) | [DeepRetrieval](https://arxiv.org/abs/2503.00223) (UIUC / Korea Univ.) | Mar 2025 | Trains an LLM to rewrite queries (boolean, expanded NL, or SQL) with RL, using the retrieval metric achieved by the real search engine as the reward, with no reference queries: `<think>`/`<answer>` protocol, tiered Recall@K / H@N / NDCG / execution-accuracy rewards, PPO with GAE + KL to the initial model. 3B model hits 65.07% / 63.18% Recall@3K on PubMed / ClinicalTrials.gov (prior SOTA 24.68% / 32.11%) and beats GPT-4o on BIRD. The folder runs the method on a toy retrieval world (no LLM/GPU), with PPO and GRPO variants. |
| [`GR2`](GR2) | [GR2 Technical Report](https://arxiv.org/abs/2606.31984) (Meta AI) | Jun 2026 | Generative Reasoning Re-Ranker: Semantic-ID mid-training, teacher-distilled chain-of-thought reasoning (SFT / On-Policy Distillation), and DAPO-based RL with a de-hacked verifiable reward for LLM-based re-ranking. +18.7% R@1 on industrial traffic. |
| [`AdvertiserPredictor`](AdvertiserPredictor) | [Fine-Tuned LLM as a Complementary Predictor](https://arxiv.org/abs/2605.27856) (Pinterest) | May 2026 | Uses a fine-tuned LLM not as a ranker but as an ads-specific ancillary predictor of likely next advertisers/interests (GRPO-trained, Semantic-ID-enhanced), fed into both retrieval (candidate generation) and ranking (features). +4.94% RoAS online. |

### Foundation-model-to-VM transfer frameworks

Frameworks for transferring a large, separately-trained foundation model's knowledge into the small vertical models that actually serve traffic, under strict latency/streaming-data constraints.

| Folder | Paper | Date | Summary |
|---|---|---|---|
| [`ExFM`](ExFM) | [External Large Foundation Model](https://arxiv.org/abs/2502.17494) (Meta AI) | Feb 2025 | External distillation + a Data Augmentation Service that amortizes FM inference across VMs; an Auxiliary Head (with Gradient/Label Scaling) to reduce cross-domain bias transfer; a Student Adapter to close the FM-VM freshness gap. |
| [`PinFM`](PinFM) | [PinFM](https://arxiv.org/abs/2507.12704) (Pinterest) | Jul 2025 | Pretrains a 20B+ param causal transformer once on cross-application user activity sequences, fine-tuned into each surface's existing ranking model via early-fusion cross-attention. Deduplicated Cross-Attention Transformer (DCAT) caches per-user context K/V once and reuses it across all scored candidates for +600% serving throughput. |
| [`LoopFM`](LoopFM) | [LoopFM](https://arxiv.org/abs/2605.29280) (Meta AI) | May 2026 | Opens a second, high-bandwidth transfer channel beyond scalar KD: materializes the FM's own historical embeddings as a user-keyed input sequence for the VM (Matryoshka-compressed, INT4-quantized), roughly doubling the FM→VM transfer ratio. |
| [`Rec-Distill`](Rec-Distill) | [Rec-Distill](https://arxiv.org/abs/2605.29755) (ByteDance AML) | May 2026 | A decoupled "1-to-N" teacher-student distillation pipeline: a black-box CE distillation loss, a fault-isolated decoupled-tower student, and a sampling-aware cross-debias correction for when teacher/student are sampled differently. Scales teachers to 24B params / 20K-length sequences with >60% transferability. |
| [`sum_user_modeling`](sum_user_modeling) | [Scaling User Modeling](https://arxiv.org/pdf/2311.09544) (Meta Platforms) | Nov 2023 | Large-scale, reusable online user representations shared across many downstream ads-personalization models. |

### Autonomous ML-engineering agents

Systems that automate the *development* of ranking models (hypothesis → experiment → debug → iterate) rather than the model architecture itself. These are agent systems, not neural networks, so the implementations are dependency-free reference designs with a simulated training cluster.

| Folder | Source | Date | Summary |
|---|---|---|---|
| [`REA`](REA) | [Ranking Engineer Agent (REA)](https://engineering.fb.com/2026/03/17/developer-tools/ranking-engineer-agent-rea-autonomous-ai-system-accelerating-meta-ads-ranking-innovation/) (Meta Engineering blog post) | Mar 2026 | Autonomous agent that runs the end-to-end ML-experimentation lifecycle over multi-week workflows: a hibernate-and-wake executor (checkpoint and resume across long training jobs), a dual-source hypothesis engine (historical insights DB + ML research agent), three-phase planning (Validation → Combination → Exploitation) inside an engineer-approved compute budget, and a failure runbook that adapts within guardrails. 2x model accuracy over baseline across six models; 5x engineering output. The post discloses no code, so this is a reference design of the described architecture; see the folder README for what is and isn't specified. |

---

## Getting Started

```bash
git clone https://github.com/doogkong/ad_ranking.git
cd ad_ranking
```

Every implementation is self-contained (PyTorch only, except `semantic_id/` which uses `scikit-learn` and `REA/` which uses only the Python standard library). From any paper's folder:

```bash
cd HSTU                        # or any other folder
python3 hstu.py                # runs a smoke test end-to-end
python3 -m pytest test_hstu.py -v   # runs the full test suite
```

---

## References

Each folder's `README.md` links directly to its paper; see the tables above for the full list.
