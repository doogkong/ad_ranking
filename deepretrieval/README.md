# DeepRetrieval

Reference implementation of **DeepRetrieval: Hacking Real Search Engines and Retrievers with Large Language Models via Reinforcement Learning** (Jiang, Lin, Cao, Tian, Kang, Wang, Sun, Han; UIUC / Korea University; arXiv 2503.00223v3).

Paper: https://arxiv.org/abs/2503.00223 · Official code: https://github.com/pat-jj/DeepRetrieval

> **Read this first — what this is and isn't.** The paper trains a 3B-parameter LLM (Qwen2.5-3B-Instruct) with PPO on real PubMed / ClinicalTrials.gov / BM25 / dense-retriever / SQL backends, using verl + vLLM on A100s. None of that fits in a reference folder. This folder implements the **method** — output protocol, retrieval-metric rewards (the exact tiered tables from the paper), the PPO/GAE/KL machinery, and a GRPO variant — against a **toy retrieval world** (synthetic corpus, boolean+BM25 engine, tiny GRU policy) so the whole loop runs on a laptop CPU in ~30 s. Toy numbers show the *mechanism* learns; they say nothing about the paper's results.

---

## Summary

### Problem

A user's query is an imperfect expression of their information need. LLM-based query augmentation helps, but prior methods train it with **supervised learning or distillation** on reference queries (human-written, or distilled from GPT-4-class models). That is expensive, caps quality at the teacher, and optimises *imitation* of a reference query instead of *retrieval performance*.

### Idea

Treat query generation as an RL problem and use the **retrieval metric itself as the reward**, with trial and error against the real search backend. No reference queries at all.

* **State** — the user's query *q*. **Action** — the generated query *q′*. **Reward** — the metric achieved when *q′* is sent to the retriever (Recall@K, H@N, NDCG@K, SQL execution accuracy).
* **Structured output** (borrowed from DeepSeek-R1): the model writes reasoning in `<think>…</think>`, then the final query in `<answer>…</answer>`.
* **Algorithm-agnostic** — they use PPO (GAE, separate critic, KL to the initial model); GRPO etc. also fit.
* **Task-agnostic** — switching the task means switching the reward function and the output format (boolean query / expanded natural language / SQL).

### Objective and reward

```
r(q, q')  =  r_retrieval(q, q')  +  r_format(q')                          (Eq. 1)

π* = argmax_π  E_{q~ρ, q'~π(·|q)} [ r(q,q')  −  β · log π(q'|q) / π_ref(q'|q) ]   (Eq. 2)

L_PPO(θ) = L_CLIP(θ) − c1·L_VF(θ) + c2·S[π_θ]                            (Eq. 3)
L_CLIP   = E_t[ min( r_t(θ)·A_t , clip(r_t(θ), 1−ε, 1+ε)·A_t ) ]         (Eq. 4)
```

Retrieval rewards are **tiered** (Table 8 of the paper), so partial progress is rewarded and a total miss is penalised:

| Task | Metric | Reward |
|---|---|---|
| Literature search | Recall@K (K=3000 for PubMed / ClinicalTrials.gov) | 5.0 if ≥0.7 · 4.0 if ≥0.5 · 3.0 if ≥0.4 · 1.0 if ≥0.3 · 0.5 if ≥0.1 · 0.1 if ≥0.05 · **−3.5** otherwise |
| Evidence-seeking (NQ, TriviaQA, SQuAD, BM25) | H@N: rank of first doc containing an answer span | 5.0 if ≤5 · 4.0 if ≤20 · 2.0 if ≤50 · 1.0 if ≤100 · 0.5 if ≤1000 · 0.1 if ≤3000 · **−3.5** otherwise |
| Sparse / dense retrieval (BEIR, MS-MARCO subsets) | NDCG@10 | the NDCG value itself |
| SQL search (BIRD, Spider) | Execution accuracy | the accuracy; BIRD adds **+0.3** if the SQL merely executes (no syntax error / missing table) to get RL off the ground |

For NDCG-based rewards the paper picks a larger k (3000) than at evaluation time, to avoid sparse zero-reward signals.

### Training setup (Appendix B)

Qwen2.5-3B-Instruct as **both actor and critic** (also Llama-3.2-3B for literature search; Qwen2.5-Coder-3B/7B for SQL). Actor lr 1e-6, critic lr 1e-5, KL coef 0.001, temperature 0.6, batch 64, PPO mini-batch 16; max response 500/350/512/512 tokens for literature / evidence-seeking / classic IR / SQL. verl (HybridFlow) + vLLM + Ray on 2×A100-80GB (4× for the 7B coder).

### Results (as reported)

| Task | Result |
|---|---|
| Literature search (real PubMed / ClinicalTrials.gov), Recall@3K | **65.07%** / **63.18%** vs. previous SOTA LEADS 24.68% / 32.11% (LEADS is SFT on GPT-4o-distilled, human-annotated queries); GPT-4o 17.59 / 16.25, Sonnet-3.5 20.94 / 18.33; original query 10.36 / 18.01 |
| Evidence-seeking, BM25, H@5 | NQ 57.5 / TriviaQA 73.2 / SQuAD 59.4 — on par with GPT-4o / Claude-3.5-Sonnet on NQ and TriviaQA and ahead on SQuAD (paper Table 1) |
| Classic IR, NDCG@10 | Best on 7/8 datasets with BM25 and 6/8 with dense retrievers (E5, BGE, Contriever). Gains largest for retrievers not trained on the target data (e.g. ≈+5 on MS-S with vanilla Contriever); small where the retriever saw the training split (BGE on FEVER: 82.5 → 84.1) |
| SQL, execution accuracy | Coder-3B: BIRD 49.02 / Spider 74.85 vs. Qwen2.5-Coder-3B zero-shot 30.77 / 50.97, i.e. +18.25 / +23.88 points; 7B reaches 56.00 / 76.01. RL from scratch beats SFT on GPT-4o-distilled SQL |
| Efficiency | BM25 + DeepRetrieval ≈ matches or beats dense + DeepRetrieval on several sets, and BM25 retrieval was **~34× faster** (352 s vs 12,232 s over 5.42M docs / 13,332 queries) |

### Findings worth remembering

1. **RL > SFT/distillation** for retrieval, even with a 3B model. RL optimises for what the engine rewards, which can differ from what a human or GPT-4o writes (e.g. DeepRetrieval learns boolean structures with operator grouping that human experts would not write).
2. **Reasoning is an exploration aid, not the solution.** Removing `<think>` hurt literature search; the model without it let queries balloon in length (Fig. 4b). Unlike R1's "aha moment", think length *shrank* as training proceeded: once a good query strategy was found, reasoning was less needed.
3. **Adaptive knowledge injection.** The model learned to inject answer-like knowledge into queries when it helps (41.5% of TriviaQA queries, 22.1% NQ) and rarely when it doesn't (4.6% SQuAD, where gains come from fitting the corpus distribution).
4. **Retriever-agnostic**; helps most for un-adapted retrievers, least for ones already near-optimal on the dataset.
5. **Cold-start SFT helps when the base model lacks a skill** (Qwen2.5-3B-Base on SQL) but is not needed for coder models.

### Limits (stated or implied)

Each retriever/dataset needs its own RL run and a reward that can be computed (ground-truth documents, answer spans, or gold SQL). Zero-reward starts are a real risk (hence tiered rewards, large reward-k, and the SQL executability bonus). Real search APIs make rollout slow; "no-think" runs were 8× slower in decoding and were stopped early.

---

## What is implemented

| Component | Paper | Code |
|---|---|---|
| `<think>/<answer>` protocol, JSON `{"query": ...}` answers | §2.2, App. H | `parse_response`, `extract_query` |
| Search backend: boolean `AND/OR/()` + BM25 ranking, top-K | stands in for PubMed / BM25 | `parse_boolean_query`, `BooleanBM25Engine` |
| Metrics | App. F | `recall_at_k`, `first_answer_rank` (H@N), `ndcg_at_k`, `execution_accuracy` |
| Tiered rewards (exact Table 8) + SQL +0.3 bonus | Table 8, §F | `literature_search_reward`, `evidence_seeking_reward`, `ndcg_reward`, `sql_reward` |
| `r = r_retrieval + r_format` | Eq. 1 | `compose_reward`, `ToyRetrievalTask.reward` |
| Actor + separate critic, KL against frozen reference | §2.2, B.2 | `DeepRetrievalTrainer`, `make_policy`, `make_critic` |
| Token-level KL penalty, GAE (γ=λ=1), advantage whitening | Eq. 2, B.1 | `compute_gae`, `masked_whiten` |
| PPO clipped surrogate, clipped value loss, entropy bonus | Eqs. 3–4 | `ppo_policy_loss`, `ppo_value_loss`, `PPOConfig.entropy_coef` |
| GRPO variant (group-relative advantage, k3 KL loss, no critic) | "algorithm-agnostic" | `grpo_advantages`, `kl_k3`, `PPOConfig(advantage="grpo")` |
| Think/query length monitoring | Fig. 4 | `history[*]["think_len" / "query_len"]` |
| *(ours)* Toy world, GRU policy, format warm-up | — | `ToyIRWorld`, `GRULM`, `pretrain_format` |

### Choices the paper does not pin down (ours)

* **Format reward values**: the paper shows only a check/cross; we use 0 / −1. A malformed answer also gets the −3.5 retrieval floor (it retrieves nothing), so malformed can never beat well-formed-but-wrong.
* **"Instruct model" stand-in**: the toy policy gets a short supervised warm-up on *random* well-formed responses that sometimes copy words from the user query. It learns the protocol and nothing about which terms retrieve well — analogous to starting from Qwen-Instruct. Without this the RL loop never sees a valid answer (zero reward).
* **Truncated responses** (no `<eos>` before the length cap) count as malformed.
* **Toy learning rates** are larger than the paper's 1e-6 / 1e-5 (they suit a 3B pretrained LLM, not a from-scratch GRU); the critic still learns 10× faster than the actor, as in the paper.
* **Model-written SQL** is executed on an in-memory SQLite connection with `PRAGMA query_only=ON`, single `SELECT`/`WITH` statements only.
* Not implemented: the knowledge-injection analysis (App. E, which uses an auxiliary LLM), dense retrievers, real data loaders, distributed rollout.

---

## Usage

```bash
python3 deepretrieval.py                 # warm-up -> 500 PPO steps -> eval (~30 s, CPU)
python3 -m pytest test_deepretrieval.py -v
```

Smoke-test output (seed 1; Recall@20 on a 3-topic, 60-document toy corpus):

```
original query     Recall@20: 0.567      <- what the user typed
before RL (greedy) Recall@20: 0.142      <- knows the format, not the task
after  RL (greedy) Recall@20: 0.717      <- trained only on retrieval reward
```

The learned policy copies the user's cue word and ORs in other words of the same topic — query expansion discovered purely from search-engine feedback. This toy task is fragile in the way small-scale RL usually is: some seeds plateau at a deterministic local optimum (in our sweeps PPO improved on 2 of 3 seeds and GRPO on 3 of 3 at 400 steps, but some runs stall far below the best), which is why the tests use fixed seeds and assert on improvement over the pre-RL policy rather than on absolute numbers.

```python
import random, torch
from deepretrieval import (ToyIRWorld, ToyRetrievalTask, make_policy, pretrain_format,
                           DeepRetrievalTrainer, PPOConfig)

task = ToyRetrievalTask(ToyIRWorld(seed=0))
policy = make_policy(len(task.vocab))
pretrain_format(policy, task)                                # "instruct" stage: format only
trainer = DeepRetrievalTrainer(policy, task, PPOConfig(advantage="gae"))   # or "grpo"
trainer.train(500, log_every=50)
print(trainer.evaluate())
```

### Plugging in a real task

`DeepRetrievalTrainer` needs a `task` with `vocab`, `sample_batch(n, rng)`, `prompt_ids(example)` and `reward(example, response_text) -> RewardInfo`. To target a different retrieval problem, build the reward from the provided pieces — e.g. SQL:

```python
r_ret = sql_reward(conn, extract_query(parsed.answer), gold_sql, executable_bonus=0.3)  # BIRD
total, ret, fmt = compose_reward(r_ret, parsed.well_formed)
```

and for a real LLM replace `GRULM` with a causal LM exposing `forward(ids, h) -> (logits, state)`-style scoring (or port `compute_gae` / `ppo_policy_loss` into verl/TRL, which is what the authors did).

---

## Tests

`test_deepretrieval.py` (108 tests, ~35 s): response/query parsing and malformed cases; boolean grammar (precedence, implicit OR, depth/length limits) and engine semantics; metrics; **every tier boundary of the paper's reward tables**; SQL execution accuracy, the executability bonus and read-only safety; toy-world sanity (oracle expansion ≈ full recall, original query under-specified); generation masks/EOS/greedy determinism, log-prob causality and temperature handling; GAE against hand-computed values (λ=1, λ=0, padding), whitening, GRPO group normalisation; PPO clipping gradients, value clipping, KL-k3; trainer invariants (frozen reference, KL = 0 before any update, critic present only for PPO); and end-to-end checks that **retrieval reward alone** raises reward and recall for both PPO and GRPO.
