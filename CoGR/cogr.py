"""CoGR: Co-Evolving Generative Retriever (Dai et al., Apple / UNC, 2026).

Reference implementation of the algorithmic core of the paper
(https://arxiv.org/abs/2609.00638):

  * keyword-set retrieval through an inverted index        (Sec. 2.1)
  * BM25 ranking of the retrieved items                    (Sec. 2.1)
  * SFT target construction                                (Alg. 1)
  * query-side reward  R_q  = F1 with a keyword budget     (Eq. 2.1-2.2)
  * item-side counterfactual marginal reward R_i           (Eq. 2.3)
  * its O(|affected queries|) incremental computation      (Appendix B.3)
  * GRPO group-normalised advantages + clipped loss        (Sec. 2.3)
  * the alternating co-evolving loop                       (Alg. 2)

The paper's generators are Qwen3 LLMs trained with verl. Here the generator is
a pluggable `KeywordGenerator` interface; `ToyKeywordGenerator` is a tiny
torch policy (bag-of-tokens encoder -> per-keyword Bernoulli) so the whole
pipeline runs on CPU in seconds. Everything that is *not* the LLM (retrieval,
rewards, GRPO math, SFT targets, the loop) follows the paper.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Keywords = Sequence[str]


# ---------------------------------------------------------------------------
# Retrieval: inverted index over generated keywords (Sec. 2.1)
# ---------------------------------------------------------------------------

def representation(entity: str, keywords: Iterable[str]) -> Set[str]:
    """Matching representation of an entity: S ∪ {entity} (the raw string
    itself always participates in matching, per I_ret definition)."""
    return set(keywords) | {entity}


class InvertedIndex:
    """keyword -> set of entity ids. Entity ids are the raw strings (query /
    item title), which also act as their own always-present keyword."""

    def __init__(self, keyword_sets: Dict[str, Keywords]):
        self.reps: Dict[str, Set[str]] = {
            e: representation(e, kws) for e, kws in keyword_sets.items()
        }
        self.postings: Dict[str, Set[str]] = defaultdict(set)
        for e, rep in self.reps.items():
            for kw in rep:
                self.postings[kw].add(e)

    def lookup(self, rep: Iterable[str]) -> Set[str]:
        """All entities whose representation overlaps `rep`."""
        out: Set[str] = set()
        for kw in rep:
            out |= self.postings.get(kw, set())
        return out

    def retrieve(self, entity: str, keywords: Keywords) -> Set[str]:
        return self.lookup(representation(entity, keywords))

    def __len__(self) -> int:
        return len(self.reps)

    @property
    def vocab_size(self) -> int:
        return len(self.postings)


def bm25_rank(
    query_rep: Iterable[str],
    candidates: Iterable[str],
    item_index: InvertedIndex,
    k1: float = 1.2,
    b: float = 0.75,
) -> List[str]:
    """Rank retrieved items by BM25, treating each side's keyword set as a bag
    of words (Sec. 2.1). Term frequency is 0/1 since reps are sets."""
    n_docs = len(item_index)
    avg_len = sum(len(r) for r in item_index.reps.values()) / max(n_docs, 1)
    q_terms = set(query_rep)
    scores: Dict[str, float] = {}
    for item in candidates:
        rep = item_index.reps[item]
        norm = k1 * (1 - b + b * len(rep) / max(avg_len, 1e-9))
        s = 0.0
        for t in q_terms & rep:
            df = len(item_index.postings[t])
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            s += idf * (1 * (k1 + 1)) / (1 + norm)
        scores[item] = s
    return sorted(scores, key=lambda i: (-scores[i], i))


# ---------------------------------------------------------------------------
# Metrics and query-side reward (Eq. 2.1, 2.2)
# ---------------------------------------------------------------------------

def precision_recall_f1(retrieved: Set[str], relevant: Set[str]) -> Tuple[float, float, float]:
    tp = len(retrieved & relevant)
    p = tp / len(retrieved) if retrieved else 0.0
    r = tp / len(relevant) if relevant else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def f1_from_counts(n_tp: int, n_ret: int, n_rel: int) -> float:
    """F1 = 2*tp / (|ret| + |rel|)  (the identity used in Appendix B.3)."""
    denom = n_ret + n_rel
    return 2.0 * n_tp / denom if denom > 0 else 0.0


def query_reward(
    keywords: Keywords, retrieved: Set[str], relevant: Set[str], k_max: int
) -> float:
    """Eq. 2.2: F1 of the retrieved set, 0 if the keyword budget is exceeded."""
    if len(set(keywords)) > k_max:
        return 0.0
    return precision_recall_f1(retrieved, relevant)[2]


def mrr_at_k(ranked: Sequence[str], relevant: Set[str], k: int = 100) -> float:
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Set[str], k: int = 100) -> float:
    dcg = sum(1.0 / math.log2(r + 1) for r, it in enumerate(ranked[:k], start=1) if it in relevant)
    ideal = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate(
    queries: Sequence[str],
    query_keywords: Dict[str, Keywords],
    item_index: InvertedIndex,
    rel: Dict[str, Set[str]],
    k: int = 100,
) -> Dict[str, float]:
    """Macro-averaged P/R/F1 over the full retrieved set plus MRR/NDCG@k, with
    BM25 ranking over generated keywords (the paper's evaluation protocol)."""
    agg = Counter()
    for q in queries:
        rep = representation(q, query_keywords[q])
        ret = item_index.lookup(rep)
        p, r, f1 = precision_recall_f1(ret, rel[q])
        ranked = bm25_rank(rep, ret, item_index)
        agg["P"] += p
        agg["R"] += r
        agg["F1"] += f1
        agg[f"MRR@{k}"] += mrr_at_k(ranked, rel[q], k)
        agg[f"NDCG@{k}"] += ndcg_at_k(ranked, rel[q], k)
    n = max(len(queries), 1)
    return {m: v / n for m, v in agg.items()}


# ---------------------------------------------------------------------------
# Item-side counterfactual reward (Eq. 2.3 + Appendix B.3)
# ---------------------------------------------------------------------------

def item_reward_naive(
    item: str,
    cand_keywords: Keywords,
    queries: Sequence[str],
    query_index: InvertedIndex,
    item_keywords: Dict[str, Keywords],
    rel: Dict[str, Set[str]],
    k_max: int,
) -> float:
    """Eq. 2.3 computed literally: rebuild the item index with only `item`'s
    keywords replaced, re-run retrieval for *every* query, sum F1 differences.
    Slow; used as the oracle for `ItemRewardCache`."""
    if len(set(cand_keywords)) > k_max:
        return -1.0
    ref_index = InvertedIndex(item_keywords)
    cand_kws = dict(item_keywords)
    cand_kws[item] = list(cand_keywords)
    cand_index = InvertedIndex(cand_kws)
    r_new = r_old = 0.0
    for q in queries:
        rep = query_index.reps[q]
        r_old += precision_recall_f1(ref_index.lookup(rep), rel[q])[2]
        r_new += precision_recall_f1(cand_index.lookup(rep), rel[q])[2]
    return r_new - r_old


class ItemRewardCache:
    """Appendix B.3: constant-per-affected-query item reward.

    Built once per item-side round from the frozen query index and the
    reference item index. Caches per-query (n_ret, n_tp, n_rel) and, per item,
    the set of queries that retrieve it under its reference keywords. A
    candidate keyword set then costs one lookup in the query index plus O(1)
    F1 updates for the symmetric difference of the two query sets.
    """

    def __init__(
        self,
        queries: Sequence[str],
        query_index: InvertedIndex,
        item_index: InvertedIndex,
        rel: Dict[str, Set[str]],
        k_max: int,
    ):
        self.query_index = query_index
        self.rel = rel
        self.k_max = k_max
        self.n_ret: Dict[str, int] = {}
        self.n_tp: Dict[str, int] = {}
        self.n_rel: Dict[str, int] = {}
        self.q_ref: Dict[str, Set[str]] = defaultdict(set)  # item -> queries retrieving it
        for q in queries:
            ret = item_index.lookup(query_index.reps[q])
            self.n_ret[q] = len(ret)
            self.n_tp[q] = len(ret & rel[q])
            self.n_rel[q] = len(rel[q])
            for it in ret:
                self.q_ref[it].add(q)

    def reward(self, item: str, cand_keywords: Keywords) -> float:
        if len(set(cand_keywords)) > self.k_max:
            return -1.0
        q_cand = self.query_index.lookup(representation(item, cand_keywords))
        q_ref = self.q_ref.get(item, set())
        total = 0.0
        for q in q_cand - q_ref:  # item newly retrieved
            hit = 1 if item in self.rel[q] else 0
            new = f1_from_counts(self.n_tp[q] + hit, self.n_ret[q] + 1, self.n_rel[q])
            old = f1_from_counts(self.n_tp[q], self.n_ret[q], self.n_rel[q])
            total += new - old
        for q in q_ref - q_cand:  # item no longer retrieved
            hit = 1 if item in self.rel[q] else 0
            new = f1_from_counts(self.n_tp[q] - hit, self.n_ret[q] - 1, self.n_rel[q])
            old = f1_from_counts(self.n_tp[q], self.n_ret[q], self.n_rel[q])
            total += new - old
        return total


# ---------------------------------------------------------------------------
# Phase 1: SFT target construction (Alg. 1)
# ---------------------------------------------------------------------------

def build_sft_query_targets(
    queries: Sequence[str],
    item_keywords: Dict[str, Keywords],
    rel: Dict[str, Set[str]],
    top_n: int,
) -> Dict[str, List[str]]:
    """Alg. 1 lines 4-7: pool (multiset union) the initial keywords of every
    relevant item, keep the top-N most frequent as the query's target."""
    targets: Dict[str, List[str]] = {}
    for q in queries:
        bag: Counter = Counter()
        for it in rel[q]:
            bag.update(set(item_keywords[it]))
        # deterministic tie-break on the keyword string
        ranked = sorted(bag.items(), key=lambda kv: (-kv[1], kv[0]))
        targets[q] = [kw for kw, _ in ranked[:top_n]]
    return targets


# ---------------------------------------------------------------------------
# GRPO (Sec. 2.3)
# ---------------------------------------------------------------------------

def grpo_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Group-relative advantages. rewards: [G, n] (n rollouts per prompt);
    normalised within each group: (r - mean) / (std + eps)."""
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, unbiased=False, keepdim=True)
    return (rewards - mean) / (std + eps)


def grpo_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """PPO-style clipped surrogate with group advantages (no KL term, matching
    `use_kl_loss=False` in Table 6)."""
    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

class KeywordGenerator:
    """Interface that an LLM wrapper would implement."""

    def sample(self, texts: Sequence[str], n: int) -> Tuple[List[List[List[str]]], torch.Tensor]:
        """Return (keyword_sets[len(texts)][n], logp[len(texts), n])."""
        raise NotImplementedError

    def log_prob(self, texts: Sequence[str], keyword_sets: List[List[List[str]]]) -> torch.Tensor:
        raise NotImplementedError

    def generate(self, texts: Sequence[str]) -> Dict[str, List[str]]:
        """Deterministic (greedy) keyword sets used to build indexes."""
        raise NotImplementedError


class ToyKeywordGenerator(KeywordGenerator, nn.Module):
    """Stand-in for the LLM: hashed bag-of-tokens encoder -> independent
    Bernoulli over a closed keyword vocabulary. A sampled set is the set of
    "on" keywords, so |S| is naturally capped by `k_max` at reward time.
    """

    def __init__(self, keyword_vocab: Sequence[str], dim: int = 32, n_buckets: int = 2048, seed: int = 0):
        nn.Module.__init__(self)
        g = torch.Generator().manual_seed(seed)
        self.vocab = list(keyword_vocab)
        self.kw_id = {k: i for i, k in enumerate(self.vocab)}
        self.n_buckets = n_buckets
        self.emb = nn.EmbeddingBag(n_buckets, dim, mode="mean")
        self.out = nn.Linear(dim, len(self.vocab))
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn(n_buckets, dim, generator=g) * 0.1)
            self.out.weight.copy_(torch.randn(len(self.vocab), dim, generator=g) * 0.1)
            self.out.bias.fill_(-2.0)  # sparse keyword sets at init

    def _encode(self, texts: Sequence[str]) -> torch.Tensor:
        ids, offsets = [], []
        for t in texts:
            offsets.append(len(ids))
            toks = t.lower().split() or ["<empty>"]
            ids.extend(self._hash(tok) for tok in toks)
        return self.emb(torch.tensor(ids), torch.tensor(offsets))

    def _hash(self, tok: str) -> int:
        h = 0
        for ch in tok:  # stable across processes (python's hash() is salted)
            h = (h * 131 + ord(ch)) % 1_000_003
        return h % self.n_buckets

    def logits(self, texts: Sequence[str]) -> torch.Tensor:
        return self.out(self._encode(texts))  # [B, V]

    def _multi_hot(self, keyword_sets: List[List[List[str]]]) -> torch.Tensor:
        B, n = len(keyword_sets), len(keyword_sets[0])
        m = torch.zeros(B, n, len(self.vocab))
        for b in range(B):
            for j in range(n):
                for kw in keyword_sets[b][j]:
                    if kw in self.kw_id:
                        m[b, j, self.kw_id[kw]] = 1.0
        return m

    @torch.no_grad()
    def sample(self, texts, n):
        probs = torch.sigmoid(self.logits(texts))  # [B, V]
        draws = torch.bernoulli(probs.unsqueeze(1).expand(-1, n, -1))  # [B, n, V]
        sets = [[[self.vocab[v] for v in draws[b, j].nonzero().flatten().tolist()] for j in range(n)]
                for b in range(len(texts))]
        return sets, self.log_prob(texts, sets)

    def log_prob(self, texts, keyword_sets):
        logits = self.logits(texts).unsqueeze(1)  # [B, 1, V]
        y = self._multi_hot(keyword_sets)  # [B, n, V]
        ll = -F.binary_cross_entropy_with_logits(logits.expand_as(y), y, reduction="none")
        return ll.sum(-1)  # [B, n]

    @torch.no_grad()
    def generate(self, texts, k_max: Optional[int] = None, threshold: float = 0.5):
        probs = torch.sigmoid(self.logits(texts))
        out = {}
        for t, p in zip(texts, probs):
            idx = (p > threshold).nonzero().flatten().tolist()
            idx.sort(key=lambda i: -p[i].item())
            if k_max is not None:
                idx = idx[:k_max]
            out[t] = [self.vocab[i] for i in idx]
        return out


def sft_step(gen: ToyKeywordGenerator, optim, texts: Sequence[str], targets: Sequence[Keywords]) -> float:
    """One SFT step: maximise log-likelihood of the target keyword set."""
    y = gen._multi_hot([[list(t)] for t in targets]).squeeze(1)
    loss = F.binary_cross_entropy_with_logits(gen.logits(texts), y, reduction="none").sum(-1).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()


def sft_train(gen, texts, targets, epochs=50, lr=0.05, batch_size=64) -> float:
    optim = torch.optim.Adam(gen.parameters(), lr=lr)
    loss = float("nan")
    idx = list(range(len(texts)))
    for _ in range(epochs):
        random.shuffle(idx)
        for s in range(0, len(idx), batch_size):
            b = idx[s : s + batch_size]
            loss = sft_step(gen, optim, [texts[i] for i in b], [targets[i] for i in b])
    return loss


# ---------------------------------------------------------------------------
# Phase 2: co-evolving RL (Alg. 2)
# ---------------------------------------------------------------------------

def grpo_train(
    gen: ToyKeywordGenerator,
    texts: Sequence[str],
    reward_fn,  # (text, keyword_list) -> float
    epochs: int,
    n_rollouts: int = 8,
    batch_size: int = 64,
    lr: float = 0.02,
    clip_eps: float = 0.2,
) -> float:
    """Fully-online GRPO: fresh rollouts from the current policy each step
    (rollout.n = 8 as in Table 6), advantages normalised within each prompt's
    group. Returns the mean reward of the last epoch."""
    optim = torch.optim.Adam(gen.parameters(), lr=lr)
    idx = list(range(len(texts)))
    last = 0.0
    for _ in range(epochs):
        random.shuffle(idx)
        ep_rewards = []
        for s in range(0, len(idx), batch_size):
            batch = [texts[i] for i in idx[s : s + batch_size]]
            sets, old_logp = gen.sample(batch, n_rollouts)
            rewards = torch.tensor(
                [[reward_fn(t, sets[b][j]) for j in range(n_rollouts)] for b, t in enumerate(batch)]
            )
            adv = grpo_advantages(rewards)
            logp = gen.log_prob(batch, sets)
            loss = grpo_loss(logp, old_logp, adv, clip_eps)
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_rewards.append(rewards.mean().item())
        last = sum(ep_rewards) / len(ep_rewards)
    return last


def co_evolve(
    query_gen: ToyKeywordGenerator,
    item_gen: ToyKeywordGenerator,
    train_queries: Sequence[str],
    items: Sequence[str],
    rel: Dict[str, Set[str]],
    rounds: int = 5,
    k_max: int = 30,
    query_epochs: int = 10,
    item_epochs: int = 5,
    n_rollouts: int = 8,
    log=None,
) -> List[Dict[str, float]]:
    """Alg. 2. Each round: (1) train the query generator against the frozen
    item index with R_q; (2) rebuild the query index; (3) train the item
    generator against the frozen query index with the counterfactual marginal
    reward R_i; (4) rebuild the item index."""
    history = []
    item_kws = item_gen.generate(items, k_max)
    for rnd in range(1, rounds + 1):
        item_index = InvertedIndex(item_kws)

        def q_reward(q, kws, _idx=item_index):
            return query_reward(kws, _idx.retrieve(q, kws), rel[q], k_max)

        q_r = grpo_train(query_gen, train_queries, q_reward, query_epochs, n_rollouts)
        query_kws = query_gen.generate(train_queries, k_max)
        query_index = InvertedIndex(query_kws)

        cache = ItemRewardCache(train_queries, query_index, item_index, rel, k_max)
        i_r = grpo_train(item_gen, items, cache.reward, item_epochs, n_rollouts)
        item_kws = item_gen.generate(items, k_max)

        stats = {"round": rnd, "query_reward": q_r, "item_reward": i_r}
        history.append(stats)
        if log:
            log(stats)
    return history
