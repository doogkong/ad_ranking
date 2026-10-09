"""Hybrid GPU-CPU retrieval for personalized search at ultra-large scale.

Reference implementation of "Hybrid GPU-CPU Retrieval for Personalized Search at
Ultra-Large Scale" (Fu, Sun, Zhu, Liu, Shi, Lu, ... Kong; Meta Platforms;
KDD '27; arXiv 2609.21281).

The personalization-scale paradox: deep, interaction-heavy scoring wants the
inventory in GPU memory (HBM), but HBM cannot hold the full online inventory;
CPUs hold it cheaply but cannot run the interaction model within the latency
budget. The paper resolves this by co-serving two independently selected,
independently versioned pathways that meet at an aggregator:

  * GPU pathway (depth, ~1B docs): two-tower ANN retrieval + DeepFM-style
    interaction pre-ranking fused on the accelerator; INT8 distance kernel;
    pool selected for *search value* by a lifecycle-aware filter (Sec. 4).
  * CPU pathway (breadth, ~20x larger inventory): dedicated embedding index
    (IVF over centroids), eager nearest-neighbour evaluation (term-at-a-time
    scan + prefetch, then other rules), bulk hit scanning, and a lightweight
    personalized unified two-tower model (Sec. 5).
  * Co-serving (Sec. 3): parallel branches with their own deadlines, dedup +
    source attribution at the aggregator, matched model-index snapshots,
    independent publish / disable / rollback.

This file implements every one of those pieces at laptop scale, plus the
paper's evaluation machinery (DCG@K, GSRR, conservative cross-day interval,
overlap/composition analysis, capacity-plan arithmetic), and a synthetic
personalized-search world to exercise them. Not implemented: XLM-V, real ANN
hardware kernels (AVX-512 / MI300X), lexical retrieval, the downstream ranker
(a noisy oracle stands in), query understanding. Synthetic numbers do not
reflect production results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Evaluation metrics (Sec. 6.1, App. A.2, A.5)
# ---------------------------------------------------------------------------

def dcg_at_k(gains: Sequence[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(list(gains)[:k]))


def ndcg_at_k(ranked_gains: Sequence[float], ideal_gains: Sequence[float], k: int) -> float:
    ideal = dcg_at_k(sorted(ideal_gains, reverse=True), k)
    return dcg_at_k(ranked_gains, k) / ideal if ideal > 0 else 0.0


def gsrr(sessions: Sequence[Sequence[Tuple[str, float]]], thresholds: Dict[str, float]) -> float:
    """Good Search Result Rate (Eqs. 3-4): fraction of eligible sessions containing at least one
    event whose strength v(e) reaches its event-type threshold tau_type(e) (a sufficiently long
    view or an explicit positive action). Each session is a list of (event_type, strength)."""
    if not sessions:
        return 0.0
    good = sum(any(v >= thresholds[t] for t, v in events) for events in sessions)
    return good / len(sessions)


def relative_lift(treatment: float, control: float) -> float:
    """100 * (mu_t / mu_c - 1), in percent."""
    return 100.0 * (treatment / control - 1.0)


def conservative_interval(daily_lifts: Sequence[float], daily_lo: Sequence[float],
                          daily_hi: Sequence[float]) -> Tuple[float, float, float]:
    """The paper's cross-day-dependence-robust interval (App. A.5).

    Persistent assignment makes daily estimates dependent, so they are not combined by sqrt(n).
    By Cauchy-Schwarz, the half-width of the mean is at most the mean of the daily half-widths
    (exact under perfect correlation). Each day's half-width is the LARGER side of its interval.
    Returns (mean lift, lower, upper).
    """
    lifts = np.asarray(daily_lifts, dtype=float)
    half = np.maximum(np.asarray(daily_hi) - lifts, lifts - np.asarray(daily_lo)).mean()
    m = float(lifts.mean())
    return m, m - float(half), m + float(half)


def overlap_metrics(gpu_set: Set[int], cpu_set: Set[int]) -> Dict[str, float]:
    """App. A.2 Eqs. 5-6: GPU-side |G∩C|/|G|, CPU-side |G∩C|/|C|, Jaccard."""
    inter = len(gpu_set & cpu_set)
    return {"gpu_side": inter / len(gpu_set) if gpu_set else 0.0,
            "cpu_side": inter / len(cpu_set) if cpu_set else 0.0,
            "jaccard": inter / len(gpu_set | cpu_set) if (gpu_set or cpu_set) else 0.0}


def source_composition(by_source: Dict[str, Set[int]]) -> Dict[str, int]:
    """Table 8: exclusive per-source counts plus the cross-group overlap."""
    counts: Dict[str, int] = {}
    owners: Dict[int, List[str]] = {}
    for src, ids in by_source.items():
        for i in ids:
            owners.setdefault(i, []).append(src)
    for src in by_source:
        counts[src] = sum(1 for o in owners.values() if o == [src])
    counts["overlap"] = sum(1 for o in owners.values() if len(o) > 1)
    counts["total"] = len(owners)
    return counts


# ---------------------------------------------------------------------------
# 2. Capacity-plan arithmetic (Sec. 6.5, App. A.4, Table 6)
# ---------------------------------------------------------------------------

@dataclass
class HardwareUnit:
    vector_capacity: float     # vectors one unit can hold
    qps: float                 # sustained queries/s per unit
    cost: float = 1.0          # unit-capacity cost (relative)


def capacity_plan(n_vectors: float, qps: float, unit: HardwareUnit, regions: int = 3) -> Dict[str, float]:
    """Units needed per region = max(storage-bound, throughput-bound); the plan replicates every region."""
    storage_units = math.ceil(n_vectors / unit.vector_capacity)
    throughput_units = math.ceil(qps / unit.qps)
    per_region = max(storage_units, throughput_units)
    return {"storage_units": storage_units, "throughput_units": throughput_units,
            "units_per_region": per_region, "total_units": per_region * regions,
            "cost": per_region * regions * unit.cost,
            "bound": "storage" if storage_units >= throughput_units else "throughput"}


def compare_plans(n_vectors: float, qps: float, cpu: HardwareUnit, accel: HardwareUnit,
                  regions: int = 3) -> Dict[str, float]:
    """Normalise to the CPU plan = 1.0 (as Table 6 does)."""
    c, a = capacity_plan(n_vectors, qps, cpu, regions), capacity_plan(n_vectors, qps, accel, regions)
    return {"cpu_units_per_region": 1.0, "accel_units_per_region": a["units_per_region"] / c["units_per_region"],
            "unit_capacity_cost": accel.cost / cpu.cost,
            "accel_plan_cost": a["cost"] / c["cost"], "accel_bound": a["bound"], "cpu_bound": c["bound"]}


# ---------------------------------------------------------------------------
# 3. Synthetic personalized-search world
# ---------------------------------------------------------------------------

CATEGORIES = ["news", "tutorial", "local", "entertainment", "evergreen"]
# (quality Beta, virality Beta, share): entertainment is viral-but-shallow, tutorials/local are the reverse
_CAT_SPEC = {
    "news": ((2, 2), (3, 2), 0.20),
    "tutorial": ((4, 1.5), (1.5, 4), 0.15),
    "local": ((4, 1.5), (1.5, 4), 0.10),
    "entertainment": ((1.5, 4), (4, 1.5), 0.35),
    "evergreen": ((4, 2), (2, 3), 0.20),
}


class SearchWorld:
    """Documents with topic, format (sub-topic), category, quality, virality, age; users with a format taste.

    relevance(d | query topic t, user taste s) = eligible * [topic_d == t] * quality_d * (1 + [sub_d == s]),
    discounted for stale news. Engagement is driven by virality AND relevance -- so an engagement-driven pool
    (the recommendation default) keeps the wrong documents for search, which is the paper's Sec. 4.1 point.
    """

    def __init__(self, n_docs: int = 20000, n_topics: int = 100, n_sub: int = 4, dim: int = 16, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.rng = rng
        self.N, self.T, self.S, self.dim = n_docs, n_topics, n_sub, dim
        self.topic = rng.randint(0, n_topics, n_docs)
        self.sub = rng.randint(0, n_sub, n_docs)
        shares = np.array([_CAT_SPEC[c][2] for c in CATEGORIES])
        self.cat = rng.choice(len(CATEGORIES), n_docs, p=shares / shares.sum())
        self.quality = np.zeros(n_docs)
        self.virality = np.zeros(n_docs)
        for ci, c in enumerate(CATEGORIES):
            m = self.cat == ci
            (qa, qb), (va, vb), _ = _CAT_SPEC[c]
            self.quality[m] = rng.beta(qa, qb, m.sum())
            self.virality[m] = rng.beta(va, vb, m.sum())
        self.age = np.where(self.cat == 0, rng.exponential(10.0, n_docs), rng.exponential(120.0, n_docs))
        self.lang_ok = rng.rand(n_docs) > 0.05
        self.spam = rng.rand(n_docs) < 0.04
        self.eligible = self.lang_ok & ~self.spam
        # what the selectors can observe (noisy proxies for the hidden quality / virality)
        self.sv_est = np.clip(self.quality + 0.12 * rng.randn(n_docs), 0, 1)
        self.eng_est = np.clip(0.8 * self.virality + 0.2 * self.quality + 0.08 * rng.randn(n_docs), 0, 1)
        self.proto = rng.randn(n_topics, dim)
        self.sub_proto = 0.5 * rng.randn(n_sub, dim)
        self.content = torch.tensor(self.proto[self.topic] + self.sub_proto[self.sub]
                                    + 0.3 * rng.randn(n_docs, dim), dtype=torch.float32)
        fresh = np.exp(-self.age / 30.0)
        onehot = np.eye(len(CATEGORIES))[self.cat]
        # dense doc features visible to the GPU pathway (CPU doc vectors are content-only)
        self.doc_dense = torch.tensor(np.column_stack([self.sv_est, self.eng_est, fresh, onehot]), dtype=torch.float32)
        self.news_decay = np.where(self.cat == 0, np.exp(-self.age / 30.0), 1.0)
        self.docs_by_topic = [np.where(self.topic == t)[0] for t in range(n_topics)]

    # -- queries ------------------------------------------------------------
    def sample_queries(self, n: int, seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        rng = np.random.RandomState(seed) if seed is not None else self.rng
        topics = rng.randint(0, self.T, n)
        tastes = rng.randint(0, self.S, n)
        qvec = self.proto[topics] + 0.3 * rng.randn(n, self.dim)
        profile = np.column_stack([np.eye(self.S)[tastes] + 0.1 * rng.randn(n, self.S), 0.5 * rng.randn(n, 4)])
        return {"topic": torch.tensor(topics), "taste": torch.tensor(tastes),
                "qvec": torch.tensor(qvec, dtype=torch.float32), "profile": torch.tensor(profile, dtype=torch.float32)}

    def relevance(self, topic: int, taste: int) -> np.ndarray:
        """Ground-truth graded relevance of EVERY document to a (topic, taste) query."""
        rel = np.zeros(self.N)
        idx = self.docs_by_topic[topic]
        rel[idx] = self.quality[idx] * (1.0 + (self.sub[idx] == taste)) * self.news_decay[idx]
        return rel * self.eligible

    def engagement_prob(self, rel: np.ndarray, idx: np.ndarray) -> np.ndarray:
        z = 4.0 * (rel[idx] - 0.4) + 3.0 * (self.virality[idx] - 0.5)
        return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# 4. Lifecycle-aware pool selection (Sec. 4.1)
# ---------------------------------------------------------------------------

@dataclass
class LifecycleConfig:
    pool_size: int = 1000
    early_days: float = 7.0              # first-week early filters
    early_min_value: float = 0.15        # drop low-value items in week one
    engagement_prune: float = 0.25       # engagement-pruning threshold...
    bypass_value: float = 0.6            # ...which high search-value items bypass
    old_days: float = 365.0
    old_keep_fraction: float = 0.001     # "<0.1% of content older than one year" for the evergreen pool
    half_life_days: Dict[str, float] = field(default_factory=lambda: {
        "news": 7.0, "entertainment": 45.0, "tutorial": 1500.0, "local": 1500.0, "evergreen": 4000.0})


def select_search_value_pool(world: SearchWorld, cfg: LifecycleConfig) -> np.ndarray:
    """Three-stage lifecycle filter + category-dependent adaptive retention -> top pool_size by search value.

      1. ingest: hard metadata gates (language, originality / spam).
      2. week 1: early filters remove spam and low-value items (no long-term evidence yet).
      3. day >= 7: long-term utility = search-value estimate x category decay. Engagement pruning applies
         EXCEPT to items with high search value (niche tutorials / local guides bypass it); news decays fast.
      Items older than a year are capped to a tiny fraction (the "highly selective evergreen pool").
    """
    w = world
    keep = w.eligible.copy()
    young = w.age < cfg.early_days
    keep &= ~(young & (w.sv_est < cfg.early_min_value))
    half = np.array([cfg.half_life_days[CATEGORIES[c]] for c in w.cat])
    utility = w.sv_est * np.exp(-np.log(2) * w.age / half)
    prune = (~young) & (w.eng_est < cfg.engagement_prune) & (w.sv_est < cfg.bypass_value)
    keep &= ~prune
    old = keep & (w.age > cfg.old_days)
    if old.any():
        n_keep_old = int(np.floor(cfg.old_keep_fraction * (w.eligible & (w.age > cfg.old_days)).sum()))
        old_idx = np.where(old)[0]
        drop = old_idx[np.argsort(-utility[old_idx])[n_keep_old:]]
        keep[drop] = False
    cand = np.where(keep)[0]
    return np.sort(cand[np.argsort(-utility[cand])[:cfg.pool_size]])


def select_engagement_pool(world: SearchWorld, pool_size: int) -> np.ndarray:
    """The recommendation-style baseline: keep what is most engaging, ignoring search utility."""
    cand = np.where(world.eligible)[0]
    return np.sort(cand[np.argsort(-world.eng_est[cand])[:pool_size]])


def select_broad_inventory(world: SearchWorld, min_value: float = 0.1) -> np.ndarray:
    """CPU inventory (tens of billions): principal languages, no spam, minimum quality only."""
    return np.where(world.eligible & (world.sv_est >= min_value))[0]


def pool_value(world: SearchWorld, pool: np.ndarray, n_queries: int = 300, k: int = 20, seed: int = 1) -> float:
    """Mean nDCG@k an oracle ranker could reach using only this pool (coverage of search value)."""
    q = world.sample_queries(n_queries, seed)
    in_pool = np.zeros(world.N, dtype=bool)
    in_pool[pool] = True
    scores = []
    for t, s in zip(q["topic"].tolist(), q["taste"].tolist()):
        rel = world.relevance(t, s)
        scores.append(ndcg_at_k(sorted(rel[in_pool], reverse=True)[:k], rel, k))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 5. INT8 fused distance + cluster index (Sec. 4.3)
# ---------------------------------------------------------------------------

def int8_quantize(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-vector symmetric INT8: x ≈ q * scale, scale = max|x| / 127."""
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    return torch.round(x / scale).clamp(-127, 127).to(torch.int8), scale.squeeze(-1)


def int8_scores(q_i8: torch.Tensor, q_scale: torch.Tensor, docs_i8: torch.Tensor,
                doc_scale: torch.Tensor) -> torch.Tensor:
    """Fused quantized dot product: integer accumulate, then one rescale -- no float copy of the corpus."""
    acc = docs_i8.to(torch.int64) @ q_i8.to(torch.int64).t()          # [N, nq] exact integer dot products
    return (acc.t().to(torch.float32) * q_scale.unsqueeze(1)) * doc_scale.unsqueeze(0)


class Int8ClusterIndex:
    """GPU-pathway ANN: clustered, INT8-quantized embeddings; probe the nearest clusters, score in INT8."""

    def __init__(self, doc_ids: np.ndarray, embeddings: torch.Tensor, centroids: torch.Tensor):
        emb = F.normalize(embeddings, dim=-1)
        self.centroids = F.normalize(centroids, dim=-1)
        assign = (emb @ self.centroids.t()).argmax(1)
        self.lists = []
        for c in range(len(centroids)):
            m = (assign == c).nonzero().squeeze(1)
            q, s = int8_quantize(emb[m])
            self.lists.append((torch.tensor(doc_ids)[m], q, s))
        self.n = len(doc_ids)

    def search(self, query: torch.Tensor, k: int, n_probe: int = 4) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """query [d] -> (doc_ids, scores, n_scanned) of the top-k by INT8 cosine."""
        qn = F.normalize(query, dim=-1)
        probe = (self.centroids @ qn).topk(min(n_probe, len(self.lists))).indices.tolist()
        q_i8, q_s = int8_quantize(qn.unsqueeze(0))
        ids, scores = [], []
        for c in probe:
            lid, ld, ls = self.lists[c]
            if len(lid):
                ids.append(lid)
                scores.append(int8_scores(q_i8, q_s, ld, ls).squeeze(0))
        if not ids:
            return torch.empty(0, dtype=torch.long), torch.empty(0), 0
        ids, scores = torch.cat(ids), torch.cat(scores)
        top = scores.topk(min(k, len(ids)))
        return ids[top.indices], top.values, len(ids)


# ---------------------------------------------------------------------------
# 6. GPU pathway models: two-tower + interaction pre-ranker, joint training (Sec. 4.2)
# ---------------------------------------------------------------------------

class ResidualMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.inp = nn.Linear(d_in, d_hidden)
        self.block = nn.Sequential(nn.ReLU(), nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_hidden))
        self.out = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        h = self.inp(x)
        return self.out(F.relu(h + self.block(h)))


class GPUTwoTower(nn.Module):
    """Query tower (semantic query vector + user profile -> residual MLP) and document tower
    (content + dense features -> residual MLP); L2-normalised output for nearest-neighbour retrieval."""

    def __init__(self, content_dim: int, profile_dim: int, doc_dense_dim: int, d: int = 32, hidden: int = 96):
        super().__init__()
        self.query_tower = ResidualMLP(content_dim + profile_dim, hidden, d)
        self.doc_tower = ResidualMLP(content_dim + doc_dense_dim, hidden, d)
        self.d = d

    def encode_query(self, qvec, profile):
        return F.normalize(self.query_tower(torch.cat([qvec, profile], -1)), dim=-1)

    def encode_doc(self, content, dense):
        return F.normalize(self.doc_tower(torch.cat([content, dense], -1)), dim=-1)


class InteractionPreRanker(nn.Module):
    """DeepFM-style interaction module: FM second-order terms over field embeddings
    (query, user profile, document, document features) plus a deep MLP; two heads
    p_rel (relevance regression) and p_eng (engagement logit)."""

    def __init__(self, d: int, profile_dim: int, doc_dense_dim: int, k: int = 16, hidden: int = 64):
        super().__init__()
        self.k = k
        self.f_query, self.f_user = nn.Linear(d, k), nn.Linear(profile_dim, k)
        self.f_doc, self.f_dense = nn.Linear(d, k), nn.Linear(doc_dense_dim, k)
        self.deep = nn.Sequential(nn.Linear(4 * k + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head_rel = nn.Linear(hidden + k + 1, 1)
        self.head_eng = nn.Linear(hidden + k + 1, 1)

    def forward(self, q_emb, profile, d_emb, dense) -> Tuple[torch.Tensor, torch.Tensor]:
        fields = torch.stack([self.f_query(q_emb), self.f_user(profile), self.f_doc(d_emb), self.f_dense(dense)], 1)
        sum_sq = fields.sum(1) ** 2
        sq_sum = (fields ** 2).sum(1)
        fm = 0.5 * (sum_sq - sq_sum)                                   # [B,k] pairwise-interaction vector
        sim = (q_emb * d_emb).sum(-1, keepdim=True) * 5.0           # the retrieval score rides along as a feature
        h = torch.cat([self.deep(torch.cat([fields.flatten(1), sim], -1)), fm, sim], -1)
        return self.head_rel(h).squeeze(-1), self.head_eng(h).squeeze(-1)


def value_model(p_rel: torch.Tensor, p_eng: torch.Tensor, lambdas: Tuple[float, float] = (1.0, 0.3)) -> torch.Tensor:
    """S_final = sum_t lambda_t * p_t. Weights pick the operating point without retraining."""
    return lambdas[0] * p_rel + lambdas[1] * torch.sigmoid(p_eng)


def info_nce_cross_session(q: torch.Tensor, d_pos: torch.Tensor, topics: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Eq. (2): in-batch InfoNCE; documents of OTHER sessions are negatives (same-topic ones are masked as false negatives)."""
    logits = q @ d_pos.t() / tau
    same_topic = (topics.unsqueeze(0) == topics.unsqueeze(1)) & ~torch.eye(len(topics), dtype=torch.bool)
    return F.cross_entropy(logits.masked_fill(same_topic, float("-inf")), torch.arange(len(q)))


def info_nce_within_session(q: torch.Tensor, d_sess: torch.Tensor, engaged: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Within-session InfoNCE: engaged candidates are positives, relevant-but-not-engaged are the (hard,
    session-local) negatives. q [B,d], d_sess [B,M,d], engaged [B,M] bool. Sessions with no positive are skipped."""
    logits = torch.einsum("bd,bmd->bm", q, d_sess) / tau
    pos = torch.logsumexp(logits.masked_fill(~engaged, float("-inf")), dim=1)
    has = engaged.any(1) & (~engaged).any(1)
    if not has.any():
        return logits.sum() * 0.0
    return (torch.logsumexp(logits, dim=1) - pos)[has].mean()


@dataclass
class JointWeights:
    w1: float = 1.0      # InfoNCE (retrieval)
    w2: float = 1.0      # Smooth L1 (relevance)
    w3: float = 0.5      # BCE (engagement)


class GPUModels(nn.Module):
    def __init__(self, world: SearchWorld, d: int = 32):
        super().__init__()
        self.tower = GPUTwoTower(world.dim, 8, world.doc_dense.size(1), d)
        self.ranker = InteractionPreRanker(d, 8, world.doc_dense.size(1))


def _sample_sessions(world: SearchWorld, pool_by_topic: Dict[int, np.ndarray], B: int, M: int, rng: np.random.RandomState):
    topics_avail = np.array(sorted(pool_by_topic))
    topics = rng.choice(topics_avail, B)
    tastes = rng.randint(0, world.S, B)
    qvec = world.proto[topics] + 0.3 * rng.randn(B, world.dim)
    profile = np.column_stack([np.eye(world.S)[tastes] + 0.1 * rng.randn(B, world.S), 0.5 * rng.randn(B, 4)])
    docs = np.stack([rng.choice(pool_by_topic[t], M) for t in topics])                # [B,M] same-topic candidates
    rel = np.stack([world.relevance(t, s)[d] for t, s, d in zip(topics, tastes, docs)])
    p_eng = 1 / (1 + np.exp(-(4 * (rel - 0.4) + 3 * (world.virality[docs] - 0.5))))
    engaged = rng.rand(B, M) < p_eng
    # two random off-topic docs per query supply rel = 0 negatives for the regression/BCE heads
    off = rng.randint(0, world.N, (B, 2))
    return topics, tastes, qvec, profile, docs, rel, engaged, off


def train_gpu_models(world: SearchWorld, pool: np.ndarray, steps: int = 300, batch: int = 128, session: int = 6,
                     lr: float = 3e-3, weights: JointWeights = JointWeights(), seed: int = 0) -> GPUModels:
    """Joint training of the two-tower retriever and the interaction pre-ranker (Eq. 1):
    L = w1 L_InfoNCE + w2 L_SmoothL1 + w3 L_BCE."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    pool_by_topic = {t: np.intersect1d(world.docs_by_topic[t], pool) for t in range(world.T)}
    pool_by_topic = {t: v for t, v in pool_by_topic.items() if len(v)}
    m = GPUModels(world)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        topics, tastes, qvec, profile, docs, rel, engaged, off = _sample_sessions(world, pool_by_topic, batch, session, rng)
        qv, pr = torch.tensor(qvec, dtype=torch.float32), torch.tensor(profile, dtype=torch.float32)
        q = m.tower.encode_query(qv, pr)
        d_sess = m.tower.encode_doc(world.content[docs], world.doc_dense[docs])               # [B,M,d]
        eng_t = torch.tensor(engaged)
        # cross-session positive: an engaged doc if any, else the most relevant one
        pos_idx = np.where(engaged.any(1), (engaged * (rel + 1e-3)).argmax(1), rel.argmax(1))
        d_pos = d_sess[torch.arange(batch), torch.tensor(pos_idx)]
        l_nce = info_nce_cross_session(q, d_pos, torch.tensor(topics)) + \
            info_nce_within_session(q, d_sess, eng_t)
        # pre-ranker on session docs + off-topic negatives
        all_docs = np.concatenate([docs, off], 1)
        rel_t = torch.tensor(np.concatenate([rel, np.zeros_like(off, dtype=float)], 1), dtype=torch.float32)
        eng_all = torch.cat([eng_t.float(), torch.zeros(batch, off.shape[1])], 1)
        d_all = m.tower.encode_doc(world.content[all_docs], world.doc_dense[all_docs])
        B, K = all_docs.shape
        p_rel, p_eng = m.ranker(q.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1), pr.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1),
                                d_all.reshape(B * K, -1), world.doc_dense[all_docs].reshape(B * K, -1))
        loss = weights.w1 * l_nce + weights.w2 * F.smooth_l1_loss(p_rel, rel_t.reshape(-1)) + \
            weights.w3 * F.binary_cross_entropy_with_logits(p_eng, eng_all.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m


# ---------------------------------------------------------------------------
# 7. CPU pathway: dedicated IVF embedding index, eager evaluation, lightweight model (Sec. 5)
# ---------------------------------------------------------------------------

def kmeans(x: torch.Tensor, k: int, iters: int = 15, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = (F.normalize(x, dim=-1) @ F.normalize(c, dim=-1).t()).argmax(1)
        for j in range(k):
            m = assign == j
            if m.any():
                c[j] = x[m].mean(0)
    return c


@dataclass
class VersionedCentroids:
    """Output of the decoupled, daily distributed centroid trainer (Sec. 5.2). Serving shards ingest a
    published version instead of re-clustering at index-build time."""
    version: int
    centroids: torch.Tensor


def train_centroids(embeddings: torch.Tensor, n_centroids: int, version: int, seed: int = 0) -> VersionedCentroids:
    return VersionedCentroids(version, F.normalize(kmeans(embeddings, n_centroids, seed=seed), dim=-1))


@dataclass
class AccessStats:
    distance_ops: int = 0          # vector distance computations
    sequential_blocks: int = 0     # contiguous list scans
    random_jumps: int = 0          # switches between posting lists while visiting docs in id order


class EmbeddingIndex:
    """Dedicated embedding index: pretrained centroids -> inverted lists of contiguous document vectors.

    Query path: pick nearest centroids (coarse pruning), scan their lists by cosine. Two execution plans:

      "eager" (term-at-a-time): scan each probed list contiguously with one bulk matmul, take a global
          top (k x prefetch_multiplier), THEN apply the other filtering rules to that small set. All heavy
          vector work finishes before filtering, so reads are sequential (Fig. 4).
      "daat" (document-at-a-time, the legacy plan): visit candidate docs in id order across all lists, apply
          the filters first, and compute a distance only for survivors. Fewer distance computations, but
          successive docs come from different lists => scattered, cache-unfriendly access.
    """

    def __init__(self, doc_ids: np.ndarray, vectors: torch.Tensor, centroids: VersionedCentroids):
        self.cv = centroids
        self.dim = vectors.size(1)
        v = F.normalize(vectors, dim=-1)
        assign = (v @ centroids.centroids.t()).argmax(1)
        self.ids: List[torch.Tensor] = []
        self.vecs: List[torch.Tensor] = []
        for c in range(len(centroids.centroids)):
            m = (assign == c).nonzero().squeeze(1)
            self.ids.append(torch.tensor(doc_ids)[m])
            self.vecs.append(v[m])
        self.deleted: Set[int] = set()
        self.stats = AccessStats()

    def __len__(self) -> int:
        return sum(len(i) for i in self.ids) - len(self.deleted)

    # -- live updates (no re-clustering) --------------------------------------
    def add(self, doc_ids: np.ndarray, vectors: torch.Tensor) -> None:
        v = F.normalize(vectors, dim=-1)
        assign = (v @ self.cv.centroids.t()).argmax(1)
        for c in assign.unique().tolist():
            m = (assign == c).nonzero().squeeze(1)
            self.ids[c] = torch.cat([self.ids[c], torch.tensor(doc_ids)[m]])
            self.vecs[c] = torch.cat([self.vecs[c], v[m]])

    def remove(self, doc_ids: Sequence[int]) -> None:
        self.deleted.update(int(i) for i in doc_ids)

    # -- search ---------------------------------------------------------------
    def probe_lists(self, q: torch.Tensor, n_probe: int) -> List[int]:
        return (self.cv.centroids @ F.normalize(q, dim=-1)).topk(min(n_probe, len(self.ids))).indices.tolist()

    def search(self, q: torch.Tensor, k: int, n_probe: int = 8, mode: str = "eager", prefetch_multiplier: int = 2,
               filter_fn: Optional[Callable[[int], bool]] = None) -> Tuple[List[int], int]:
        """Returns (top-k doc ids by cosine that pass filter_fn, number of docs scanned)."""
        qn = F.normalize(q.detach(), dim=-1)
        lists = self.probe_lists(qn, n_probe)
        keep = (lambda i: True) if filter_fn is None else filter_fn
        scanned = sum(len(self.ids[c]) for c in lists)
        if mode == "eager":
            all_ids, all_sims = [], []
            for c in lists:                                             # TAAT: contiguous bulk scan per list
                if len(self.ids[c]):
                    all_sims.append(self.vecs[c] @ qn)
                    all_ids.append(self.ids[c])
                    self.stats.sequential_blocks += 1
                    self.stats.distance_ops += len(self.ids[c])
            if not all_ids:
                return [], 0
            ids, sims = torch.cat(all_ids), torch.cat(all_sims)
            top = sims.topk(min(k * prefetch_multiplier, len(ids))).indices      # global top BEFORE other rules
            out = []
            for i in ids[top].tolist():
                if i not in self.deleted and keep(i):
                    out.append(i)
                if len(out) == k:
                    break
            return out, scanned
        if mode == "daat":
            entries = [(int(i), c, j) for c in lists for j, i in enumerate(self.ids[c].tolist())]
            entries.sort()                                              # merge in document-id order
            last_list, scored = None, []
            for doc, c, j in entries:
                if last_list is not None and c != last_list:
                    self.stats.random_jumps += 1
                last_list = c
                if doc in self.deleted or not keep(doc):
                    continue
                scored.append((float(self.vecs[c][j] @ qn), doc))      # one distance per surviving doc
                self.stats.distance_ops += 1
            scored.sort(reverse=True)
            return [d for _, d in scored[:k]], scanned
        raise ValueError(mode)


def recall_candidate_frontier(index: EmbeddingIndex, queries: torch.Tensor, truth: List[Set[int]], k: int,
                              n_probes: Sequence[int]) -> List[Tuple[float, float]]:
    """Fig. 5: (avg candidates scanned, recall@k vs. exact neighbours) as n_probe grows."""
    out = []
    for p in n_probes:
        rec, scan = [], []
        for q, t in zip(queries, truth):
            ids, scanned = index.search(q, k, n_probe=p)
            rec.append(len(set(ids) & t) / max(len(t), 1))
            scan.append(scanned)
        out.append((float(np.mean(scan)), float(np.mean(rec))))
    return out


class LightweightTwoTower(nn.Module):
    """CPU-pathway unified model: ONE shared backbone encodes queries and documents; the query side adds an
    attention-style fusion with a pooled lookup of sparse context features; documents stay precomputable and
    interact with the query only by dot product (no cross features, no document dense features)."""

    def __init__(self, content_dim: int, n_ctx: int, d: int = 24, hidden: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(content_dim, hidden), nn.ReLU(), nn.Linear(hidden, d))
        self.ctx = nn.Embedding(n_ctx, d)
        self.gate = nn.Linear(2 * d, d)

    def encode_query(self, qvec: torch.Tensor, ctx_ids: torch.Tensor) -> torch.Tensor:
        h, c = self.backbone(qvec), self.ctx(ctx_ids)
        return F.normalize(h + torch.sigmoid(self.gate(torch.cat([h, c], -1))) * c, dim=-1)

    def encode_doc(self, content: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(content), dim=-1)


def bidirectional_info_nce(q: torch.Tensor, d: torch.Tensor, topics: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    logits = q @ d.t() / tau
    mask = (topics.unsqueeze(0) == topics.unsqueeze(1)) & ~torch.eye(len(topics), dtype=torch.bool)
    logits = logits.masked_fill(mask, float("-inf"))
    tgt = torch.arange(len(q))
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t().masked_fill(mask, float("-inf")), tgt))


def train_cpu_model(world: SearchWorld, inventory: np.ndarray, steps: int = 300, batch: int = 128, lr: float = 3e-3,
                    seed: int = 0) -> LightweightTwoTower:
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    inv_by_topic = {t: np.intersect1d(world.docs_by_topic[t], inventory) for t in range(world.T)}
    avail = np.array([t for t, v in inv_by_topic.items() if len(v)])
    model = LightweightTwoTower(world.dim, world.S)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        topics = rng.choice(avail, batch)
        tastes = rng.randint(0, world.S, batch)
        qvec = torch.tensor(world.proto[topics] + 0.3 * rng.randn(batch, world.dim), dtype=torch.float32)
        pos = []
        for t, s in zip(topics, tastes):                                 # engaged doc ~ relevance-weighted
            cand = inv_by_topic[t]
            w = world.relevance(t, s)[cand] + 0.05
            pos.append(rng.choice(cand, p=w / w.sum()))
        loss = bidirectional_info_nce(model.encode_query(qvec, torch.tensor(tastes)),
                                      model.encode_doc(world.content[np.array(pos)]), torch.tensor(topics))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


# ---------------------------------------------------------------------------
# 8. Pathways, aggregation, versioned release (Sec. 3)
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    doc_id: int
    sources: Set[str]
    score: float = 0.0


@dataclass
class BranchResult:
    name: str
    candidates: List[Tuple[int, float]]       # (doc_id, score), best first
    latency_ms: float


@dataclass
class ModelIndexSnapshot:
    """A model and the index built from ITS embeddings, published together (Sec. 3.2, 6.2). Embeddings,
    ANN configuration and features form one deployment unit -- an embedding alone is not a release unit."""
    version: int
    model_version: int
    index: object


class GPUPathway:
    name = "gpu"

    def __init__(self, world: SearchWorld, models: GPUModels, pool: np.ndarray, n_centroids: int = 16,
                 n_probe: int = 4, k_prefetch: int = 100, k_final: int = 30, lambdas=(1.0, 0.3),
                 model_version: int = 1):
        self.world, self.models, self.pool = world, models, pool
        self.n_probe, self.k_prefetch, self.k_final, self.lambdas = n_probe, k_prefetch, k_final, lambdas
        self.model_version = model_version
        with torch.no_grad():
            self.doc_emb = models.tower.encode_doc(world.content[pool], world.doc_dense[pool])
        cents = kmeans(self.doc_emb, n_centroids)
        self.snapshot = ModelIndexSnapshot(1, model_version, Int8ClusterIndex(pool, self.doc_emb, cents))
        self._pos = {int(d): i for i, d in enumerate(pool)}
        self.enabled = True

    def publish(self, snapshot: ModelIndexSnapshot) -> None:
        if snapshot.model_version != self.model_version:
            raise ValueError("snapshot's index was built by a different model version than the one serving")
        self.snapshot = snapshot

    @torch.no_grad()
    def retrieve(self, qvec: torch.Tensor, profile: torch.Tensor) -> BranchResult:
        q = self.models.tower.encode_query(qvec, profile)
        ids, _, _ = self.snapshot.index.search(q[0], self.k_prefetch, self.n_probe)          # ANN pre-fetch (INT8)
        if len(ids) == 0:
            return BranchResult(self.name, [], 0.0)
        pos = torch.tensor([self._pos[int(i)] for i in ids])
        n = len(ids)
        p_rel, p_eng = self.models.ranker(q.expand(n, -1), profile.expand(n, -1), self.doc_emb[pos],
                                          self.world.doc_dense[ids])                      # interaction pre-ranking
        s = value_model(p_rel, p_eng, self.lambdas)
        top = s.topk(min(self.k_final, n))
        return BranchResult(self.name, [(int(ids[i]), float(v)) for i, v in zip(top.indices, top.values)], 0.0)


class CPUPathway:
    name = "cpu"

    def __init__(self, world: SearchWorld, model: LightweightTwoTower, inventory: np.ndarray, centroids: VersionedCentroids,
                 n_probe: int = 8, k: int = 40, mode: str = "eager", min_value: float = 0.2):
        self.world, self.model, self.n_probe, self.k, self.mode = world, model, n_probe, k, mode
        with torch.no_grad():
            vecs = model.encode_doc(world.content[inventory])
        self.index = EmbeddingIndex(inventory, vecs, centroids)
        self.filter_fn = lambda i: bool(world.sv_est[i] >= min_value)                      # the lighter filter
        self.enabled = True

    @torch.no_grad()
    def retrieve(self, qvec: torch.Tensor, taste: torch.Tensor) -> BranchResult:
        q = self.model.encode_query(qvec, taste)
        ids, _ = self.index.search(q[0], self.k, self.n_probe, self.mode, filter_fn=self.filter_fn)
        # No interaction score on this branch; carry a rank-derived score so the aggregator has something to keep.
        return BranchResult(self.name, [(i, 1.0 / (r + 1)) for r, i in enumerate(ids)], 0.0)


class Aggregator:
    """Merges whatever branches returned in time. Each branch has its own deadline, so one slow or failed
    branch never discards the other's candidates. Deduplicates by doc id and keeps source attribution."""

    def __init__(self, deadlines_ms: Dict[str, float]):
        self.deadlines = deadlines_ms

    def merge(self, results: Sequence[BranchResult]) -> List[Candidate]:
        by_doc: Dict[int, Candidate] = {}
        for r in results:
            if r.latency_ms > self.deadlines.get(r.name, float("inf")):
                continue                                                   # missed its deadline: dropped alone
            for doc, score in r.candidates:
                c = by_doc.setdefault(doc, Candidate(doc, set(), score))
                c.sources.add(r.name)
                c.score = max(c.score, score)
        return list(by_doc.values())


class HybridRetriever:
    """Fork-join over independently enableable pathways. Either (or both) may run per request; disabling one
    is a config flip that leaves the aggregation interface untouched (rollback)."""

    def __init__(self, gpu: Optional[GPUPathway], cpu: Optional[CPUPathway], deadlines_ms: Optional[Dict[str, float]] = None,
                 latency_fn: Optional[Callable[[str], float]] = None):
        self.gpu, self.cpu = gpu, cpu
        self.agg = Aggregator(deadlines_ms or {"gpu": 200.0, "cpu": 200.0})
        self.latency_fn = latency_fn or (lambda name: 0.0)

    def retrieve(self, q: Dict[str, torch.Tensor], i: int) -> List[Candidate]:
        results = []
        if self.gpu is not None and self.gpu.enabled:
            r = self.gpu.retrieve(q["qvec"][i:i + 1], q["profile"][i:i + 1])
            r.latency_ms = self.latency_fn("gpu")
            results.append(r)
        if self.cpu is not None and self.cpu.enabled:
            r = self.cpu.retrieve(q["qvec"][i:i + 1], q["taste"][i:i + 1])
            r.latency_ms = self.latency_fn("cpu")
            results.append(r)
        return self.agg.merge(results)


class NoisyOracleRanker:
    """Stand-in for the shared downstream ranker (out of the paper's scope): relevance plus noise."""

    def __init__(self, world: SearchWorld, noise: float = 0.15, seed: int = 0):
        self.world, self.noise, self.rng = world, noise, np.random.RandomState(seed)

    def rank(self, cands: Sequence[Candidate], topic: int, taste: int, k: int = 20) -> List[int]:
        if not cands:
            return []
        rel = self.world.relevance(topic, taste)
        ids = np.array([c.doc_id for c in cands])
        s = rel[ids] + self.noise * self.rng.randn(len(ids))
        return ids[np.argsort(-s)][:k].tolist()


def evaluate_retrievers(world: SearchWorld, retrievers: Dict[str, HybridRetriever], n_queries: int = 150,
                        k: int = 20, seed: int = 7) -> Dict[str, Dict[str, float]]:
    """nDCG@k (vs the whole corpus) after the shared ranker, candidate-set size, and `rel_recall`: the share of
    the ideal top-k relevance mass that the CANDIDATE SET (before ranking) already contains."""
    q = world.sample_queries(n_queries, seed)
    out = {n: {"ndcg": [], "cands": [], "rel_recall": []} for n in retrievers}
    for name, r in retrievers.items():
        ranker = NoisyOracleRanker(world)
        for i in range(n_queries):
            t, s = int(q["topic"][i]), int(q["taste"][i])
            rel = world.relevance(t, s)
            cands = r.retrieve(q, i)
            ranked = ranker.rank(cands, t, s, k)
            out[name]["ndcg"].append(ndcg_at_k([rel[d] for d in ranked], rel, k))
            out[name]["cands"].append(len(cands))
            ideal_mass = np.sort(rel)[::-1][:k].sum()
            got_mass = sum(sorted((rel[c.doc_id] for c in cands), reverse=True)[:k])
            out[name]["rel_recall"].append(got_mass / ideal_mass if ideal_mass > 0 else 0.0)
    return {n: {m: float(np.mean(v)) for m, v in d.items()} for n, d in out.items()}


@torch.no_grad()
def modeling_depth_ndcg(world: SearchWorld, pool: np.ndarray, gpu: GPUPathway, cpu_model: LightweightTwoTower,
                        n_queries: int = 150, k: int = 20, seed: int = 7) -> Dict[str, float]:
    """Isolates MODELING DEPTH from inventory size: both models order the SAME pool and are scored by nDCG@k
    against the best achievable from that pool (their own ordering, no shared ranker).
      * "gpu"      = ANN pre-fetch + interaction pre-ranker + value model (the GPU pathway as deployed)
      * "gpu_ann"  = the GPU two-tower alone, no interaction scoring
      * "cpu_model"= the lightweight dot-product model
    """
    q = world.sample_queries(n_queries, seed)
    with torch.no_grad():
        cpu_docs = cpu_model.encode_doc(world.content[pool])
        gpu_docs = gpu.doc_emb
    res = {"gpu": [], "gpu_ann": [], "cpu_model": []}
    for i in range(n_queries):
        t, s = int(q["topic"][i]), int(q["taste"][i])
        rel = world.relevance(t, s)[pool]
        ideal = rel
        g = gpu.retrieve(q["qvec"][i:i + 1], q["profile"][i:i + 1]).candidates[:k]
        pos = {int(d): j for j, d in enumerate(pool)}
        res["gpu"].append(ndcg_at_k([rel[pos[d]] for d, _ in g], ideal, k))
        qe = gpu.models.tower.encode_query(q["qvec"][i:i + 1], q["profile"][i:i + 1])[0]
        top = (gpu_docs @ qe).topk(k).indices.tolist()
        res["gpu_ann"].append(ndcg_at_k([rel[j] for j in top], ideal, k))
        ce = cpu_model.encode_query(q["qvec"][i:i + 1], q["taste"][i:i + 1])[0]
        top = (cpu_docs @ ce).topk(k).indices.tolist()
        res["cpu_model"].append(ndcg_at_k([rel[j] for j in top], ideal, k))
    return {n: float(np.mean(v)) for n, v in res.items()}


# ---------------------------------------------------------------------------
# 9. Demo
# ---------------------------------------------------------------------------

def main(seed: int = 0) -> Dict[str, object]:
    world = SearchWorld(seed=seed)
    cfg = LifecycleConfig(pool_size=1000, old_keep_fraction=0.2)
    pool = select_search_value_pool(world, cfg)
    eng_pool = select_engagement_pool(world, cfg.pool_size)
    inventory = select_broad_inventory(world)
    print(f"corpus {world.N}; GPU pool {len(pool)}; CPU inventory {len(inventory)} "
          f"({len(inventory) / len(pool):.0f}x larger)")
    print("== pool selection (oracle nDCG@20 reachable from the pool) ==")
    print(f"  search-value pool {pool_value(world, pool):.3f}   engagement pool {pool_value(world, eng_pool):.3f}")

    gpu_models = train_gpu_models(world, pool, steps=300, seed=seed)
    cpu_model = train_cpu_model(world, inventory, steps=300, seed=seed)
    with torch.no_grad():
        cvecs = cpu_model.encode_doc(world.content[inventory])
    centroids = train_centroids(cvecs, 128, version=1)
    gpu = GPUPathway(world, gpu_models, pool)
    cpu = CPUPathway(world, cpu_model, inventory, centroids)
    res = evaluate_retrievers(world, {"gpu only": HybridRetriever(gpu, None), "cpu only": HybridRetriever(None, cpu),
                                      "hybrid": HybridRetriever(gpu, cpu)})
    print("== end-to-end (shared noisy-oracle ranker; nDCG@20 against the whole corpus) ==")
    for n, r in res.items():
        print(f"  {n:9s} nDCG@20 {r['ndcg']:.3f}   candidates/query {r['cands']:.0f}")

    depth = modeling_depth_ndcg(world, pool, gpu, cpu_model)
    print("== modeling depth on the SAME pool (own ordering, nDCG@20 vs best reachable from the pool) ==")
    print(f"  GPU two-tower + interaction pre-ranker {depth['gpu']:.3f}   GPU two-tower only {depth['gpu_ann']:.3f}   "
          f"lightweight CPU model {depth['cpu_model']:.3f}")
    q = world.sample_queries(100, 11)
    g_set, c_set = set(), set()
    per_src = {"gpu": set(), "cpu": set()}
    for i in range(100):
        g = {d for d, _ in gpu.retrieve(q["qvec"][i:i + 1], q["profile"][i:i + 1]).candidates}
        c = {d for d, _ in cpu.retrieve(q["qvec"][i:i + 1], q["taste"][i:i + 1]).candidates}
        per_src["gpu"] |= {(i, d) for d in g}
        per_src["cpu"] |= {(i, d) for d in c}
    comp = source_composition(per_src)
    print(f"== candidate diversity == exclusive GPU {comp['gpu']}, exclusive CPU {comp['cpu']}, "
          f"overlap {comp['overlap']} of {comp['total']}")
    print("== CPU index: eager (TAAT + prefetch) vs document-at-a-time, 100 filtered queries ==")
    qs = world.sample_queries(100, 13)
    stats = {}
    for mode in ("eager", "daat"):
        idx = EmbeddingIndex(inventory, cvecs, centroids)
        for i in range(100):
            q = cpu_model.encode_query(qs["qvec"][i:i + 1], qs["taste"][i:i + 1])[0]
            idx.search(q, 40, 8, mode, 50, cpu.filter_fn)
        stats[mode] = idx.stats
        print(f"  {mode:6s} distance ops {idx.stats.distance_ops:8d}   sequential list scans "
              f"{idx.stats.sequential_blocks:5d}   random list-to-list jumps {idx.stats.random_jumps:7d}")
    cpu_u, acc_u = HardwareUnit(1.0, 1.0, 1.0), HardwareUnit(3.0, 3.5, 12.0)
    plan = compare_plans(1000, 100, cpu_u, acc_u)
    print(f"== capacity plan (storage-dominated, 3 regions) == accelerator units/region {plan['accel_units_per_region']:.2f}x, "
          f"unit cost {plan['unit_capacity_cost']:.0f}x, plan cost {plan['accel_plan_cost']:.1f}x the CPU plan")
    return {"pool": (pool_value(world, pool), pool_value(world, eng_pool)), "e2e": res, "composition": comp, "depth": depth}


if __name__ == "__main__":
    main()
