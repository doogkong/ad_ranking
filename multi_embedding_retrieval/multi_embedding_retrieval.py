"""Multi-embedding retrieval conditioned on implicit and explicit user interests.

Reference implementation of "Synergizing Implicit and Explicit User Interests:
A Multi-Embedding Retrieval Framework at Pinterest" (Fan, Lin, Chen, Deng, Xia,
Yan, Li; KDD 2025; arXiv 2506.23060).

A single two-tower user embedding is dominated by the user's top interests, so
torso/tail interests are rarely retrieved. The paper generates K user embeddings,
each *conditioned* on one interest, from two complementary sources:

  * Implicit interests (Sec. 3.2) -- mined from the engagement sequence by the
    Differentiable Clustering Module (DCM): a Capsule-Network style routing with
      - Validity-Aware Farthest Point Initialization (VA-FPI, Eqs. 5-6) and
      - Single-Assignment Routing (SAR, Eq. 7)
    Condition association is internal: the positive item is paired with the
    user embedding that scores it highest (argmax, Eq. 8) and trained with a
    sampled-softmax + logQ loss on that embedding only (Eq. 9).
  * Explicit interests (Sec. 3.3) -- followed topics, via Conditional Retrieval
    (CR): the topic embedding enters the user tower's embedding layer and passes
    through the feature-crossing layers; association happens at *logging* time
    (the topic that sourced the engagement), plus an explicit relevance filter.

Serving (Sec. 3.5): ANN search per embedding, per-embedding budgets (implicit:
proportional to routing mass sum_i b_ij; explicit: equal), round-robin merge
with dedup.

Also implemented, as the paper's baselines/ablations: vanilla Capsule Network
(MIND: Gaussian init + multi-assignment), multi-head self-attention
(ComiRec-SA), interest tokens (MVKE-style), and PinnerFormer-subsequence-style
concept pooling (PFS), with the Straight-Through Gumbel-Softmax association the
paper uses to stop self-attention/interest-token embeddings from collapsing.

Not implemented: Pinterest's data, PinSage/PinnerFormer features, ANN/serving
infrastructure, the production DHEN with ParallelMaskNet at full scale (a small
Transformer+MLP / MaskNet+MLP version is here). The synthetic world at the
bottom stands in for the data; its numbers do not reflect the paper's results.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Feature crossing (Appendix A: DHEN-style, two hierarchies of parallel modules)
# ---------------------------------------------------------------------------

class MaskNetBlock(nn.Module):
    """Instance-guided multiplicative mask over the flattened fields (MaskNet)."""

    def __init__(self, dim: int, n_blocks: int = 4, ratio: float = 0.5):
        super().__init__()
        hidden = max(int(dim * ratio), 1)
        self.norm = nn.LayerNorm(dim)
        self.masks = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim)) for _ in range(n_blocks))
        self.out = nn.Linear(dim * n_blocks, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # [B, dim]
        xn = self.norm(x)
        return self.out(torch.cat([m(x) * xn for m in self.masks], dim=-1))


class FeatureCrossing(nn.Module):
    """Two hierarchies; each sums parallel submodules and feeds the next.

    hierarchy 1: Transformer encoder over field tokens  +  MLP on concatenated fields
    hierarchy 2: MaskNet                                +  MLP
    Input [B, n_fields, d] -> output [B, d] (all fields mixed, then projected).
    """

    def __init__(self, n_fields: int, d: int, n_heads: int = 4, n_layers: int = 2, mlp_hidden: int = 128):
        super().__init__()
        self.n_fields, self.d = n_fields, d
        flat = n_fields * d
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.mlp1 = nn.Sequential(nn.Linear(flat, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, flat))
        self.norm1 = nn.LayerNorm(flat)
        self.mask_net = MaskNetBlock(flat)
        self.mlp2 = nn.Sequential(nn.Linear(flat, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, flat))
        self.norm2 = nn.LayerNorm(flat)
        self.out = nn.Linear(flat, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        flat = x.reshape(B, -1)
        h1 = self.norm1(flat + self.transformer(x).reshape(B, -1) + self.mlp1(flat))
        h2 = self.norm2(h1 + self.mask_net(h1) + self.mlp2(h1))
        return self.out(h2)


# ---------------------------------------------------------------------------
# 2. Towers
# ---------------------------------------------------------------------------

class ItemTower(nn.Module):
    """psi(i): item features -> L2-normalised embedding."""

    def __init__(self, item_feat_dim: int, d: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(item_feat_dim, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, item_feats: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(item_feats), dim=-1)


class UserTower(nn.Module):
    """phi(u, c): user features + ONE condition vector -> normalised embedding.

    The condition is one more field token fed through the feature-crossing
    layers (Sec. 3.3.1), so condition x user-feature interactions are learned.
    Called for all K conditions at once by folding K into the batch.
    """

    def __init__(self, user_feat_dim: int, d: int, n_tokens: int = 4, **crossing_kw):
        super().__init__()
        self.n_tokens, self.d = n_tokens, d
        self.embed = nn.Linear(user_feat_dim, n_tokens * d)
        self.cond_proj = nn.Linear(d, d)
        self.crossing = FeatureCrossing(n_tokens + 1, d, **crossing_kw)

    def forward(self, user_feats: torch.Tensor, conds: torch.Tensor) -> torch.Tensor:
        """user_feats [B,Fu], conds [B,K,d] -> [B,K,d]."""
        B, K, d = conds.shape
        tokens = self.embed(user_feats).view(B, 1, self.n_tokens, d).expand(B, K, self.n_tokens, d)
        cond_tok = self.cond_proj(conds).unsqueeze(2)
        x = torch.cat([tokens, cond_tok], dim=2).reshape(B * K, self.n_tokens + 1, d)
        return F.normalize(self.crossing(x), dim=-1).view(B, K, d)


# ---------------------------------------------------------------------------
# 3. Implicit condition construction
# ---------------------------------------------------------------------------

def squash(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Eq. (3): non-linearity whose output norm in [0,1) encodes the cluster's presence."""
    n2 = (v * v).sum(-1, keepdim=True)
    return (n2 / (1.0 + n2)) * v / torch.sqrt(n2 + eps)


@dataclass
class Conditions:
    """Output of every implicit-condition builder."""
    conds: torch.Tensor        # [B,K,d]
    mask: torch.Tensor         # [B,K] bool: this condition is real (non-empty cluster / valid)
    importance: torch.Tensor   # [B,K] routing mass; drives serving budgets


class ItemSummarizer(nn.Module):
    """Eq. (4): e_i = W2 GELU(W1 concat(item features))."""

    def __init__(self, item_feat_dim: int, d: int, hidden: int = 64):
        super().__init__()
        self.w1, self.w2 = nn.Linear(item_feat_dim, hidden), nn.Linear(hidden, d)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(feats)))


def validity_aware_fpi(e: torch.Tensor, valid: torch.Tensor, k: int,
                       use_validity: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """VA-FPI (Eqs. 5-6). e [B,L,d], valid [B,L] -> centroids [B,K,d], distinct [B,K].

    First centroid: a random valid item. Then repeatedly the item whose maximum
    similarity to the chosen centroids is smallest (the farthest point), among
    valid, not-yet-chosen items. `distinct` is False where fewer than K usable
    items existed (the centroid is then a repeat and should be ignored).
    With use_validity=False every non-padding item may be chosen -- including
    out-of-distribution ones, which is what the ablation shows collapsing.
    """
    B, L, _ = e.shape
    ed = e.detach()
    ar = torch.arange(B)
    first = torch.rand(B, L).masked_fill(~valid, -1.0).argmax(1)
    chosen = torch.zeros(B, L, dtype=torch.bool)
    chosen[ar, first] = True
    idx = [first]
    distinct = [valid.any(1)]
    max_sim = torch.einsum("bld,bd->bl", ed, ed[ar, first])
    for _ in range(k - 1):
        cand = max_sim.masked_fill(~valid | chosen, float("inf"))
        nxt = cand.argmin(1)
        ok = torch.isfinite(cand[ar, nxt])
        nxt = torch.where(ok, nxt, idx[-1])
        chosen[ar, nxt] = True
        idx.append(nxt)
        distinct.append(ok & distinct[-1])
        max_sim = torch.maximum(max_sim, torch.einsum("bld,bd->bl", ed, ed[ar, nxt]))
    idx_t = torch.stack(idx, 1)                                        # [B,K]
    return e[ar.unsqueeze(1), idx_t], torch.stack(distinct, 1)


class DifferentiableClusteringModule(nn.Module):
    """DCM (Sec. 3.2.2) -- and, via flags, the vanilla Capsule Network (MIND).

    init="vafpi"    + single_assignment=True  -> DCM
    init="gaussian" + single_assignment=False -> MIND-style capsule routing
    The four combinations reproduce the paper's Table 6 ablation grid.
    """

    def __init__(self, item_feat_dim: int, d: int, n_interests: int, iters: int = 3,
                 init: str = "vafpi", single_assignment: bool = True, use_validity: bool = True):
        super().__init__()
        assert init in ("vafpi", "gaussian")
        self.k, self.iters, self.init = n_interests, iters, init
        self.single_assignment, self.use_validity = single_assignment, use_validity
        self.summarize = ItemSummarizer(item_feat_dim, d)
        # Shared bilinear map S (Eq. 1); identity start so S e_i ~ e_i at init (cf. Eq. 5 using e_i).
        self.S = nn.Parameter(torch.eye(d))

    def forward(self, item_feats: torch.Tensor, seq_mask: torch.Tensor,
                item_valid: Optional[torch.Tensor] = None) -> Conditions:
        """item_feats [B,L,F]; seq_mask [B,L] real (non-padding); item_valid [B,L] features usable."""
        e = self.summarize(item_feats)
        B, L, d = e.shape
        valid = seq_mask if (item_valid is None or not self.use_validity) else (seq_mask & item_valid)
        Se = e @ self.S.t()

        if self.init == "vafpi":
            c, distinct = validity_aware_fpi(e, valid, self.k)
        else:
            c = torch.randn(B, self.k, d) / math.sqrt(d)
            distinct = valid.any(1, keepdim=True).expand(B, self.k)

        vmask = valid.unsqueeze(1).float()                              # [B,1,L]
        for _ in range(self.iters):
            logits = torch.einsum("bkd,bld->bkl", c, Se)                # c_j^T S e_i
            b = F.softmax(logits, dim=1)                                # Eq. (1): over centroids
            if self.single_assignment:                                  # Eq. (7)
                b = b * (logits == logits.max(dim=1, keepdim=True).values).float()
            b = b * vmask
            c = squash(torch.einsum("bkl,bld->bkd", b, Se))             # Eq. (2)
        importance = b.sum(-1)                                          # [B,K]
        mask = distinct & (importance > 1e-6)
        return Conditions(c, mask, importance)


class SelfAttentionConditions(nn.Module):
    """ComiRec-SA: K attention heads over the sequence, one interest per head."""

    def __init__(self, item_feat_dim: int, d: int, n_interests: int, hidden: int = 64):
        super().__init__()
        self.summarize = ItemSummarizer(item_feat_dim, d)
        self.w1, self.w2 = nn.Linear(d, hidden), nn.Linear(hidden, n_interests, bias=False)

    def forward(self, item_feats, seq_mask, item_valid=None) -> Conditions:
        e = self.summarize(item_feats)
        att = self.w2(torch.tanh(self.w1(e))).masked_fill(~seq_mask.unsqueeze(-1), float("-inf"))
        att = F.softmax(att, dim=1).nan_to_num(0.0).transpose(1, 2)    # [B,K,L]
        conds = att @ e
        return Conditions(conds, seq_mask.any(1, keepdim=True).expand_as(conds[..., 0]), att.sum(-1))


class InterestTokenConditions(nn.Module):
    """Learnable query tokens cross-attend to the sequence (MVKE minus the virtual-kernel gating)."""

    def __init__(self, item_feat_dim: int, d: int, n_interests: int):
        super().__init__()
        self.summarize = ItemSummarizer(item_feat_dim, d)
        self.tokens = nn.Parameter(torch.randn(n_interests, d) / math.sqrt(d))
        self.attn = nn.MultiheadAttention(d, 1, batch_first=True)

    def forward(self, item_feats, seq_mask, item_valid=None) -> Conditions:
        e = self.summarize(item_feats)
        B = e.size(0)
        q = self.tokens.unsqueeze(0).expand(B, -1, -1)
        out, w = self.attn(q, e, e, key_padding_mask=~seq_mask, need_weights=True)
        any_item = seq_mask.any(1, keepdim=True).expand(B, self.tokens.size(0))
        return Conditions(out.nan_to_num(0.0), any_item, w.nan_to_num(0.0).sum(-1))


def kmeans(x: torch.Tensor, k: int, iters: int = 20, seed: int = 0) -> torch.Tensor:
    """Plain Lloyd's k-means (used to define PFS concepts offline)."""
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(x.size(0), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = torch.cdist(x, c).argmin(1)
        for j in range(k):
            if (assign == j).any():
                c[j] = x[assign == j].mean(0)
    return c


class ConceptPoolConditions(nn.Module):
    """PFS-style: split the sequence into subsequences by fixed offline concept
    clusters; each subsequence's mean-pooled embedding is a condition. Not
    end-to-end (clusters are fixed), which the paper cites as why DCM wins."""

    def __init__(self, item_feat_dim: int, d: int, concept_centroids: torch.Tensor):
        super().__init__()
        self.summarize = ItemSummarizer(item_feat_dim, d)
        self.register_buffer("concepts", concept_centroids)

    def forward(self, item_feats, seq_mask, item_valid=None) -> Conditions:
        e = self.summarize(item_feats)
        B, L, _ = e.shape
        k = self.concepts.size(0)
        assign = torch.cdist(item_feats.reshape(B * L, -1), self.concepts).argmin(1).view(B, L)
        onehot = F.one_hot(assign, k).float() * seq_mask.unsqueeze(-1).float()   # [B,L,K]
        count = onehot.sum(1)                                                   # [B,K]
        conds = torch.einsum("blk,bld->bkd", onehot, e) / count.clamp(min=1).unsqueeze(-1)
        return Conditions(conds, count > 0, count)


# ---------------------------------------------------------------------------
# 4. Condition association + loss (Eqs. 8-9)
# ---------------------------------------------------------------------------

class StreamingFrequencyEstimator:
    """Item sampling-probability estimate for logQ correction (Yi et al. 2019, ref. [36]).

    Keeps, per hash bucket, the last step an item was seen (A) and a moving
    average of the gap between occurrences (B); p = 1/B. Bucket = id % size.
    """

    def __init__(self, num_buckets: int = 100_000, alpha: float = 0.05, init_gap: float = 1000.0):
        self.alpha, self.size = alpha, num_buckets
        self.last = np.zeros(num_buckets, dtype=np.float64)
        self.gap = np.full(num_buckets, init_gap, dtype=np.float64)
        self.t = 0

    def update(self, item_ids: torch.Tensor) -> None:
        for i in item_ids.tolist():
            self.t += 1
            h = i % self.size
            if self.last[h] > 0:
                self.gap[h] = (1 - self.alpha) * self.gap[h] + self.alpha * (self.t - self.last[h])
            self.last[h] = self.t

    def prob(self, item_ids: torch.Tensor) -> torch.Tensor:
        h = (item_ids % self.size).cpu().numpy()
        return torch.from_numpy(1.0 / self.gap[h]).float().to(item_ids.device)


def select_embedding(user_embs: torch.Tensor, cond_mask: torch.Tensor, pos_emb: torch.Tensor,
                     mode: str = "argmax", tau: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Implicit condition association (Eq. 8). Returns (selected [B,d], index [B]).

    "argmax":  hard pick j* = argmax_j o_u^j . o_y; only that embedding gets gradient
               (fine for DCM, whose conditions are already diverse by construction).
    "gumbel":  Straight-Through Gumbel-Softmax: forward is the one-hot pick, backward
               flows to every embedding -- the paper's fix for collapse in the
               self-attention / interest-token baselines.
    """
    scores = torch.einsum("bkd,bd->bk", user_embs, pos_emb).masked_fill(~cond_mask, float("-inf"))
    idx = scores.argmax(1)
    if mode == "argmax":
        return user_embs[torch.arange(user_embs.size(0)), idx], idx
    if mode != "gumbel":
        raise ValueError(mode)
    safe = scores.masked_fill(~cond_mask, -1e9)
    w = F.gumbel_softmax(safe / tau, tau=1.0, hard=True)
    return torch.einsum("bk,bkd->bd", w, user_embs), w.argmax(1)


def sampled_softmax_logq_loss(user_emb: torch.Tensor, pos_emb: torch.Tensor, pos_ids: torch.Tensor,
                              pos_logp: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Eq. (9): in-batch sampled softmax with logQ correction.

    Logit(u, k) = o_u . o_k / T - log p_k for every in-batch target k; the
    positive sits on the diagonal. Other rows with the same item id are masked
    (they would be false negatives). `user_emb` is the single embedding picked
    by condition association; negatives never need the other K-1 embeddings.
    """
    logits = user_emb @ pos_emb.t() / temperature - pos_logp.unsqueeze(0)
    same = pos_ids.unsqueeze(0) == pos_ids.unsqueeze(1)
    off_diag_dup = same & ~torch.eye(len(pos_ids), dtype=torch.bool)
    logits = logits.masked_fill(off_diag_dup, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(pos_ids)))


# ---------------------------------------------------------------------------
# 5. The two models
# ---------------------------------------------------------------------------

def make_condition_builder(kind: str, item_feat_dim: int, d: int, k: int, **kw) -> nn.Module:
    kind = kind.lower()
    if kind == "dcm":
        return DifferentiableClusteringModule(item_feat_dim, d, k, init="vafpi", single_assignment=True, **kw)
    if kind == "mind":
        return DifferentiableClusteringModule(item_feat_dim, d, k, init="gaussian", single_assignment=False,
                                              use_validity=False, **kw)
    if kind == "self_attention":
        return SelfAttentionConditions(item_feat_dim, d, k)
    if kind == "interest_token":
        return InterestTokenConditions(item_feat_dim, d, k)
    raise ValueError(f"unknown implicit model {kind!r} (use dcm|mind|self_attention|interest_token, "
                     "or build ConceptPoolConditions directly)")


class ImplicitInterestModel(nn.Module):
    """History -> K conditions -> K user embeddings; trained with argmax/Gumbel association."""

    def __init__(self, item_feat_dim: int, user_feat_dim: int, d: int = 32, n_interests: int = 4,
                 builder: Optional[nn.Module] = None, kind: str = "dcm", temperature: float = 0.07,
                 select: Optional[str] = None, **builder_kw):
        super().__init__()
        self.d, self.k, self.temperature = d, n_interests, temperature
        self.builder = builder or make_condition_builder(kind, item_feat_dim, d, n_interests, **builder_kw)
        self.user_tower = UserTower(user_feat_dim, d)
        self.item_tower = ItemTower(item_feat_dim, d)
        # DCM/MIND are diverse by construction; attention/token baselines need ST-Gumbel (Sec. 3.2.4).
        self.select = select or ("argmax" if isinstance(self.builder, (DifferentiableClusteringModule,
                                                                         ConceptPoolConditions)) else "gumbel")

    def user_embeddings(self, hist_feats, hist_mask, user_feats, hist_valid=None) -> Tuple[torch.Tensor, Conditions]:
        cond = self.builder(hist_feats, hist_mask, hist_valid)
        return self.user_tower(user_feats, cond.conds), cond

    def loss(self, hist_feats, hist_mask, user_feats, pos_feats, pos_ids, pos_logp, hist_valid=None):
        embs, cond = self.user_embeddings(hist_feats, hist_mask, user_feats, hist_valid)
        pos = self.item_tower(pos_feats)
        sel, _ = select_embedding(embs, cond.mask, pos, self.select)
        return sampled_softmax_logq_loss(sel, pos, pos_ids, pos_logp, self.temperature)


class ExplicitInterestModel(nn.Module):
    """Conditional Retrieval: condition = followed-topic embedding; association done at logging time."""

    def __init__(self, item_feat_dim: int, user_feat_dim: int, n_topics: int, d: int = 32,
                 temperature: float = 0.07):
        super().__init__()
        self.d, self.temperature = d, temperature
        self.topic_embedding = nn.Embedding(n_topics, d)
        self.user_tower = UserTower(user_feat_dim, d)
        self.item_tower = ItemTower(item_feat_dim, d)

    def user_embeddings(self, user_feats: torch.Tensor, topic_ids: torch.Tensor) -> torch.Tensor:
        """topic_ids [B] or [B,K] -> [B,K,d]."""
        if topic_ids.dim() == 1:
            topic_ids = topic_ids.unsqueeze(1)
        return self.user_tower(user_feats, self.topic_embedding(topic_ids))

    def loss(self, user_feats, topic_ids, pos_feats, pos_ids, pos_logp):
        u = self.user_embeddings(user_feats, topic_ids)[:, 0]
        return sampled_softmax_logq_loss(u, self.item_tower(pos_feats), pos_ids, pos_logp, self.temperature)


# ---------------------------------------------------------------------------
# 6. Serving: budgets, round-robin merge, filters, overlap  (Sec. 3.5)
# ---------------------------------------------------------------------------

def allocate_budget(weights: Sequence[float], total: int) -> List[int]:
    """Split `total` retrieval slots across embeddings proportionally to `weights`
    (largest-remainder rounding, sums to exactly `total`). Zero-weight embeddings get 0."""
    w = np.clip(np.asarray(weights, dtype=np.float64), 0, None)
    if total <= 0 or w.sum() <= 0:
        return [0] * len(w)
    raw = w / w.sum() * total
    base = np.floor(raw).astype(int)
    rem = total - int(base.sum())
    order = np.argsort(-(raw - base), kind="stable")
    for j in order[:rem]:
        base[j] += 1
    return base.tolist()


def round_robin_merge(lists: Sequence[Sequence[int]], total: int) -> List[int]:
    """Interleave ranked lists (best of each first), dropping duplicates, up to `total` items."""
    out, seen, pos = [], set(), [0] * len(lists)
    while len(out) < total and any(p < len(l) for p, l in zip(pos, lists)):
        for j, l in enumerate(lists):
            while pos[j] < len(l) and l[pos[j]] in seen:
                pos[j] += 1
            if pos[j] < len(l):
                out.append(l[pos[j]])
                seen.add(l[pos[j]])
                pos[j] += 1
                if len(out) == total:
                    break
    return out


class ItemIndex:
    """Exact inner-product index standing in for the ANN service."""

    def __init__(self, embeddings: torch.Tensor):
        self.emb = embeddings

    def search(self, q: torch.Tensor, k: int) -> List[List[int]]:
        """q [n,d] -> n ranked id lists of length <= k."""
        k = min(k, self.emb.size(0))
        return (q @ self.emb.t()).topk(k, dim=1).indices.tolist() if k > 0 else [[] for _ in range(q.size(0))]


def retrieve_multi_embedding(user_embs: torch.Tensor, budgets: Sequence[int], index: ItemIndex,
                             total: int) -> List[int]:
    """One ANN lookup per embedding with its own budget, then round-robin merge with dedup."""
    keep = [j for j, b in enumerate(budgets) if b > 0]
    if not keep:
        return []
    lists = [l[:budgets[j]] for j, l in zip(keep, index.search(user_embs[keep], max(budgets[j] for j in keep)))]
    return round_robin_merge(lists, total)


def filter_by_topic(item_ids: Sequence[int], item_topics: Sequence[int], allowed_topics) -> List[int]:
    """Explicit relevance filter (Sec. 3.3.3): keep items whose topic signals match the condition."""
    allowed = set(allowed_topics)
    return [i for i in item_ids if item_topics[i] in allowed]


def candidate_overlap(a: Sequence[int], b: Sequence[int]) -> float:
    """|A ∩ B| / |A ∪ B| -- the paper reports only 3.2% between implicit and explicit candidates."""
    sa, sb = set(a), set(b)
    return len(sa & sb) / max(len(sa | sb), 1)


# ---------------------------------------------------------------------------
# 7. Evaluation
# ---------------------------------------------------------------------------

def multi_embedding_scores(user_embs: torch.Tensor, mask: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
    """Item score = max over a user's valid embeddings ([3, 12]). [B,K,d] x [N,d] -> [B,N]."""
    s = torch.einsum("bkd,nd->bkn", user_embs, item_embs).masked_fill(~mask.unsqueeze(-1), float("-inf"))
    return s.max(1).values


def hit_rate(scores: torch.Tensor, pos_ids: torch.Tensor, ks: Sequence[int]) -> Dict[int, float]:
    """HR@k: positive's rank among the corpus (rank = #items scoring strictly higher + 1)."""
    pos = scores.gather(1, pos_ids.unsqueeze(1))
    rank = (scores > pos).sum(1) + 1
    return {k: (rank <= k).float().mean().item() for k in ks}


def embedding_diversity(embs: torch.Tensor, mask: torch.Tensor) -> float:
    """Mean pairwise cosine among a user's valid embeddings (lower = more diverse; 1 = collapsed)."""
    sims = torch.einsum("bkd,bjd->bkj", embs, embs)
    pair = mask.unsqueeze(2) & mask.unsqueeze(1) & ~torch.eye(embs.size(1), dtype=torch.bool)
    return (sims * pair).sum().item() / max(pair.sum().item(), 1)


# ---------------------------------------------------------------------------
# 8. Synthetic world (stands in for Pinterest data)
# ---------------------------------------------------------------------------

class SyntheticWorld:
    """Topics with prototype feature vectors; Zipf popularity inside each topic.

    A user has `n_hist` history interests with skewed weights (0.6/0.3/0.1 for 3)
    that drive the engagement sequence, plus followed topics: one that also
    appears in the history and one that does NOT (a "forgotten" long-term or new
    interest only explicit modeling can recover). Retrieval targets are drawn
    uniformly over a user's interests, so torso/tail interests matter as much as
    the dominant one -- which is exactly what a single embedding fails at.
    ~5% of items have invalid (out-of-distribution) features.
    """

    def __init__(self, n_topics: int = 12, items_per_topic: int = 40, item_feat_dim: int = 16,
                 user_feat_dim: int = 8, hist_len: int = 30, n_hist: int = 3, noise: float = 0.5,
                 invalid_frac: float = 0.05, seed: int = 0):
        self.rng = np.random.RandomState(seed)
        self.T, self.per, self.Fi, self.Fu, self.L, self.n_hist = n_topics, items_per_topic, item_feat_dim, \
            user_feat_dim, hist_len, n_hist
        proto = self.rng.randn(n_topics, item_feat_dim)
        self.item_topic = np.repeat(np.arange(n_topics), items_per_topic)
        self.n_items = len(self.item_topic)
        feats = proto[self.item_topic] + noise * self.rng.randn(self.n_items, item_feat_dim)
        self.item_valid = self.rng.rand(self.n_items) > invalid_frac
        feats[~self.item_valid] = 5.0 * self.rng.randn((~self.item_valid).sum(), item_feat_dim)
        self.item_feats = torch.tensor(feats, dtype=torch.float32)
        self.valid_t = torch.tensor(self.item_valid)
        zipf = 1.0 / (1.0 + np.arange(items_per_topic))
        self.pop_in_topic = zipf / zipf.sum()
        self.hist_weights = np.array([0.6, 0.3, 0.1] if n_hist == 3 else
                                     np.linspace(1, 0.2, n_hist) / np.linspace(1, 0.2, n_hist).sum())

    def _item_from_topic(self, topics: np.ndarray) -> np.ndarray:
        return topics * self.per + self.rng.choice(self.per, size=topics.shape, p=self.pop_in_topic)

    def sample_users(self, n: int) -> Dict[str, torch.Tensor]:
        hist_topics = np.argsort(self.rng.rand(n, self.T), axis=1)[:, :self.n_hist]   # distinct topics
        pick = self.rng.choice(self.n_hist, size=(n, self.L), p=self.hist_weights)
        seq_topics = np.take_along_axis(hist_topics, pick, 1)
        hist_ids = self._item_from_topic(seq_topics.reshape(-1)).reshape(n, self.L)
        in_hist = np.zeros((n, self.T), dtype=bool)
        np.put_along_axis(in_hist, hist_topics, True, axis=1)
        followed = np.stack([
            hist_topics[np.arange(n), self.rng.randint(0, self.n_hist, n)],      # also in history
            np.where(in_hist, -1.0, self.rng.rand(n, self.T)).argmax(1),         # NOT in history
        ], axis=1)
        return {
            "hist_ids": torch.tensor(hist_ids), "hist_topics": torch.tensor(hist_topics),
            "followed": torch.tensor(followed),
            "user_feats": torch.tensor(self.rng.randn(n, self.Fu), dtype=torch.float32),
        }

    def hist_inputs(self, users) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = users["hist_ids"]
        return self.item_feats[ids], torch.ones_like(ids, dtype=torch.bool), self.valid_t[ids]

    def implicit_targets(self, users) -> torch.Tensor:
        """Uniform over the user's history interests (tail matters as much as head)."""
        n = users["hist_ids"].size(0)
        j = self.rng.randint(0, self.n_hist, n)
        topics = users["hist_topics"].numpy()[np.arange(n), j]
        return torch.tensor(self._item_from_topic(topics))

    def explicit_samples(self, users) -> Tuple[torch.Tensor, torch.Tensor]:
        """(condition topic, item): logging-time association -- item comes from the followed topic."""
        n = users["hist_ids"].size(0)
        j = self.rng.randint(0, 2, n)
        topics = users["followed"].numpy()[np.arange(n), j]
        return torch.tensor(topics), torch.tensor(self._item_from_topic(topics))

    def joint_targets(self, users) -> torch.Tensor:
        """Uniform over history interests AND followed topics (the full set of user interests)."""
        n = users["hist_ids"].size(0)
        all_topics = np.concatenate([users["hist_topics"].numpy(), users["followed"].numpy()], 1)
        topics = all_topics[np.arange(n), self.rng.randint(0, all_topics.shape[1], n)]
        return torch.tensor(self._item_from_topic(topics))


# ---------------------------------------------------------------------------
# 9. Training / evaluation helpers + demo
# ---------------------------------------------------------------------------

def train_implicit(model: ImplicitInterestModel, world: SyntheticWorld, steps: int = 300, batch: int = 256,
                   lr: float = 2e-3, log_every: int = 0) -> List[float]:
    est, opt, losses = StreamingFrequencyEstimator(world.n_items * 2), torch.optim.Adam(model.parameters(), lr=lr), []
    for s in range(steps):
        users = world.sample_users(batch)
        pos = world.implicit_targets(users)
        est.update(pos)
        hf, hm, hv = world.hist_inputs(users)
        loss = model.loss(hf, hm, users["user_feats"], world.item_feats[pos], pos,
                          est.prob(pos).clamp(min=1e-6).log(), hv)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if log_every and (s + 1) % log_every == 0:
            print(f"  implicit step {s + 1:4d}  loss {loss.item():.3f}")
    return losses


def train_explicit(model: ExplicitInterestModel, world: SyntheticWorld, steps: int = 300, batch: int = 256,
                   lr: float = 2e-3, log_every: int = 0) -> List[float]:
    est, opt, losses = StreamingFrequencyEstimator(world.n_items * 2), torch.optim.Adam(model.parameters(), lr=lr), []
    for s in range(steps):
        users = world.sample_users(batch)
        topics, pos = world.explicit_samples(users)
        est.update(pos)
        loss = model.loss(users["user_feats"], topics, world.item_feats[pos], pos,
                          est.prob(pos).clamp(min=1e-6).log())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if log_every and (s + 1) % log_every == 0:
            print(f"  explicit step {s + 1:4d}  loss {loss.item():.3f}")
    return losses


@torch.no_grad()
def eval_implicit(model: ImplicitInterestModel, world: SyntheticWorld, users, targets,
                  ks=(10, 50)) -> Dict[int, float]:
    model.eval()
    hf, hm, hv = world.hist_inputs(users)
    embs, cond = model.user_embeddings(hf, hm, users["user_feats"], hv)
    scores = multi_embedding_scores(embs, cond.mask, model.item_tower(world.item_feats))
    return hit_rate(scores, targets, ks)


@torch.no_grad()
def explicit_user_embeddings(model: ExplicitInterestModel, users) -> Tuple[torch.Tensor, torch.Tensor]:
    """One embedding per followed topic: [B,2,d] and an all-true mask."""
    embs = model.user_embeddings(users["user_feats"], users["followed"])
    return embs, torch.ones(embs.shape[:2], dtype=torch.bool)


@torch.no_grad()
def eval_joint(implicit: ImplicitInterestModel, explicit: ExplicitInterestModel, world: SyntheticWorld,
               users, targets, ks=(10, 50)) -> Dict[str, Dict[int, float]]:
    """HR when scoring with implicit embeddings only, explicit only, and both together."""
    implicit.eval()
    explicit.eval()
    hf, hm, hv = world.hist_inputs(users)
    ie, icond = implicit.user_embeddings(hf, hm, users["user_feats"], hv)
    ee, emask = explicit_user_embeddings(explicit, users)
    i_scores = multi_embedding_scores(ie, icond.mask, implicit.item_tower(world.item_feats))
    e_scores = multi_embedding_scores(ee, emask, explicit.item_tower(world.item_feats))
    return {"implicit": hit_rate(i_scores, targets, ks), "explicit": hit_rate(e_scores, targets, ks),
            # Each source has its own item tower, so combine by taking the better rank per source:
            # an item is "covered" if either retriever puts it in its top-k.
            "both": _union_hit_rate(i_scores, e_scores, targets, ks)}


def _union_hit_rate(a: torch.Tensor, b: torch.Tensor, pos_ids: torch.Tensor, ks) -> Dict[int, float]:
    """Hit if the positive is in a's top-k/2 or b's top-k/2 -- equal budget to a single retriever's top-k."""
    out = {}
    for k in ks:
        h = k // 2
        ra = (a > a.gather(1, pos_ids[:, None])).sum(1) + 1
        rb = (b > b.gather(1, pos_ids[:, None])).sum(1) + 1
        out[k] = ((ra <= h) | (rb <= h)).float().mean().item()
    return out


def main(steps: int = 300, seed: int = 0) -> Dict[str, Dict]:
    torch.manual_seed(seed)
    world = SyntheticWorld(seed=seed)
    test_users = world.sample_users(1000)
    imp_targets = world.implicit_targets(test_users)
    joint_targets = world.joint_targets(test_users)
    results: Dict[str, Dict] = {}

    print("== implicit interest modeling (HR over a 480-item corpus; targets uniform over 3 history interests) ==")
    for name, kind, k in [("single embedding (K=1)", "dcm", 1), ("DCM (K=3)", "dcm", 3),
                          ("DCM (K=4)", "dcm", 4), ("MIND capsule (K=4)", "mind", 4),
                          ("self-attention (K=4)", "self_attention", 4)]:
        torch.manual_seed(seed)
        m = ImplicitInterestModel(world.Fi, world.Fu, d=32, n_interests=k, kind=kind)
        train_implicit(m, world, steps)
        results[name] = eval_implicit(m, world, test_users, imp_targets)
        print(f"  {name:24s} HR@10 {results[name][10]:.3f}   HR@50 {results[name][50]:.3f}")

    print("== synergy: targets uniform over history interests + followed topics ==")
    torch.manual_seed(seed)
    imp = ImplicitInterestModel(world.Fi, world.Fu, d=32, n_interests=4, kind="dcm")
    train_implicit(imp, world, steps)
    exp = ExplicitInterestModel(world.Fi, world.Fu, world.T, d=32)
    train_explicit(exp, world, steps)
    joint = eval_joint(imp, exp, world, test_users, joint_targets)
    for name, hr in joint.items():
        print(f"  {name:10s} HR@10 {hr[10]:.3f}   HR@50 {hr[50]:.3f}")
    results["joint"] = joint
    return results


if __name__ == "__main__":
    main()
