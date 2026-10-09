"""Real-time cluster-based hard negative sampling for two-tower retrieval (Cluster GOOBS).

Reference implementation of "Real-Time Hard Negative Sampling via LLM-based
Clustering for Large-Scale Two-Tower Retrieval" (Ji, Hu, Zhao, Huang, Zhang,
Fan, Singh; Meta; OARS @ RecSys 2026; arXiv 2607.00448).

Two-tower training with in-batch negatives (+LogQ) gives easy negatives: most
negatives are semantically far from the positive, produce near-zero gradients
and the model quickly stops learning from them. The paper draws extra negatives
from the SAME SEMANTIC CLUSTER as each positive item, which are hard by
construction (Sec. 3.2-3.3), and makes it deployable with a real-time item pool:

  * Clusters (Sec. 3.4): k-means over LLM-derived multimodal content embeddings;
    new items are assigned online to the nearest centroid. (`ContentClusterer`;
    the LLM encoder itself is out of scope -- bring any content embedding.)
  * Global Out-Of-Batch Sampling, GOOBS (Sec. 4): a preallocated tensor pool of
    item features split into per-cluster *segments* of S slots. Update engine
    (Alg. 1): slot = cluster_id * S + item_id % S; a later item overwrites an
    earlier one by design (freshness + bounded memory). Sample engine (Alg. 2):
    for each in-batch example, pick a random slot inside its positive's cluster
    segment. (`ClusterItemPool`)
  * Loss (Sec. 3.2): softmax cross-entropy over {positive} U negatives.

Also implemented, as the paper's comparison set (Table 1): in-batch + LogQ +
false-negative mask (baseline), DNS, CBNS, ANCE, GOOBS (random OOB) and
Cluster GOOBS, behind one `NegativeSampler` interface; a popularity-bias report
(Table 3); and a synthetic interaction dataset standing in for MovieLens /
Amazon / the proprietary production data.

Not implemented: the LLM content encoder, distributed/GPU-resident pool, and
the production feature set. Numbers from the synthetic data do not reflect the
paper's results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Clusters (Sec. 3.4)
# ---------------------------------------------------------------------------

def kmeans(x: torch.Tensor, k: int, iters: int = 25, seed: int = 0) -> torch.Tensor:
    """Lloyd's k-means with k-means++-style spread initialisation. Returns [k, d] centroids."""
    g = torch.Generator().manual_seed(seed)
    n = x.size(0)
    c = [x[torch.randint(n, (1,), generator=g)].squeeze(0)]
    for _ in range(k - 1):
        d2 = torch.cdist(x, torch.stack(c)).min(1).values ** 2
        p = d2 / d2.sum() if d2.sum() > 0 else torch.full_like(d2, 1.0 / n)
        c.append(x[torch.multinomial(p, 1, generator=g)].squeeze(0))
    c = torch.stack(c)
    for _ in range(iters):
        assign = torch.cdist(x, c).argmin(1)
        for j in range(k):
            m = assign == j
            if m.any():
                c[j] = x[m].mean(0)
    return c


class ContentClusterer:
    """k-means over content embeddings; newly arriving items get the nearest centroid online.

    In the paper the embeddings come from a fine-tuned multimodal LLM encoder. The
    granularity k matters: too fine -> in-cluster items become effectively relevant
    (false negatives); too coarse -> negatives are not hard. They use ~300 clusters
    of >=10^4 items each for a corpus of billions.
    """

    def __init__(self, n_clusters: int, seed: int = 0):
        self.k, self.seed = n_clusters, seed
        self.centroids: Optional[torch.Tensor] = None

    def fit(self, content_emb: torch.Tensor) -> "ContentClusterer":
        self.centroids = kmeans(content_emb, self.k, seed=self.seed)
        return self

    def assign(self, content_emb: torch.Tensor) -> torch.Tensor:
        assert self.centroids is not None, "fit() first"
        return torch.cdist(content_emb, self.centroids).argmin(1)

    def sizes(self, content_emb: torch.Tensor) -> torch.Tensor:
        return torch.bincount(self.assign(content_emb), minlength=self.k)


# ---------------------------------------------------------------------------
# 2. Cluster-segmented out-of-batch item pool (Sec. 4, Algorithms 1-2)
# ---------------------------------------------------------------------------

class ClusterItemPool:
    """Preallocated item-feature tensors, one segment of `slots` per cluster.

    Layout: slot = cluster_id * slots + item_id % slots (Fig. 3). Collisions
    overwrite, so the pool stays fresh and its memory is fixed at
    n_clusters * slots rows regardless of corpus size.
    """

    def __init__(self, n_clusters: int, slots_per_cluster: int, feat_dim: int):
        self.C, self.S, self.F = n_clusters, slots_per_cluster, feat_dim
        n = n_clusters * slots_per_cluster
        self.item_ids = torch.full((n,), -1, dtype=torch.long)
        self.cluster_ids = torch.zeros(n, dtype=torch.long)
        self.feats = torch.zeros(n, feat_dim)
        self.filled = torch.zeros(n, dtype=torch.bool)

    def __len__(self) -> int:
        return self.item_ids.numel()

    def slot(self, item_ids: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        return cluster_ids * self.S + item_ids % self.S

    # -- Algorithm 1: update engine -----------------------------------------
    @torch.no_grad()
    def update(self, item_ids: torch.Tensor, cluster_ids: torch.Tensor, feats: torch.Tensor) -> None:
        s = self.slot(item_ids, cluster_ids)
        self.item_ids[s] = item_ids          # duplicate slots in one call: last write wins
        self.cluster_ids[s] = cluster_ids
        self.feats[s] = feats
        self.filled[s] = True

    def preload(self, item_ids: torch.Tensor, cluster_ids: torch.Tensor, feats: torch.Tensor) -> None:
        """Warm start from an item table so early sampling already hits (Sec. 4)."""
        self.update(item_ids, cluster_ids, feats)

    # -- Algorithm 2: sample engine -----------------------------------------
    @torch.no_grad()
    def sample_in_cluster(self, cluster_ids: torch.Tensor, n: int = 1,
                          generator: Optional[torch.Generator] = None):
        """For each cluster id draw `n` random slots in its segment.

        Returns (item_ids, feats, valid), shapes [B,n], [B,n,F], [B,n]. `valid` is
        False where the drawn slot has not been filled yet.
        """
        r = torch.randint(0, self.S, (cluster_ids.numel(), n), generator=generator)
        s = cluster_ids.view(-1, 1) * self.S + r
        return self.item_ids[s], self.feats[s], self.filled[s]

    @torch.no_grad()
    def sample_random(self, shape: Tuple[int, ...], generator: Optional[torch.Generator] = None):
        """Uniform draws over FILLED slots anywhere in the pool (the plain GOOBS negatives).

        Returns (item_ids, cluster_ids, feats, valid) with leading shape `shape`.
        """
        filled = self.filled.nonzero().squeeze(1)
        if filled.numel() == 0:
            return (torch.full(shape, -1, dtype=torch.long), torch.zeros(shape, dtype=torch.long),
                    torch.zeros(*shape, self.F), torch.zeros(shape, dtype=torch.bool))
        idx = filled[torch.randint(0, filled.numel(), shape, generator=generator)]
        return self.item_ids[idx], self.cluster_ids[idx], self.feats[idx], torch.ones(shape, dtype=torch.bool)

    def fill_fraction(self) -> float:
        return self.filled.float().mean().item()


# ---------------------------------------------------------------------------
# 3. Two-tower model
# ---------------------------------------------------------------------------

class TwoTower(nn.Module):
    """s(x, y) = <u(x), v(y)> / tau with L2-normalised towers (tau is ours; the paper writes v^T u)."""

    def __init__(self, n_users: int, n_items: int, n_clusters: int, user_profile_dim: int,
                 item_feat_dim: int, d: int = 32, tau: float = 0.1, hidden: int = 64, use_item_id: bool = True):
        super().__init__()
        self.tau, self.use_item_id = tau, use_item_id
        self.item_id_emb = nn.Embedding(n_items, 16)
        self.user_emb = nn.Embedding(n_users, d)
        self.user_mlp = nn.Sequential(nn.Linear(d + user_profile_dim, hidden), nn.ReLU(), nn.Linear(hidden, d))
        self.cluster_emb = nn.Embedding(n_clusters, 8)
        self.item_mlp = nn.Sequential(nn.Linear(item_feat_dim + 8 + 16, hidden), nn.ReLU(), nn.Linear(hidden, d))

    def encode_users(self, user_ids: torch.Tensor, profile: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.user_emb(user_ids), profile], dim=-1)
        return F.normalize(self.user_mlp(h), dim=-1)

    def encode_items(self, feats: torch.Tensor, cluster_ids: torch.Tensor,
                     item_ids: torch.Tensor) -> torch.Tensor:
        h = torch.cat([feats, self.cluster_emb(cluster_ids), self.item_id_emb(item_ids.clamp(min=0)) * float(self.use_item_id)], dim=-1)
        return F.normalize(self.item_mlp(h), dim=-1)


# ---------------------------------------------------------------------------
# 4. Negatives and the loss (Sec. 3.2)
# ---------------------------------------------------------------------------

@dataclass
class Negatives:
    """Extra negatives for one training batch.

    emb:   [B,K,d] per-example negatives, or [M,d] shared by the whole batch
    ids:   matching item ids, used to mask accidental positives (false negatives)
    valid: matching bool mask (False = empty pool slot / unusable)
    """
    emb: torch.Tensor
    ids: torch.Tensor
    valid: torch.Tensor


def contrastive_loss(user_emb: torch.Tensor, pos_emb: torch.Tensor, pos_ids: torch.Tensor,
                     tau: float, item_logq: Optional[torch.Tensor] = None,
                     extra: Optional[List[Negatives]] = None,
                     labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """L = -r_i * log( e^{s(x_i,y_i)} / (e^{s(x_i,y_i)} + sum_k e^{s(x_i, y_ik^-)}) ).

    Candidates per row: the row's own positive (diagonal), the other rows'
    positives (in-batch, with LogQ correction `- log p` and a false-negative mask
    for equal item ids), and every `Negatives` block in `extra` (the OOB / hard
    negatives). Rows with label r_i = 0 contribute no loss and are not used as
    in-batch negatives.
    """
    B = user_emb.size(0)
    r = torch.ones(B) if labels is None else labels.float()
    in_logits = user_emb @ pos_emb.t() / tau
    if item_logq is not None:
        in_logits = in_logits - item_logq.unsqueeze(0)
    eye = torch.eye(B, dtype=torch.bool)
    in_logits = in_logits.masked_fill((pos_ids.unsqueeze(0) == pos_ids.unsqueeze(1)) & ~eye, float("-inf"))
    in_logits = in_logits.masked_fill((r.unsqueeze(0) == 0) & ~eye, float("-inf"))
    blocks = [in_logits]
    for neg in extra or []:
        if neg.emb.dim() == 3:
            lg = torch.einsum("bd,bkd->bk", user_emb, neg.emb) / tau
            bad = ~neg.valid | (neg.ids == pos_ids.unsqueeze(1))
        else:
            lg = user_emb @ neg.emb.t() / tau
            bad = (~neg.valid).unsqueeze(0) | (neg.ids.unsqueeze(0) == pos_ids.unsqueeze(1))
        blocks.append(lg.masked_fill(bad, float("-inf")))
    logits = torch.cat(blocks, dim=1)
    nll = F.cross_entropy(logits, torch.arange(B), reduction="none")
    return (r * nll).sum() / r.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# 5. Negative samplers (Table 1 comparison set)
# ---------------------------------------------------------------------------

class NegativeSampler:
    """Base: in-batch negatives only (the production baseline)."""

    name = "in-batch"

    def negatives(self, model: TwoTower, batch: Dict[str, torch.Tensor],
                  user_emb: torch.Tensor) -> List[Negatives]:
        return []

    def after_step(self, model: TwoTower, batch: Dict[str, torch.Tensor], pos_emb: torch.Tensor) -> None:
        pass


class ItemCatalog:
    """All items' features/clusters; used by samplers that need the whole corpus (DNS, ANCE, preload)."""

    def __init__(self, feats: torch.Tensor, cluster_ids: torch.Tensor):
        self.feats, self.cluster_ids = feats, cluster_ids
        self.n = feats.size(0)

    def encode(self, model: TwoTower, ids: torch.Tensor) -> torch.Tensor:
        return model.encode_items(self.feats[ids], self.cluster_ids[ids], ids)


class DNSSampler(NegativeSampler):
    """Dynamic Negative Sampling: from a random candidate set keep the items the CURRENT model
    scores highest for this user (the items it wrongly believes are most relevant)."""

    name = "DNS"

    def __init__(self, catalog: ItemCatalog, n_candidates: int = 64, k: int = 4):
        self.cat, self.C, self.k = catalog, n_candidates, k

    def negatives(self, model, batch, user_emb):
        B = user_emb.size(0)
        cand = torch.randint(0, self.cat.n, (B, self.C))
        with torch.no_grad():
            ce = self.cat.encode(model, cand.reshape(-1)).view(B, self.C, -1)
            scores = torch.einsum("bd,bcd->bc", user_emb, ce).masked_fill(cand == batch["items"].unsqueeze(1), -2)
            top = scores.topk(self.k, dim=1).indices
        ids = cand.gather(1, top)
        emb = self.cat.encode(model, ids.reshape(-1)).view(B, self.k, -1)   # re-encode WITH grad
        return [Negatives(emb, ids, torch.ones_like(ids, dtype=torch.bool))]


class CBNSSampler(NegativeSampler):
    """Cross-Batch Negative Sampling: reuse (detached) item embeddings of recent mini-batches.
    Chosen by recency, not by semantics."""

    name = "CBNS"

    def __init__(self, queue_size: int = 1024):
        self.size = queue_size
        self.emb: Optional[torch.Tensor] = None
        self.ids: Optional[torch.Tensor] = None

    def negatives(self, model, batch, user_emb):
        if self.emb is None:
            return []
        return [Negatives(self.emb, self.ids, torch.ones_like(self.ids, dtype=torch.bool))]

    def after_step(self, model, batch, pos_emb):
        e, i = pos_emb.detach(), batch["items"]
        self.emb = e if self.emb is None else torch.cat([self.emb, e])[-self.size:]
        self.ids = i if self.ids is None else torch.cat([self.ids, i])[-self.size:]


class ANCESampler(NegativeSampler):
    """ANCE: a global index of item embeddings, asynchronously refreshed; negatives are the
    user's nearest items in the (stale) index. Geometrically hardest, but needs the global index."""

    name = "ANCE"

    def __init__(self, catalog: ItemCatalog, refresh_every: int = 50, k: int = 4, skip: int = 0):
        self.cat, self.refresh, self.k, self.skip = catalog, refresh_every, k, skip
        self.index: Optional[torch.Tensor] = None
        self.step = 0

    def _rebuild(self, model):
        with torch.no_grad():
            self.index = self.cat.encode(model, torch.arange(self.cat.n))

    def negatives(self, model, batch, user_emb):
        if self.index is None or self.step % self.refresh == 0:
            self._rebuild(model)
        with torch.no_grad():
            s = (user_emb @ self.index.t()).scatter(1, batch["items"].unsqueeze(1), -2.0)
            ids = s.topk(self.k + self.skip, dim=1).indices[:, self.skip:]
        emb = self.cat.encode(model, ids.reshape(-1)).view(ids.size(0), self.k, -1)
        return [Negatives(emb, ids, torch.ones_like(ids, dtype=torch.bool))]

    def after_step(self, model, batch, pos_emb):
        self.step += 1


class GOOBSampler(NegativeSampler):
    """Global Out-of-Batch Sampling: `n_random` uniformly random items from the maintained pool
    for every example. The pool is refreshed with each batch's items (Algorithm 1)."""

    name = "GOOBS"

    def __init__(self, pool: ClusterItemPool, n_random: int = 16):
        self.pool, self.n_random = pool, n_random

    def _random_block(self, model: TwoTower, B: int) -> Negatives:
        ids, cl, feats, valid = self.pool.sample_random((B, self.n_random))
        return Negatives(model.encode_items(feats, cl, ids), ids, valid)

    def negatives(self, model, batch, user_emb):
        return [self._random_block(model, user_emb.size(0))]

    def after_step(self, model, batch, pos_emb):
        self.pool.update(batch["items"], batch["item_clusters"], batch["item_feats"])


class ClusterGOOBSampler(GOOBSampler):
    """Cluster GOOBS: random OOB negatives PLUS `n_cluster` hard negatives per example drawn
    from the positive's own cluster segment (Algorithm 2).

    `random_per_cluster` is the paper's cluster:random mixing ratio (1:15 on MovieLens,
    1:31 on Amazon): each example gets n_cluster in-cluster and
    random_per_cluster * n_cluster random OOB negatives.
    """

    name = "Cluster GOOBS"

    def __init__(self, pool: ClusterItemPool, n_cluster: int = 1, random_per_cluster: int = 15):
        super().__init__(pool, n_random=n_cluster * random_per_cluster)
        self.n_cluster = n_cluster

    def negatives(self, model, batch, user_emb):
        B = user_emb.size(0)
        clusters = batch["item_clusters"]
        ids, feats, valid = self.pool.sample_in_cluster(clusters, self.n_cluster)
        cl = clusters.unsqueeze(1).expand(B, self.n_cluster)
        hard = Negatives(model.encode_items(feats, cl, ids), ids, valid)
        return [hard, self._random_block(model, B)]


def make_sampler(name: str, catalog: ItemCatalog, pool: Optional[ClusterItemPool] = None, **kw) -> NegativeSampler:
    name = name.lower()
    if name in ("in-batch", "baseline"):
        return NegativeSampler()
    if name == "dns":
        return DNSSampler(catalog, **kw)
    if name == "cbns":
        return CBNSSampler(**kw)
    if name == "ance":
        return ANCESampler(catalog, **kw)
    if name == "goobs":
        return GOOBSampler(pool, **kw)
    if name in ("cluster_goobs", "cluster goobs"):
        return ClusterGOOBSampler(pool, **kw)
    raise ValueError(f"unknown sampler {name!r}")


# ---------------------------------------------------------------------------
# 6. Synthetic interaction data (stands in for MovieLens / Amazon / production logs)
# ---------------------------------------------------------------------------

class SyntheticInteractions:
    """Items live in a latent space: z_i = cluster centroid + within-cluster offset.

    P(i | u) ∝ exp(p_u . z_i + popularity_i). Coarse (cluster-level) taste is easy
    to learn; telling apart items INSIDE a cluster is the hard part and is exactly
    what in-cluster negatives train. Clusters used by the sampler are recovered by
    k-means over a noisy `content embedding` (the stand-in for the LLM encoder),
    not read off from ground truth.
    """

    def __init__(self, n_users: int = 2000, n_true_clusters: int = 150, items_per_cluster: int = 40,
                 latent_dim: int = 6, cluster_scale: float = 0.7, fine_scale: float = 1.0, feature_noise: float = 1.0,
                 pop_strength: float = 2.0, n_train: int = 30, n_test: int = 8, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.n_users, self.n_items = n_users, n_true_clusters * items_per_cluster
        self.latent_dim = latent_dim
        self.true_cluster = np.repeat(np.arange(n_true_clusters), items_per_cluster)
        mu = cluster_scale * rng.randn(n_true_clusters, latent_dim)
        z = mu[self.true_cluster] + fine_scale * rng.randn(self.n_items, latent_dim)
        pop = -pop_strength * np.log1p(rng.permutation(self.n_items) / 10.0)          # Zipf-like head/tail
        p = rng.randn(n_users, latent_dim)
        logits = p @ z.T + pop[None, :]
        probs = np.exp(logits - logits.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        inter = np.stack([rng.choice(self.n_items, n_train + n_test, replace=False, p=probs[u])
                          for u in range(n_users)])
        self.train_items, self.test_items = inter[:, :n_train], inter[:, n_train:]
        self.item_feats = torch.tensor(z + feature_noise * rng.randn(*z.shape), dtype=torch.float32)  # observed features
        self.content_emb = torch.tensor(z + 0.5 * rng.randn(*z.shape), dtype=torch.float32)       # "LLM" content embedding
        self.train_pairs = torch.tensor([(u, i) for u in range(n_users) for i in self.train_items[u]])
        counts = np.bincount(self.train_pairs[:, 1].numpy(), minlength=self.n_items) + 1.0
        self.item_logq = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32)
        self.user_profile = torch.stack([self.item_feats[torch.tensor(self.train_items[u])].mean(0)
                                         for u in range(n_users)])
        self.test_pairs = torch.tensor([(u, i) for u in range(n_users) for i in self.test_items[u]])
        self.train_mask = torch.zeros(n_users, self.n_items, dtype=torch.bool)
        self.train_mask[self.train_pairs[:, 0], self.train_pairs[:, 1]] = True
        self.relevant = torch.zeros(n_users, self.n_items, dtype=torch.bool)    # all of a user's true positives
        self.relevant[self.train_mask.nonzero()[:, 0], self.train_mask.nonzero()[:, 1]] = True
        self.relevant[self.test_pairs[:, 0], self.test_pairs[:, 1]] = True


def false_negative_rate(data: SyntheticInteractions, clusters: torch.Tensor, n_samples: int = 20000,
                        seed: int = 0) -> float:
    """P(a random same-cluster item is one the user actually interacted with) -- the false-negative
    risk that rises as clusters get finer (Sec. 3.4). Measured over (user, positive) pairs."""
    g = torch.Generator().manual_seed(seed)
    members = [(clusters == c).nonzero().squeeze(1) for c in range(int(clusters.max()) + 1)]
    pairs = data.train_pairs[torch.randint(0, len(data.train_pairs), (n_samples,), generator=g)]
    hits = 0
    for u, i in pairs.tolist():
        m = members[int(clusters[i])]
        j = m[torch.randint(0, len(m), (1,), generator=g)].item()
        hits += int(j != i and data.relevant[u, j])
    return hits / n_samples


# ---------------------------------------------------------------------------
# 7. Training / evaluation
# ---------------------------------------------------------------------------

def build_pool(data: SyntheticInteractions, clusters: torch.Tensor, n_clusters: int,
               slots: int = 20, preload: bool = True) -> ClusterItemPool:
    pool = ClusterItemPool(n_clusters, slots, data.item_feats.size(1))
    if preload:
        pool.preload(torch.arange(data.n_items), clusters, data.item_feats)
    return pool


def train(model: TwoTower, data: SyntheticInteractions, clusters: torch.Tensor, sampler: NegativeSampler,
          steps: int = 1000, batch_size: int = 128, lr: float = 3e-3, seed: int = 0,
          log_every: int = 0) -> List[float]:
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, n = [], len(data.train_pairs)
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=g)
        users, items = data.train_pairs[idx, 0], data.train_pairs[idx, 1]
        batch = {"users": users, "items": items, "item_clusters": clusters[items],
                 "item_feats": data.item_feats[items]}
        u = model.encode_users(users, data.user_profile[users])
        v = model.encode_items(batch["item_feats"], batch["item_clusters"], items)
        loss = contrastive_loss(u, v, items, model.tau, data.item_logq[items],
                                sampler.negatives(model, batch, u))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sampler.after_step(model, batch, v)
        losses.append(loss.item())
        if log_every and (step + 1) % log_every == 0:
            print(f"  [{sampler.name}] step {step + 1:4d}  loss {loss.item():.3f}")
    return losses


@torch.no_grad()
def retrieval_scores(model: TwoTower, data: SyntheticInteractions, clusters: torch.Tensor) -> torch.Tensor:
    """Full-corpus scores [n_users, n_items] with each user's training items masked out."""
    model.eval()
    u = model.encode_users(torch.arange(data.n_users), data.user_profile)
    v = model.encode_items(data.item_feats, clusters, torch.arange(data.n_items))
    return (u @ v.t()).masked_fill(data.train_mask, float("-inf"))


def evaluate(model: TwoTower, data: SyntheticInteractions, clusters: torch.Tensor,
             ks=(50, 100)) -> Dict[int, float]:
    """Global HR@K: fraction of held-out interactions ranked in the user's top-K over the WHOLE
    corpus (no sampled negatives, no k-core filtering -- the paper's protocol)."""
    scores = retrieval_scores(model, data, clusters)
    users, items = data.test_pairs[:, 0], data.test_pairs[:, 1]
    rank = (scores[users] > scores[users, items].unsqueeze(1)).sum(1) + 1
    return {k: (rank <= k).float().mean().item() for k in ks}


def popularity_report(scores: torch.Tensor, top_k: int = 50, min_impressions: int = 20,
                      head: int = 20) -> Dict[str, float]:
    """Table 3 analogue. Treat each user's top-K as the impressions it would receive:
    `head_share` = share of all impressions going to the `head` most-exposed items;
    `items_over_threshold` = how many distinct items got >= `min_impressions`."""
    top = scores.topk(top_k, dim=1).indices.reshape(-1)
    counts = torch.bincount(top, minlength=scores.size(1)).float()
    return {"head_share": (counts.sort(descending=True).values[:head].sum() / counts.sum()).item(),
            "items_over_threshold": float((counts >= min_impressions).sum().item()),
            "coverage": float((counts > 0).float().mean().item())}


# ---------------------------------------------------------------------------
# 8. Demo
# ---------------------------------------------------------------------------

def run_method(name: str, data: SyntheticInteractions, clusters: torch.Tensor, n_clusters: int,
               steps: int = 1000, seed: int = 0, **kw) -> Tuple[TwoTower, Dict[int, float]]:
    torch.manual_seed(seed)
    model = TwoTower(data.n_users, data.n_items, n_clusters, data.item_feats.size(1), data.item_feats.size(1))
    catalog = ItemCatalog(data.item_feats, clusters)
    pool = build_pool(data, clusters, n_clusters) if name in ("goobs", "cluster_goobs") else None
    sampler = make_sampler(name, catalog, pool, **kw)
    train(model, data, clusters, sampler, steps=steps, seed=seed)
    return model, evaluate(model, data, clusters)


def main(steps: int = 1000, seed: int = 0) -> Dict[str, Dict[int, float]]:
    data = SyntheticInteractions(seed=seed)
    n_clusters = 150
    clusters = ContentClusterer(n_clusters, seed=seed).fit(data.content_emb).assign(data.content_emb)
    sizes = torch.bincount(clusters, minlength=n_clusters)
    print(f"{data.n_users} users, {data.n_items} items, {n_clusters} k-means clusters "
          f"(size min/median/max = {sizes.min().item()}/{int(sizes.median())}/{sizes.max().item()})")
    print(f"false-negative rate of a same-cluster draw: {false_negative_rate(data, clusters):.4f}")
    configs = [("in-batch", {}), ("dns", {}), ("cbns", {}), ("ance", {}), ("goobs", dict(n_random=16)),
               ("cluster_goobs", dict(n_cluster=1, random_per_cluster=15))]
    results = {}
    for name, kw in configs:
        model, hr = run_method(name, data, clusters, n_clusters, steps, seed, **kw)
        results[name] = hr
        base = results["in-batch"]
        rel = " ".join(f"{(hr[k] / base[k] - 1) * 100:+6.1f}%" for k in (50, 100))
        pop = popularity_report(retrieval_scores(model, data, clusters))
        print(f"  {name:14s} HR@50 {hr[50]:.4f}  HR@100 {hr[100]:.4f}   vs in-batch: {rel}   "
              f"top-20 share {pop['head_share']:.3f}  coverage {pop['coverage']:.3f}")
    return results


if __name__ == "__main__":
    main()
