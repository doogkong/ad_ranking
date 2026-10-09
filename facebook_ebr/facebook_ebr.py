"""Embedding-based retrieval (EBR) for a personalized social search engine.

Reference implementation of "Embedding-based Retrieval in Facebook Search"
(Huang, Sharma, Sun, Xia, Zhang, Pronin, Padmanabhan, Ottaviano, Yang;
KDD 2020; arXiv 2006.11632).

Facebook search is personalized: the same query "john smith" means different
people to different searchers, so embeddings must encode the searcher's social
context and location, not just text. The paper covers the full stack, all of
which is implemented here at small scale:

  Modeling (Secs. 2-3, 6)
    * Unified embedding model: query tower over (text n-grams, searcher location,
      searcher social features) and document tower over (text, location, social
      cluster); cosine similarity; triplet loss with margin (Eq. 2).
    * Feature engineering: character n-grams (+ hashed word n-grams), location
      and weighted multi-hot social features.
    * Training-data mining: random vs non-click-impression negatives, click vs
      impression positives (`neg="random" | "non_click"`).
    * Hard negative mining: online (in-batch hardest), offline (rank-range
      selection over the whole index), mixed easy/hard blending, hard -> easy
      transfer. Hard positive mining from failed search sessions.
    * Embedding ensemble: weighted concatenation (Eqs. 4-6) and cascade models.
  Serving (Sec. 4)
    * ANN: IVF and IMI coarse quantization, product quantization (PQ), OPQ and
      PCA transforms; recall vs. scanned-documents tuning; pq_bytes = d/4.
    * Unicorn-style integration: embeddings as first-class index terms (coarse
      cluster = term, PQ residual = payload), an `nn` operator with radius or
      top-K mode inside Boolean queries -> hybrid retrieval.
    * Query and index selection.
  Later-stage optimization (Sec. 5)
    * Embedding similarity as a ranking feature (cosine / Hadamard / raw), and
      a human-rating feedback loop that filters irrelevant EBR results.

Not implemented: Faiss/Unicorn themselves (re-implemented minimally), the social-graph embedding
model, real data, GPU training. The synthetic search log at the top is ours; its
numbers do not reflect Facebook's results.
"""

from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Text features: character n-grams and hashed word n-grams (Sec. 3)
# ---------------------------------------------------------------------------

def char_ngrams(text: str, n: int = 3) -> List[str]:
    t = f"#{text}#"
    return [t[i:i + n] for i in range(len(t) - n + 1)]


def word_ngrams(text: str) -> List[str]:
    w = text.split()
    return w + [" ".join(w[i:i + 2]) for i in range(len(w) - 1)]


def hash_id(token: str, buckets: int, salt: str = "") -> int:
    """Stable hash into [1, buckets]; 0 is reserved for padding. Word n-gram cardinality is huge
    (352M query trigram-words in the paper), hence hashing despite collisions."""
    return zlib.crc32((salt + token).encode()) % buckets + 1


def text_feature_ids(text: str, buckets: int = 4096, use_word_ngrams: bool = True) -> List[int]:
    ids = [hash_id(g, buckets, "c") for g in char_ngrams(text)]
    if use_word_ngrams:
        ids += [hash_id(g, buckets, "w") for g in word_ngrams(text)]
    return ids


def pad_ids(rows: Sequence[Sequence[int]]) -> torch.Tensor:
    width = max((len(r) for r in rows), default=1)
    return torch.tensor([list(r) + [0] * (width - len(r)) for r in rows], dtype=torch.long)


# ---------------------------------------------------------------------------
# 2. Synthetic personalized social-search world
# ---------------------------------------------------------------------------

def _syllable_word(rng: random.Random, n_syl: int) -> str:
    cons, vow = "bcdfghjklmnprstvwz", "aeiou"
    return "".join(rng.choice(cons) + rng.choice(vow) for _ in range(n_syl))


def add_typo(text: str, rng: random.Random) -> str:
    if len(text) < 3:
        return text
    i = rng.randrange(1, len(text) - 1)
    kind = rng.choice(["delete", "swap", "sub"])
    if kind == "delete":
        return text[:i] + text[i + 1:]
    if kind == "swap":
        return text[:i] + text[i + 1] + text[i] + text[i + 2:]
    return text[:i] + rng.choice("abcdefghijklmnopqrstuvwxyz") + text[i + 1:]


@dataclass
class Sessions:
    """A search log. Everything per session i."""
    query_text: List[str]
    loc: np.ndarray             # searcher location
    soc: np.ndarray             # [n,2] searcher social clusters (primary, secondary)
    target: np.ndarray          # clicked entity (positive)
    name_idx: np.ndarray        # intended name
    impressions: List[np.ndarray]   # shown results (includes the target)
    has_typo: np.ndarray
    has_extra: np.ndarray

    def __len__(self) -> int:
        return len(self.query_text)

    def subset(self, idx: Sequence[int]) -> "Sessions":
        idx = np.asarray(idx)
        return Sessions([self.query_text[i] for i in idx], self.loc[idx], self.soc[idx], self.target[idx],
                        self.name_idx[idx], [self.impressions[i] for i in idx], self.has_typo[idx], self.has_extra[idx])


class SocialSearchWorld:
    """People/entities with a name, a location and a social cluster.

    Many entities share a name (hundreds of "John Smith"s). The searcher's intended target among same-name
    entities is the one with the highest affinity to the searcher: same primary social cluster (2.0), same
    secondary cluster (1.0), same location (1.5). So text alone cannot resolve the query -- the searcher's
    context must be in the embedding, which is the paper's central modeling claim. Queries carry typos
    (fuzzy match) and sometimes an optional extra term (e.g. "nw") that exact term matching cannot ignore.
    """

    OPTIONAL_TOKENS = ["nw", "fb", "page"]

    def __init__(self, n_entities: int = 3000, n_names: int = 600, n_locations: int = 15, n_clusters: int = 30,
                 typo_rate: float = 0.25, extra_rate: float = 0.10, seed: int = 0):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.L, self.C, self.n_entities = n_locations, n_clusters, n_entities
        firsts = sorted({_syllable_word(self.rng, self.rng.choice([2, 3])) for _ in range(60)})[:40]
        lasts = sorted({_syllable_word(self.rng, self.rng.choice([2, 3])) for _ in range(60)})[:40]
        combos = [f"{a} {b}" for a in firsts for b in lasts]
        self.rng.shuffle(combos)
        self.names = combos[:n_names]
        self.entity_name = self.np_rng.randint(0, n_names, n_entities)
        self.entity_loc = self.np_rng.randint(0, n_locations, n_entities)
        self.entity_cluster = self.np_rng.randint(0, n_clusters, n_entities)
        self.entity_text = [self.names[i] for i in self.entity_name]
        self.by_name: Dict[int, np.ndarray] = {n: np.where(self.entity_name == n)[0] for n in range(n_names)}
        self.typo_rate, self.extra_rate = typo_rate, extra_rate
        first_tok = [n.split()[0] for n in self.names]
        self._by_first: Dict[str, List[int]] = {}
        for i, f in enumerate(first_tok):
            self._by_first.setdefault(f, []).append(i)

    def affinity(self, ents: np.ndarray, loc: int, s1: int, s2: int, noise: float = 0.0,
                 rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        a = 2.0 * (self.entity_cluster[ents] == s1) + 1.0 * (self.entity_cluster[ents] == s2) + \
            1.5 * (self.entity_loc[ents] == loc)
        return a + noise * (rng or self.np_rng).gumbel(size=len(ents))

    def sample_sessions(self, n: int, seed: Optional[int] = None) -> Sessions:
        rng = random.Random(seed) if seed is not None else self.rng
        nrng = np.random.RandomState(seed) if seed is not None else self.np_rng
        qt, locs, socs, tgts, names, imps, typ, ext = [], [], [], [], [], [], [], []
        while len(qt) < n:
            name = int(nrng.randint(0, len(self.names)))
            cands = self.by_name[name]
            if len(cands) == 0:
                continue
            loc, s1, s2 = int(nrng.randint(self.L)), int(nrng.randint(self.C)), int(nrng.randint(self.C))
            target = int(cands[np.argmax(self.affinity(cands, loc, s1, s2, noise=0.3, rng=nrng))])
            text = self.names[name]
            typo, extra = rng.random() < self.typo_rate, rng.random() < self.extra_rate
            if typo:
                text = add_typo(text, rng)
            if extra:
                text = f"{text} {rng.choice(self.OPTIONAL_TOKENS)}"
            # production-ranker impressions: target + same-name others (hard) + similar first-name others
            others = [int(e) for e in cands if e != target]
            rng.shuffle(others)
            fuzzy_names = [m for m in self._by_first.get(self.names[name].split()[0], []) if m != name]
            fuzzy = [int(self.by_name[m][0]) for m in fuzzy_names[:3] if len(self.by_name[m])]
            shown = [target] + others[:3] + fuzzy[:2]
            rng.shuffle(shown)
            qt.append(text); locs.append(loc); socs.append([s1, s2]); tgts.append(target); names.append(name)
            imps.append(np.array(shown)); typ.append(typo); ext.append(extra)
        return Sessions(qt, np.array(locs), np.array(socs), np.array(tgts), np.array(names), imps,
                        np.array(typ), np.array(ext))

    def boolean_match(self, query_text: str) -> np.ndarray:
        """Production-style exact term matching: every query term must appear in the document text.
        Fails on typos and on extra optional terms."""
        toks = query_text.split()
        return np.array([i for i, t in enumerate(self.entity_text) if all(w in t.split() for w in toks)], dtype=int)


# ---------------------------------------------------------------------------
# 3. Unified embedding model (Sec. 2.3, Fig. 2)
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    use_location: bool = True
    use_social: bool = True
    use_word_ngrams: bool = True
    text_buckets: int = 4096


class FeatureStore:
    """Pre-extracted feature tensors for the entity index and for a session log."""

    def __init__(self, world: SocialSearchWorld, cfg: FeatureConfig):
        self.world, self.cfg = world, cfg
        self.doc_text = pad_ids([text_feature_ids(t, cfg.text_buckets, cfg.use_word_ngrams) for t in world.entity_text])
        self.doc_loc = torch.tensor(world.entity_loc)
        self.doc_soc_idx = torch.tensor(np.stack([world.entity_cluster, world.entity_cluster], 1))
        self.doc_soc_w = torch.tensor(np.tile([1.0, 0.0], (world.n_entities, 1)), dtype=torch.float32)

    def queries(self, s: Sessions) -> Dict[str, torch.Tensor]:
        c = self.cfg
        return {"text": pad_ids([text_feature_ids(t, c.text_buckets, c.use_word_ngrams) for t in s.query_text]),
                "loc": torch.tensor(s.loc), "soc_idx": torch.tensor(s.soc),
                "soc_w": torch.tensor(np.tile([1.0, 0.5], (len(s), 1)), dtype=torch.float32)}   # primary 1.0, secondary 0.5

    def docs(self, ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        d = {"text": self.doc_text, "loc": self.doc_loc, "soc_idx": self.doc_soc_idx, "soc_w": self.doc_soc_w}
        return d if ids is None else {k: v[ids] for k, v in d.items()}


class SideEncoder(nn.Module):
    """One tower. Each feature group is embedded and projected on its own -- text n-gram embedding bag -> MLP,
    location embedding, weighted multi-hot social embedding -- L2-normalised, and concatenated. The tower's output
    cosine is therefore an average of per-group matches (text match + location match + social match), which keeps the
    contextual signal learnable next to a strong text signal. (How the paper's encoders combine the groups is not
    specified; this is our design.)"""

    def __init__(self, cfg: FeatureConfig, n_loc: int, n_soc: int, d_text: int = 48, d_cat: int = 12, d_out: int = 32,
                 hidden: int = 96):
        super().__init__()
        self.cfg = cfg
        self.text = nn.EmbeddingBag(cfg.text_buckets + 1, d_text, mode="mean", padding_idx=0)
        self.text_mlp = nn.Sequential(nn.Linear(d_text, hidden), nn.ReLU(), nn.Linear(hidden, d_out))
        self.loc = nn.Embedding(n_loc, d_cat)
        self.loc_proj = nn.Linear(d_cat, d_out // 2)
        self.soc = nn.EmbeddingBag(n_soc, d_cat, mode="sum")            # weighted combination of several embeddings
        self.soc_proj = nn.Linear(d_cat, d_out // 2)

    def forward(self, f: Dict[str, torch.Tensor]) -> torch.Tensor:
        t = F.normalize(self.text_mlp(self.text(f["text"])), dim=-1)
        loc = F.normalize(self.loc_proj(self.loc(f["loc"])), dim=-1) * float(self.cfg.use_location)
        soc = F.normalize(self.soc_proj(self.soc(f["soc_idx"], per_sample_weights=f["soc_w"])), dim=-1) * float(self.cfg.use_social)
        return F.normalize(torch.cat([t, loc, soc], -1), dim=-1)


class UnifiedEmbeddingModel(nn.Module):
    """Query encoder f and document encoder g (separate parameters by default; `shared=True` shares them)."""

    def __init__(self, world: SocialSearchWorld, cfg: Optional[FeatureConfig] = None, shared: bool = False, d_out: int = 32,
                 d_cat: int = 12):
        super().__init__()
        self.cfg = cfg or FeatureConfig()
        self.f = SideEncoder(self.cfg, world.L, world.C, d_cat=d_cat, d_out=d_out)
        self.g = self.f if shared else SideEncoder(self.cfg, world.L, world.C, d_cat=d_cat, d_out=d_out)
        self.d_out = d_out

    def encode_query(self, f):
        return self.f(f)

    def encode_doc(self, f):
        return self.g(f)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - (a * b).sum(-1)


def triplet_loss(q: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor, margin: float) -> torch.Tensor:
    """Eq. (2): sum max(0, D(q,d+) - D(q,d-) + m). Returned as the mean over triplets."""
    return F.relu(cosine_distance(q, pos) - cosine_distance(q, neg) + margin).mean()


# ---------------------------------------------------------------------------
# 4. Evaluation (Sec. 2.1)
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_index(model: UnifiedEmbeddingModel, store: FeatureStore) -> torch.Tensor:
    model.eval()
    return model.encode_doc(store.docs())


@torch.no_grad()
def embed_queries(model: UnifiedEmbeddingModel, store: FeatureStore, s: Sessions) -> torch.Tensor:
    model.eval()
    return model.encode_query(store.queries(s))


def recall_at_k(q: torch.Tensor, index: torch.Tensor, s: Sessions, k: int = 10) -> float:
    """Eq. (1) with one target per session: recall@K = fraction of sessions whose target is in the top K of a
    KNN search over the whole index."""
    top = (q @ index.t()).topk(k, dim=1).indices
    hit = (top == torch.tensor(s.target).unsqueeze(1)).any(1)
    return hit.float().mean().item()


def evaluate(model, store, s: Sessions, ks: Sequence[int] = (1, 10, 50)) -> Dict[int, float]:
    q, idx = embed_queries(model, store, s), embed_index(model, store)
    return {k: recall_at_k(q, idx, s, k) for k in ks}


# ---------------------------------------------------------------------------
# 5. Training with data-mining and hard-mining options (Secs. 2.4, 6.1)
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    steps: int = 300
    batch: int = 256
    lr: float = 3e-3
    margin: float = 0.3
    neg: str = "random"          # "random" | "non_click" | "offline"
    n_easy: int = 1              # random negatives per query
    n_online_hard: int = 0       # hardest in-batch negatives per query (paper: <= 2 is best)
    seed: int = 0


def mine_online_hard(q: torch.Tensor, pos: torch.Tensor, pos_ids: torch.Tensor, h: int) -> torch.Tensor:
    """Online HNM (Sec. 6.1.1): for each query, rank the OTHER positives in the batch by similarity and take
    the `h` most similar as hard negatives. Returns indices [B,h] into the batch. Columns holding the same
    entity as the query's own positive are excluded."""
    sims = q @ pos.t()
    sims = sims.masked_fill(pos_ids.unsqueeze(0) == pos_ids.unsqueeze(1), float("-inf"))
    return sims.topk(h, dim=1).indices


def train_model(world: SocialSearchWorld, store: FeatureStore, sessions: Sessions, cfg: TrainConfig,
                model: Optional[UnifiedEmbeddingModel] = None, offline_negs: Optional[np.ndarray] = None,
                shared: bool = False) -> UnifiedEmbeddingModel:
    """Triplet training. Positives are the clicked results. Negative sources:
      * "random":    random entities from the index (the initial study's winner)
      * "non_click": a random non-clicked impression of the same session (biased to hard cases; hurts recall)
      * "offline":   `offline_negs[i]` = an entity mined offline for session i (plus `n_easy` random ones)
    optionally blended with in-batch online hard negatives.
    """
    g = torch.Generator().manual_seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = model or UnifiedEmbeddingModel(world, store.cfg, shared=shared)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    qf_all = store.queries(sessions)
    n = len(sessions)
    non_click = [np.array([e for e in imp if e != t]) for imp, t in zip(sessions.impressions, sessions.target)]
    for _ in range(cfg.steps):
        idx = torch.randint(0, n, (cfg.batch,), generator=g)
        qf = {k: v[idx] for k, v in qf_all.items()}
        pos_ids = torch.tensor(sessions.target[idx.numpy()])
        q, pos = model.encode_query(qf), model.encode_doc(store.docs(pos_ids))
        losses = []
        if cfg.neg == "random":
            negs = [torch.tensor(rng.randint(0, world.n_entities, cfg.batch)) for _ in range(cfg.n_easy)]
        elif cfg.neg == "non_click":
            negs = [torch.tensor([rng.choice(non_click[i]) if len(non_click[i]) else rng.randint(world.n_entities)
                                  for i in idx.tolist()])]
        elif cfg.neg == "offline":
            assert offline_negs is not None
            negs = [torch.tensor(offline_negs[idx.numpy()])]
            negs += [torch.tensor(rng.randint(0, world.n_entities, cfg.batch)) for _ in range(cfg.n_easy)]
        else:
            raise ValueError(cfg.neg)
        for nid in negs:
            ok = (nid != pos_ids).float()
            losses.append((F.relu(cosine_distance(q, pos) - cosine_distance(q, model.encode_doc(store.docs(nid)))
                                  + cfg.margin) * ok).sum() / ok.sum().clamp(min=1))
        if cfg.n_online_hard > 0:
            hard = mine_online_hard(q.detach(), pos.detach(), pos_ids, cfg.n_online_hard)
            for j in range(cfg.n_online_hard):
                losses.append(triplet_loss(q, pos, pos[hard[:, j]], cfg.margin))
        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def mine_offline_negatives(model, store: FeatureStore, sessions: Sessions, rank_lo: int, rank_hi: int,
                           seed: int = 0) -> np.ndarray:
    """Offline HNM (Sec. 6.1.1): retrieve each training query's top results over the WHOLE index, then pick one
    negative uniformly from rank positions [rank_lo, rank_hi] (1-based, the target skipped). The paper found
    the very hardest ranks are NOT best; a band further down the list gave the best recall."""
    rng = np.random.RandomState(seed)
    q, idx = embed_queries(model, store, sessions), embed_index(model, store)
    top = (q @ idx.t()).topk(min(rank_hi + 1, idx.size(0)), dim=1).indices.numpy()
    out = np.empty(len(sessions), dtype=int)
    for i in range(len(sessions)):
        ranked = [e for e in top[i] if e != sessions.target[i]][rank_lo - 1:rank_hi]
        out[i] = rng.choice(ranked) if ranked else rng.randint(store.world.n_entities)
    return out


def mine_hard_positives(world: SocialSearchWorld, sessions: Sessions) -> np.ndarray:
    """Hard positive mining (Sec. 6.1.2): sessions where the production retrieval (exact term matching) FAILED to
    return the target, yet the searcher went on to engage with it (found after reformulating). Returns indices."""
    fail = []
    for i, (text, tgt) in enumerate(zip(sessions.query_text, sessions.target)):
        if int(tgt) not in set(world.boolean_match(text).tolist()):
            fail.append(i)
    return np.array(fail, dtype=int)


# ---------------------------------------------------------------------------
# 6. Embedding ensemble (Sec. 6.2)
# ---------------------------------------------------------------------------

def weighted_ensemble_vectors(vq: Sequence[torch.Tensor], vd: Sequence[torch.Tensor],
                              alphas: Sequence[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eqs. (4)-(5). Weighted concatenation: E_Q = (a_1 V_Q1/|V_Q1|, ..., a_n V_Qn/|V_Qn|), E_D unweighted.
    Applying the weights to ONE side suffices for the served metric to equal the weighted ensemble similarity,
    so a single ANN index over E_D can serve n models at once."""
    eq = torch.cat([a * F.normalize(v, dim=-1) for a, v in zip(alphas, vq)], dim=-1)
    ed = torch.cat([F.normalize(u, dim=-1) for u in vd], dim=-1)
    return eq, ed


def ensemble_similarity(vq, vd, alphas) -> torch.Tensor:
    """S_w(Q, D) = sum_i a_i cos(V_Qi, U_Di)  (the quantity the concatenation is proportional to)."""
    return sum(a * (F.normalize(q, dim=-1) * F.normalize(d, dim=-1)).sum(-1) for a, q, d in zip(alphas, vq, vd))


def cascade_rerank(candidates: torch.Tensor, q2: torch.Tensor, index2: torch.Tensor, k: int) -> torch.Tensor:
    """Cascade (Sec. 6.2): the second-stage model re-ranks only the first stage's candidates.
    candidates [B,K1] entity ids, q2 [B,d], index2 [N,d] -> [B,k] ids ordered by the second model."""
    sims = torch.einsum("bd,bkd->bk", q2, index2[candidates])
    order = sims.topk(min(k, candidates.size(1)), dim=1).indices
    return candidates.gather(1, order)


# ---------------------------------------------------------------------------
# 7. ANN: k-means, PQ, OPQ, PCA, IVF-PQ, IMI (Sec. 4.1)
# ---------------------------------------------------------------------------

def kmeans(x: torch.Tensor, k: int, iters: int = 15, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    k = min(k, len(x))
    c = x[torch.randperm(len(x), generator=g)[:k]].clone()
    for _ in range(iters):
        assign = torch.cdist(x, c).argmin(1)
        for j in range(k):
            m = assign == j
            if m.any():
                c[j] = x[m].mean(0)
    return c


class ProductQuantizer:
    """Split a d-dim vector into m sub-vectors, quantize each with its own 2^nbits-word codebook.
    `m` is the paper's pq_bytes (one byte per sub-quantizer at nbits=8)."""

    def __init__(self, d: int, m: int, nbits: int = 8, seed: int = 0):
        assert d % m == 0, "d must be divisible by m"
        self.d, self.m, self.dsub, self.ksub, self.seed = d, m, d // m, 2 ** nbits, seed
        self.codebooks: Optional[torch.Tensor] = None      # [m, ksub, dsub]

    def train(self, x: torch.Tensor) -> "ProductQuantizer":
        books = []
        for j in range(self.m):
            c = kmeans(x[:, j * self.dsub:(j + 1) * self.dsub], self.ksub, seed=self.seed + j)
            if len(c) < self.ksub:                       # fewer training points than codewords
                c = torch.cat([c, c[:1].expand(self.ksub - len(c), -1)])
            books.append(c)
        self.codebooks = torch.stack(books)
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        codes = [torch.cdist(x[:, j * self.dsub:(j + 1) * self.dsub], self.codebooks[j]).argmin(1) for j in range(self.m)]
        return torch.stack(codes, 1).to(torch.uint8 if self.ksub <= 256 else torch.int32)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        c = codes.long()
        return torch.cat([self.codebooks[j][c[:, j]] for j in range(self.m)], dim=1)

    def distance_table(self, q: torch.Tensor) -> torch.Tensor:
        """Asymmetric distance computation (ADC) table [m, ksub] of squared sub-distances for one query."""
        return torch.stack([((q[j * self.dsub:(j + 1) * self.dsub] - self.codebooks[j]) ** 2).sum(-1) for j in range(self.m)])

    def adc(self, table: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        c = codes.long()
        return sum(table[j][c[:, j]] for j in range(self.m))


class PCATransform:
    def __init__(self, d_out: int):
        self.d_out = d_out
        self.mean: Optional[torch.Tensor] = None
        self.proj: Optional[torch.Tensor] = None

    def train(self, x: torch.Tensor) -> "PCATransform":
        self.mean = x.mean(0)
        _, _, vt = torch.linalg.svd(x - self.mean, full_matrices=False)
        self.proj = vt[:self.d_out].t()
        return self

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) @ self.proj


class OPQTransform:
    """Optimized PQ: learn an orthogonal rotation R minimising PQ reconstruction error, by alternating
    (1) train PQ on x R and (2) solve the orthogonal Procrustes problem R = argmin ||x R - decode(encode(x R))||.
    Rotating balances variance across sub-spaces, which is why the paper found OPQ generally beats PCA."""

    def __init__(self, d: int, m: int, nbits: int = 8, iters: int = 4, seed: int = 0):
        self.d, self.m, self.nbits, self.iters, self.seed = d, m, nbits, iters, seed
        self.R = torch.eye(d)

    def train(self, x: torch.Tensor) -> "OPQTransform":
        for it in range(self.iters):
            xr = x @ self.R
            pq = ProductQuantizer(self.d, self.m, self.nbits, self.seed + it).train(xr)
            recon = pq.decode(pq.encode(xr))
            u, _, vt = torch.linalg.svd(x.t() @ recon)
            self.R = u @ vt
        return self

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.R


class IVFIndex:
    """Coarse quantization (inverted file): k-means cells; at query time probe the `nprobe` nearest cells.
    With a ProductQuantizer the cell contents are PQ codes of the RESIDUAL to the cell centroid (as in Faiss
    IVFPQ); without one, exact distances (IVFFlat). Optional pre-transform (PCA / OPQ)."""

    def __init__(self, vectors: torch.Tensor, nlist: int, pq_bytes: Optional[int] = None, nbits: int = 8,
                 transform=None, seed: int = 0):
        self.transform = transform
        x = vectors if transform is None else transform.apply(vectors)
        self.x_dim = x.size(1)
        self.centroids = kmeans(x, nlist, seed=seed)
        assign = torch.cdist(x, self.centroids).argmin(1)
        self.lists = [(assign == c).nonzero().squeeze(1) for c in range(len(self.centroids))]
        self.pq: Optional[ProductQuantizer] = None
        if pq_bytes is not None:
            resid = x - self.centroids[assign]
            self.pq = ProductQuantizer(x.size(1), pq_bytes, nbits, seed).train(resid)
            self.codes = self.pq.encode(resid)
        else:
            self.x = x
        self.n = len(vectors)

    def list_sizes(self) -> List[int]:
        return [len(l) for l in self.lists]

    def search(self, q: torch.Tensor, k: int, nprobe: int) -> Tuple[torch.Tensor, int]:
        """-> (ids of the approximate top-k, number of documents scanned)."""
        if self.transform is not None:
            q = self.transform.apply(q.unsqueeze(0)).squeeze(0)
        probes = torch.cdist(q.unsqueeze(0), self.centroids).squeeze(0).topk(min(nprobe, len(self.centroids)), largest=False).indices
        ids, dist = [], []
        for c in probes.tolist():
            lid = self.lists[c]
            if len(lid) == 0:
                continue
            if self.pq is None:
                d = ((self.x[lid] - q) ** 2).sum(1)
            else:
                d = self.pq.adc(self.pq.distance_table(q - self.centroids[c]), self.codes[lid])
            ids.append(lid)
            dist.append(d)
        if not ids:
            return torch.empty(0, dtype=torch.long), 0
        ids, dist = torch.cat(ids), torch.cat(dist)
        return ids[dist.topk(min(k, len(ids)), largest=False).indices], len(ids)


class IMIIndex:
    """Inverted Multi-Index (2 x K): the vector is split in two halves, each quantized with K centroids; a cell is a
    pair (i, j) so there are K^2 cells from only 2K centroids. Cells are visited in order of the sum of the two
    sub-distances. Prone to very uneven cell sizes (the paper saw about half the cells nearly empty)."""

    def __init__(self, vectors: torch.Tensor, k_sub: int, seed: int = 0):
        d = vectors.size(1)
        self.h, self.k = d // 2, k_sub
        self.c1, self.c2 = kmeans(vectors[:, :self.h], k_sub, seed=seed), kmeans(vectors[:, self.h:], k_sub, seed=seed + 1)
        a1 = torch.cdist(vectors[:, :self.h], self.c1).argmin(1)
        a2 = torch.cdist(vectors[:, self.h:], self.c2).argmin(1)
        self.cell_of = a1 * len(self.c2) + a2
        self.x = vectors
        self.cells: Dict[int, torch.Tensor] = {int(c): (self.cell_of == c).nonzero().squeeze(1) for c in self.cell_of.unique()}
        self.n_cells = len(self.c1) * len(self.c2)

    def search(self, q: torch.Tensor, k: int, nprobe: int) -> Tuple[torch.Tensor, int]:
        d1 = ((q[:self.h] - self.c1) ** 2).sum(1)
        d2 = ((q[self.h:] - self.c2) ** 2).sum(1)
        order = (d1.unsqueeze(1) + d2.unsqueeze(0)).flatten().argsort()
        ids, taken = [], 0
        for cell in order.tolist():
            if cell in self.cells:
                ids.append(self.cells[cell])
                taken += 1
                if taken == nprobe:
                    break
        if not ids:
            return torch.empty(0, dtype=torch.long), 0
        ids = torch.cat(ids)
        d = ((self.x[ids] - q) ** 2).sum(1)
        return ids[d.topk(min(k, len(ids)), largest=False).indices], len(ids)

    def fraction_small_cells(self, thresh: int = 2) -> float:
        sizes = [len(self.cells.get(c, [])) for c in range(self.n_cells)]
        return sum(s <= thresh for s in sizes) / self.n_cells


def one_recall_at_k(index, vectors: torch.Tensor, queries: torch.Tensor, nprobe: int, k: int = 10) -> Tuple[float, float]:
    """The paper's ANN metric: 1-recall@10 = P(the exact nearest neighbour is in the approximate top 10), plus
    the percentage of the index scanned (the quantity ANN settings are compared at -- not nprobe)."""
    hit, scanned = 0, 0
    for q in queries:
        exact = ((vectors - q) ** 2).sum(1).argmin().item()
        ids, n = index.search(q, k, nprobe)
        hit += int(exact in ids.tolist())
        scanned += n
    return hit / len(queries), scanned / len(queries) / len(vectors)


# ---------------------------------------------------------------------------
# 8. Unicorn-style hybrid retrieval: `nn` as a first-class index operator (Sec. 4.2)
# ---------------------------------------------------------------------------

def parse_sexpr(text: str):
    toks = text.replace("(", " ( ").replace(")", " ) ").split()
    pos = 0

    def parse():
        nonlocal pos
        if pos >= len(toks):
            raise ValueError("unexpected end of expression")
        t = toks[pos]
        pos += 1
        if t == "(":
            out = []
            while pos < len(toks) and toks[pos] != ")":
                out.append(parse())
            if pos >= len(toks):
                raise ValueError("missing ')'")
            pos += 1
            return out
        if t == ")":
            raise ValueError("unexpected ')'")
        return t

    node = parse()
    if pos != len(toks):
        raise ValueError("trailing tokens")
    return node


class UnicornLite:
    """A tiny retrieval engine in the spirit of Unicorn: each document is a bag of terms; queries are Boolean
    s-expressions. Embeddings are first-class: at index time a document's embedding is quantized into
    (coarse cluster -> an ordinary TERM `emb:<key>:c<id>`, PQ residual -> a payload). The `nn` operator is rewritten
    at query time into an OR over the terms of the `nprobe` clusters nearest the query embedding, and each matched
    document's payload is used to verify the radius. Real-time updates, planning and multi-hop queries are therefore
    inherited from the term machinery.

        (and (or (term location:seattle) (term location:menlo_park))
             (nn model-1 :radius 0.24 :nprobe 16))
    """

    def __init__(self, key: str, vectors: torch.Tensor, doc_terms: Dict[int, Set[str]], nlist: int = 16,
                 pq_bytes: int = 4, nbits: int = 6, seed: int = 0):
        self.key, self.n = key, len(vectors)
        self.vectors_dim = vectors.size(1)
        self.postings: Dict[str, Set[int]] = {}
        self.ivf = IVFIndex(vectors, nlist, pq_bytes=pq_bytes, nbits=nbits, seed=seed)
        for doc, terms in doc_terms.items():
            for t in terms:
                self.postings.setdefault(t, set()).add(doc)
        for c, lid in enumerate(self.ivf.lists):
            self.postings.setdefault(f"emb:{key}:c{c}", set()).update(lid.tolist())
        self.cluster_of = torch.zeros(self.n, dtype=torch.long)
        self.row_in_list = torch.zeros(self.n, dtype=torch.long)
        for c, lid in enumerate(self.ivf.lists):
            self.cluster_of[lid] = c
            self.row_in_list[lid] = torch.arange(len(lid))

    def _approx_dist(self, q: torch.Tensor, doc: int) -> float:
        """Cosine distance from the query to the document's reconstructed (centroid + PQ residual) vector."""
        c = int(self.cluster_of[doc])
        code = self.ivf.codes[self.ivf.lists[c][self.row_in_list[doc]]].unsqueeze(0)
        recon = self.ivf.centroids[c] + self.ivf.pq.decode(code)[0]
        return 1.0 - float(F.normalize(q, dim=-1) @ F.normalize(recon, dim=-1))

    def _nn(self, args, embeddings: Dict[str, torch.Tensor]) -> Set[int]:
        key = args[0]
        opts = {args[i]: args[i + 1] for i in range(1, len(args) - 1, 2)}
        q = embeddings[key]
        nprobe = int(opts.get(":nprobe", 8))
        probes = torch.cdist(q.unsqueeze(0), self.ivf.centroids).squeeze(0).topk(min(nprobe, len(self.ivf.centroids)), largest=False).indices
        cand: Set[int] = set()
        for c in probes.tolist():                                           # the (nn) -> (or (term cluster)...) rewrite
            cand |= self.postings.get(f"emb:{self.key}:c{c}", set())
        dists = {d: self._approx_dist(q, d) for d in cand}
        if ":radius" in opts:                                               # constrained NN: payload verifies the radius
            return {d for d, x in dists.items() if x <= float(opts[":radius"])}
        k = int(opts.get(":topk", 10))                                      # top-K mode: pick K nearest FIRST
        return set(sorted(dists, key=dists.get)[:k])

    def evaluate(self, expr, embeddings: Dict[str, torch.Tensor]) -> Set[int]:
        if isinstance(expr, str):
            raise ValueError(f"bare atom {expr!r}")
        op, args = expr[0], expr[1:]
        if op == "term":
            return set(self.postings.get(args[0], set()))
        if op == "and":
            sets = [self.evaluate(a, embeddings) for a in args]
            return set.intersection(*sets) if sets else set()
        if op == "or":
            return set().union(*[self.evaluate(a, embeddings) for a in args]) if args else set()
        if op == "nn":
            return self._nn(args, embeddings)
        raise ValueError(f"unknown operator {op!r}")

    def search(self, text: str, embeddings: Dict[str, torch.Tensor]) -> Set[int]:
        return self.evaluate(parse_sexpr(text), embeddings)


# ---------------------------------------------------------------------------
# 9. Query / index selection (Sec. 4.3)
# ---------------------------------------------------------------------------

def should_trigger_ebr(query: str, searcher_clicked_queries: Set[str], min_chars: int = 2) -> bool:
    """Skip EBR where it adds little: searcher is re-finding a specific target they searched and clicked before
    (navigational), or the query is too short/empty. Avoids over-triggering, capacity cost and junkiness."""
    q = query.strip().lower()
    return len(q) >= min_chars and q not in {c.strip().lower() for c in searcher_clicked_queries}


def select_index_docs(monthly_active: np.ndarray, age_days: np.ndarray, popularity: np.ndarray,
                      max_age: float = 30.0, pop_quantile: float = 0.8) -> np.ndarray:
    """Index selection: only monthly-active entities, recent events and popular pages/groups, so searching is faster."""
    keep = monthly_active.astype(bool) & ((age_days <= max_age) | (popularity >= np.quantile(popularity, pop_quantile)))
    return np.where(keep)[0]


# ---------------------------------------------------------------------------
# 10. Later-stage optimization (Sec. 5)
# ---------------------------------------------------------------------------

def embedding_features(q: torch.Tensor, d: torch.Tensor, kind: str = "cosine") -> torch.Tensor:
    """Ways to feed an embedding pair to a ranker: cosine similarity (1 number), Hadamard product (d numbers), or
    the raw embeddings concatenated (2d numbers). The paper found cosine consistently best."""
    if kind == "cosine":
        return (q * d).sum(-1, keepdim=True)
    if kind == "hadamard":
        return q * d
    if kind == "raw":
        return torch.cat([q, d], -1)
    raise ValueError(kind)


class RelevanceFilter:
    """Training-data feedback loop (Sec. 5): EBR trades precision for recall, so results it returns are logged, sent
    to human raters, and a relevance model trained on the labels filters out the irrelevant ones while keeping the
    relevant ones. Logistic regression over [cosine, token overlap, char-trigram Jaccard]."""

    def __init__(self):
        self.w = torch.zeros(3)
        self.b = torch.zeros(1)

    @staticmethod
    def features(query: str, doc: str, cos: float) -> torch.Tensor:
        qt, dt = set(query.split()), set(doc.split())
        qc, dc = set(char_ngrams(query)), set(char_ngrams(doc))
        return torch.tensor([cos, len(qt & dt) / max(len(qt), 1), len(qc & dc) / max(len(qc | dc), 1)])

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 300, lr: float = 0.5) -> "RelevanceFilter":
        w, b = self.w.clone().requires_grad_(True), self.b.clone().requires_grad_(True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        for _ in range(epochs):
            loss = F.binary_cross_entropy_with_logits(X @ w + b, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.w, self.b = w.detach(), b.detach()
        return self

    def score(self, X: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(X @ self.w + self.b)


# ---------------------------------------------------------------------------
# 11. Demo
# ---------------------------------------------------------------------------

def main(seed: int = 0) -> Dict[str, object]:
    world = SocialSearchWorld(seed=seed)
    train_s, test_s = world.sample_sessions(8000, seed=1), world.sample_sessions(1500, seed=2)
    print(f"{world.n_entities} entities, {len(world.names)} names (~{world.n_entities / len(world.names):.1f} entities per "
          f"name -> text alone gives recall@1 of about {len(world.names) / world.n_entities:.2f}); {len(train_s)} training sessions")
    out: Dict[str, object] = {}
    fmt = lambda r: f"recall@1 {r[1]:.3f}  recall@10 {r[10]:.3f}"
    recipe = dict(steps=600, n_online_hard=2)

    print("== unified embedding vs text-only (Table 1 analogue; trained with online hard negatives) ==")
    for name, fc in [("text only", FeatureConfig(use_location=False, use_social=False)),
                     ("+ location", FeatureConfig(use_location=True, use_social=False)),
                     ("+ location + social", FeatureConfig())]:
        store = FeatureStore(world, fc)
        r = evaluate(train_model(world, store, train_s, TrainConfig(seed=seed, **recipe)), store, test_s)
        out[name] = r
        print(f"  {name:22s} {fmt(r)}")

    store = FeatureStore(world, FeatureConfig())
    print("== training-data mining (random-negative baseline, 600 steps) ==")
    base = train_model(world, store, train_s, TrainConfig(steps=600, seed=seed))
    print(f"  {'random negatives':36s} {fmt(evaluate(base, store, test_s))}")
    m = train_model(world, store, train_s, TrainConfig(steps=600, neg="non_click", seed=seed))
    print(f"  {'non-click impressions as negatives':36s} {fmt(evaluate(m, store, test_s))}")
    hp = mine_hard_positives(world, train_s)
    m_hp = train_model(world, store, train_s.subset(hp), TrainConfig(steps=600, seed=seed))
    print(f"  {'hard positives only (%d%% of data)' % round(100 * len(hp) / len(train_s)):36s} {fmt(evaluate(m_hp, store, test_s))}")

    print("== hard negative mining ==")
    for h in (2, 8):
        m = train_model(world, store, train_s, TrainConfig(steps=600, n_online_hard=h, seed=seed))
        print(f"  {'online HNM, %d hardest in batch' % h:36s} {fmt(evaluate(m, store, test_s))}")
    for lo, hi in [(2, 6), (7, 30)]:
        negs = mine_offline_negatives(base, store, train_s, lo, hi, seed)
        mixed = train_model(world, store, train_s, TrainConfig(steps=300, neg="offline", n_easy=3, seed=seed),
                            model=base, offline_negs=negs)
        print(f"  {'offline HNM ranks %d-%d, mixed 3:1' % (lo, hi):36s} {fmt(evaluate(mixed, store, test_s))}   (continued from the baseline)")

    print("== ANN tuning on the learned embeddings (1-recall@10 vs % index scanned) ==")
    idx_vecs = embed_index(base, store)
    qv = embed_queries(base, store, test_s.subset(range(200)))
    for name, kw in [("IVF flat", dict(pq_bytes=None)), ("IVF + PQ (d/4 bytes)", dict(pq_bytes=idx_vecs.size(1) // 4)),
                     ("IVF + PQ (2 bytes)", dict(pq_bytes=2))]:
        ivf = IVFIndex(idx_vecs, 32, nbits=6, **kw)
        r, sc = one_recall_at_k(ivf, idx_vecs, qv, nprobe=4)
        print(f"  {name:22s} 1-recall@10 {r:.3f}  scanned {sc:.1%}")
    return out


if __name__ == "__main__":
    main()
