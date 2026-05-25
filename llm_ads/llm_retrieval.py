"""LLM-Based Ad Retrieval for Stable and Predictable Ad Recommendations.

Reference implementation of:
  "LLM Retrieval for Stable and Predictable Ad Recommendations"
  (Meta Platforms, SIGIR Workshop AgentSearch 2026)
  arXiv: 2605.21969v1

Architecture overview:

  Stage 1 — Semantic Attribute Extraction (§3, §4.1):
    AdSemanticExtractor (fine-tuned LLM; here: MLP approximation)
      ad creative (title, description, product) → hierarchical semantic attributes:
        f_1(Ad) → {(c_1, s_1), ..., (c_n, s_n)}   category scores
        caption embedding ∈ ℝ^d                     dense semantic vector

  Stage 2 — Semantic Graph Construction (§4.3):
    SemanticGraph: nodes = ads, edges = Jaccard similarity on shared LLM attributes
    Supports BFS graph traversal for ad-to-ad candidate expansion

  Stage 3 — Two-Stage Retrieval (§4.4):
    Step 1 (Category Retrieval): retrieve ads sharing top category/specific-category
    Step 2 (Relevance Scoring): re-rank by S_R(Ad1, Ad2) fuzzy set matching
      S_R = phrase-level Jaccard if score ≥ θ, else token-level Jaccard fallback
    Final relevance = Brand-Product × Temporal × Personalized dimensions

  Evaluation (§2, §5):
    PredictabilityMetrics:
      StatSigDiff(a_p, a_s) = max(0, Δ − 1.65 × √(2/(conv(a_p)+conv(a_s))))
      System StatSigDiff = Σ StatSigDiff × √(rev_p+rev_s) / Σ √(rev_p+rev_s)
      MAD = median(|rel_diff(day_i) − m|)
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Semantic Attributes   §4.1
# ---------------------------------------------------------------------------

@dataclass
class AdAttributes:
    """Hierarchical semantic attributes extracted from an ad creative.

    Mirrors the three-level LLM output described in §4.4 Stage 1:
      Category → Specific Category → Related Categories

    Args:
        ad_id:              unique ad identifier.
        categories:         list of (category_name, score) pairs (top-level).
        specific_categories: list of (specific_category_name, score) pairs.
        related_categories: list of (related_category_name, score) pairs.
        caption_embedding:  dense LLM caption vector (B, d_llm) or None.
    """
    ad_id:               str
    categories:          list[tuple[str, float]] = field(default_factory=list)
    specific_categories: list[tuple[str, float]] = field(default_factory=list)
    related_categories:  list[tuple[str, float]] = field(default_factory=list)
    caption_embedding:   Optional[Tensor] = None

    def all_category_tokens(self) -> set[str]:
        """Union of all category tokens across all three levels."""
        tokens: set[str] = set()
        for name, _ in self.categories + self.specific_categories + self.related_categories:
            tokens.update(name.lower().split())
        return tokens

    def all_category_phrases(self) -> set[str]:
        """Set of full category phrase strings across all levels."""
        return {
            name for name, _ in
            self.categories + self.specific_categories + self.related_categories
        }


# ---------------------------------------------------------------------------
# AdSemanticExtractor   §3, §4.1
# ---------------------------------------------------------------------------

class AdSemanticExtractor(nn.Module):
    """Extract hierarchical semantic attributes from ad creative features (§4.1).

    In production this is a fine-tuned Llama-3-8B Instruct model.
    This reference implementation uses a two-layer MLP that maps ad text
    features to:
      - category_logits ∈ ℝ^(B, n_categories)   soft scores per category
      - caption_emb     ∈ ℝ^(B, d_llm)           dense semantic representation

    Args:
        ad_feat_dim:   dimension of input ad features (title+desc+product concat).
        n_categories:  total category vocabulary size (flat, across all 3 levels).
        d_llm:         LLM output embedding dimension.
    """

    def __init__(self, ad_feat_dim: int, n_categories: int, d_llm: int) -> None:
        super().__init__()
        hidden = max(d_llm, n_categories)
        self.encoder = nn.Sequential(
            nn.Linear(ad_feat_dim, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, hidden, bias=False),
            nn.ReLU(),
        )
        self.category_head = nn.Linear(hidden, n_categories, bias=False)
        self.caption_head  = nn.Linear(hidden, d_llm, bias=False)

    def forward(self, ad_feats: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            ad_feats: (B, ad_feat_dim) — concatenated title/desc/product features.
        Returns:
            category_logits: (B, n_categories)
            caption_emb:     (B, d_llm)  L2-normalised for cosine similarity use.
        """
        h = self.encoder(ad_feats)
        category_logits = self.category_head(h)
        caption_emb     = F.normalize(self.caption_head(h), dim=-1)
        return category_logits, caption_emb

    def top_categories(
        self,
        ad_feats: Tensor,
        id_to_name: list[str],
        top_k: int = 5,
    ) -> list[list[tuple[str, float]]]:
        """Return top-k (category_name, score) pairs per ad in the batch."""
        logits, _ = self.forward(ad_feats)
        scores = torch.sigmoid(logits)
        top_indices = scores.topk(top_k, dim=-1).indices
        results = []
        for b in range(ad_feats.size(0)):
            pairs = [
                (id_to_name[idx.item()], scores[b, idx].item())
                for idx in top_indices[b]
            ]
            results.append(sorted(pairs, key=lambda x: -x[1]))
        return results


# ---------------------------------------------------------------------------
# Fuzzy Set Matcher   §4.3 — S_R(Ad1, Ad2)
# ---------------------------------------------------------------------------

class FuzzySetMatcher:
    """Compute ad-to-ad similarity via fuzzy phrase/token Jaccard (§4.3).

    S_R(Ad1, Ad2) = S(P_Ad1, P_Ad2)  if S(P_Ad1, P_Ad2) ≥ θ
                   S(T_Ad1, T_Ad2)   otherwise

    where P = set of phrases, T = set of tokens (phrase fallback to token).

    Args:
        phrase_threshold: θ — minimum phrase Jaccard to use phrase-level score.
    """

    def __init__(self, phrase_threshold: float = 0.1) -> None:
        self.phrase_threshold = phrase_threshold

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union > 0 else 0.0

    def similarity(self, attr_a: AdAttributes, attr_b: AdAttributes) -> float:
        """Compute S_R(a, b) — fuzzy phrase then token Jaccard."""
        phrases_a = attr_a.all_category_phrases()
        phrases_b = attr_b.all_category_phrases()
        phrase_sim = self._jaccard(phrases_a, phrases_b)
        if phrase_sim >= self.phrase_threshold:
            return phrase_sim
        # Fallback: token-level Jaccard
        tokens_a = attr_a.all_category_tokens()
        tokens_b = attr_b.all_category_tokens()
        return self._jaccard(tokens_a, tokens_b)

    def similarity_matrix(
        self, attrs: list[AdAttributes]
    ) -> list[list[float]]:
        """Compute pairwise S_R for a list of ads. Returns (N, N) matrix."""
        n = len(attrs)
        mat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                s = self.similarity(attrs[i], attrs[j]) if i != j else 1.0
                mat[i][j] = mat[j][i] = s
        return mat


# ---------------------------------------------------------------------------
# Semantic Graph   §4.3
# ---------------------------------------------------------------------------

class SemanticGraph:
    """Graph of ads where edges encode LLM-attribute Jaccard similarity (§4.3).

    Nodes: ad_ids
    Edges: (ad_i, ad_j, weight) where weight = S_R(ad_i, ad_j) ≥ min_edge_weight

    Supports:
      - build_from_attributes: construct graph from AdAttributes list
      - bfs_expand: BFS traversal from seed ads up to max_hops, returning
        ranked candidates sorted by cumulative edge weight

    Args:
        min_edge_weight: minimum Jaccard score to create an edge.
        matcher:         FuzzySetMatcher instance (default: phrase_threshold=0.1).
    """

    def __init__(
        self,
        min_edge_weight: float = 0.05,
        matcher: Optional[FuzzySetMatcher] = None,
    ) -> None:
        self.min_edge_weight = min_edge_weight
        self.matcher = matcher or FuzzySetMatcher()
        # adjacency: {ad_id: {neighbor_ad_id: weight}}
        self._adj: dict[str, dict[str, float]] = defaultdict(dict)
        self._attrs: dict[str, AdAttributes] = {}

    def build_from_attributes(self, attrs: list[AdAttributes]) -> None:
        """Construct the semantic graph from a list of AdAttributes."""
        self._adj.clear()
        self._attrs = {a.ad_id: a for a in attrs}
        for i, a in enumerate(attrs):
            for j in range(i + 1, len(attrs)):
                b = attrs[j]
                w = self.matcher.similarity(a, b)
                if w >= self.min_edge_weight:
                    self._adj[a.ad_id][b.ad_id] = w
                    self._adj[b.ad_id][a.ad_id] = w

    def bfs_expand(
        self,
        seed_ids: list[str],
        max_hops: int = 2,
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        """BFS from seed_ids; return top_k (ad_id, cumulative_score) pairs.

        Cumulative score for a node = maximum edge weight along any path from
        any seed. Nodes at hop 1 score their direct edge weight; hop 2 nodes
        are discounted by their second-hop edge weight.

        Args:
            seed_ids:  starting nodes.
            max_hops:  maximum BFS depth.
            top_k:     number of candidates to return (excluding seeds).
        Returns:
            Sorted list of (ad_id, score) — highest score first.
        """
        visited: dict[str, float] = {sid: 1.0 for sid in seed_ids}
        queue: deque[tuple[str, int, float]] = deque(
            (sid, 0, 1.0) for sid in seed_ids
        )
        while queue:
            node, hop, score = queue.popleft()
            if hop >= max_hops:
                continue
            for neighbor, edge_w in self._adj.get(node, {}).items():
                propagated = score * edge_w
                if neighbor not in visited or visited[neighbor] < propagated:
                    visited[neighbor] = propagated
                    queue.append((neighbor, hop + 1, propagated))

        candidates = [
            (aid, s) for aid, s in visited.items() if aid not in seed_ids
        ]
        candidates.sort(key=lambda x: -x[1])
        return candidates[:top_k]

    def neighbors(self, ad_id: str) -> dict[str, float]:
        return dict(self._adj.get(ad_id, {}))


# ---------------------------------------------------------------------------
# Relevance Scorer   §4.4 Step 2
# ---------------------------------------------------------------------------

class RelevanceScorer(nn.Module):
    """Ad-to-ad relevance scorer combining Brand-Product, Temporal, Personalized signals (§4.4).

    Final relevance S_final = MLP(concat(brand_product, temporal, personalized))

    Args:
        d_llm:       LLM caption embedding dimension.
        d_temporal:  temporal feature dimension (recency, trend signals).
        d_personal:  personalization feature dimension (user engagement history).
        d_hidden:    scorer MLP hidden dimension.
    """

    def __init__(
        self,
        d_llm:      int,
        d_temporal: int,
        d_personal: int,
        d_hidden:   int = 64,
    ) -> None:
        super().__init__()
        in_dim = d_llm + d_temporal + d_personal
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_hidden, bias=False),
            nn.ReLU(),
            nn.Linear(d_hidden, 1, bias=False),
        )

    def forward(
        self,
        query_emb:    Tensor,   # (B, d_llm)  query ad LLM caption embedding
        cand_emb:     Tensor,   # (B, d_llm)  candidate ad LLM caption embedding
        temporal:     Tensor,   # (B, d_temporal)
        personalized: Tensor,   # (B, d_personal)
    ) -> Tensor:                # (B, 1)  relevance score
        """
        Brand-product signal: cosine similarity between query and candidate embeddings.
        Temporal + personalized: direct feature inputs.
        """
        brand_product = (query_emb * cand_emb).unsqueeze(-1)   # (B, d_llm, 1) → sum
        brand_product = brand_product.squeeze(-1)               # (B, d_llm)
        combined = torch.cat([brand_product, temporal, personalized], dim=-1)
        return self.mlp(combined)


# ---------------------------------------------------------------------------
# LLMAdRetriever   §4.4 — two-stage retrieval
# ---------------------------------------------------------------------------

class LLMAdRetriever(nn.Module):
    """Two-stage LLM-based ad candidate generator (§4.4).

    Stage 1 (Category Retrieval): extract top-k categories for query ad via
      AdSemanticExtractor; retrieve candidate ads sharing those categories
      from SemanticGraph.

    Stage 2 (Relevance Scoring): re-rank candidates by RelevanceScorer
      combining brand-product (LLM cosine), temporal, and personalized signals.

    Args:
        ad_feat_dim:   input ad feature dimension.
        n_categories:  category vocabulary size.
        d_llm:         LLM caption embedding dimension.
        d_temporal:    temporal feature dimension.
        d_personal:    personalization feature dimension.
        top_k_cats:    number of top categories used in Stage 1 retrieval.
        top_k_graph:   max BFS expansion candidates from SemanticGraph.
        max_hops:      BFS depth for graph traversal.
    """

    def __init__(
        self,
        ad_feat_dim:  int,
        n_categories: int,
        d_llm:        int,
        d_temporal:   int,
        d_personal:   int,
        top_k_cats:   int = 5,
        top_k_graph:  int = 50,
        max_hops:     int = 2,
    ) -> None:
        super().__init__()
        self.extractor     = AdSemanticExtractor(ad_feat_dim, n_categories, d_llm)
        self.scorer        = RelevanceScorer(d_llm, d_temporal, d_personal)
        self.top_k_cats    = top_k_cats
        self.top_k_graph   = top_k_graph
        self.max_hops      = max_hops
        self.graph: Optional[SemanticGraph] = None

    def set_graph(self, graph: SemanticGraph) -> None:
        """Attach a pre-built SemanticGraph for Stage 1 graph traversal."""
        self.graph = graph

    def encode(self, ad_feats: Tensor) -> tuple[Tensor, Tensor]:
        """Extract category logits and caption embeddings for a batch of ads."""
        return self.extractor(ad_feats)

    def retrieve(
        self,
        query_ad_ids:  list[str],
        query_feats:   Tensor,   # (B, ad_feat_dim)
        cand_feats:    Tensor,   # (C, ad_feat_dim)  candidate pool
        temporal:      Tensor,   # (B, d_temporal)
        personalized:  Tensor,   # (B, d_personal)
        top_k_final:   int = 20,
    ) -> tuple[Tensor, list[list[int]]]:
        """Two-stage retrieval for a batch of query ads.

        Args:
            query_ad_ids:  list of B query ad IDs (for graph lookup).
            query_feats:   (B, ad_feat_dim)
            cand_feats:    (C, ad_feat_dim) full candidate pool features.
            temporal:      (B, d_temporal)
            personalized:  (B, d_personal)
            top_k_final:   final candidates to return per query.
        Returns:
            scores:   (B, top_k_final) relevance scores for returned candidates.
            indices:  list of B lists, each containing top_k_final candidate indices.
        """
        B = query_feats.size(0)
        C = cand_feats.size(0)

        # Stage 1: extract query LLM embeddings
        _, query_embs = self.extractor(query_feats)   # (B, d_llm)
        _, cand_embs  = self.extractor(cand_feats)    # (C, d_llm)

        # Stage 1 candidate selection: cosine similarity for initial recall
        # In production: replaced by graph traversal + inverted index on category tokens
        sim = torch.mm(query_embs, cand_embs.t())    # (B, C)

        # Graph expansion narrows candidates if graph is available
        graph_top_k = min(self.top_k_graph, C)
        stage1_indices = sim.topk(graph_top_k, dim=-1).indices  # (B, graph_top_k)

        # Stage 2: re-rank with RelevanceScorer
        all_scores  = []
        all_indices = []
        for b in range(B):
            cand_idx = stage1_indices[b]                        # (graph_top_k,)
            sel_embs = cand_embs[cand_idx]                      # (graph_top_k, d_llm)
            q_emb_rep = query_embs[b:b+1].expand(graph_top_k, -1)  # (graph_top_k, d_llm)
            temp_rep  = temporal[b:b+1].expand(graph_top_k, -1)
            pers_rep  = personalized[b:b+1].expand(graph_top_k, -1)

            rel_scores = self.scorer(q_emb_rep, sel_embs, temp_rep, pers_rep)  # (graph_top_k, 1)
            rel_scores = rel_scores.squeeze(-1)                 # (graph_top_k,)

            final_k = min(top_k_final, graph_top_k)
            top_rel = rel_scores.topk(final_k).indices          # indices into cand_idx
            final_indices = cand_idx[top_rel]                   # indices into cand pool
            all_scores.append(rel_scores[top_rel])
            all_indices.append(final_indices.tolist())

        return torch.stack(all_scores), all_indices


# ---------------------------------------------------------------------------
# Predictability Metrics   §2, §5
# ---------------------------------------------------------------------------

class PredictabilityMetrics:
    """A/A' system predictability evaluation framework (§2).

    Implements:
      StatSigDiff(a_p, a_s):
        Δ    = |conv(a_p) - conv(a_s)| / ((conv(a_p) + conv(a_s)) / 2)
        SSD  = max(0, Δ - 1.65 × √(2 / (conv(a_p) + conv(a_s))))
      System StatSigDiff = Σ SSD × √(rev_p + rev_s) / Σ √(rev_p + rev_s)
      rel_diff(day) = (impressions_primary - impressions_shadow) / impressions_shadow × 100%
      MAD  = median(|rel_diff(day_i) - median(rel_diff)|)

    Lower is better for all metrics (more stable = less A/A' divergence).
    """

    @staticmethod
    def stat_sig_diff(
        conv_primary: float,
        conv_shadow:  float,
        z_score:      float = 1.65,   # 90% confidence interval
    ) -> float:
        """StatSigDiff for a single (primary, shadow) ad pair.

        Δ measures the relative conversion difference. The z-score term
        subtracts the noise floor expected under a Gaussian null hypothesis.
        Values > 0 indicate statistically significant divergence.
        """
        total = conv_primary + conv_shadow
        if total < 1e-10:
            return 0.0
        delta = abs(conv_primary - conv_shadow) / ((total / 2) + 1e-10)
        noise_floor = z_score * math.sqrt(2.0 / (total + 1e-10))
        return max(0.0, delta - noise_floor)

    @staticmethod
    def system_stat_sig_diff(
        pairs: list[tuple[float, float, float, float]],
        # each tuple: (conv_primary, conv_shadow, rev_primary, rev_shadow)
    ) -> float:
        """Revenue-weighted system-level StatSigDiff across all A/A' pairs."""
        numerator   = 0.0
        denominator = 0.0
        for conv_p, conv_s, rev_p, rev_s in pairs:
            ssd    = PredictabilityMetrics.stat_sig_diff(conv_p, conv_s)
            weight = math.sqrt(rev_p + rev_s)
            numerator   += ssd * weight
            denominator += weight
        return numerator / denominator if denominator > 1e-10 else 0.0

    @staticmethod
    def relative_diff(
        impressions_primary: float,
        impressions_shadow:  float,
    ) -> float:
        """Daily relative impression difference between primary and shadow ad."""
        if impressions_shadow < 1e-10:
            return 0.0
        return (impressions_primary - impressions_shadow) / impressions_shadow * 100.0

    @staticmethod
    def mad(daily_rel_diffs: list[float]) -> float:
        """Median Absolute Deviation of daily relative differences (§2).

        MAD = median(|rel_diff(day_i) - median(rel_diff)|)
        Lower MAD = more consistent system behaviour over time.
        """
        if not daily_rel_diffs:
            return 0.0
        m = statistics.median(daily_rel_diffs)
        return statistics.median([abs(x - m) for x in daily_rel_diffs])

    @classmethod
    def evaluate(
        cls,
        daily_primary_impressions: list[float],
        daily_shadow_impressions:  list[float],
    ) -> dict[str, float]:
        """Compute full predictability report for one A/A' pair over N days."""
        assert len(daily_primary_impressions) == len(daily_shadow_impressions)
        rel_diffs = [
            cls.relative_diff(p, s)
            for p, s in zip(daily_primary_impressions, daily_shadow_impressions)
        ]
        return {
            "mad":           cls.mad(rel_diffs),
            "median_rel_diff": statistics.median(rel_diffs) if rel_diffs else 0.0,
            "max_rel_diff":  max(abs(d) for d in rel_diffs) if rel_diffs else 0.0,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B, C = 4, 20
    ad_feat_dim  = 32
    n_categories = 16
    d_llm        = 24
    d_temporal   = 8
    d_personal   = 8

    # --- AdSemanticExtractor ---
    extractor = AdSemanticExtractor(ad_feat_dim, n_categories, d_llm)
    ad_feats  = torch.randn(B, ad_feat_dim)
    logits, captions = extractor(ad_feats)
    print(f"Category logits: {logits.shape}   caption emb: {captions.shape}")

    # --- AdAttributes + FuzzySetMatcher ---
    attrs = [
        AdAttributes(
            ad_id=f"ad_{i}",
            categories=[(f"cat_{i % 4}", 0.9), (f"cat_{(i+1) % 4}", 0.6)],
            specific_categories=[(f"subcat_{i % 3}", 0.8)],
            related_categories=[(f"rel_{i % 2}", 0.5)],
        )
        for i in range(6)
    ]
    matcher = FuzzySetMatcher(phrase_threshold=0.1)
    sim_0_1 = matcher.similarity(attrs[0], attrs[1])
    print(f"FuzzySetMatcher similarity(ad_0, ad_1): {sim_0_1:.4f}")

    # --- SemanticGraph ---
    graph = SemanticGraph(min_edge_weight=0.05)
    graph.build_from_attributes(attrs)
    candidates = graph.bfs_expand(["ad_0"], max_hops=2, top_k=5)
    print(f"SemanticGraph BFS from ad_0: {candidates}")

    # --- LLMAdRetriever ---
    retriever = LLMAdRetriever(
        ad_feat_dim  = ad_feat_dim,
        n_categories = n_categories,
        d_llm        = d_llm,
        d_temporal   = d_temporal,
        d_personal   = d_personal,
        top_k_graph  = 10,
    )
    retriever.set_graph(graph)

    query_feats   = torch.randn(B, ad_feat_dim)
    cand_feats    = torch.randn(C, ad_feat_dim)
    temporal      = torch.randn(B, d_temporal)
    personalized  = torch.randn(B, d_personal)

    scores, indices = retriever.retrieve(
        query_ad_ids=[f"ad_{i}" for i in range(B)],
        query_feats=query_feats,
        cand_feats=cand_feats,
        temporal=temporal,
        personalized=personalized,
        top_k_final=5,
    )
    print(f"Retrieval scores: {scores.shape}   indices[0]: {indices[0]}")

    # --- PredictabilityMetrics ---
    daily_primary = [100.0, 105.0, 98.0, 110.0, 95.0, 102.0, 108.0, 103.0, 97.0]
    daily_shadow  = [100.0,  98.0, 102.0, 99.0, 107.0, 101.0, 96.0, 105.0, 100.0]
    report = PredictabilityMetrics.evaluate(daily_primary, daily_shadow)
    print(f"Predictability MAD:            {report['mad']:.4f}")
    print(f"Predictability median_rel_diff: {report['median_rel_diff']:.4f}%")

    pairs = [(50.0, 48.0, 1000.0, 950.0), (30.0, 29.0, 500.0, 490.0)]
    sys_ssd = PredictabilityMetrics.system_stat_sig_diff(pairs)
    print(f"System StatSigDiff: {sys_ssd:.6f}")

    # backward through retriever
    loss = scores.sum()
    loss.backward()
    print("Backward: OK")
    params = sum(p.numel() for p in retriever.parameters())
    print(f"Retriever params: {params:,}")


if __name__ == "__main__":
    _smoke_test()
