"""Tests for llm_retrieval.py — LLM-based ad retrieval and predictability metrics."""

import math

import pytest
import torch
import torch.nn.functional as F

from llm_retrieval import (
    AdAttributes,
    AdSemanticExtractor,
    FuzzySetMatcher,
    LLMAdRetriever,
    PredictabilityMetrics,
    RelevanceScorer,
    SemanticGraph,
)


# ---------------------------------------------------------------------------
# AdAttributes
# ---------------------------------------------------------------------------

class TestAdAttributes:
    def test_all_category_tokens_merges_levels(self):
        attr = AdAttributes(
            ad_id="a",
            categories=[("running shoes", 0.9)],
            specific_categories=[("trail running", 0.8)],
            related_categories=[("sports gear", 0.5)],
        )
        tokens = attr.all_category_tokens()
        assert tokens == {"running", "shoes", "trail", "sports", "gear"}

    def test_all_category_phrases_merges_levels(self):
        attr = AdAttributes(
            ad_id="a",
            categories=[("running shoes", 0.9)],
            specific_categories=[("trail running", 0.8)],
            related_categories=[("sports gear", 0.5)],
        )
        assert attr.all_category_phrases() == {"running shoes", "trail running", "sports gear"}

    def test_tokens_lowercased(self):
        attr = AdAttributes(ad_id="a", categories=[("Running Shoes", 0.9)])
        assert "running" in attr.all_category_tokens()
        assert "shoes" in attr.all_category_tokens()
        assert "Running" not in attr.all_category_tokens()

    def test_empty_attributes(self):
        attr = AdAttributes(ad_id="a")
        assert attr.all_category_tokens() == set()
        assert attr.all_category_phrases() == set()

    def test_duplicate_phrases_deduplicated(self):
        attr = AdAttributes(
            ad_id="a",
            categories=[("shoes", 0.9)],
            specific_categories=[("shoes", 0.8)],
        )
        assert len(attr.all_category_phrases()) == 1

    def test_tokens_split_on_spaces(self):
        attr = AdAttributes(ad_id="a", categories=[("a b c", 0.9)])
        assert attr.all_category_tokens() == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# AdSemanticExtractor
# ---------------------------------------------------------------------------

class TestAdSemanticExtractor:
    @pytest.fixture
    def extractor(self):
        torch.manual_seed(42)
        return AdSemanticExtractor(ad_feat_dim=32, n_categories=16, d_llm=24)

    def test_output_shapes(self, extractor):
        logits, captions = extractor(torch.randn(4, 32))
        assert logits.shape == (4, 16)
        assert captions.shape == (4, 24)

    def test_caption_l2_normalised(self, extractor):
        _, captions = extractor(torch.randn(8, 32))
        norms = captions.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_single_item_batch(self, extractor):
        logits, captions = extractor(torch.randn(1, 32))
        assert logits.shape == (1, 16)
        assert captions.shape == (1, 24)

    def test_backward(self, extractor):
        logits, captions = extractor(torch.randn(4, 32))
        (logits.sum() + captions.sum()).backward()

    def test_top_categories_length(self, extractor):
        id_to_name = [f"cat_{i}" for i in range(16)]
        results = extractor.top_categories(torch.randn(3, 32), id_to_name, top_k=5)
        assert len(results) == 3
        assert all(len(per_ad) == 5 for per_ad in results)

    def test_top_categories_descending_score(self, extractor):
        id_to_name = [f"cat_{i}" for i in range(16)]
        results = extractor.top_categories(torch.randn(2, 32), id_to_name, top_k=5)
        for per_ad in results:
            scores = [s for _, s in per_ad]
            assert scores == sorted(scores, reverse=True)

    def test_top_categories_names_valid(self, extractor):
        id_to_name = [f"cat_{i}" for i in range(16)]
        results = extractor.top_categories(torch.randn(2, 32), id_to_name, top_k=3)
        for per_ad in results:
            for name, score in per_ad:
                assert name in id_to_name
                assert 0.0 <= score <= 1.0  # sigmoid output


# ---------------------------------------------------------------------------
# FuzzySetMatcher
# ---------------------------------------------------------------------------

class TestFuzzySetMatcher:
    def test_identical_ads_phrase_sim_one(self):
        attr = AdAttributes(ad_id="a", categories=[("running shoes", 0.9)])
        assert FuzzySetMatcher().similarity(attr, attr) == pytest.approx(1.0)

    def test_disjoint_ads_zero(self):
        a = AdAttributes(ad_id="a", categories=[("shoes", 0.9)])
        b = AdAttributes(ad_id="b", categories=[("electronics", 0.9)])
        # no shared phrases or tokens
        assert FuzzySetMatcher().similarity(a, b) == pytest.approx(0.0)

    def test_shared_phrase_preferred(self):
        a = AdAttributes(ad_id="a", categories=[("running shoes", 0.9)])
        b = AdAttributes(ad_id="b", categories=[("running shoes", 0.7)])
        sim = FuzzySetMatcher(phrase_threshold=0.1).similarity(a, b)
        assert sim == pytest.approx(1.0)  # phrase Jaccard = 1/1

    def test_token_fallback_when_phrase_below_threshold(self):
        # phrases differ: "running shoes" vs "running gear" → phrase Jaccard = 0
        # tokens: {"running","shoes"} vs {"running","gear"} → 1/3
        a = AdAttributes(ad_id="a", categories=[("running shoes", 0.9)])
        b = AdAttributes(ad_id="b", categories=[("running gear", 0.9)])
        sim = FuzzySetMatcher(phrase_threshold=0.5).similarity(a, b)
        assert sim == pytest.approx(1 / 3, abs=1e-5)

    def test_empty_sets_zero(self):
        a = AdAttributes(ad_id="a")
        b = AdAttributes(ad_id="b")
        assert FuzzySetMatcher().similarity(a, b) == pytest.approx(0.0)

    def test_similarity_matrix_shape(self):
        attrs = [AdAttributes(ad_id=f"ad_{i}", categories=[(f"cat_{i}", 0.9)]) for i in range(5)]
        mat = FuzzySetMatcher().similarity_matrix(attrs)
        assert len(mat) == 5
        assert all(len(row) == 5 for row in mat)

    def test_similarity_matrix_diagonal_one(self):
        attrs = [AdAttributes(ad_id=f"ad_{i}", categories=[(f"cat_{i}", 0.9)]) for i in range(4)]
        mat = FuzzySetMatcher().similarity_matrix(attrs)
        for i in range(4):
            assert mat[i][i] == pytest.approx(1.0)

    def test_similarity_matrix_symmetric(self):
        attrs = [
            AdAttributes(
                ad_id=f"ad_{i}",
                categories=[(f"cat_{i % 3}", 0.9), (f"cat_{(i + 1) % 3}", 0.5)],
            )
            for i in range(5)
        ]
        mat = FuzzySetMatcher().similarity_matrix(attrs)
        for i in range(5):
            for j in range(5):
                assert mat[i][j] == pytest.approx(mat[j][i], abs=1e-6)

    def test_jaccard_partial_overlap(self):
        a = AdAttributes(ad_id="a", categories=[("cat_a", 0.9), ("cat_b", 0.8)])
        b = AdAttributes(ad_id="b", categories=[("cat_b", 0.7), ("cat_c", 0.6)])
        # phrase Jaccard: {"cat_a","cat_b"} ∩ {"cat_b","cat_c"} / union = 1/3
        sim = FuzzySetMatcher(phrase_threshold=0.1).similarity(a, b)
        assert sim == pytest.approx(1 / 3, abs=1e-5)


# ---------------------------------------------------------------------------
# SemanticGraph
# ---------------------------------------------------------------------------

class TestSemanticGraph:
    @pytest.fixture
    def overlapping_attrs(self):
        return [
            AdAttributes(ad_id="ad_0", categories=[("running shoes", 0.9), ("trail", 0.6)]),
            AdAttributes(ad_id="ad_1", categories=[("running shoes", 0.8), ("gym", 0.5)]),
            AdAttributes(ad_id="ad_2", categories=[("electronics", 0.9)]),
        ]

    def test_edge_created_for_similar_ads(self, overlapping_attrs):
        graph = SemanticGraph(min_edge_weight=0.1)
        graph.build_from_attributes(overlapping_attrs)
        assert "ad_1" in graph.neighbors("ad_0")
        assert "ad_0" in graph.neighbors("ad_1")

    def test_no_edge_for_dissimilar_ads(self, overlapping_attrs):
        graph = SemanticGraph(min_edge_weight=0.1)
        graph.build_from_attributes(overlapping_attrs)
        assert "ad_2" not in graph.neighbors("ad_0")
        assert "ad_2" not in graph.neighbors("ad_1")

    def test_edge_weight_symmetric(self, overlapping_attrs):
        graph = SemanticGraph(min_edge_weight=0.05)
        graph.build_from_attributes(overlapping_attrs)
        w01 = graph.neighbors("ad_0").get("ad_1", 0.0)
        w10 = graph.neighbors("ad_1").get("ad_0", 0.0)
        assert w01 == pytest.approx(w10)

    def test_bfs_excludes_seeds(self, overlapping_attrs):
        graph = SemanticGraph(min_edge_weight=0.05)
        graph.build_from_attributes(overlapping_attrs)
        results = graph.bfs_expand(["ad_0"], max_hops=2, top_k=10)
        assert "ad_0" not in [r[0] for r in results]

    def test_bfs_returns_at_most_top_k(self):
        attrs = [
            AdAttributes(ad_id=f"ad_{i}", categories=[("shoes", 0.9), (f"extra_{i}", 0.5)])
            for i in range(10)
        ]
        graph = SemanticGraph(min_edge_weight=0.05)
        graph.build_from_attributes(attrs)
        results = graph.bfs_expand(["ad_0"], max_hops=2, top_k=3)
        assert len(results) <= 3

    def test_bfs_scores_descending(self):
        attrs = [
            AdAttributes(ad_id=f"ad_{i}", categories=[("shoes", 0.9), (f"x_{i}", 0.5)])
            for i in range(6)
        ]
        graph = SemanticGraph(min_edge_weight=0.05)
        graph.build_from_attributes(attrs)
        results = graph.bfs_expand(["ad_0"], max_hops=2, top_k=10)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_bfs_empty_graph(self):
        graph = SemanticGraph(min_edge_weight=0.05)
        graph.build_from_attributes([])
        assert graph.bfs_expand(["unknown"], max_hops=2, top_k=10) == []

    def test_neighbors_unknown_node(self):
        graph = SemanticGraph()
        assert graph.neighbors("unknown") == {}

    def test_rebuild_clears_old_state(self):
        attrs1 = [
            AdAttributes(ad_id="a", categories=[("shoes", 0.9)]),
            AdAttributes(ad_id="b", categories=[("shoes", 0.8)]),
        ]
        attrs2 = [
            AdAttributes(ad_id="x", categories=[("electronics", 0.9)]),
            AdAttributes(ad_id="y", categories=[("clothing", 0.9)]),
        ]
        graph = SemanticGraph(min_edge_weight=0.05)
        graph.build_from_attributes(attrs1)
        graph.build_from_attributes(attrs2)
        assert graph.neighbors("a") == {}  # old edges cleared


# ---------------------------------------------------------------------------
# RelevanceScorer
# ---------------------------------------------------------------------------

class TestRelevanceScorer:
    @pytest.fixture
    def scorer(self):
        return RelevanceScorer(d_llm=24, d_temporal=8, d_personal=8)

    def test_output_shape(self, scorer):
        B = 6
        out = scorer(
            F.normalize(torch.randn(B, 24), dim=-1),
            F.normalize(torch.randn(B, 24), dim=-1),
            torch.randn(B, 8),
            torch.randn(B, 8),
        )
        assert out.shape == (B, 1)

    def test_backward(self, scorer):
        B = 4
        out = scorer(
            F.normalize(torch.randn(B, 24), dim=-1),
            F.normalize(torch.randn(B, 24), dim=-1),
            torch.randn(B, 8),
            torch.randn(B, 8),
        )
        out.sum().backward()

    def test_batch_size_one(self, scorer):
        out = scorer(
            F.normalize(torch.randn(1, 24), dim=-1),
            F.normalize(torch.randn(1, 24), dim=-1),
            torch.randn(1, 8),
            torch.randn(1, 8),
        )
        assert out.shape == (1, 1)


# ---------------------------------------------------------------------------
# LLMAdRetriever
# ---------------------------------------------------------------------------

class TestLLMAdRetriever:
    @pytest.fixture
    def retriever(self):
        torch.manual_seed(0)
        return LLMAdRetriever(
            ad_feat_dim=32,
            n_categories=16,
            d_llm=24,
            d_temporal=8,
            d_personal=8,
            top_k_graph=10,
        )

    def _run(self, retriever, B=4, C=20, top_k=5):
        return retriever.retrieve(
            query_ad_ids=[f"q_{i}" for i in range(B)],
            query_feats=torch.randn(B, 32),
            cand_feats=torch.randn(C, 32),
            temporal=torch.randn(B, 8),
            personalized=torch.randn(B, 8),
            top_k_final=top_k,
        )

    def test_output_shapes(self, retriever):
        scores, indices = self._run(retriever, B=4, C=20, top_k=5)
        assert scores.shape == (4, 5)
        assert len(indices) == 4
        assert all(len(idx) == 5 for idx in indices)

    def test_indices_in_candidate_range(self, retriever):
        C = 15
        _, indices = self._run(retriever, B=3, C=C, top_k=5)
        for per_query in indices:
            assert all(0 <= idx < C for idx in per_query)

    def test_top_k_capped_at_pool_size(self, retriever):
        C = 3
        scores, indices = self._run(retriever, B=2, C=C, top_k=10)
        assert all(len(idx) <= C for idx in indices)
        assert scores.shape[1] <= C

    def test_backward(self, retriever):
        scores, _ = self._run(retriever)
        scores.sum().backward()

    def test_encode_shapes(self, retriever):
        logits, captions = retriever.encode(torch.randn(4, 32))
        assert logits.shape == (4, 16)
        assert captions.shape == (4, 24)

    def test_single_query(self, retriever):
        scores, indices = self._run(retriever, B=1, C=20, top_k=5)
        assert scores.shape == (1, 5)
        assert len(indices) == 1


# ---------------------------------------------------------------------------
# PredictabilityMetrics
# ---------------------------------------------------------------------------

class TestPredictabilityMetrics:
    # --- stat_sig_diff ---

    def test_ssd_identical_conversions(self):
        assert PredictabilityMetrics.stat_sig_diff(100.0, 100.0) == pytest.approx(0.0)

    def test_ssd_zero_total(self):
        assert PredictabilityMetrics.stat_sig_diff(0.0, 0.0) == pytest.approx(0.0)

    def test_ssd_large_difference_positive(self):
        assert PredictabilityMetrics.stat_sig_diff(1000.0, 1.0) > 0.0

    def test_ssd_always_non_negative(self):
        for a, b in [(10, 9), (5, 5), (100, 80), (1, 1000), (0, 0)]:
            assert PredictabilityMetrics.stat_sig_diff(float(a), float(b)) >= 0.0

    def test_ssd_symmetric(self):
        ssd_ab = PredictabilityMetrics.stat_sig_diff(80.0, 100.0)
        ssd_ba = PredictabilityMetrics.stat_sig_diff(100.0, 80.0)
        assert ssd_ab == pytest.approx(ssd_ba)

    # --- system_stat_sig_diff ---

    def test_system_ssd_all_equal(self):
        pairs = [(100.0, 100.0, 500.0, 500.0), (50.0, 50.0, 200.0, 200.0)]
        assert PredictabilityMetrics.system_stat_sig_diff(pairs) == pytest.approx(0.0)

    def test_system_ssd_empty(self):
        assert PredictabilityMetrics.system_stat_sig_diff([]) == pytest.approx(0.0)

    def test_system_ssd_zero_revenue_pair_excluded(self):
        pairs = [
            (1000.0, 1.0, 1000.0, 1000.0),  # large diff, high revenue
            (10.0, 10.0, 0.0, 0.0),          # no diff, zero revenue (zero weight)
        ]
        result = PredictabilityMetrics.system_stat_sig_diff(pairs)
        expected = PredictabilityMetrics.stat_sig_diff(1000.0, 1.0)
        assert result == pytest.approx(expected, rel=0.01)

    def test_system_ssd_single_pair(self):
        ssd = PredictabilityMetrics.stat_sig_diff(80.0, 100.0)
        result = PredictabilityMetrics.system_stat_sig_diff([(80.0, 100.0, 10.0, 10.0)])
        assert result == pytest.approx(ssd)

    # --- relative_diff ---

    def test_relative_diff_equal(self):
        assert PredictabilityMetrics.relative_diff(100.0, 100.0) == pytest.approx(0.0)

    def test_relative_diff_zero_shadow(self):
        assert PredictabilityMetrics.relative_diff(100.0, 0.0) == pytest.approx(0.0)

    def test_relative_diff_primary_higher(self):
        assert PredictabilityMetrics.relative_diff(110.0, 100.0) == pytest.approx(10.0)

    def test_relative_diff_primary_lower(self):
        assert PredictabilityMetrics.relative_diff(90.0, 100.0) == pytest.approx(-10.0)

    def test_relative_diff_double(self):
        assert PredictabilityMetrics.relative_diff(200.0, 100.0) == pytest.approx(100.0)

    # --- mad ---

    def test_mad_constant_series(self):
        assert PredictabilityMetrics.mad([5.0, 5.0, 5.0, 5.0]) == pytest.approx(0.0)

    def test_mad_empty(self):
        assert PredictabilityMetrics.mad([]) == pytest.approx(0.0)

    def test_mad_single_element(self):
        assert PredictabilityMetrics.mad([7.0]) == pytest.approx(0.0)

    def test_mad_known_value(self):
        # median([1,2,3,4,5]) = 3, deviations = [2,1,0,1,2], MAD = median([2,1,0,1,2]) = 1
        assert PredictabilityMetrics.mad([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(1.0)

    def test_mad_two_elements(self):
        # median([0,10])=5, deviations=[5,5], MAD=5
        assert PredictabilityMetrics.mad([0.0, 10.0]) == pytest.approx(5.0)

    # --- evaluate ---

    def test_evaluate_returns_required_keys(self):
        report = PredictabilityMetrics.evaluate([100.0, 105.0], [100.0, 98.0])
        assert {"mad", "median_rel_diff", "max_rel_diff"} == set(report.keys())

    def test_evaluate_equal_series_zero_mad(self):
        series = [100.0] * 7
        report = PredictabilityMetrics.evaluate(series, series)
        assert report["mad"] == pytest.approx(0.0)
        assert report["median_rel_diff"] == pytest.approx(0.0)
        assert report["max_rel_diff"] == pytest.approx(0.0)

    def test_evaluate_max_rel_diff(self):
        primary = [100.0, 200.0, 100.0]
        shadow = [100.0, 100.0, 100.0]
        report = PredictabilityMetrics.evaluate(primary, shadow)
        assert report["max_rel_diff"] == pytest.approx(100.0)  # 200/100 - 1 = 100%

    def test_evaluate_mismatched_lengths_raises(self):
        with pytest.raises(AssertionError):
            PredictabilityMetrics.evaluate([1.0, 2.0], [1.0])
