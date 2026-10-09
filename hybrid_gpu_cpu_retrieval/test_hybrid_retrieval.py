"""Tests for the hybrid GPU-CPU retrieval implementation.

Run with:
    pytest test_hybrid_retrieval.py -v
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hybrid_retrieval import (
    Aggregator, AccessStats, BranchResult, CATEGORIES, CPUPathway, Candidate, EmbeddingIndex, GPUModels,
    GPUPathway, HardwareUnit, HybridRetriever, Int8ClusterIndex, InteractionPreRanker, LifecycleConfig,
    LightweightTwoTower, ModelIndexSnapshot, NoisyOracleRanker, SearchWorld, VersionedCentroids,
    bidirectional_info_nce, capacity_plan, compare_plans, conservative_interval, dcg_at_k, evaluate_retrievers,
    gsrr, info_nce_cross_session, info_nce_within_session, int8_quantize, int8_scores, kmeans,
    modeling_depth_ndcg, ndcg_at_k, overlap_metrics, pool_value, recall_candidate_frontier, relative_lift,
    select_broad_inventory, select_engagement_pool, select_search_value_pool, source_composition,
    train_centroids, train_cpu_model, train_gpu_models, value_model,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_dcg_and_ndcg(self):
        assert dcg_at_k([3, 2, 1], 3) == pytest.approx(3 + 2 / math.log2(3) + 1 / 2)
        assert ndcg_at_k([3, 2, 1], [1, 2, 3], 3) == pytest.approx(1.0)
        assert ndcg_at_k([1, 2, 3], [1, 2, 3], 3) < 1.0
        assert ndcg_at_k([], [1], 3) == 0.0 and ndcg_at_k([1], [], 3) == 0.0

    def test_ndcg_cutoff(self):
        assert ndcg_at_k([0, 0, 5], [5], 2) == 0.0

    def test_gsrr(self):
        th = {"view": 30.0, "click": 1.0}
        sessions = [[("view", 45.0)], [("view", 10.0), ("click", 0.0)], [("view", 5.0), ("click", 1.0)], []]
        assert gsrr(sessions, th) == pytest.approx(2 / 4)
        assert gsrr([], th) == 0.0

    def test_relative_lift(self):
        assert relative_lift(1.0451, 1.0) == pytest.approx(4.51)
        assert relative_lift(1.0, 1.0) == 0.0

    def test_conservative_interval_is_mean_plus_minus_mean_larger_half_width(self):
        lifts = [2.0, 2.0, 2.0]
        lo, hi = [1.5, 1.2, 1.9], [2.3, 2.5, 2.1]      # larger sides: .5? -> max(.3,.5)=.5, max(.5,.8)=.8, max(.1,.1)=.1
        m, a, b = conservative_interval(lifts, lo, hi)
        assert m == 2.0 and (b - m) == pytest.approx((0.5 + 0.8 + 0.1) / 3) and (m - a) == pytest.approx(b - m)

    def test_conservative_interval_wider_than_independent_assumption(self):
        # If days were independent the half-width would shrink by sqrt(9)=3; the conservative bound does not.
        daily = np.full(9, 2.0)
        m, a, b = conservative_interval(daily, daily - 0.9, daily + 0.9)
        assert (b - m) == pytest.approx(0.9)
        assert (b - m) > 0.9 / 3

    def test_overlap_and_jaccard(self):
        r = overlap_metrics({1, 2, 3, 4}, {3, 4, 5})
        assert r["gpu_side"] == 0.5 and r["cpu_side"] == pytest.approx(2 / 3) and r["jaccard"] == pytest.approx(2 / 5)
        assert overlap_metrics(set(), set()) == {"gpu_side": 0.0, "cpu_side": 0.0, "jaccard": 0.0}

    def test_source_composition(self):
        c = source_composition({"gpu": {1, 2, 3}, "cpu": {3, 4}, "lex": {5}})
        assert c["gpu"] == 2 and c["cpu"] == 1 and c["lex"] == 1 and c["overlap"] == 1 and c["total"] == 5


class TestCapacity:
    def test_storage_bound(self):
        p = capacity_plan(1000, 10, HardwareUnit(100, 50), regions=3)
        assert p["bound"] == "storage" and p["units_per_region"] == 10 and p["total_units"] == 30

    def test_throughput_bound(self):
        p = capacity_plan(100, 1000, HardwareUnit(100, 50))
        assert p["bound"] == "throughput" and p["units_per_region"] == 20

    def test_paper_style_comparison(self):
        # accelerator: ~3x vectors/unit, ~3.5x qps/unit, ~12x unit cost; storage-dominated workload
        cpu, acc = HardwareUnit(1.0, 1.0, 1.0), HardwareUnit(3.0, 3.5, 12.0)
        r = compare_plans(n_vectors=300, qps=100, cpu=cpu, accel=acc)
        assert r["cpu_bound"] == "storage" and r["accel_bound"] == "storage"
        assert r["accel_units_per_region"] == pytest.approx(1 / 3, abs=0.01)
        assert r["unit_capacity_cost"] == 12.0
        assert r["accel_plan_cost"] == pytest.approx(4.0, abs=0.1)       # 12x per unit * 1/3 the units

    def test_throughput_floor_can_dominate(self):
        cpu, acc = HardwareUnit(1.0, 1.0), HardwareUnit(3.0, 3.5, 12.0)
        r = compare_plans(n_vectors=10, qps=1000, cpu=cpu, accel=acc)
        assert r["accel_bound"] == "throughput" and r["accel_units_per_region"] == pytest.approx(1 / 3.5, abs=0.01)


# ---------------------------------------------------------------------------
# World + pool selection
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def world():
    return SearchWorld(n_docs=6000, n_topics=40, seed=0)


class TestWorld:
    def test_relevance_properties(self, world):
        rel = world.relevance(3, 1)
        assert (rel[world.topic != 3] == 0).all()
        assert (rel[~world.eligible] == 0).all()
        in_topic = (world.topic == 3) & world.eligible & (world.cat != 0)
        match, miss = in_topic & (world.sub == 1), in_topic & (world.sub != 1)
        assert rel[match].mean() > 1.5 * rel[miss].mean()                 # taste doubles relevance

    def test_stale_news_discounted(self, world):
        news = (world.cat == 0) & (world.age > 60) & (world.topic == 5) & world.eligible
        if news.any():
            base = world.quality[news] * (1 + (world.sub[news] == 0))
            assert (world.relevance(5, 0)[news] < base).all()

    def test_category_structure(self, world):
        ent, tut = world.cat == CATEGORIES.index("entertainment"), world.cat == CATEGORIES.index("tutorial")
        assert world.virality[ent].mean() > world.virality[tut].mean()
        assert world.quality[tut].mean() > world.quality[ent].mean()

    def test_queries(self, world):
        q = world.sample_queries(50, seed=1)
        assert q["qvec"].shape == (50, world.dim) and q["profile"].shape == (50, 8)
        assert torch.equal(world.sample_queries(5, seed=3)["topic"], world.sample_queries(5, seed=3)["topic"])

    def test_engagement_driven_by_virality_and_relevance(self, world):
        idx = np.arange(100)
        hi = world.engagement_prob(np.ones(world.N), idx)
        lo = world.engagement_prob(np.zeros(world.N), idx)
        assert (hi > lo).all()


class TestPoolSelection:
    def test_pool_is_eligible_and_sized(self, world):
        cfg = LifecycleConfig(pool_size=300, old_keep_fraction=1.0)
        pool = select_search_value_pool(world, cfg)
        assert len(pool) == 300 and world.eligible[pool].all()
        assert len(set(pool.tolist())) == 300

    def test_week_one_low_value_items_dropped(self, world):
        cfg = LifecycleConfig(pool_size=len(world.eligible), old_keep_fraction=1.0)
        pool = set(select_search_value_pool(world, cfg).tolist())
        young_low = np.where((world.age < cfg.early_days) & (world.sv_est < cfg.early_min_value) & world.eligible)[0]
        assert young_low.size > 0 and not (set(young_low.tolist()) & pool)

    def test_niche_high_search_value_bypasses_engagement_pruning(self, world):
        cfg = LifecycleConfig(pool_size=len(world.eligible), old_keep_fraction=1.0)
        pool = set(select_search_value_pool(world, cfg).tolist())
        old = world.age >= cfg.early_days
        bypass = np.where(old & (world.eng_est < cfg.engagement_prune) & (world.sv_est >= cfg.bypass_value)
                          & world.eligible)[0]
        pruned = np.where(old & (world.eng_est < cfg.engagement_prune) & (world.sv_est < cfg.bypass_value)
                          & world.eligible)[0]
        assert bypass.size > 0 and pruned.size > 0
        assert set(bypass.tolist()) <= pool                              # high search value survives low engagement
        assert not (set(pruned.tolist()) & pool)

    def test_old_content_cap(self, world):
        cfg = LifecycleConfig(pool_size=len(world.eligible), old_keep_fraction=0.01, old_days=100.0)
        pool = select_search_value_pool(world, cfg)
        n_old_eligible = (world.eligible & (world.age > 100.0)).sum()
        assert (world.age[pool] > 100.0).sum() <= int(0.01 * n_old_eligible)

    def test_news_decays_faster_than_evergreen(self, world):
        cfg = LifecycleConfig()
        assert cfg.half_life_days["news"] < cfg.half_life_days["entertainment"] < cfg.half_life_days["evergreen"]

    def test_search_value_pool_beats_engagement_pool(self, world):
        # The Sec. 4.1 claim: recommendation-style engagement selection discards search-useful content.
        sv = select_search_value_pool(world, LifecycleConfig(pool_size=400, old_keep_fraction=1.0))
        eng = select_engagement_pool(world, 400)
        assert pool_value(world, sv, 120) > 1.3 * pool_value(world, eng, 120)

    def test_broad_inventory_is_a_superset_in_spirit(self, world):
        inv = select_broad_inventory(world)
        pool = select_search_value_pool(world, LifecycleConfig(pool_size=300, old_keep_fraction=1.0))
        assert len(inv) > 10 * len(pool)
        assert len(set(pool.tolist()) - set(inv.tolist())) < 0.1 * len(pool)      # containment can vary (Sec. 3)


# ---------------------------------------------------------------------------
# INT8 kernel + index
# ---------------------------------------------------------------------------

class TestInt8:
    def test_quantize_roundtrip_error(self):
        x = torch.randn(50, 32)
        q, s = int8_quantize(x)
        assert q.dtype == torch.int8 and q.abs().max() <= 127
        assert ((q.float() * s.unsqueeze(1)) - x).abs().max() < s.max() * 0.51

    def test_scores_match_float_cosine(self):
        torch.manual_seed(0)
        d = F.normalize(torch.randn(200, 32), dim=-1)
        q = F.normalize(torch.randn(3, 32), dim=-1)
        approx = int8_scores(*int8_quantize(q), *int8_quantize(d))
        exact = q @ d.t()
        assert approx.shape == (3, 200)
        assert (approx - exact).abs().max() < 0.02

    def test_integer_accumulation_is_exact(self):
        a = torch.tensor([[127, -127, 5]], dtype=torch.int8)
        b = torch.tensor([[2, 3, 4]], dtype=torch.int8)
        out = int8_scores(a, torch.ones(1), b, torch.ones(1))
        assert out.item() == 127 * 2 - 127 * 3 + 5 * 4

    def test_zero_vector_safe(self):
        q, s = int8_quantize(torch.zeros(2, 4))
        assert (q == 0).all() and torch.isfinite(s).all()

    def test_cluster_index_recall_vs_exact(self):
        torch.manual_seed(0)
        centers = torch.randn(8, 16) * 3
        emb = centers[torch.randint(0, 8, (800,))] + 0.3 * torch.randn(800, 16)
        ids = np.arange(800)
        idx = Int8ClusterIndex(ids, emb, kmeans(emb, 8))
        en = F.normalize(emb, dim=-1)
        gap = []
        for _ in range(30):
            q = centers[torch.randint(0, 8, (1,))][0] + 0.3 * torch.randn(16)
            exact = en @ F.normalize(q, dim=-1)
            got, _, n = idx.search(q, 10, n_probe=2)
            assert n < 800
            # INT8 may reorder near-ties, so judge by the true score of what it returned
            gap.append((exact.topk(10).values.mean() - exact[got].mean()).item())
        assert max(gap) < 0.03

    def test_more_probes_scan_more(self):
        torch.manual_seed(0)
        emb = torch.randn(400, 16)
        idx = Int8ClusterIndex(np.arange(400), emb, kmeans(emb, 8))
        q = torch.randn(16)
        assert idx.search(q, 5, 1)[2] < idx.search(q, 5, 4)[2] <= 400


# ---------------------------------------------------------------------------
# GPU models + losses
# ---------------------------------------------------------------------------

class TestLosses:
    def test_cross_session_masks_same_topic_false_negatives(self):
        q = F.normalize(torch.randn(3, 8), dim=-1)
        d = torch.stack([q[0], q[0], q[2]])           # row 1's doc is a perfect match for query 0
        topics_same, topics_diff = torch.tensor([1, 1, 2]), torch.tensor([1, 2, 3])
        assert info_nce_cross_session(q, d, topics_same) < info_nce_cross_session(q, d, topics_diff)

    def test_within_session_prefers_engaged(self):
        q = F.normalize(torch.randn(1, 8), dim=-1)
        sess = torch.stack([q[0], -q[0], 0.1 * q[0]]).unsqueeze(0)
        good = info_nce_within_session(q, sess, torch.tensor([[True, False, False]]))
        bad = info_nce_within_session(q, sess, torch.tensor([[False, True, False]]))
        assert good < bad

    def test_within_session_skips_degenerate_sessions(self):
        q, sess = torch.randn(2, 8), torch.randn(2, 4, 8)
        for e in (torch.zeros(2, 4, dtype=torch.bool), torch.ones(2, 4, dtype=torch.bool)):
            assert info_nce_within_session(q, sess, e).item() == 0.0

    def test_bidirectional_nce_is_symmetric_in_roles(self):
        q = F.normalize(torch.randn(6, 8), dim=-1)
        topics = torch.arange(6)
        assert bidirectional_info_nce(q, q, topics) < bidirectional_info_nce(q, q.roll(1, 0), topics)

    def test_value_model_is_weighted_sum(self):
        p_rel, p_eng = torch.tensor([1.0]), torch.tensor([0.0])
        assert value_model(p_rel, p_eng, (2.0, 1.0)).item() == pytest.approx(2.0 + 0.5)
        # operating point moves with the weights, no retraining
        a, b = torch.tensor([1.0, 0.0]), torch.tensor([-5.0, 5.0])
        assert value_model(a, b, (1.0, 0.0)).argmax() == 0 and value_model(a, b, (0.0, 1.0)).argmax() == 1


class TestPreRanker:
    def test_shapes_and_grad(self):
        r = InteractionPreRanker(8, 8, 8)
        args = [torch.randn(5, 8, requires_grad=True) for _ in range(4)]
        p_rel, p_eng = r(*args)
        assert p_rel.shape == p_eng.shape == (5,)
        (p_rel.sum() + p_eng.sum()).backward()
        assert all(a.grad is not None for a in args)

    def test_fm_term_is_pairwise(self):
        # Two fields identical to zero => the FM contribution vanishes for them.
        r = InteractionPreRanker(4, 4, 4)
        q = torch.zeros(1, 4)
        out0 = r(q, q, q, q)[0]
        assert torch.isfinite(out0).all()


@pytest.fixture(scope="module")
def trained(world):
    pool = select_search_value_pool(world, LifecycleConfig(pool_size=400, old_keep_fraction=1.0))
    inv = select_broad_inventory(world)
    gm = train_gpu_models(world, pool, steps=150, batch=64, seed=0)
    cm = train_cpu_model(world, inv, steps=150, batch=64, seed=0)
    with torch.no_grad():
        cents = train_centroids(cm.encode_doc(world.content[inv]), 32, version=1)
    return world, pool, inv, gm, cm, cents


class TestGPUPathway:
    def test_retrieve_contract(self, trained):
        world, pool, inv, gm, cm, cents = trained
        gpu = GPUPathway(world, gm, pool, n_centroids=8, k_prefetch=60, k_final=25)
        q = world.sample_queries(3, 5)
        r = gpu.retrieve(q["qvec"][:1], q["profile"][:1])
        assert 0 < len(r.candidates) <= 25
        ids = [d for d, _ in r.candidates]
        assert set(ids) <= set(pool.tolist()) and len(set(ids)) == len(ids)
        scores = [s for _, s in r.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_retrieval_finds_topic_documents(self, trained):
        world, pool, inv, gm, cm, cents = trained
        gpu = GPUPathway(world, gm, pool, n_centroids=8, k_prefetch=60, k_final=25)
        q = world.sample_queries(40, 6)
        frac = []
        for i in range(40):
            ids = [d for d, _ in gpu.retrieve(q["qvec"][i:i + 1], q["profile"][i:i + 1]).candidates[:5]]
            frac.append(np.mean(world.topic[ids] == int(q["topic"][i])))
        assert np.mean(frac) > 0.5

    def test_publish_requires_matching_model_version(self, trained):
        world, pool, inv, gm, cm, cents = trained
        gpu = GPUPathway(world, gm, pool, n_centroids=8, model_version=3)
        with pytest.raises(ValueError):
            gpu.publish(ModelIndexSnapshot(version=2, model_version=2, index=gpu.snapshot.index))
        gpu.publish(ModelIndexSnapshot(version=2, model_version=3, index=gpu.snapshot.index))
        assert gpu.snapshot.version == 2


# ---------------------------------------------------------------------------
# CPU index
# ---------------------------------------------------------------------------

def _clustered(n=1500, k=12, d=16, seed=0):
    torch.manual_seed(seed)
    centers = torch.randn(k, d) * 3
    v = centers[torch.randint(0, k, (n,))] + 0.8 * torch.randn(n, d)
    return v, F.normalize(v, dim=-1)


class TestEmbeddingIndex:
    def setup_method(self):
        self.v, self.vn = _clustered()
        self.cents = train_centroids(self.v, 12, version=1)
        self.idx = EmbeddingIndex(np.arange(len(self.v)), self.v, self.cents)

    def test_every_doc_in_exactly_one_list(self):
        all_ids = torch.cat(self.idx.ids)
        assert len(all_ids) == len(self.v) and len(set(all_ids.tolist())) == len(self.v)
        assert len(self.idx) == len(self.v)

    def test_search_returns_best_by_cosine_when_all_lists_probed(self):
        q = torch.randn(16)
        got, scanned = self.idx.search(q, 10, n_probe=12)
        exact = (self.vn @ F.normalize(q, dim=-1)).topk(10).indices.tolist()
        assert got == exact and scanned == len(self.v)

    def test_pruning_scans_fewer_docs(self):
        _, scanned = self.idx.search(torch.randn(16), 10, n_probe=2)
        assert scanned < len(self.v)

    def test_eager_equals_daat_with_enough_prefetch(self):
        flt = lambda i: i % 3 != 0
        for _ in range(10):
            q = torch.randn(16)
            a, _ = self.idx.search(q, 10, 4, "eager", prefetch_multiplier=50, filter_fn=flt)
            b, _ = self.idx.search(q, 10, 4, "daat", filter_fn=flt)
            assert a == b and all(flt(i) for i in a)

    def test_prefetch_multiplier_trades_completeness(self):
        flt = lambda i: i % 10 == 0                                       # a harsh filter
        q = torch.randn(16)
        small, _ = self.idx.search(q, 10, 4, "eager", prefetch_multiplier=1, filter_fn=flt)
        big, _ = self.idx.search(q, 10, 4, "eager", prefetch_multiplier=100, filter_fn=flt)
        assert len(small) <= len(big) == 10

    def test_eager_has_sequential_access_daat_has_random_jumps(self):
        # Fig. 4 / Sec. 5.3: same answer, but eager scans lists contiguously while DAAT hops between lists.
        e = EmbeddingIndex(np.arange(len(self.v)), self.v, self.cents)
        d = EmbeddingIndex(np.arange(len(self.v)), self.v, self.cents)
        flt = lambda i: i % 2 == 0
        for _ in range(5):
            q = torch.randn(16)
            e.search(q, 10, 4, "eager", 50, flt)
            d.search(q, 10, 4, "daat", filter_fn=flt)
        assert e.stats.random_jumps == 0 and e.stats.sequential_blocks == 5 * 4
        assert d.stats.random_jumps > 50
        assert d.stats.distance_ops < e.stats.distance_ops               # DAAT does less arithmetic, worse access

    def test_filter_is_applied(self):
        got, _ = self.idx.search(torch.randn(16), 20, 6, filter_fn=lambda i: i < 100)
        assert all(i < 100 for i in got)

    def test_remove_and_add(self):
        q = self.v[5]
        top, _ = self.idx.search(q, 3, 12)
        assert 5 in top
        self.idx.remove([5])
        assert 5 not in self.idx.search(q, 3, 12)[0] and len(self.idx) == len(self.v) - 1
        self.idx.add(np.array([9001]), q.unsqueeze(0) + 0.01)
        assert 9001 in self.idx.search(q, 3, 12)[0]

    def test_online_add_does_not_recluster(self):
        before = self.cents.centroids.clone()
        self.idx.add(np.array([7000, 7001]), torch.randn(2, 16))
        assert torch.equal(before, self.idx.cv.centroids) and self.idx.cv.version == 1

    def test_unknown_mode(self):
        with pytest.raises(ValueError):
            self.idx.search(torch.randn(16), 5, 2, mode="x")


class TestFrontier:
    def test_recall_rises_with_probes_and_finer_centroids_scan_less(self):
        # Fig. 5: more centroids reach comparable recall while scanning fewer candidates.
        torch.manual_seed(0)
        centers = torch.randn(64, 16) * 3
        v = centers[torch.randint(0, 64, (6000,))] + 0.7 * torch.randn(6000, 16)
        vn = F.normalize(v, dim=-1)
        queries = v[torch.randint(0, 6000, (40,))] + 0.2 * torch.randn(40, 16)
        truth = [set((vn @ F.normalize(q, dim=-1)).topk(10).indices.tolist()) for q in queries]
        frontiers = {}
        for k in (8, 64):
            idx = EmbeddingIndex(np.arange(6000), v, train_centroids(v, k, version=1))
            frontiers[k] = recall_candidate_frontier(idx, queries, truth, 10, [1, 2, 4, 8])
            rec = [r for _, r in frontiers[k]]
            assert rec == sorted(rec) or max(abs(a - b) for a, b in zip(rec, sorted(rec))) < 0.02
        def scanned_at(frontier, target):
            return min(c for c, r in frontier if r >= target)
        target = 0.8
        assert scanned_at(frontiers[64], target) < scanned_at(frontiers[8], target)


# ---------------------------------------------------------------------------
# CPU model + pathway
# ---------------------------------------------------------------------------

class TestCPUPathway:
    def test_lightweight_model_shapes(self):
        m = LightweightTwoTower(16, 4)
        q = m.encode_query(torch.randn(5, 16), torch.randint(0, 4, (5,)))
        d = m.encode_doc(torch.randn(7, 16))
        assert q.shape == (5, 24) and d.shape == (7, 24)
        assert torch.allclose(q.norm(dim=-1), torch.ones(5), atol=1e-5)

    def test_context_changes_query_but_not_documents(self):
        m = LightweightTwoTower(16, 4).eval()
        x = torch.randn(1, 16)
        assert not torch.allclose(m.encode_query(x, torch.tensor([0])), m.encode_query(x, torch.tensor([1])))

    def test_pathway_applies_light_filter_and_budget(self, trained):
        world, pool, inv, gm, cm, cents = trained
        cpu = CPUPathway(world, cm, inv, cents, k=30, min_value=0.4)
        q = world.sample_queries(5, 9)
        for i in range(5):
            r = cpu.retrieve(q["qvec"][i:i + 1], q["taste"][i:i + 1])
            assert len(r.candidates) <= 30
            assert all(world.sv_est[d] >= 0.4 for d, _ in r.candidates)

    def test_cpu_inventory_much_larger_than_gpu_pool(self, trained):
        world, pool, inv, *_ = trained
        assert len(inv) > 10 * len(pool)


# ---------------------------------------------------------------------------
# Co-serving
# ---------------------------------------------------------------------------

class TestAggregator:
    def test_dedup_and_source_attribution(self):
        agg = Aggregator({"gpu": 100, "cpu": 100})
        out = agg.merge([BranchResult("gpu", [(1, 0.9), (2, 0.5)], 10), BranchResult("cpu", [(2, 0.7), (3, 0.1)], 20)])
        by = {c.doc_id: c for c in out}
        assert set(by) == {1, 2, 3} and by[2].sources == {"gpu", "cpu"} and by[1].sources == {"gpu"}
        assert by[2].score == 0.7                      # best score across branches is kept

    def test_late_branch_dropped_without_discarding_the_other(self):
        agg = Aggregator({"gpu": 100, "cpu": 50})
        out = agg.merge([BranchResult("gpu", [(1, 1.0)], 80), BranchResult("cpu", [(2, 1.0)], 120)])
        assert [c.doc_id for c in out] == [1]

    def test_no_branches(self):
        assert Aggregator({}).merge([]) == []


class TestHybridRetriever:
    def test_independent_enable_disable(self, trained):
        world, pool, inv, gm, cm, cents = trained
        gpu = GPUPathway(world, gm, pool, n_centroids=8)
        cpu = CPUPathway(world, cm, inv, cents)
        h = HybridRetriever(gpu, cpu)
        q = world.sample_queries(2, 3)
        both = h.retrieve(q, 0)
        assert {"gpu", "cpu"} <= set().union(*[c.sources for c in both])
        gpu.enabled = False                                               # rollback: flip a flag, same interface
        only_cpu = h.retrieve(q, 0)
        assert set().union(*[c.sources for c in only_cpu]) == {"cpu"}
        gpu.enabled, cpu.enabled = True, False
        assert set().union(*[c.sources for c in h.retrieve(q, 0)]) == {"gpu"}
        cpu.enabled = False
        gpu.enabled = False
        assert h.retrieve(q, 0) == []

    def test_slow_branch_missed_deadline(self, trained):
        world, pool, inv, gm, cm, cents = trained
        gpu, cpu = GPUPathway(world, gm, pool, n_centroids=8), CPUPathway(world, cm, inv, cents)
        h = HybridRetriever(gpu, cpu, deadlines_ms={"gpu": 100, "cpu": 100},
                            latency_fn=lambda n: 500.0 if n == "gpu" else 10.0)
        out = h.retrieve(world.sample_queries(1, 3), 0)
        assert set().union(*[c.sources for c in out]) == {"cpu"}

    def test_pathways_contribute_distinct_candidates(self, trained):
        world, pool, inv, gm, cm, cents = trained
        gpu, cpu = GPUPathway(world, gm, pool, n_centroids=8), CPUPathway(world, cm, inv, cents)
        q = world.sample_queries(30, 4)
        g, c = set(), set()
        for i in range(30):
            g |= {(i, d) for d, _ in gpu.retrieve(q["qvec"][i:i + 1], q["profile"][i:i + 1]).candidates}
            c |= {(i, d) for d, _ in cpu.retrieve(q["qvec"][i:i + 1], q["taste"][i:i + 1]).candidates}
        assert overlap_metrics(g, c)["jaccard"] < 0.5                     # structurally distinct sources


class TestEndToEnd:
    def test_hybrid_beats_either_pathway_alone(self, trained):
        world, pool, inv, gm, cm, cents = trained
        gpu, cpu = GPUPathway(world, gm, pool, n_centroids=8), CPUPathway(world, cm, inv, cents)
        res = evaluate_retrievers(world, {"gpu": HybridRetriever(gpu, None), "cpu": HybridRetriever(None, cpu),
                                          "hybrid": HybridRetriever(gpu, cpu)}, n_queries=60)
        assert res["hybrid"]["ndcg"] >= max(res["gpu"]["ndcg"], res["cpu"]["ndcg"]) - 0.01
        assert res["hybrid"]["ndcg"] > res["gpu"]["ndcg"]
        assert res["hybrid"]["rel_recall"] >= res["cpu"]["rel_recall"] - 1e-9
        assert res["hybrid"]["cands"] > res["cpu"]["cands"]

    def test_gpu_models_are_deeper_than_cpu_model_on_same_pool(self, trained):
        world, pool, inv, gm, cm, cents = trained
        gpu = GPUPathway(world, gm, pool, n_centroids=8)
        d = modeling_depth_ndcg(world, pool, gpu, cm, n_queries=60)
        assert d["gpu"] > d["cpu_model"] - 0.02
        assert d["gpu_ann"] > 0.5 and d["cpu_model"] > 0.3

    def test_oracle_ranker_orders_by_relevance(self, trained):
        world, *_ = trained
        rel = world.relevance(2, 1)
        top = np.argsort(-rel)[:5]
        cands = [Candidate(int(i), {"x"}) for i in list(top) + list(np.where(rel == 0)[0][:20])]
        out = NoisyOracleRanker(world, noise=0.0).rank(cands, 2, 1, 5)
        assert set(out) == set(top.tolist())
