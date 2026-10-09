"""Tests for the Facebook EBR implementation.

Run with:
    pytest test_facebook_ebr.py -v
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from facebook_ebr import (
    FeatureConfig, FeatureStore, IMIIndex, IVFIndex, OPQTransform, PCATransform, ProductQuantizer, RelevanceFilter,
    SocialSearchWorld, TrainConfig, UnicornLite, UnifiedEmbeddingModel, add_typo, cascade_rerank, char_ngrams,
    cosine_distance, embed_index, embed_queries, embedding_features, ensemble_similarity, evaluate, hash_id, kmeans,
    mine_hard_positives, mine_offline_negatives, mine_online_hard, one_recall_at_k, pad_ids, parse_sexpr, recall_at_k,
    select_index_docs, should_trigger_ebr, text_feature_ids, train_model, triplet_loss, weighted_ensemble_vectors,
    word_ngrams,
)
import random


# ---------------------------------------------------------------------------
# Text features
# ---------------------------------------------------------------------------

class TestTextFeatures:
    def test_char_trigrams(self):
        assert char_ngrams("ab") == ["#ab", "ab#"]
        assert char_ngrams("abc") == ["#ab", "abc", "bc#"]

    def test_word_ngrams(self):
        assert word_ngrams("john smith") == ["john", "smith", "john smith"]
        assert word_ngrams("x") == ["x"]

    def test_hash_is_stable_in_range_and_nonzero(self):
        ids = [hash_id(t, 100) for t in ["a", "b", "john"]]
        assert ids == [hash_id(t, 100) for t in ["a", "b", "john"]]
        assert all(1 <= i <= 100 for i in ids)

    def test_typo_shares_most_trigrams(self):
        a, b = set(char_ngrams("kasiecreations")), set(char_ngrams("kasiecraetions"))   # one swap
        assert len(a & b) / len(a | b) > 0.5

    def test_word_ngrams_optional(self):
        assert len(text_feature_ids("john smith", use_word_ngrams=True)) > len(text_feature_ids("john smith", use_word_ngrams=False))

    def test_pad(self):
        t = pad_ids([[1, 2, 3], [4]])
        assert t.shape == (2, 3) and t[1].tolist() == [4, 0, 0]


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def world():
    return SocialSearchWorld(n_entities=1200, n_names=240, seed=0)


class TestWorld:
    def test_many_entities_share_names(self, world):
        sizes = [len(v) for v in world.by_name.values()]
        assert np.mean(sizes) == pytest.approx(5.0, abs=0.5) and max(sizes) > 5

    def test_target_has_above_average_affinity_among_same_name_entities(self, world):
        s = world.sample_sessions(300, seed=3)
        gain = []
        for i in range(len(s)):
            cands = world.by_name[int(s.name_idx[i])]
            aff = world.affinity(cands, int(s.loc[i]), int(s.soc[i, 0]), int(s.soc[i, 1]))
            gain.append(aff[list(cands).index(s.target[i])] - aff.mean())
        assert np.mean(gain) > 0.5                                    # context, not chance, picks the target

    def test_context_resolves_same_name_ambiguity(self, world):
        s = world.sample_sessions(400, seed=4)
        names = {}
        for i in range(len(s)):
            names.setdefault(int(s.name_idx[i]), set()).add(int(s.target[i]))
        assert any(len(v) > 1 for v in names.values())                # same name, different targets for different searchers

    def test_impressions_contain_target_and_hard_cases(self, world):
        s = world.sample_sessions(100, seed=5)
        for i in range(len(s)):
            assert s.target[i] in s.impressions[i]
        same_name = np.mean([np.mean(world.entity_name[s.impressions[i]] == s.name_idx[i]) for i in range(len(s))])
        assert same_name > 0.3

    def test_exact_matching_fails_on_typos_and_extra_terms(self, world):
        s = world.sample_sessions(600, seed=6)
        ok = np.array([int(t) in set(world.boolean_match(q).tolist()) for q, t in zip(s.query_text, s.target)])
        clean = ~s.has_typo & ~s.has_extra
        assert ok[clean].all()
        assert not ok[s.has_typo | s.has_extra].any() or ok[s.has_typo | s.has_extra].mean() < 0.2

    def test_typo_changes_string_by_one_edit(self):
        rng = random.Random(0)
        for _ in range(50):
            t = add_typo("johnson", rng)
            assert abs(len(t) - 7) <= 1

    def test_subset(self, world):
        s = world.sample_sessions(20, seed=7)
        sub = s.subset([2, 5])
        assert len(sub) == 2 and sub.query_text == [s.query_text[2], s.query_text[5]]
        assert sub.target.tolist() == [s.target[2], s.target[5]]

    def test_seeded_sessions_reproducible(self, world):
        a, b = world.sample_sessions(10, seed=9), world.sample_sessions(10, seed=9)
        assert a.query_text == b.query_text and (a.target == b.target).all()


# ---------------------------------------------------------------------------
# Model + losses
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sessions(world):
    return world.sample_sessions(2500, seed=1), world.sample_sessions(500, seed=2)


class TestModel:
    def test_encoders_unit_norm_and_shapes(self, world, sessions):
        store = FeatureStore(world, FeatureConfig())
        m = UnifiedEmbeddingModel(world)
        q = m.encode_query(store.queries(sessions[0].subset(range(8))))
        d = m.encode_doc(store.docs())
        assert q.shape == (8, 64) and d.shape == (world.n_entities, 64)      # 32 text + 16 location + 16 social
        assert torch.allclose(q.norm(dim=-1), torch.ones(8), atol=1e-5)

    def test_shared_towers(self, world):
        m = UnifiedEmbeddingModel(world, shared=True)
        assert m.f is m.g and UnifiedEmbeddingModel(world).f is not UnifiedEmbeddingModel(world).g

    def test_feature_flags_remove_context_dependence(self, world, sessions):
        s = sessions[0].subset(range(6))
        cfg = FeatureConfig(use_location=False, use_social=False)
        store = FeatureStore(world, cfg)
        m = UnifiedEmbeddingModel(world, cfg).eval()
        f = store.queries(s)
        g = dict(f, loc=(f["loc"] + 3) % world.L, soc_idx=(f["soc_idx"] + 5) % world.C)
        assert torch.allclose(m.encode_query(f), m.encode_query(g))
        on = UnifiedEmbeddingModel(world, FeatureConfig()).eval()
        assert not torch.allclose(on.encode_query(f), on.encode_query(g))

    def test_social_weights_matter(self, world, sessions):
        store = FeatureStore(world, FeatureConfig())
        m = UnifiedEmbeddingModel(world).eval()
        f = store.queries(sessions[0].subset(range(5)))
        g = dict(f, soc_w=torch.tensor([[1.0, 0.0]] * 5))
        assert not torch.allclose(m.encode_query(f), m.encode_query(g))

    def test_triplet_loss_values(self):
        q = torch.tensor([[1.0, 0.0]])
        pos, neg_far, neg_close = torch.tensor([[1.0, 0.0]]), torch.tensor([[-1.0, 0.0]]), torch.tensor([[1.0, 0.0]])
        assert triplet_loss(q, pos, neg_far, 0.3).item() == 0.0              # margin satisfied
        assert triplet_loss(q, pos, neg_close, 0.3).item() == pytest.approx(0.3)
        assert cosine_distance(q, neg_far).item() == pytest.approx(2.0)

    def test_training_improves_over_untrained(self, world, sessions):
        store = FeatureStore(world, FeatureConfig())
        fresh = UnifiedEmbeddingModel(world)
        trained = train_model(world, store, sessions[0], TrainConfig(steps=150, batch=128))
        assert evaluate(trained, store, sessions[1], (10,))[10] > evaluate(fresh, store, sessions[1], (10,))[10] + 0.3

    def test_unified_beats_text_only_on_personalized_recall_at_1(self, world, sessions):
        # The paper's central modeling claim: searcher context belongs in the embedding.
        tc = TrainConfig(steps=400, batch=128, n_online_hard=2)
        res = {}
        for name, cfg in [("text", FeatureConfig(use_location=False, use_social=False)), ("full", FeatureConfig())]:
            store = FeatureStore(world, cfg)
            res[name] = evaluate(train_model(world, store, sessions[0], tc), store, sessions[1], (1,))[1]
        assert res["full"] > res["text"] + 0.08

    def test_recall_at_k_definition(self, world, sessions):
        s = sessions[1].subset(range(4))
        index = torch.zeros(world.n_entities, 4)
        for i, tgt in enumerate(s.target):
            index[tgt, i] = 1.0                               # each target owns one axis
        q = torch.eye(4)
        assert recall_at_k(q, index, s, 1) == 1.0
        assert recall_at_k(q.roll(1, 0), index, s, 1) == 0.0   # queries pointing at the wrong axes
        assert recall_at_k(q.roll(1, 0), index, s, world.n_entities) == 1.0


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------

class TestMining:
    def test_online_hardest_negatives(self):
        q = F.normalize(torch.randn(4, 8), dim=-1)
        pos = F.normalize(torch.randn(4, 8), dim=-1)
        ids = torch.tensor([0, 1, 2, 3])
        h = mine_online_hard(q, pos, ids, 2)
        assert h.shape == (4, 2)
        for i in range(4):
            sims = q[i] @ pos.t()
            sims[i] = -9
            assert set(h[i].tolist()) == set(sims.topk(2).indices.tolist())

    def test_online_hard_excludes_same_entity(self):
        q = F.normalize(torch.randn(3, 4), dim=-1)
        pos = q.clone()                                       # q_i is closest to pos_i (and pos_j when ids equal)
        h = mine_online_hard(q, pos, torch.tensor([7, 7, 9]), 1)
        assert h[0, 0] == 2 and h[1, 0] == 2 and h[2, 0] != 2

    def test_offline_negatives_in_requested_rank_band(self, world, sessions):
        store = FeatureStore(world, FeatureConfig())
        m = train_model(world, store, sessions[0], TrainConfig(steps=60, batch=64))
        s = sessions[0].subset(range(100))
        negs = mine_offline_negatives(m, store, s, 3, 8)
        assert len(negs) == 100 and (negs != s.target).all()
        q, idx = embed_queries(m, store, s), embed_index(m, store)
        order = (q @ idx.t()).argsort(1, descending=True).numpy()
        for i in range(100):
            ranked = [e for e in order[i] if e != s.target[i]]
            assert int(negs[i]) in ranked[2:8]

    def test_hard_positives_are_exactly_the_failed_sessions(self, world, sessions):
        s = sessions[0]
        hp = set(mine_hard_positives(world, s).tolist())
        for i in range(0, len(s), 25):
            failed = int(s.target[i]) not in set(world.boolean_match(s.query_text[i]).tolist())
            assert (i in hp) == failed
        assert all(s.has_typo[i] or s.has_extra[i] for i in hp)

    def test_non_click_impressions_as_negatives_is_much_worse(self, world, sessions):
        # Sec. 2.4: all-hard negatives misrepresent the retrieval task (paper: -55% recall).
        store = FeatureStore(world, FeatureConfig())
        rnd = train_model(world, store, sessions[0], TrainConfig(steps=200, batch=128, neg="random"))
        hard = train_model(world, store, sessions[0], TrainConfig(steps=200, batch=128, neg="non_click"))
        r_rnd, r_hard = evaluate(rnd, store, sessions[1], (10,))[10], evaluate(hard, store, sessions[1], (10,))[10]
        assert r_hard < 0.5 * r_rnd

    def test_unknown_negative_mode(self, world, sessions):
        with pytest.raises(ValueError):
            train_model(world, FeatureStore(world, FeatureConfig()), sessions[0], TrainConfig(steps=1, neg="x"))

    def test_offline_mode_requires_negatives(self, world, sessions):
        with pytest.raises(AssertionError):
            train_model(world, FeatureStore(world, FeatureConfig()), sessions[0], TrainConfig(steps=1, neg="offline"))


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class TestEnsemble:
    def test_concatenation_cosine_is_proportional_to_weighted_sum(self):
        torch.manual_seed(0)
        vq = [torch.randn(5, 8), torch.randn(5, 6)]
        vd = [torch.randn(5, 8), torch.randn(5, 6)]
        alphas = [0.7, 1.9]
        eq, ed = weighted_ensemble_vectors(vq, vd, alphas)
        cos = F.cosine_similarity(eq, ed)
        sw = ensemble_similarity(vq, vd, alphas)
        expected = sw / (np.sqrt(sum(a * a for a in alphas)) * np.sqrt(len(alphas)))        # Eq. (6)
        assert torch.allclose(cos, expected, atol=1e-5)

    def test_weights_on_one_side_only(self):
        vq, vd = [torch.randn(3, 4)] * 2, [torch.randn(3, 4)] * 2
        eq, ed = weighted_ensemble_vectors(vq, vd, [2.0, 1.0])
        assert ed.shape == (3, 8) and torch.allclose(ed.norm(dim=-1), torch.full((3,), 2 ** 0.5), atol=1e-5)

    def test_equal_weights_reduce_to_mean_cosine(self):
        vq, vd = [torch.randn(4, 5), torch.randn(4, 5)], [torch.randn(4, 5), torch.randn(4, 5)]
        eq, ed = weighted_ensemble_vectors(vq, vd, [1.0, 1.0])
        mean_cos = 0.5 * (F.cosine_similarity(vq[0], vd[0]) + F.cosine_similarity(vq[1], vd[1]))
        assert torch.allclose(F.cosine_similarity(eq, ed), mean_cos, atol=1e-5)

    def test_cascade_reorders_only_stage_one_candidates(self):
        index2 = torch.eye(6)
        q2 = torch.zeros(1, 6)
        q2[0, 5] = 1.0
        cands = torch.tensor([[0, 2, 3]])                      # entity 5 would be the best, but stage one never returned it
        out = cascade_rerank(cands, q2, index2, 2)
        assert set(out[0].tolist()) <= {0, 2, 3} and out.shape == (1, 2)
        q2[0, 3], q2[0, 5] = 2.0, 0.0
        assert cascade_rerank(cands, q2, index2, 1)[0, 0] == 3


# ---------------------------------------------------------------------------
# ANN
# ---------------------------------------------------------------------------

def _clustered(n=1200, d=16, k=10, seed=0):
    torch.manual_seed(seed)
    centers = torch.randn(k, d) * 3
    return centers[torch.randint(0, k, (n,))] + 0.6 * torch.randn(n, d)


class TestPQ:
    def test_reconstruction_error_falls_with_more_subquantizers(self):
        x = _clustered()
        errs = []
        for m in (2, 4, 8):
            pq = ProductQuantizer(16, m, nbits=5).train(x)
            errs.append(((pq.decode(pq.encode(x)) - x) ** 2).sum(1).mean().item())
        assert errs[0] > errs[1] > errs[2]

    def test_adc_matches_distance_to_decoded_vector(self):
        x = _clustered()
        pq = ProductQuantizer(16, 4, nbits=5).train(x)
        codes, q = pq.encode(x[:50]), x[100]
        adc = pq.adc(pq.distance_table(q), codes)
        exact = ((pq.decode(codes) - q) ** 2).sum(1)
        assert torch.allclose(adc, exact, atol=1e-3)

    def test_dimension_must_divide(self):
        with pytest.raises(AssertionError):
            ProductQuantizer(10, 3)

    def test_codes_are_bytes(self):
        pq = ProductQuantizer(16, 4, nbits=8).train(_clustered(600))
        assert pq.encode(torch.randn(3, 16)).dtype == torch.uint8


class TestTransforms:
    def _anisotropic(self):
        torch.manual_seed(0)
        scales = torch.tensor([6.0] * 4 + [0.3] * 12)             # variance concentrated in few dims (bad for PQ)
        x = torch.randn(1500, 16) * scales
        q, _ = torch.linalg.qr(torch.randn(16, 16))
        return x @ q

    def test_opq_rotation_is_orthogonal_and_lowers_pq_error(self):
        x = self._anisotropic()
        opq = OPQTransform(16, 4, nbits=5, iters=4).train(x)
        assert torch.allclose(opq.R @ opq.R.t(), torch.eye(16), atol=1e-4)
        def err(z):
            pq = ProductQuantizer(16, 4, nbits=5).train(z)
            return ((pq.decode(pq.encode(z)) - z) ** 2).sum(1).mean().item()
        assert err(opq.apply(x)) < err(x)

    def test_pca_reduces_dimension_and_keeps_variance(self):
        x = self._anisotropic()
        p = PCATransform(4).train(x)
        z = p.apply(x)
        assert z.shape == (1500, 4)
        assert z.var(0).sum() > 0.9 * x.var(0).sum()


class TestIVF:
    def setup_method(self):
        self.x = _clustered(1500)
        self.qs = self.x[torch.randperm(1500)[:60]] + 0.1 * torch.randn(60, 16)

    def test_every_vector_in_one_list(self):
        ivf = IVFIndex(self.x, 20)
        assert sum(ivf.list_sizes()) == 1500

    def test_all_probes_flat_is_exact(self):
        ivf = IVFIndex(self.x, 20)
        r, sc = one_recall_at_k(ivf, self.x, self.qs, nprobe=20)
        assert r == 1.0 and sc == pytest.approx(1.0)

    def test_recall_and_scan_grow_with_nprobe(self):
        ivf = IVFIndex(self.x, 40)
        pts = [one_recall_at_k(ivf, self.x, self.qs, p) for p in (1, 4, 16)]
        assert pts[0][0] <= pts[1][0] <= pts[2][0] and pts[0][1] < pts[1][1] < pts[2][1]

    def test_more_pq_bytes_means_better_recall(self):
        torch.manual_seed(1)
        x = torch.randn(1500, 16)
        qs = x[:60] + 0.3 * torch.randn(60, 16)
        res = {m: one_recall_at_k(IVFIndex(x, 20, pq_bytes=m, nbits=6), x, qs, 20)[0] for m in (1, 4, 8)}
        assert res[1] < res[8] and res[1] <= res[4] + 0.05 and res[4] <= res[8] + 0.05

    def test_transform_applied_consistently(self):
        opq = OPQTransform(16, 4, nbits=5, iters=2).train(self.x)
        ivf = IVFIndex(self.x, 20, pq_bytes=4, nbits=5, transform=opq)
        assert one_recall_at_k(ivf, self.x, self.qs, nprobe=20)[0] > 0.5

    def test_empty_probe_safe(self):
        ivf = IVFIndex(self.x[:30], 50)
        ids, n = ivf.search(self.x[0], 5, 3)
        assert len(ids) <= 5


class TestIMI:
    def test_cells_cover_all_vectors(self):
        x = _clustered(800)
        imi = IMIIndex(x, 8)
        assert sum(len(v) for v in imi.cells.values()) == 800 and imi.n_cells == 64

    def test_search_quality_and_scan_growth(self):
        x = _clustered(1200)
        imi = IMIIndex(x, 8)
        qs = x[:40] + 0.05 * torch.randn(40, 16)
        r1, s1 = one_recall_at_k(imi, x, qs, 2)
        r2, s2 = one_recall_at_k(imi, x, qs, 12)
        assert r2 >= r1 and s2 > s1

    def test_cells_are_imbalanced(self):
        # Sec. 4.1: with a multi-index many of the K^2 cells hold (almost) nothing -> compare at equal % scanned.
        x = _clustered(1200)
        assert IMIIndex(x, 12).fraction_small_cells(2) > 0.4


# ---------------------------------------------------------------------------
# Unicorn-style hybrid retrieval
# ---------------------------------------------------------------------------

class TestSexpr:
    def test_parse(self):
        assert parse_sexpr("(and (term a) (nn m :radius 0.2))") == ["and", ["term", "a"], ["nn", "m", ":radius", "0.2"]]

    @pytest.mark.parametrize("bad", ["", "(and", ")", "(term a) (term b)"])
    def test_errors(self, bad):
        with pytest.raises(ValueError):
            parse_sexpr(bad)


class TestUnicornLite:
    def setup_method(self):
        torch.manual_seed(0)
        self.vecs = F.normalize(_clustered(600, 16, 6), dim=-1)
        self.terms = {i: ({"location:seattle"} if i % 3 == 0 else {"location:menlo_park"} if i % 3 == 1 else set())
                      for i in range(600)}
        self.u = UnicornLite("m1", self.vecs, self.terms, nlist=12, pq_bytes=4, nbits=5)

    def test_term_and_or(self):
        assert self.u.search("(term location:seattle)", {}) == {i for i in range(600) if i % 3 == 0}
        assert self.u.search("(and (term location:seattle) (term location:menlo_park))", {}) == set()
        assert len(self.u.search("(or (term location:seattle) (term location:menlo_park))", {})) == 400

    def test_cluster_terms_are_ordinary_terms(self):
        total = set()
        for c in range(12):
            total |= self.u.search(f"(term emb:m1:c{c})", {})
        assert total == set(range(600))

    def test_radius_nn_finds_close_documents_only(self):
        q = self.vecs[5]
        got = self.u.search("(nn m1 :radius 0.1 :nprobe 12)", {"m1": q})
        true_close = set(((1 - self.vecs @ q) <= 0.1).nonzero().squeeze(1).tolist())
        assert 5 in got
        assert len(got & true_close) / len(true_close) > 0.8
        assert all(1 - float(self.vecs[d] @ q) < 0.25 for d in got)

    def test_larger_radius_is_a_superset(self):
        q = self.vecs[7]
        a = self.u.search("(nn m1 :radius 0.002 :nprobe 6)", {"m1": q})
        b = self.u.search("(nn m1 :radius 0.3 :nprobe 6)", {"m1": q})
        assert a <= b and len(b) > len(a)

    def test_more_probes_never_lose_candidates(self):
        q = self.vecs[9]
        a = self.u.search("(nn m1 :radius 0.3 :nprobe 2)", {"m1": q})
        b = self.u.search("(nn m1 :radius 0.3 :nprobe 8)", {"m1": q})
        assert a <= b

    def test_radius_mode_keeps_constrained_matches_that_topk_mode_loses(self):
        # Sec. 4.2.2: top-K picks the K nearest FIRST, then the rest of the query prunes them -> fewer results.
        q = self.vecs[3]
        radius = self.u.search("(and (term location:menlo_park) (nn m1 :radius 0.3 :nprobe 8))", {"m1": q})
        topk = self.u.search("(and (term location:menlo_park) (nn m1 :topk 10 :nprobe 8))", {"m1": q})
        assert len(radius) > len(topk)
        assert topk <= self.u.search("(term location:menlo_park)", {})

    def test_hybrid_fuzzy_match_rescues_a_misspelled_term(self):
        # "john smithe" fails an exact term match but its embedding is near "john smith".
        terms = {0: {"text:john", "text:smith", "location:seattle"}}
        vecs = F.normalize(torch.randn(50, 16), dim=-1)
        for i in range(1, 50):
            terms[i] = {f"text:other{i}"}
        q = F.normalize(vecs[0] + 0.05 * torch.randn(16), dim=-1)
        u = UnicornLite("m", vecs, terms, nlist=4, pq_bytes=4, nbits=5)
        assert 0 not in u.search("(and (term text:john) (term text:smithe))", {})
        hybrid = "(and (or (term location:seattle) (term location:menlo_park)) (or (and (term text:john) (term text:smithe)) (nn m :radius 0.2 :nprobe 4)))"
        assert 0 in u.search(hybrid, {"m": q})

    def test_unknown_operator(self):
        with pytest.raises(ValueError):
            self.u.search("(xor (term a) (term b))", {})


# ---------------------------------------------------------------------------
# Selection + later-stage
# ---------------------------------------------------------------------------

class TestSelection:
    def test_should_trigger(self):
        assert should_trigger_ebr("equipment for sale", {"john smith"})
        assert not should_trigger_ebr("John Smith ", {"john smith"})          # re-finding a known target
        assert not should_trigger_ebr("a", set())

    def test_index_selection(self):
        active = np.array([1, 1, 0, 1, 1])
        age = np.array([5, 400, 1, 500, 10])
        pop = np.array([0.1, 0.2, 0.9, 0.95, 0.3])
        sel = select_index_docs(active, age, pop, max_age=30, pop_quantile=0.8)
        assert sel.tolist() == [0, 3, 4]                       # inactive 2 is out; old-but-popular 3 stays; old-unpopular 1 out


class TestRankingFeatures:
    def test_shapes(self):
        q, d = torch.randn(5, 8), torch.randn(5, 8)
        assert embedding_features(q, d, "cosine").shape == (5, 1)
        assert embedding_features(q, d, "hadamard").shape == (5, 8)
        assert embedding_features(q, d, "raw").shape == (5, 16)
        with pytest.raises(ValueError):
            embedding_features(q, d, "x")

    def test_cosine_feature_is_dot_of_normalized_embeddings(self):
        q, d = F.normalize(torch.randn(3, 8), dim=-1), F.normalize(torch.randn(3, 8), dim=-1)
        assert torch.allclose(embedding_features(q, d)[:, 0], F.cosine_similarity(q, d), atol=1e-6)


class TestFeedbackLoop:
    def test_filter_raises_precision(self):
        torch.manual_seed(0)
        n = 600
        y = (torch.rand(n) < 0.5).float()
        cos = 0.55 * y + 0.3 * torch.rand(n) + 0.1
        overlap = 0.6 * y + 0.4 * torch.rand(n)
        jac = 0.5 * y + 0.4 * torch.rand(n)
        X = torch.stack([cos, overlap, jac], 1)
        flt = RelevanceFilter().fit(X[:400], y[:400])
        keep = flt.score(X[400:]) > 0.5
        yt = y[400:]
        precision_before = yt.mean().item()
        precision_after = yt[keep].mean().item()
        recall_kept = (keep & (yt == 1)).sum().item() / (yt == 1).sum().item()
        assert precision_after > precision_before + 0.2 and recall_kept > 0.8

    def test_features(self):
        f = RelevanceFilter.features("john smith", "john smith", 0.9)
        assert f.tolist() == [pytest.approx(0.9), 1.0, 1.0]
        g = RelevanceFilter.features("john smith", "mary jones", 0.1)
        assert g[1] == 0.0 and g[2] < 0.3
