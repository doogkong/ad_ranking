"""Tests for the multi-embedding retrieval implementation.

Run with:
    pytest test_multi_embedding_retrieval.py -v
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from multi_embedding_retrieval import (
    ConceptPoolConditions, DifferentiableClusteringModule, ExplicitInterestModel, FeatureCrossing,
    ImplicitInterestModel, InterestTokenConditions, ItemIndex, ItemTower, SelfAttentionConditions,
    StreamingFrequencyEstimator, SyntheticWorld, UserTower,
    allocate_budget, candidate_overlap, embedding_diversity, eval_implicit, eval_joint, filter_by_topic,
    hit_rate, kmeans, make_condition_builder, multi_embedding_scores, retrieve_multi_embedding,
    round_robin_merge, sampled_softmax_logq_loss, select_embedding, squash, train_explicit,
    train_implicit, validity_aware_fpi,
)

D = 16
B = 5
L = 12
FI = 6
FU = 4


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TestFeatureCrossing:
    def test_shape_and_grad(self):
        fc = FeatureCrossing(5, D, n_heads=4)
        x = torch.randn(B, 5, D, requires_grad=True)
        out = fc(x)
        assert out.shape == (B, D)
        out.sum().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

    def test_mixes_fields(self):
        fc = FeatureCrossing(3, D).eval()
        x = torch.randn(2, 3, D)
        x2 = x.clone()
        x2[:, 0] += 1.0
        assert not torch.allclose(fc(x), fc(x2))


class TestTowers:
    def test_item_tower_unit_norm(self):
        out = ItemTower(FI, D)(torch.randn(B, FI))
        assert torch.allclose(out.norm(dim=-1), torch.ones(B), atol=1e-5)

    def test_user_tower_conditions_change_output(self):
        ut = UserTower(FU, D).eval()
        uf = torch.randn(B, FU)
        c = torch.randn(B, 3, D)
        out = ut(uf, c)
        assert out.shape == (B, 3, D)
        assert torch.allclose(out.norm(dim=-1), torch.ones(B, 3), atol=1e-5)
        assert not torch.allclose(out[:, 0], out[:, 1])          # different conditions -> different embeddings

    def test_user_tower_k_folding_matches_single(self):
        ut = UserTower(FU, D).eval()
        uf, c = torch.randn(B, FU), torch.randn(B, 3, D)
        both = ut(uf, c)
        single = ut(uf, c[:, 1:2])
        assert torch.allclose(both[:, 1], single[:, 0], atol=1e-5)


class TestSquash:
    def test_norm_below_one_and_monotone(self):
        small, big = squash(torch.tensor([[0.1, 0.0]])), squash(torch.tensor([[10.0, 0.0]]))
        assert 0 < small.norm() < big.norm() < 1

    def test_direction_preserved(self):
        v = torch.randn(4, D)
        assert torch.allclose(F.normalize(squash(v), dim=-1), F.normalize(v, dim=-1), atol=1e-5)

    def test_zero_is_finite(self):
        assert torch.isfinite(squash(torch.zeros(1, D))).all()


# ---------------------------------------------------------------------------
# VA-FPI
# ---------------------------------------------------------------------------

def _three_cluster_seq(n_per=6):
    centers = torch.tensor([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]])
    pts = torch.cat([c + 0.05 * torch.randn(n_per, 3) for c in centers])
    return pts[torch.randperm(len(pts))].unsqueeze(0), centers


class TestVAFPI:
    def test_picks_one_per_cluster(self):
        torch.manual_seed(0)
        e, centers = _three_cluster_seq()
        valid = torch.ones(1, e.size(1), dtype=torch.bool)
        c, distinct = validity_aware_fpi(e, valid, 3)
        assert distinct.all()
        which = torch.cdist(c[0], centers).argmin(1)
        assert sorted(which.tolist()) == [0, 1, 2]                    # farthest-point => one per cluster

    def test_centroids_are_diverse(self):
        torch.manual_seed(1)
        e, _ = _three_cluster_seq()
        c, _ = validity_aware_fpi(e, torch.ones(1, e.size(1), dtype=torch.bool), 3)
        sims = F.normalize(c[0], dim=-1) @ F.normalize(c[0], dim=-1).t()
        assert (sims - torch.eye(3)).max() < 0.2

    def test_invalid_items_never_chosen(self):
        torch.manual_seed(0)
        e = torch.randn(4, 10, D).abs() + 1.0
        e[:, 0] = -1000.0                                             # out-of-distribution outlier
        valid = torch.ones(4, 10, dtype=torch.bool)
        valid[:, 0] = False
        c, _ = validity_aware_fpi(e, valid, 4)
        assert (c.abs().max(-1).values < 500).all()

    def test_without_validity_outlier_is_chosen(self):
        # The failure the validity filter prevents: the farthest point IS the outlier.
        torch.manual_seed(0)
        e = torch.randn(1, 10, D).abs() + 1.0
        e[:, 0] = -1000.0                                             # far from everything in dot-product terms
        valid = torch.ones(1, 10, dtype=torch.bool)
        c, _ = validity_aware_fpi(e, valid, 2)
        assert (c.abs().max(-1).values > 500).any()

    def test_fewer_items_than_k_marks_duplicates(self):
        e = torch.randn(1, 5, D)
        valid = torch.tensor([[True, True, False, False, False]])
        _, distinct = validity_aware_fpi(e, valid, 4)
        assert distinct[0].tolist() == [True, True, False, False]

    def test_no_valid_items(self):
        _, distinct = validity_aware_fpi(torch.randn(1, 4, D), torch.zeros(1, 4, dtype=torch.bool), 3)
        assert not distinct.any()


# ---------------------------------------------------------------------------
# DCM and baseline condition builders
# ---------------------------------------------------------------------------

def _seq(batch=B, length=L):
    return torch.randn(batch, length, FI), torch.ones(batch, length, dtype=torch.bool), torch.ones(batch, length, dtype=torch.bool)


class TestDCM:
    def test_shapes(self):
        dcm = DifferentiableClusteringModule(FI, D, 4)
        f, m, v = _seq()
        out = dcm(f, m, v)
        assert out.conds.shape == (B, 4, D) and out.mask.shape == (B, 4) and out.importance.shape == (B, 4)

    def test_gradients_reach_item_features_and_S(self):
        dcm = DifferentiableClusteringModule(FI, D, 3)
        f, m, v = _seq()
        f.requires_grad_(True)
        dcm(f, m, v).conds.sum().backward()
        assert f.grad.abs().sum() > 0 and dcm.S.grad.abs().sum() > 0

    def test_single_assignment_gives_each_item_one_cluster(self):
        torch.manual_seed(0)
        sar = DifferentiableClusteringModule(FI, D, 3, single_assignment=True)
        f, m, v = _seq()
        # importance mass is the sum of routing weights; with SAR each item contributes <= 1 total.
        out = sar(f, m, v)
        assert (out.importance.sum(1) <= L + 1e-4).all()
        multi = DifferentiableClusteringModule(FI, D, 3, init="gaussian", single_assignment=False)
        out_m = multi(f, m, v)
        assert torch.allclose(out_m.importance.sum(1), torch.full((B,), float(L)), atol=1e-3)  # softmax rows sum to 1

    def test_padding_and_invalid_items_are_ignored(self):
        torch.manual_seed(0)
        dcm = DifferentiableClusteringModule(FI, D, 3).eval()
        f, m, v = _seq(2, 10)
        m[:, 6:] = False
        torch.manual_seed(5)
        a = dcm(f, m, v)
        f2 = f.clone()
        f2[:, 6:] = 99.0                                               # change only padded positions
        torch.manual_seed(5)
        b = dcm(f2, m, v)
        assert torch.allclose(a.conds, b.conds, atol=1e-5)

    def test_invalid_item_features_do_not_leak(self):
        dcm = DifferentiableClusteringModule(FI, D, 3).eval()
        f, m, v = _seq(2, 10)
        v[:, 3] = False
        torch.manual_seed(5)
        a = dcm(f, m, v)
        f2 = f.clone()
        f2[:, 3] = 1e3
        torch.manual_seed(5)
        b = dcm(f2, m, v)
        assert torch.allclose(a.conds, b.conds, atol=1e-5)

    def test_short_sequence_marks_empty_conditions(self):
        dcm = DifferentiableClusteringModule(FI, D, 4)
        f, m, v = _seq(1, 3)
        out = dcm(f, m, v)
        assert out.mask.sum() <= 3                                     # cannot have more clusters than items

    def test_empty_history(self):
        dcm = DifferentiableClusteringModule(FI, D, 3)
        f, m, v = _seq(1, 4)
        out = dcm(f, torch.zeros_like(m), v)
        assert not out.mask.any() and torch.isfinite(out.conds).all()

    def test_dcm_clusters_are_more_diverse_than_vanilla_capsules(self):
        # Items from 3 well-separated topics: DCM (VA-FPI + SAR) should yield more diverse conditions
        # than Gaussian init + multi-assignment (the paper's Fig. 5 / Table 6 argument).
        torch.manual_seed(0)
        protos = torch.randn(3, FI) * 3
        feats = (protos[torch.randint(0, 3, (64, 18))] + 0.1 * torch.randn(64, 18, FI))
        m = torch.ones(64, 18, dtype=torch.bool)
        div = {}
        for name, kw in {"dcm": dict(init="vafpi", single_assignment=True),
                         "capsule": dict(init="gaussian", single_assignment=False)}.items():
            torch.manual_seed(1)
            mod = DifferentiableClusteringModule(FI, D, 3, **kw).eval()
            with torch.no_grad():
                out = mod(feats, m, m)
            div[name] = embedding_diversity(F.normalize(out.conds, dim=-1), out.mask)
        assert div["dcm"] < div["capsule"]


class TestOtherBuilders:
    @pytest.mark.parametrize("kind", ["dcm", "mind", "self_attention", "interest_token"])
    def test_factory_shapes(self, kind):
        b = make_condition_builder(kind, FI, D, 3)
        f, m, v = _seq()
        out = b(f, m, v)
        assert out.conds.shape == (B, 3, D) and out.mask.shape == (B, 3) and out.importance.shape == (B, 3)
        assert torch.isfinite(out.conds).all()

    def test_factory_rejects_unknown(self):
        with pytest.raises(ValueError):
            make_condition_builder("nope", FI, D, 3)

    def test_self_attention_weights_sum_to_one(self):
        out = SelfAttentionConditions(FI, D, 3)(*_seq())
        assert torch.allclose(out.importance, torch.ones(B, 3), atol=1e-4)

    def test_interest_token_handles_padding(self):
        b = InterestTokenConditions(FI, D, 3)
        f, m, v = _seq(2, 8)
        m[:, 4:] = False
        out = b(f, m, v)
        assert torch.isfinite(out.conds).all()

    def test_concept_pool_assigns_by_concept(self):
        concepts = torch.tensor([[10.0] * FI, [-10.0] * FI])
        b = ConceptPoolConditions(FI, D, concepts)
        f = torch.cat([torch.full((1, 4, FI), 10.0), torch.full((1, 2, FI), -10.0)], 1)
        out = b(f, torch.ones(1, 6, dtype=torch.bool))
        assert out.importance[0].tolist() == [4.0, 2.0] and out.mask.all()

    def test_concept_pool_empty_concept_masked(self):
        concepts = torch.tensor([[10.0] * FI, [-10.0] * FI])
        out = ConceptPoolConditions(FI, D, concepts)(torch.full((1, 3, FI), 10.0), torch.ones(1, 3, dtype=torch.bool))
        assert out.mask[0].tolist() == [True, False]

    def test_kmeans_recovers_clusters(self):
        torch.manual_seed(0)
        x = torch.cat([torch.randn(50, 2) * 0.1 + 5, torch.randn(50, 2) * 0.1 - 5])
        c = kmeans(x, 2)
        assert sorted(c[:, 0].round().tolist()) == [-5.0, 5.0]


# ---------------------------------------------------------------------------
# Association + loss
# ---------------------------------------------------------------------------

class TestAssociation:
    def test_argmax_picks_most_similar_valid_embedding(self):
        embs = F.normalize(torch.randn(B, 4, D), dim=-1)
        pos = embs[:, 2].clone()
        mask = torch.ones(B, 4, dtype=torch.bool)
        sel, idx = select_embedding(embs, mask, pos, "argmax")
        assert (idx == 2).all() and torch.allclose(sel, pos)

    def test_invalid_embeddings_never_selected(self):
        embs = F.normalize(torch.randn(B, 4, D), dim=-1)
        pos = embs[:, 2].clone()
        mask = torch.ones(B, 4, dtype=torch.bool)
        mask[:, 2] = False
        _, idx = select_embedding(embs, mask, pos, "argmax")
        assert (idx != 2).all()
        _, idx_g = select_embedding(embs, mask, pos, "gumbel")
        assert (idx_g != 2).all()

    def test_argmax_gradient_only_to_winner(self):
        embs = F.normalize(torch.randn(1, 4, D), dim=-1).requires_grad_(True)
        sel, idx = select_embedding(embs, torch.ones(1, 4, dtype=torch.bool), embs[:, 1].detach(), "argmax")
        sel.sum().backward()
        loser = [j for j in range(4) if j != idx.item()]
        assert embs.grad[0, idx.item()].abs().sum() > 0 and embs.grad[0, loser].abs().sum() == 0

    def test_gumbel_forward_is_hard_but_gradient_reaches_all(self):
        torch.manual_seed(0)
        embs = torch.randn(1, 4, D).requires_grad_(True)
        sel, idx = select_embedding(embs, torch.ones(1, 4, dtype=torch.bool), torch.randn(1, D), "gumbel")
        assert torch.allclose(sel[0], embs[0, idx.item()].detach(), atol=1e-6)
        (sel * torch.randn(1, D)).sum().backward()
        assert (embs.grad[0].abs().sum(-1) > 0).sum() > 1             # straight-through reaches non-winners

    def test_unknown_mode(self):
        with pytest.raises(ValueError):
            select_embedding(torch.randn(1, 2, D), torch.ones(1, 2, dtype=torch.bool), torch.randn(1, D), "x")


class TestLoss:
    def test_aligned_beats_misaligned(self):
        pos = F.normalize(torch.randn(8, D), dim=-1)
        ids = torch.arange(8)
        logp = torch.full((8,), -5.0)
        good = sampled_softmax_logq_loss(pos.clone(), pos, ids, logp)
        bad = sampled_softmax_logq_loss(pos.roll(1, 0), pos, ids, logp)
        assert good < bad

    def test_logq_discounts_popular_negatives(self):
        # A popular item's logit is reduced (-log p), so its presence as a negative hurts the positive less.
        u = F.normalize(torch.randn(2, D), dim=-1)
        pos = torch.stack([u[0], u[0]])                               # item 1 is a perfect negative for user 0
        ids = torch.tensor([0, 1])
        rare = sampled_softmax_logq_loss(u, pos, ids, torch.log(torch.tensor([0.5, 0.001])))
        popular = sampled_softmax_logq_loss(u, pos, ids, torch.log(torch.tensor([0.5, 0.5])))
        assert rare > popular       # a *rare* hard negative is penalised more once the correction is applied

    def test_duplicate_ids_masked(self):
        u = F.normalize(torch.randn(2, D), dim=-1)
        pos = torch.stack([u[0], u[0]])
        logp = torch.zeros(2)
        dup = sampled_softmax_logq_loss(u, pos, torch.tensor([7, 7]), logp)
        distinct = sampled_softmax_logq_loss(u, pos, torch.tensor([7, 8]), logp)
        assert dup < distinct          # same item is not treated as a negative of itself

    def test_gradient_flows(self):
        u = torch.randn(4, D, requires_grad=True)
        sampled_softmax_logq_loss(u, torch.randn(4, D), torch.arange(4), torch.zeros(4)).backward()
        assert u.grad is not None


class TestFrequencyEstimator:
    def test_frequent_items_get_higher_probability(self):
        est = StreamingFrequencyEstimator(num_buckets=100, alpha=0.2)
        for _ in range(200):
            est.update(torch.tensor([1, 1, 1, 2]))
        p = est.prob(torch.tensor([1, 2, 3]))
        assert p[0] > p[1] > p[2]

    def test_probabilities_positive_and_bounded(self):
        est = StreamingFrequencyEstimator(num_buckets=10)
        est.update(torch.arange(30))
        p = est.prob(torch.arange(30))
        assert (p > 0).all() and (p <= 1.0).all()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestModels:
    def test_implicit_model_default_selection_mode(self):
        assert ImplicitInterestModel(FI, FU, D, 3, kind="dcm").select == "argmax"
        assert ImplicitInterestModel(FI, FU, D, 3, kind="mind").select == "argmax"
        assert ImplicitInterestModel(FI, FU, D, 3, kind="self_attention").select == "gumbel"
        assert ImplicitInterestModel(FI, FU, D, 3, kind="interest_token").select == "gumbel"

    @pytest.mark.parametrize("kind", ["dcm", "mind", "self_attention", "interest_token"])
    def test_implicit_loss_backward(self, kind):
        m = ImplicitInterestModel(FI, FU, D, 3, kind=kind)
        f, mask, v = _seq()
        loss = m.loss(f, mask, torch.randn(B, FU), torch.randn(B, FI), torch.arange(B), torch.zeros(B), v)
        loss.backward()
        assert torch.isfinite(loss)
        assert sum(p.grad.abs().sum() for p in m.parameters() if p.grad is not None) > 0

    def test_implicit_embeddings_shape(self):
        m = ImplicitInterestModel(FI, FU, D, 4)
        f, mask, v = _seq()
        embs, cond = m.user_embeddings(f, mask, torch.randn(B, FU), v)
        assert embs.shape == (B, 4, D) and cond.mask.shape == (B, 4)

    def test_explicit_model(self):
        m = ExplicitInterestModel(FI, FU, n_topics=7, d=D)
        uf = torch.randn(B, FU)
        e1 = m.user_embeddings(uf, torch.tensor([0, 1, 2, 3, 4]))
        assert e1.shape == (B, 1, D)
        e2 = m.user_embeddings(uf, torch.tensor([[0, 1]] * B))
        assert e2.shape == (B, 2, D)
        assert not torch.allclose(e2[:, 0], e2[:, 1])                  # topic changes the embedding
        loss = m.loss(uf, torch.zeros(B, dtype=torch.long), torch.randn(B, FI), torch.arange(B), torch.zeros(B))
        loss.backward()
        assert m.topic_embedding.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------

class TestBudgets:
    def test_sums_to_total_and_proportional(self):
        b = allocate_budget([6.0, 3.0, 1.0], 100)
        assert sum(b) == 100 and b == [60, 30, 10]

    @pytest.mark.parametrize("w,total", [([1, 1, 1], 10), ([0.2, 0.3, 0.5], 7), ([5, 0, 1], 13), ([1], 5)])
    def test_exact_total(self, w, total):
        assert sum(allocate_budget(w, total)) == total

    def test_zero_weight_gets_nothing(self):
        assert allocate_budget([0.0, 1.0], 5) == [0, 5]

    def test_degenerate(self):
        assert allocate_budget([0, 0], 5) == [0, 0]
        assert allocate_budget([1, 1], 0) == [0, 0]

    def test_equal_weights_equal_budgets(self):
        assert allocate_budget([1, 1, 1, 1], 8) == [2, 2, 2, 2]


class TestMerge:
    def test_round_robin_order_and_dedup(self):
        assert round_robin_merge([[1, 2, 3], [2, 4, 5], [6]], 10) == [1, 2, 6, 3, 4, 5]

    def test_total_cap(self):
        assert round_robin_merge([[1, 2, 3], [4, 5, 6]], 4) == [1, 4, 2, 5]

    def test_balanced_mix_prevents_domination(self):
        # a long head-interest list does not crowd out a short tail-interest list
        head, tail = list(range(100)), [1000, 1001]
        merged = round_robin_merge([head, tail], 6)
        assert 1000 in merged and 1001 in merged

    def test_empty(self):
        assert round_robin_merge([], 5) == [] and round_robin_merge([[], []], 5) == []


class TestRetrieval:
    def test_per_embedding_budgets_respected(self):
        torch.manual_seed(0)
        items = F.normalize(torch.randn(50, D), dim=-1)
        idx = ItemIndex(items)
        q = F.normalize(torch.randn(3, D), dim=-1)
        out = retrieve_multi_embedding(q, [5, 0, 3], idx, total=20)
        assert len(out) == len(set(out)) <= 8
        top0 = idx.search(q[:1], 5)[0]
        assert set(top0) <= set(out)

    def test_each_embedding_contributes_its_best_item(self):
        items = torch.eye(4, D)
        q = torch.eye(4, D)[:3]
        out = retrieve_multi_embedding(q, [1, 1, 1], ItemIndex(items), total=3)
        assert sorted(out) == [0, 1, 2]

    def test_all_zero_budget(self):
        assert retrieve_multi_embedding(torch.randn(2, D), [0, 0], ItemIndex(torch.randn(5, D)), 5) == []

    def test_filter_by_topic(self):
        topics = [0, 0, 1, 2, 1]
        assert filter_by_topic([0, 2, 3, 4], topics, {1}) == [2, 4]
        assert filter_by_topic([0, 1], topics, set()) == []

    def test_overlap(self):
        assert candidate_overlap([1, 2, 3], [3, 4]) == pytest.approx(1 / 4)
        assert candidate_overlap([], []) == 0.0


class TestEvaluation:
    def test_multi_embedding_score_is_max_over_valid(self):
        embs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        items = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
        s = multi_embedding_scores(embs, torch.tensor([[True, True]]), items)
        assert s[0].tolist() == pytest.approx([1.0, 1.0, 0.7])
        s2 = multi_embedding_scores(embs, torch.tensor([[True, False]]), items)
        assert s2[0, 1] == pytest.approx(0.0)                        # masked embedding contributes nothing

    def test_hit_rate(self):
        scores = torch.tensor([[0.9, 0.5, 0.1], [0.1, 0.5, 0.9]])
        hr = hit_rate(scores, torch.tensor([0, 0]), [1, 3])
        assert hr[1] == 0.5 and hr[3] == 1.0

    def test_diversity_extremes(self):
        same = F.normalize(torch.ones(1, 3, D), dim=-1)
        assert embedding_diversity(same, torch.ones(1, 3, dtype=torch.bool)) == pytest.approx(1.0, abs=1e-5)
        ortho = torch.eye(3, D).unsqueeze(0)
        assert embedding_diversity(ortho, torch.ones(1, 3, dtype=torch.bool)) == pytest.approx(0.0, abs=1e-5)
        assert embedding_diversity(same, torch.tensor([[True, False, False]])) == 0.0


# ---------------------------------------------------------------------------
# Synthetic world + end-to-end
# ---------------------------------------------------------------------------

class TestWorld:
    def setup_method(self):
        self.w = SyntheticWorld(seed=0)
        self.u = self.w.sample_users(200)

    def test_history_topics_distinct_and_followed_structure(self):
        ht, fl = self.u["hist_topics"].numpy(), self.u["followed"].numpy()
        assert all(len(set(r)) == 3 for r in ht)
        assert all(f[0] in h and f[1] not in h for f, h in zip(fl, ht))

    def test_history_is_skewed_toward_dominant_interest(self):
        topics = self.w.item_topic[self.u["hist_ids"].numpy()]
        top_share = np.mean([np.mean(t == h[0]) for t, h in zip(topics, self.u["hist_topics"].numpy())])
        assert 0.45 < top_share < 0.75

    def test_targets_cover_tail_interests(self):
        tgt = self.w.implicit_targets(self.u)
        tt = self.w.item_topic[tgt.numpy()]
        ht = self.u["hist_topics"].numpy()
        share_tail = np.mean(tt == ht[:, 2])
        assert 0.2 < share_tail < 0.45                               # ~1/3, not 10%

    def test_explicit_samples_match_condition(self):
        topics, items = self.w.explicit_samples(self.u)
        assert (torch.tensor(self.w.item_topic)[items] == topics).all()

    def test_invalid_items_have_ood_features(self):
        bad = self.w.item_feats[~self.w.valid_t]
        assert len(bad) > 0 and bad.abs().mean() > self.w.item_feats[self.w.valid_t].abs().mean() * 2

    def test_inputs_shapes(self):
        f, m, v = self.w.hist_inputs(self.u)
        assert f.shape == (200, self.w.L, self.w.Fi) and m.all() and v.shape == m.shape


class TestEndToEnd:
    def test_training_reduces_loss_and_beats_random(self):
        torch.manual_seed(0)
        w = SyntheticWorld(seed=0)
        m = ImplicitInterestModel(w.Fi, w.Fu, d=32, n_interests=3, kind="dcm")
        losses = train_implicit(m, w, steps=60, batch=128)
        assert np.mean(losses[-10:]) < np.mean(losses[:10])
        users = w.sample_users(300)
        hr = eval_implicit(m, w, users, w.implicit_targets(users), ks=(50,))
        assert hr[50] > 2 * 50 / w.n_items                             # well above random-ranking HR@50

    def test_multi_embedding_beats_single_on_tail_interests(self):
        # The paper's core motivation: one embedding favours the head interest.
        w = SyntheticWorld(seed=0)
        users = w.sample_users(600)
        targets = w.implicit_targets(users)
        hr = {}
        for k in (1, 3):
            torch.manual_seed(0)
            m = ImplicitInterestModel(w.Fi, w.Fu, d=32, n_interests=k, kind="dcm")
            train_implicit(m, w, steps=200, batch=256)
            hr[k] = eval_implicit(m, w, users, targets, ks=(50,))[50]
        assert hr[3] > hr[1] + 0.03

    def test_explicit_recovers_interest_missing_from_history(self):
        # The followed topic that never appears in the history is only reachable through CR.
        torch.manual_seed(0)
        w = SyntheticWorld(seed=0)
        imp = ImplicitInterestModel(w.Fi, w.Fu, d=32, n_interests=3, kind="dcm")
        exp = ExplicitInterestModel(w.Fi, w.Fu, w.T, d=32)
        train_implicit(imp, w, steps=150, batch=256)
        train_explicit(exp, w, steps=150, batch=256)
        users = w.sample_users(600)
        # targets drawn only from the followed topic that is NOT in the history
        new_topics = users["followed"][:, 1].numpy()
        targets = torch.tensor(w._item_from_topic(new_topics))
        res = eval_joint(imp, exp, w, users, targets, ks=(50,))
        assert res["explicit"][50] > res["implicit"][50] + 0.2
        assert res["both"][50] > res["implicit"][50]

    def test_synergy_on_full_interest_set(self):
        torch.manual_seed(0)
        w = SyntheticWorld(seed=0)
        imp = ImplicitInterestModel(w.Fi, w.Fu, d=32, n_interests=3, kind="dcm")
        exp = ExplicitInterestModel(w.Fi, w.Fu, w.T, d=32)
        train_implicit(imp, w, steps=200, batch=256)
        train_explicit(exp, w, steps=200, batch=256)
        users = w.sample_users(600)
        res = eval_joint(imp, exp, w, users, w.joint_targets(users), ks=(50,))
        assert res["both"][50] > max(res["implicit"][50], res["explicit"][50])
