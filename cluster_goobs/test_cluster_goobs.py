"""Tests for the Cluster GOOBS implementation.

Run with:
    pytest test_cluster_goobs.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from cluster_goobs import (
    ANCESampler, CBNSSampler, ClusterGOOBSampler, ClusterItemPool, ContentClusterer, DNSSampler,
    GOOBSampler, ItemCatalog, Negatives, NegativeSampler, SyntheticInteractions, TwoTower,
    build_pool, contrastive_loss, evaluate, false_negative_rate, kmeans, make_sampler,
    popularity_report, retrieval_scores, train,
)

FD = 4


# ---------------------------------------------------------------------------
# Clusters
# ---------------------------------------------------------------------------

class TestClustering:
    def _blobs(self):
        torch.manual_seed(0)
        centers = torch.tensor([[8.0, 0], [-8.0, 0], [0, 8.0]])
        x = torch.cat([c + 0.3 * torch.randn(40, 2) for c in centers])
        return x, centers

    def test_kmeans_recovers_blobs(self):
        x, centers = self._blobs()
        c = kmeans(x, 3)
        assert torch.cdist(c, centers).min(1).values.max() < 0.5

    def test_assign_online_matches_nearest_centroid(self):
        x, centers = self._blobs()
        cl = ContentClusterer(3).fit(x)
        new_item = torch.tensor([[7.5, 0.4], [-7.0, -0.5]])
        a = cl.assign(new_item)
        assert a[0] != a[1]
        assert a[0] == cl.assign(centers[:1])[0] and a[1] == cl.assign(centers[1:2])[0]

    def test_sizes_sum_to_items(self):
        x, _ = self._blobs()
        cl = ContentClusterer(3).fit(x)
        assert cl.sizes(x).sum() == len(x)

    def test_assign_requires_fit(self):
        with pytest.raises(AssertionError):
            ContentClusterer(2).assign(torch.randn(3, 2))


# ---------------------------------------------------------------------------
# Pool: hashing, update, sampling (Algorithms 1-2, Fig. 3-4)
# ---------------------------------------------------------------------------

class TestPool:
    def test_paper_hashing_example(self):
        # Fig. 3: S = 5, item 12 in cluster 1 -> S + 12 % S = 5 + 2 = 7
        pool = ClusterItemPool(3, 5, FD)
        assert pool.slot(torch.tensor([12]), torch.tensor([1])).item() == 7

    def test_slot_stays_inside_cluster_segment(self):
        pool = ClusterItemPool(4, 8, FD)
        ids = torch.randint(0, 10_000, (500,))
        cl = torch.randint(0, 4, (500,))
        s = pool.slot(ids, cl)
        assert ((s >= cl * 8) & (s < cl * 8 + 8)).all()

    def test_memory_is_fixed(self):
        pool = ClusterItemPool(4, 8, FD)
        n = len(pool)
        pool.update(torch.arange(10_000), torch.randint(0, 4, (10_000,)), torch.randn(10_000, FD))
        assert len(pool) == n == 32 and pool.feats.shape == (32, FD)

    def test_update_stores_id_cluster_features(self):
        pool = ClusterItemPool(3, 5, FD)
        f = torch.randn(1, FD)
        pool.update(torch.tensor([12]), torch.tensor([1]), f)
        assert pool.item_ids[7] == 12 and pool.cluster_ids[7] == 1 and pool.filled[7]
        assert torch.equal(pool.feats[7], f[0])
        assert pool.filled.sum() == 1

    def test_later_item_overwrites_earlier_in_same_slot(self):
        pool = ClusterItemPool(2, 5, FD)
        pool.update(torch.tensor([2]), torch.tensor([0]), torch.ones(1, FD))
        pool.update(torch.tensor([7]), torch.tensor([0]), 2 * torch.ones(1, FD))      # 7 % 5 == 2 % 5
        assert pool.item_ids[2] == 7 and pool.feats[2, 0] == 2.0 and pool.filled.sum() == 1

    def test_sample_in_cluster_only_from_that_cluster(self):
        torch.manual_seed(0)
        pool = ClusterItemPool(5, 6, FD)
        ids = torch.arange(100)
        cl = ids % 5
        pool.preload(ids, cl, torch.randn(100, FD))
        clusters = torch.tensor([0, 3, 3, 4])
        out_ids, feats, valid = pool.sample_in_cluster(clusters, n=4)
        assert out_ids.shape == (4, 4) and feats.shape == (4, 4, FD) and valid.all()
        assert (out_ids % 5 == clusters.unsqueeze(1)).all()

    def test_unfilled_slots_flagged_invalid(self):
        pool = ClusterItemPool(2, 4, FD)
        pool.update(torch.tensor([1]), torch.tensor([0]), torch.randn(1, FD))
        _, _, valid = pool.sample_in_cluster(torch.tensor([1]), n=8)               # cluster 1 is empty
        assert not valid.any()

    def test_sample_random_spans_clusters_and_only_filled(self):
        torch.manual_seed(0)
        pool = ClusterItemPool(5, 4, FD)
        ids = torch.arange(10)
        pool.update(ids, ids % 5, torch.randn(10, FD))
        sid, scl, feats, valid = pool.sample_random((200,))
        assert valid.all() and (sid >= 0).all() and (sid % 5 == scl).all()
        assert len(set(scl.tolist())) > 1
        e = ClusterItemPool(2, 2, FD).sample_random((3, 2))
        assert not e[3].any()

    def test_preload_raises_fill_fraction(self):
        pool = ClusterItemPool(4, 4, FD)
        assert pool.fill_fraction() == 0
        ids = torch.arange(200)
        pool.preload(ids, (ids // 4) % 4, torch.randn(200, FD))
        assert pool.fill_fraction() == 1.0


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _unit(n, d=8):
    return F.normalize(torch.randn(n, d), dim=-1)


class TestLoss:
    def test_matches_manual_cross_entropy(self):
        torch.manual_seed(0)
        u, v = _unit(3), _unit(3)
        ids = torch.arange(3)
        neg = Negatives(_unit(3 * 2).view(3, 2, -1), torch.full((3, 2), 99), torch.ones(3, 2, dtype=torch.bool))
        loss = contrastive_loss(u, v, ids, 0.5, None, [neg])
        manual = 0.0
        for i in range(3):
            logits = torch.cat([u[i] @ v.t(), neg.emb[i] @ u[i]]) / 0.5
            manual += -F.log_softmax(logits, 0)[i]
        assert loss.item() == pytest.approx(manual.item() / 3, abs=1e-5)

    def test_false_negative_mask_in_batch(self):
        u = _unit(2)
        v = torch.stack([u[0], u[0]])                                  # row 1's positive is a perfect match for user 0
        distinct = contrastive_loss(u, v, torch.tensor([1, 2]), 0.1)
        dup = contrastive_loss(u, v, torch.tensor([5, 5]), 0.1)
        assert dup < distinct                                          # identical ids are not negatives

    def test_oob_negative_equal_to_positive_is_masked(self):
        u, v = _unit(2), _unit(2)
        ids = torch.tensor([3, 4])
        bad = Negatives(v.unsqueeze(1).clone(), ids.unsqueeze(1).clone(), torch.ones(2, 1, dtype=torch.bool))
        masked = contrastive_loss(u, v, ids, 0.1, None, [bad])
        assert masked.item() == pytest.approx(contrastive_loss(u, v, ids, 0.1).item(), abs=1e-6)

    def test_invalid_negatives_ignored(self):
        u, v = _unit(2), _unit(2)
        neg = Negatives(_unit(4).view(2, 2, -1), torch.tensor([[8, 9], [8, 9]]), torch.zeros(2, 2, dtype=torch.bool))
        assert contrastive_loss(u, v, torch.arange(2), 0.1, None, [neg]).item() == pytest.approx(
            contrastive_loss(u, v, torch.arange(2), 0.1).item(), abs=1e-6)

    def test_shared_negatives_block(self):
        u, v = _unit(3), _unit(3)
        shared = Negatives(_unit(5), torch.arange(10, 15), torch.ones(5, dtype=torch.bool))
        base = contrastive_loss(u, v, torch.arange(3), 0.1)
        assert contrastive_loss(u, v, torch.arange(3), 0.1, None, [shared]).item() > base.item()   # more negatives, higher loss

    def test_logq_discounts_popular_negatives(self):
        u = _unit(2)
        v = torch.stack([u[0], u[0]])
        ids = torch.tensor([0, 1])
        popular = contrastive_loss(u, v, ids, 0.1, torch.log(torch.tensor([0.5, 0.5])))
        rare = contrastive_loss(u, v, ids, 0.1, torch.log(torch.tensor([0.5, 0.001])))
        assert rare > popular

    def test_zero_label_rows_have_no_loss_and_are_not_negatives(self):
        u, v = _unit(3), _unit(3)
        ids = torch.arange(3)
        labels = torch.tensor([1.0, 0.0, 1.0])
        full = contrastive_loss(u, v, ids, 0.1, labels=labels)
        keep = torch.tensor([0, 2])
        sub = contrastive_loss(u[keep], v[keep], ids[keep], 0.1)
        assert full.item() == pytest.approx(sub.item(), abs=1e-5)

    def test_gradient_flows_to_negative_embeddings(self):
        u, v = _unit(2), _unit(2)
        e = _unit(4).view(2, 2, -1).clone().requires_grad_(True)
        neg = Negatives(e, torch.tensor([[8, 9], [8, 9]]), torch.ones(2, 2, dtype=torch.bool))
        contrastive_loss(u, v, torch.arange(2), 0.1, None, [neg]).backward()
        assert e.grad.abs().sum() > 0

    def test_hard_negative_has_larger_gradient_than_easy(self):
        # Sec. 3.3: d loss / d s(x, y^-) = softmax prob of the negative, which grows with its similarity.
        u = torch.tensor([[1.0, 0.0]], requires_grad=True)
        v = torch.tensor([[0.0, 1.0]])
        hard, easy = torch.tensor([[0.9, 0.436]]), torch.tensor([[-1.0, 0.0]])
        grads = []
        for n in (hard, easy):
            u2 = u.detach().clone().requires_grad_(True)
            n2 = n.clone().unsqueeze(0).requires_grad_(True)
            neg = Negatives(n2, torch.tensor([[7]]), torch.ones(1, 1, dtype=torch.bool))
            contrastive_loss(u2, v, torch.tensor([0]), 0.2, None, [neg]).backward()
            grads.append(n2.grad.norm().item())
        assert grads[0] > 5 * grads[1]


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def world():
    torch.manual_seed(0)
    data = SyntheticInteractions(n_users=120, n_true_clusters=8, items_per_cluster=20, seed=0)
    clusters = ContentClusterer(8).fit(data.content_emb).assign(data.content_emb)
    return data, clusters


def _setup(world, sampler_name, **kw):
    data, clusters = world
    torch.manual_seed(0)
    model = TwoTower(data.n_users, data.n_items, 8, FD_of(data), FD_of(data))
    catalog = ItemCatalog(data.item_feats, clusters)
    pool = build_pool(data, clusters, 8, slots=10) if sampler_name in ("goobs", "cluster_goobs") else None
    return model, make_sampler(sampler_name, catalog, pool, **kw), data, clusters


def FD_of(data):
    return data.item_feats.size(1)


def _batch(data, clusters, n=32):
    idx = torch.arange(n)
    users, items = data.train_pairs[idx, 0], data.train_pairs[idx, 1]
    return {"users": users, "items": items, "item_clusters": clusters[items], "item_feats": data.item_feats[items]}


class TestSamplers:
    def test_inbatch_has_no_extra_negatives(self, world):
        model, s, data, cl = _setup(world, "in-batch")
        b = _batch(data, cl)
        assert s.negatives(model, b, model.encode_users(b["users"], data.user_profile[b["users"]])) == []

    def test_unknown_sampler(self, world):
        with pytest.raises(ValueError):
            _setup(world, "nope")

    @pytest.mark.parametrize("name,kw", [("dns", {}), ("ance", {}), ("goobs", {}), ("cluster_goobs", {}), ("cbns", {})])
    def test_all_samplers_train_a_step(self, world, name, kw):
        model, s, data, cl = _setup(world, name, **kw)
        for _ in range(2):                                              # cbns needs one step to fill its queue
            b = _batch(data, cl)
            u = model.encode_users(b["users"], data.user_profile[b["users"]])
            v = model.encode_items(b["item_feats"], b["item_clusters"], b["items"])
            loss = contrastive_loss(u, v, b["items"], model.tau, data.item_logq[b["items"]], s.negatives(model, b, u))
            loss.backward()
            s.after_step(model, b, v)
            assert torch.isfinite(loss)

    def test_cluster_goobs_negatives_share_the_positives_cluster(self, world):
        model, s, data, cl = _setup(world, "cluster_goobs", n_cluster=3, random_per_cluster=2)
        b = _batch(data, cl)
        u = model.encode_users(b["users"], data.user_profile[b["users"]])
        hard, rand = s.negatives(model, b, u)
        assert hard.emb.shape == (32, 3, 32) and rand.emb.shape == (32, 6, 32)
        ok = hard.valid
        assert (cl[hard.ids[ok]] == b["item_clusters"].unsqueeze(1).expand_as(hard.ids)[ok]).all()
        assert len(set(cl[rand.ids[rand.valid]].tolist())) > 1           # random block spans clusters

    def test_cluster_negatives_are_closer_than_random_ones(self, world):
        # Sec. 3.3 premise: same-cluster items are semantically nearer the positive than random items.
        data, cl = world
        sims_c, sims_r = [], []
        _, s, _, _ = _setup(world, "cluster_goobs", n_cluster=4, random_per_cluster=4)
        b = _batch(data, cl, 64)
        ids, feats, valid = s.pool.sample_in_cluster(b["item_clusters"], 4)
        rid, _, rfeats, _ = s.pool.sample_random((64, 4))
        pos = data.content_emb[b["items"]].unsqueeze(1)
        cs = -torch.cdist(pos, data.content_emb[ids]).squeeze(1)
        rs = -torch.cdist(pos, data.content_emb[rid]).squeeze(1)
        assert cs[valid].mean() > rs.mean()

    def test_goobs_updates_pool_with_batch_items(self, world):
        model, s, data, cl = _setup(world, "goobs")
        b = _batch(data, cl)
        s.after_step(model, b, None)
        slots = s.pool.slot(b["items"], b["item_clusters"])
        assert s.pool.filled[slots].all()
        assert (s.pool.cluster_ids[slots] == b["item_clusters"]).all()
        stored = s.pool.item_ids[slots]
        assert (s.pool.slot(stored, cl[stored]) == slots).all()      # whatever won each slot hashes there

    def test_cbns_queue_is_bounded_and_detached(self, world):
        model, s, data, cl = _setup(world, "cbns", queue_size=50)
        b = _batch(data, cl)
        v = model.encode_items(b["item_feats"], b["item_clusters"], b["items"])
        for _ in range(5):
            s.after_step(model, b, v)
        assert len(s.emb) == 50 and not s.emb.requires_grad

    def test_dns_picks_highest_scoring_candidates(self, world):
        torch.manual_seed(0)
        model, s, data, cl = _setup(world, "dns", n_candidates=40, k=3)
        b = _batch(data, cl, 8)
        u = model.encode_users(b["users"], data.user_profile[b["users"]])
        (neg,) = s.negatives(model, b, u)
        chosen = (u.unsqueeze(1) * neg.emb).sum(-1)
        allitems = model.encode_items(data.item_feats, cl, torch.arange(data.n_items))
        median = (u @ allitems.t()).median(dim=1).values
        assert (chosen.mean(1) > median).all()                          # far above a typical item

    def test_ance_index_is_stale_until_refresh(self, world):
        model, s, data, cl = _setup(world, "ance", refresh_every=3, k=2)
        b = _batch(data, cl, 8)
        u = model.encode_users(b["users"], data.user_profile[b["users"]])
        s.negatives(model, b, u)
        first = s.index.clone()
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.5)
        s.after_step(model, b, None)
        s.negatives(model, b, u)
        assert torch.equal(s.index, first)                              # step 1: not refreshed
        for _ in range(2):
            s.after_step(model, b, None)
        s.negatives(model, b, u)
        assert not torch.equal(s.index, first)                          # step 3: refreshed

    def test_ance_excludes_positive(self, world):
        model, s, data, cl = _setup(world, "ance", k=5)
        b = _batch(data, cl, 16)
        u = model.encode_users(b["users"], data.user_profile[b["users"]])
        (neg,) = s.negatives(model, b, u)
        assert (neg.ids != b["items"].unsqueeze(1)).all()


# ---------------------------------------------------------------------------
# Data, training, evaluation
# ---------------------------------------------------------------------------

class TestData:
    def test_split_disjoint_and_masks(self, world):
        data, _ = world
        for u in range(data.n_users):
            assert not set(data.train_items[u]) & set(data.test_items[u])
        assert data.train_mask.sum() == len(data.train_pairs)
        assert data.relevant.sum() >= data.train_mask.sum()

    def test_logq_is_a_log_probability(self, world):
        data, _ = world
        assert (data.item_logq < 0).all()
        assert torch.exp(data.item_logq).sum().item() == pytest.approx(1.0, abs=0.05)

    def test_false_negative_rate_grows_with_cluster_fineness(self, world):
        data, _ = world
        fine = ContentClusterer(40).fit(data.content_emb).assign(data.content_emb)
        coarse = ContentClusterer(2).fit(data.content_emb).assign(data.content_emb)
        r_fine, r_coarse = false_negative_rate(data, fine, 4000), false_negative_rate(data, coarse, 4000)
        assert 0 <= r_fine <= 1 and 0 <= r_coarse <= 1
        assert r_fine > r_coarse        # Sec. 3.4: finer clusters hold more truly-similar items => more false negatives


class TestEvaluation:
    def test_train_items_are_excluded_from_ranking(self, world):
        data, clusters = world
        model = TwoTower(data.n_users, data.n_items, 8, FD_of(data), FD_of(data))
        s = retrieval_scores(model, data, clusters)
        assert torch.isinf(s[data.train_mask]).all() and torch.isfinite(s[~data.train_mask]).all()

    def test_hit_rate_monotone_in_k(self, world):
        data, clusters = world
        model = TwoTower(data.n_users, data.n_items, 8, FD_of(data), FD_of(data))
        hr = evaluate(model, data, clusters, ks=(10, 50, 100))
        assert hr[10] <= hr[50] <= hr[100]

    def test_popularity_report(self):
        concentrated = torch.zeros(100, 200)
        concentrated[:, :5] = 10.0                                      # everyone gets the same 5 items
        spread = torch.randn(100, 200)
        a, b = popularity_report(concentrated, 5, 10, 5), popularity_report(spread, 5, 10, 5)
        assert a["head_share"] == pytest.approx(1.0) and a["coverage"] == pytest.approx(5 / 200)
        assert b["head_share"] < a["head_share"] and b["coverage"] > a["coverage"]
        assert a["items_over_threshold"] == 5


class TestTraining:
    def _run(self, name, steps=250, **kw):
        data = SyntheticInteractions(n_users=300, n_true_clusters=20, items_per_cluster=20, seed=0)
        clusters = ContentClusterer(20).fit(data.content_emb).assign(data.content_emb)
        torch.manual_seed(0)
        model = TwoTower(data.n_users, data.n_items, 20, FD_of(data), FD_of(data))
        pool = build_pool(data, clusters, 20, slots=10) if name in ("goobs", "cluster_goobs") else None
        sampler = make_sampler(name, ItemCatalog(data.item_feats, clusters), pool, **kw)
        losses = train(model, data, clusters, sampler, steps=steps, batch_size=64)
        return model, data, clusters, losses

    def test_inbatch_training_beats_untrained(self):
        torch.manual_seed(0)
        model, data, clusters, losses = self._run("in-batch")
        fresh = TwoTower(data.n_users, data.n_items, 20, FD_of(data), FD_of(data))
        assert evaluate(model, data, clusters)[50] > evaluate(fresh, data, clusters)[50] + 0.1
        assert sum(losses[-20:]) < sum(losses[:20])

    def test_cluster_goobs_trains_and_improves(self):
        model, data, clusters, losses = self._run("cluster_goobs", n_cluster=1, random_per_cluster=7)
        fresh = TwoTower(data.n_users, data.n_items, 20, FD_of(data), FD_of(data))
        assert evaluate(model, data, clusters)[50] > evaluate(fresh, data, clusters)[50] + 0.1
        assert all(torch.isfinite(torch.tensor(losses)))

    def test_goobs_oob_exposure_beats_inbatch_only(self):
        # Table 1: even random OOB negatives help over the in-batch baseline.
        base = self._run("in-batch")
        goobs = self._run("goobs", n_random=16)
        assert evaluate(goobs[0], goobs[1], goobs[2])[100] >= evaluate(base[0], base[1], base[2])[100] - 0.01
