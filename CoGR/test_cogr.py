"""Tests for the CoGR implementation.

Run with:
    pytest test_cogr.py -v
"""

import random

import pytest
import torch

from cogr import (
    InvertedIndex, ItemRewardCache, ToyKeywordGenerator, bm25_rank,
    build_sft_query_targets, co_evolve, evaluate, f1_from_counts,
    grpo_advantages, grpo_loss, item_reward_naive, mrr_at_k, ndcg_at_k,
    precision_recall_f1, query_reward, representation, sft_train,
)
from demo import make_data


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class TestInvertedIndex:
    def test_overlap_retrieval(self):
        idx = InvertedIndex({"a": ["x", "y"], "b": ["y"], "c": ["z"]})
        assert idx.retrieve("q", ["y"]) == {"a", "b"}
        assert idx.retrieve("q", ["nothing"]) == set()

    def test_raw_string_always_matches(self):
        # I_ret = {i : (S_q ∪ {q}) ∩ (S_i ∪ {i}) ≠ ∅}: the raw strings count
        idx = InvertedIndex({"walmart": ["shopping"]})
        assert idx.retrieve("walmart", []) == {"walmart"}

    def test_query_string_matches_item_keyword(self):
        idx = InvertedIndex({"i1": ["zoom"]})
        assert idx.retrieve("zoom", []) == {"i1"}

    def test_representation_includes_entity(self):
        assert representation("e", ["k"]) == {"e", "k"}

    def test_vocab_size(self):
        idx = InvertedIndex({"a": ["x"], "b": ["x"]})
        assert idx.vocab_size == 3  # x, a, b


class TestBM25:
    def test_rarer_term_ranks_higher(self):
        idx = InvertedIndex({"i1": ["rare"], "i2": ["common"], "i3": ["common"], "i4": ["common"]})
        ranked = bm25_rank({"rare", "common"}, {"i1", "i2"}, idx)
        assert ranked[0] == "i1"

    def test_more_overlap_ranks_higher(self):
        idx = InvertedIndex({"i1": ["a", "b"], "i2": ["a"], "i3": ["c"], "i4": ["d"]})
        assert bm25_rank({"a", "b"}, {"i1", "i2"}, idx)[0] == "i1"

    def test_returns_all_candidates(self):
        idx = InvertedIndex({"i1": ["a"], "i2": ["a"]})
        assert set(bm25_rank({"a"}, {"i1", "i2"}, idx)) == {"i1", "i2"}


# ---------------------------------------------------------------------------
# Metrics / query-side reward
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_prf1_values(self):
        p, r, f1 = precision_recall_f1({"a", "b", "c", "d"}, {"a", "b", "x", "y", "z", "w"})
        assert (p, r) == (0.5, pytest.approx(1 / 3))
        assert f1 == pytest.approx(2 * 0.5 * (1 / 3) / (0.5 + 1 / 3))

    def test_prf1_empty_retrieved(self):
        assert precision_recall_f1(set(), {"a"}) == (0.0, 0.0, 0.0)

    def test_f1_count_identity(self):
        for ret, rel, tp in [(10, 20, 5), (3, 3, 3), (7, 2, 0)]:
            retrieved = set(range(ret))
            relevant = set(range(ret - tp, ret - tp + rel))
            assert f1_from_counts(len(retrieved & relevant), ret, rel) == pytest.approx(
                precision_recall_f1(retrieved, relevant)[2])

    def test_mrr_ndcg(self):
        assert mrr_at_k(["x", "a"], {"a"}) == 0.5
        assert mrr_at_k(["x"], {"a"}) == 0.0
        assert ndcg_at_k(["a", "b"], {"a", "b"}) == pytest.approx(1.0)
        assert ndcg_at_k(["x", "a"], {"a"}) < 1.0


class TestQueryReward:
    def test_equals_f1_within_budget(self):
        assert query_reward(["a", "b"], {"i1", "i2"}, {"i1"}, k_max=5) == pytest.approx(2 / 3)

    def test_zero_over_budget(self):
        assert query_reward(["a", "b", "c"], {"i1"}, {"i1"}, k_max=2) == 0.0

    def test_at_budget_is_allowed(self):
        assert query_reward(["a", "b"], {"i1"}, {"i1"}, k_max=2) == 1.0

    def test_duplicates_dont_count_twice(self):
        assert query_reward(["a", "a"], {"i1"}, {"i1"}, k_max=1) == 1.0


# ---------------------------------------------------------------------------
# SFT targets (Alg. 1)
# ---------------------------------------------------------------------------

class TestSFTTargets:
    def test_top_n_most_frequent(self):
        item_kws = {"i1": ["a", "b"], "i2": ["a", "c"], "i3": ["a", "b"]}
        t = build_sft_query_targets(["q"], item_kws, {"q": {"i1", "i2", "i3"}}, top_n=2)
        assert t["q"] == ["a", "b"]

    def test_only_relevant_items_contribute(self):
        item_kws = {"i1": ["a"], "i2": ["z"]}
        t = build_sft_query_targets(["q"], item_kws, {"q": {"i1"}}, top_n=5)
        assert t["q"] == ["a"]

    def test_target_creates_overlap_with_relevant_items(self):
        item_kws = {"i1": ["a", "b"], "i2": ["a", "c"]}
        rel = {"q": {"i1", "i2"}}
        t = build_sft_query_targets(["q"], item_kws, rel, top_n=1)
        idx = InvertedIndex(item_kws)
        assert idx.retrieve("q", t["q"]) == {"i1", "i2"}

    def test_deterministic_tie_break(self):
        item_kws = {"i1": ["b", "a"]}
        t = build_sft_query_targets(["q"], item_kws, {"q": {"i1"}}, top_n=1)
        assert t["q"] == ["a"]


# ---------------------------------------------------------------------------
# Item-side counterfactual reward (Eq. 2.3 / App. B.3)
# ---------------------------------------------------------------------------

def random_world(seed, n_items=15, n_queries=12, n_kw=8):
    rnd = random.Random(seed)
    kws = [f"k{j}" for j in range(n_kw)]
    items = [f"item{i}" for i in range(n_items)]
    queries = [f"query{i}" for i in range(n_queries)]
    item_kws = {i: rnd.sample(kws, rnd.randint(0, 4)) for i in items}
    query_kws = {q: rnd.sample(kws, rnd.randint(0, 4)) for q in queries}
    rel = {q: set(rnd.sample(items, rnd.randint(1, 8))) for q in queries}
    return kws, items, queries, item_kws, query_kws, rel


class TestItemReward:
    @pytest.mark.parametrize("seed", range(8))
    def test_incremental_matches_naive(self, seed):
        """The central B.3 claim: cached incremental reward == full recompute."""
        rnd = random.Random(100 + seed)
        kws, items, queries, item_kws, query_kws, rel = random_world(seed)
        q_idx, i_idx = InvertedIndex(query_kws), InvertedIndex(item_kws)
        cache = ItemRewardCache(queries, q_idx, i_idx, rel, k_max=6)
        for _ in range(25):
            item = rnd.choice(items)
            cand = rnd.sample(kws, rnd.randint(0, 6))
            assert cache.reward(item, cand) == pytest.approx(
                item_reward_naive(item, cand, queries, q_idx, item_kws, rel, 6), abs=1e-9)

    def test_unchanged_keywords_give_zero(self):
        kws, items, queries, item_kws, query_kws, rel = random_world(3)
        q_idx, i_idx = InvertedIndex(query_kws), InvertedIndex(item_kws)
        cache = ItemRewardCache(queries, q_idx, i_idx, rel, k_max=6)
        for it in items:
            assert cache.reward(it, item_kws[it]) == pytest.approx(0.0)

    def test_over_budget_is_minus_one(self):
        kws, items, queries, item_kws, query_kws, rel = random_world(1)
        q_idx, i_idx = InvertedIndex(query_kws), InvertedIndex(item_kws)
        cache = ItemRewardCache(queries, q_idx, i_idx, rel, k_max=2)
        assert cache.reward(items[0], ["a", "b", "c"]) == -1.0
        assert item_reward_naive(items[0], ["a", "b", "c"], queries, q_idx, item_kws, rel, 2) == -1.0

    def test_adding_a_matching_keyword_for_relevant_item_helps(self):
        q_idx = InvertedIndex({"q": ["shop"]})
        item_kws = {"walmart": ["store"], "other": ["x"]}
        rel = {"q": {"walmart"}}
        cache = ItemRewardCache(["q"], q_idx, InvertedIndex(item_kws), rel, k_max=5)
        assert cache.reward("walmart", ["store", "shop"]) > 0

    def test_matching_irrelevant_query_hurts(self):
        q_idx = InvertedIndex({"q": ["shop"]})
        item_kws = {"walmart": ["shop"], "zoom": ["meeting"]}
        rel = {"q": {"walmart"}}
        cache = ItemRewardCache(["q"], q_idx, InvertedIndex(item_kws), rel, k_max=5)
        assert cache.reward("zoom", ["meeting", "shop"]) < 0

    def test_dropping_a_relevant_match_hurts(self):
        q_idx = InvertedIndex({"q": ["shop"]})
        item_kws = {"walmart": ["shop"], "amazon": ["shop"]}
        rel = {"q": {"walmart", "amazon"}}
        cache = ItemRewardCache(["q"], q_idx, InvertedIndex(item_kws), rel, k_max=5)
        assert cache.reward("walmart", ["other"]) < 0

    def test_only_affected_queries_matter(self):
        # q2 never touches item's old or new keywords -> reward unaffected by it
        q_idx = InvertedIndex({"q1": ["a"], "q2": ["z"]})
        item_kws = {"i": ["b"], "j": ["z"]}
        rel = {"q1": {"i"}, "q2": {"j"}}
        cache = ItemRewardCache(["q1", "q2"], q_idx, InvertedIndex(item_kws), rel, k_max=5)
        r = cache.reward("i", ["a"])
        assert r == pytest.approx(item_reward_naive("i", ["a"], ["q1", "q2"], q_idx, item_kws, rel, 5))
        assert r == pytest.approx(1.0 - 0.0)  # q1: F1 0 -> 1


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------

class TestGRPO:
    def test_advantages_zero_mean_unit_std(self):
        adv = grpo_advantages(torch.tensor([[0.1, 0.5, 0.9, 0.3], [1.0, 2.0, 3.0, 4.0]]))
        assert torch.allclose(adv.mean(-1), torch.zeros(2), atol=1e-5)
        assert torch.allclose(adv.std(-1, unbiased=False), torch.ones(2), atol=1e-3)

    def test_constant_group_gives_zero_advantage(self):
        adv = grpo_advantages(torch.full((1, 6), 0.4))
        assert torch.all(adv == 0)

    def test_groups_are_independent(self):
        a = grpo_advantages(torch.tensor([[1.0, 2.0, 3.0]]))
        b = grpo_advantages(torch.tensor([[1.0, 2.0, 3.0], [100.0, 0.0, 50.0]]))
        assert torch.allclose(a[0], b[0])

    def test_loss_at_ratio_one_is_neg_mean_advantage(self):
        lp = torch.randn(3, 4)
        adv = torch.randn(3, 4)
        assert grpo_loss(lp, lp, adv).item() == pytest.approx(-adv.mean().item(), abs=1e-6)

    def test_clipping_blocks_gradient_beyond_trust_region(self):
        old = torch.zeros(1, 1)
        logp = torch.full((1, 1), 1.0, requires_grad=True)  # ratio e >> 1.2
        grpo_loss(logp, old, torch.ones(1, 1)).backward()
        assert logp.grad.item() == 0.0

    def test_unclipped_gradient_flows_with_positive_advantage(self):
        old = torch.zeros(1, 1)
        logp = torch.zeros(1, 1, requires_grad=True)
        grpo_loss(logp, old, torch.ones(1, 1)).backward()
        assert logp.grad.item() < 0  # gradient descent raises logp


# ---------------------------------------------------------------------------
# Toy generator
# ---------------------------------------------------------------------------

VOCAB = [f"w{i}" for i in range(12)]


class TestToyGenerator:
    def test_sample_shapes_and_vocab(self):
        g = ToyKeywordGenerator(VOCAB)
        sets, lp = g.sample(["a b", "c"], 5)
        assert len(sets) == 2 and len(sets[0]) == 5 and lp.shape == (2, 5)
        assert all(k in VOCAB for s in sets for r in s for k in r)

    def test_logp_matches_resampled_logp(self):
        g = ToyKeywordGenerator(VOCAB)
        sets, lp = g.sample(["a b"], 4)
        assert torch.allclose(lp, g.log_prob(["a b"], sets), atol=1e-6)

    def test_logp_nonpositive(self):
        g = ToyKeywordGenerator(VOCAB)
        _, lp = g.sample(["x y z"], 6)
        assert (lp <= 0).all()

    def test_hash_is_deterministic(self):
        assert ToyKeywordGenerator(VOCAB)._hash("hello") == ToyKeywordGenerator(VOCAB)._hash("hello")

    def test_generate_respects_k_max(self):
        g = ToyKeywordGenerator(VOCAB)
        with torch.no_grad():
            g.out.bias.fill_(5.0)
        assert len(g.generate(["t"], k_max=3)["t"]) == 3

    def test_sft_learns_targets(self):
        torch.manual_seed(0)
        g = ToyKeywordGenerator(VOCAB)
        texts = ["alpha", "beta"]
        targets = [["w0", "w1"], ["w5"]]
        sft_train(g, texts, targets, epochs=100, lr=0.1)
        out = g.generate(texts)
        assert set(out["alpha"]) == {"w0", "w1"} and set(out["beta"]) == {"w5"}


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_evaluate_perfect_retriever(self):
        idx = InvertedIndex({"i1": ["a"], "i2": ["a"], "i3": ["b"]})
        rel = {"q": {"i1", "i2"}}
        m = evaluate(["q"], {"q": ["a"]}, idx, rel, k=10)
        assert m["P"] == 1.0 and m["R"] == 1.0 and m["F1"] == 1.0 and m["MRR@10"] == 1.0

    def test_co_evolving_improves_training_reward(self):
        random.seed(0)
        torch.manual_seed(0)
        queries, items, rel, init_kws, vocab = make_data(n_topics=4, items_per_topic=6, queries_per_topic=5)
        targets = build_sft_query_targets(queries, init_kws, rel, top_n=6)
        qg, ig = ToyKeywordGenerator(vocab, seed=1), ToyKeywordGenerator(vocab, seed=2)
        sft_train(qg, queries, [targets[q] for q in queries], epochs=30)
        sft_train(ig, items, [init_kws[i] for i in items], epochs=30)
        hist = co_evolve(qg, ig, queries, items, rel, rounds=3, query_epochs=4, item_epochs=2)
        assert len(hist) == 3
        assert hist[-1]["query_reward"] > hist[0]["query_reward"]

    def test_budget_enforced_in_loop(self):
        random.seed(1)
        torch.manual_seed(1)
        queries, items, rel, init_kws, vocab = make_data(n_topics=3, items_per_topic=4, queries_per_topic=3)
        qg, ig = ToyKeywordGenerator(vocab, seed=1), ToyKeywordGenerator(vocab, seed=2)
        co_evolve(qg, ig, queries, items, rel, rounds=1, k_max=4, query_epochs=1, item_epochs=1)
        assert all(len(v) <= 4 for v in qg.generate(queries, 4).values())
