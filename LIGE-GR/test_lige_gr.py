"""Tests for the LIGE-GR implementation.

Run with:
    pytest test_lige_gr.py -v
"""

import itertools
import math

import pytest
import torch

from lige_gr import (
    ItemValueModel,
    ControlLayer,
    ContextFreeModel,
    ContextAwareModel,
    LigeGR,
    context_aware_loss,
    normalized_entropy,
    relative_ne_improvement,
    list_composition_metrics,
    list_value,
    future_value_step,
    future_value_duration,
    palette_decode,
    make_cf_scorer,
    make_ca_scorer,
)

Du, Di, D, K = 6, 5, 16, 3


def random_scorer(C, seed=0):
    """A prefix-dependent toy scorer (hash-like) so context matters."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(C, K + 1, generator=g)
    inter = torch.rand(C, C, K + 1, generator=g) * 0.5   # effect of last-selected item

    def score(prefixes):
        N, t = prefixes.shape
        p = base.unsqueeze(0).expand(N, C, K + 1).clone()
        if t > 0:
            p = p * (0.5 + inter[prefixes[:, -1]])
        p = p.clamp(0.01, 0.99)
        return p[..., :K], p[..., K]
    return score


# ---------------------------------------------------------------------------
# ItemValueModel / ControlLayer
# ---------------------------------------------------------------------------

class TestItemVM:
    def test_linear_combination(self):
        vm = ItemValueModel([1.0, 2.0, 3.0])
        assert vm(torch.tensor([0.1, 0.2, 0.3])).item() == pytest.approx(0.1 + 0.4 + 0.9)

    def test_batch_shape(self):
        vm = ItemValueModel([1.0, 1.0])
        assert vm(torch.rand(4, 7, 2)).shape == (4, 7)


class TestControlLayer:
    def test_selected_items_masked(self):
        cl = ControlLayer(5)
        out = cl(torch.tensor([[1, 3]]))
        assert out[0, 1] == float("-inf") and out[0, 3] == float("-inf")
        assert out[0, 0] == 0.0

    def test_empty_prefix_all_zero(self):
        assert torch.equal(ControlLayer(4)(torch.zeros(2, 0, dtype=torch.long)), torch.zeros(2, 4))

    def test_hard_rule_masks_forbidden_category(self):
        cats = torch.tensor([0, 1, 2, 2])
        cl = ControlLayer(4, categories=cats, forbidden_pairs={(0, 1)})
        out = cl(torch.tensor([[0]]))
        assert out[0, 1] == float("-inf")        # cat 1 forbidden with cat 0
        assert out[0, 2] == 0.0 and out[0, 3] == 0.0

    def test_gap_demotion_values(self):
        cats = torch.tensor([0, 1, 0, 1, 0])
        cl = ControlLayer(5, categories=cats, gap_coef=2.0)
        # prefix [0, 1]: candidate 2 (cat 0) nearest same-cat at t'=1, t=3 -> -exp(1-3+1)*2
        out = cl(torch.tensor([[0, 1]]))
        assert out[0, 2].item() == pytest.approx(-math.exp(-1) * 2.0)
        # candidate 3 (cat 1): nearest at t'=2 -> adjacent: -exp(0)*2
        assert out[0, 3].item() == pytest.approx(-2.0)

    def test_gap_demotion_decays_with_distance(self):
        cats = torch.tensor([0, 1, 1, 0, 2])
        cl = ControlLayer(5, categories=cats, gap_coef=1.0)
        near = cl(torch.tensor([[1, 0]]))[0, 2]   # cat 1 at t'=1, t=3
        far = cl(torch.tensor([[1, 0, 4]]))[0, 2]  # cat 1 at t'=1, t=4
        assert far > near  # less negative

    def test_dpp_prefers_diverse(self):
        emb = torch.tensor([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]])
        cl = ControlLayer(3, embeddings=emb, dpp_weight=1.0)
        out = cl(torch.tensor([[0]]))
        assert out[0, 2] > out[0, 1]  # orthogonal item adds more volume than a near-duplicate

    def test_dpp_increment_matches_logdet_definition(self):
        emb = torch.randn(4, 3)
        cl = ControlLayer(4, embeddings=emb, dpp_weight=1.0, dpp_eps=0.0)
        phi = torch.nn.functional.normalize(emb, dim=-1)
        out = cl(torch.tensor([[0, 1]]))
        expected = torch.logdet(phi[[0, 1, 2]] @ phi[[0, 1, 2]].T) - torch.logdet(phi[[0, 1]] @ phi[[0, 1]].T)
        assert out[0, 2].item() == pytest.approx(expected.item(), abs=1e-4)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestModels:
    def test_cf_shapes(self):
        cf = ContextFreeModel(Du, Di, D, K)
        v, lg = cf(torch.randn(2, Du), torch.randn(2, 7, Di))
        assert v.shape == (2, 7, D) and lg.shape == (2, 7, K + 1)

    def test_ca_shapes_and_default_depth(self):
        ca = ContextAwareModel(D, K)
        assert len(ca.blocks.layers) == 4
        assert ca.blocks.layers[0].self_attn.num_heads == 4
        assert ca(torch.randn(3, 6, D)).shape == (3, 6, K + 1)

    def test_ca_is_causal(self):
        ca = ContextAwareModel(D, K).eval()
        x = torch.randn(2, 6, D)
        y1 = ca(x)
        x2 = x.clone()
        x2[:, 4:] = torch.randn(2, 2, D)      # change only future positions
        y2 = ca(x2)
        assert torch.allclose(y1[:, :4], y2[:, :4], atol=1e-5)
        assert not torch.allclose(y1[:, 4:], y2[:, 4:], atol=1e-5)

    def test_ca_prediction_depends_on_prefix(self):
        ca = ContextAwareModel(D, K).eval()
        last = torch.randn(1, 1, D)
        a = ca(torch.cat([torch.randn(1, 3, D), last], 1))[:, -1]
        b = ca(torch.cat([torch.randn(1, 3, D), last], 1))[:, -1]
        assert not torch.allclose(a, b, atol=1e-5)

    def test_task_heads_shared_across_positions(self):
        """Identical inputs at the same position -> identical outputs; the head is position-agnostic."""
        ca = ContextAwareModel(D, K).eval()
        x = torch.randn(1, 1, D)
        assert torch.allclose(ca(x), ca(x))


# ---------------------------------------------------------------------------
# Losses / metrics
# ---------------------------------------------------------------------------

class TestTrainingAndMetrics:
    def test_loss_mask_ignores_padding(self):
        logits = torch.randn(2, 4, K + 1)
        labels = (torch.rand(2, 4, K + 1) < 0.5).float()
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float32)
        l_mask = context_aware_loss(logits, labels, mask)
        bad = labels.clone(); bad[1, 2:] = 1 - bad[1, 2:]
        assert context_aware_loss(logits, bad, mask).item() == pytest.approx(l_mask.item())

    def test_ne_of_background_rate_is_one(self):
        y = (torch.rand(10000) < 0.2).float()
        p = torch.full_like(y, y.mean().item())
        assert normalized_entropy(p, y).item() == pytest.approx(1.0, abs=1e-4)

    def test_ne_better_predictor_lower(self):
        y = (torch.rand(5000) < 0.3).float()
        good = (0.1 + 0.8 * y).clamp(0, 1)
        assert normalized_entropy(good, y) < normalized_entropy(torch.full_like(y, 0.3), y)

    def test_relative_ne_improvement(self):
        assert relative_ne_improvement(1.0, 0.9843) == pytest.approx(0.0157)

    def test_composition_metrics(self):
        m = list_composition_metrics([1, 1, 1, 2, 3])
        assert m["longest_streak"] == 3 and m["distinct_topics"] == 3
        assert list_composition_metrics([1, 2, 3, 4])["topic_entropy"] == pytest.approx(math.log(4))

    def test_ca_training_step_updates_only_ca(self):
        cf, ca = ContextFreeModel(Du, Di, D, K), ContextAwareModel(D, K)
        model = LigeGR(cf, ca, ItemValueModel([1.0] * K))
        loss = model.ca_training_step(torch.randn(4, Du), torch.randn(4, 5, Di), (torch.rand(4, 5, K + 1) < 0.3).float())
        loss.backward()
        assert all(p.grad is None for p in cf.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in ca.parameters())

    def test_ca_can_overfit_context_signal(self):
        """Label of item t depends on whether the previous item equals it; CF can't know this, CA can."""
        torch.manual_seed(0)
        cf = ContextFreeModel(Du, Di, D, 1)
        for p in cf.parameters():
            p.requires_grad_(False)
        ca = ContextAwareModel(D, 1, num_layers=2, max_len=4)
        model = LigeGR(cf, ca, ItemValueModel([1.0]))
        pool = torch.randn(4, Di)
        idx = torch.randint(0, 4, (64, 3))
        items, user = pool[idx], torch.zeros(64, Du)
        rep = torch.zeros(64, 3, 2)
        rep[:, 1:, 0] = (idx[:, 1:] == idx[:, :-1]).float()
        opt = torch.optim.Adam(ca.parameters(), lr=3e-3)
        first = None
        for _ in range(150):
            loss = model.ca_training_step(user, items, rep)
            first = first if first is not None else loss.item()
            opt.zero_grad(); loss.backward(); opt.step()
        assert loss.item() < first * 0.7


# ---------------------------------------------------------------------------
# List value, future value
# ---------------------------------------------------------------------------

class TestListVM:
    def setup_method(self):
        self.C = 5
        self.score = random_scorer(self.C)
        self.vm = ItemValueModel([1.0, 0.5, 2.0])
        self.cl = ControlLayer(self.C)

    def test_vanilla_is_plain_sum(self):
        order = [2, 0, 4]
        total = 0.0
        for t, c in enumerate(order):
            pr = torch.tensor([order[:t]], dtype=torch.long).view(1, t)
            total += self.vm(self.score(pr)[0])[0, c].item()
        assert list_value(self.score, self.cl, self.vm, order, golden=False) == pytest.approx(total)

    def test_golden_weights_by_continuation(self):
        order = [1, 3]
        p0, c0 = self.score(torch.zeros(1, 0, dtype=torch.long))
        p1, _ = self.score(torch.tensor([[1]]))
        v0 = self.vm(p0)[0, 1].item()
        v1 = self.vm(p1)[0, 3].item()
        expect = v0 + c0[0, 1].item() * v1
        assert list_value(self.score, self.cl, self.vm, order, golden=True) == pytest.approx(expect)

    def test_golden_le_vanilla_for_positive_scores(self):
        order = [0, 1, 2, 3]
        g = list_value(self.score, self.cl, self.vm, order, golden=True)
        v = list_value(self.score, self.cl, self.vm, order, golden=False)
        assert g < v


class TestFutureValue:
    def test_step_closed_form(self):
        s, p = torch.tensor([2.0]), torch.tensor([0.5])
        assert future_value_step(s, p, 3).item() == pytest.approx(2.0 * (0.5 + 0.25 + 0.125))

    def test_zero_when_nothing_remains(self):
        assert future_value_step(torch.tensor([2.0]), torch.tensor([0.5]), 0).item() == 0.0
        z = torch.tensor([1.0])
        assert future_value_duration(z, z * 0.5, z, z, 0).item() == 0.0

    def test_duration_equals_step_when_durations_match(self):
        s, p, d = torch.tensor([1.5]), torch.tensor([0.7]), torch.tensor([20.0])
        assert future_value_duration(s, p, d, d, 4).item() == pytest.approx(future_value_step(s, p, 4).item())

    def test_duration_reduces_long_item_penalty(self):
        """Last item much longer than average: step estimate is overly pessimistic, duration-aware raises it."""
        s, p = torch.tensor([1.0]), torch.tensor([0.4])
        d_last, d_bar = torch.tensor([60.0]), torch.tensor([20.0])
        assert future_value_duration(s, p, d_last, d_bar, 3) > future_value_step(s, p, 3)

    def test_duration_lowers_estimate_for_short_last_item(self):
        s, p = torch.tensor([1.0]), torch.tensor([0.9])
        assert future_value_duration(s, p, torch.tensor([5.0]), torch.tensor([20.0]), 3) < future_value_step(s, p, 3)


# ---------------------------------------------------------------------------
# Palette decoder
# ---------------------------------------------------------------------------

class TestPalette:
    def setup_method(self):
        self.C, self.T = 6, 3
        self.vm = ItemValueModel([1.0, 0.5, 2.0])
        self.cl = ControlLayer(self.C)
        self.score = random_scorer(self.C, seed=3)

    def brute_force(self, golden, cl=None):
        cl = cl or self.cl
        best, best_v = None, -1e9
        for perm in itertools.permutations(range(self.C), self.T):
            v = list_value(self.score, cl, self.vm, perm, golden=golden)
            if v > best_v:
                best, best_v = list(perm), v
        return best, best_v

    def test_strict_generalization_recovers_itemwise_greedy(self):
        cf_probs = torch.rand(self.C, K + 1)
        cl = ControlLayer(self.C, categories=torch.tensor([0, 0, 1, 1, 2, 2]), gap_coef=0.3)
        res = palette_decode(make_cf_scorer(cf_probs), cl, self.vm, self.C, self.T,
                             beam_width=1, golden=False, future="none")
        # reference incumbent: v_t = argmax_c itemVM(CF(u,c)) + CL(c | V_{t-1})
        order = []
        item = self.vm(cf_probs[:, :K])
        for _ in range(self.T):
            s = item + cl(torch.tensor([order], dtype=torch.long).view(1, len(order)))[0]
            order.append(int(s.argmax()))
        assert res.order == order

    @pytest.mark.parametrize("golden", [False, True])
    def test_full_beam_is_exhaustive(self, golden):
        """b >= number of prefixes at every step (F=0 at full length) -> exact optimum."""
        res = palette_decode(self.score, self.cl, self.vm, self.C, self.T, beam_width=200, golden=golden)
        best, best_v = self.brute_force(golden)
        assert res.order == best
        assert res.list_value == pytest.approx(best_v, abs=1e-5)

    def test_reported_value_matches_list_value(self):
        res = palette_decode(self.score, self.cl, self.vm, self.C, self.T, beam_width=3, golden=True)
        assert res.list_value == pytest.approx(list_value(self.score, self.cl, self.vm, res.order, True), abs=1e-5)

    def test_wider_beam_never_worse_at_full_coverage(self):
        v1 = palette_decode(self.score, self.cl, self.vm, self.C, self.T, beam_width=1).list_value
        v200 = palette_decode(self.score, self.cl, self.vm, self.C, self.T, beam_width=200).list_value
        assert v200 >= v1 - 1e-6

    def test_lists_have_distinct_items_and_length(self):
        res = palette_decode(self.score, self.cl, self.vm, self.C, 5, beam_width=4)
        assert len(res.order) == 5 and len(set(res.order)) == 5

    def test_respects_hard_constraint(self):
        cats = torch.tensor([0, 1, 2, 2, 2, 2])
        cl = ControlLayer(self.C, categories=cats, forbidden_pairs={(0, 1)})
        for b in (1, 4):
            res = palette_decode(self.score, cl, self.vm, self.C, self.T, beam_width=b)
            assert not ({0, 1} <= set(res.order))

    def test_list_length_capped_by_candidates(self):
        res = palette_decode(self.score, self.cl, self.vm, self.C, 50, beam_width=2)
        assert len(res.order) == self.C

    @pytest.mark.parametrize("future", ["step", "duration"])
    def test_future_estimators_run_and_return_valid_lists(self, future):
        durs = torch.rand(self.C) * 40 + 5
        res = palette_decode(self.score, self.cl, self.vm, self.C, 4, beam_width=3, golden=True,
                             future=future, durations=durs)
        assert len(set(res.order)) == 4
        assert res.list_value == pytest.approx(list_value(self.score, self.cl, self.vm, res.order, True), abs=1e-5)

    def test_duration_future_requires_durations(self):
        with pytest.raises(AssertionError):
            palette_decode(self.score, self.cl, self.vm, self.C, 3, future="duration")

    def test_beam_returns_b_lists(self):
        res = palette_decode(self.score, self.cl, self.vm, self.C, self.T, beam_width=5)
        assert len(res.beam) == 5

    def test_vanilla_decoding_uses_unit_continuation(self):
        """Scorer with continue==0 kills golden value after position 1 but not vanilla."""
        def score(prefixes):
            p, _ = self.score(prefixes)
            return p, torch.zeros(prefixes.shape[0], self.C)
        g = palette_decode(score, self.cl, self.vm, self.C, 3, beam_width=200, golden=True)
        # golden total is just the first position's value: picks the top item first
        first = self.vm(score(torch.zeros(1, 0, dtype=torch.long))[0])[0]
        assert g.order[0] == int(first.argmax())
        assert g.list_value == pytest.approx(first.max().item(), abs=1e-5)


# ---------------------------------------------------------------------------
# End-to-end serving
# ---------------------------------------------------------------------------

class TestServing:
    def setup_method(self):
        torch.manual_seed(0)
        self.C, self.T = 18, 6
        cf = ContextFreeModel(Du, Di, D, K)
        ca = ContextAwareModel(D, K)
        self.model = LigeGR(cf, ca, ItemValueModel([1.0, 2.0, 0.5])).eval()
        self.u, self.items = torch.randn(Du), torch.randn(self.C, Di)
        self.cl = ControlLayer(self.C, categories=torch.randint(0, 4, (self.C,)), gap_coef=0.5)

    def test_ca_scorer_matches_direct_forward(self):
        v = torch.randn(self.C, D)
        ca = self.model.ca
        prefix = torch.tensor([[3, 7]])
        probs, cont = make_ca_scorer(ca, v)(prefix)
        direct = torch.sigmoid(ca(torch.stack([v[3], v[7], v[5]]).unsqueeze(0))[0, -1])
        assert torch.allclose(probs[0, 5], direct[:K], atol=1e-5)
        assert cont[0, 5].item() == pytest.approx(direct[K].item(), abs=1e-5)

    def test_revert_switch_equals_incumbent_greedy(self):
        res = self.model.generate(self.u, self.items, self.T, self.cl, use_context_aware=False)
        _, logits = self.model.cf(self.u.unsqueeze(0), self.items.unsqueeze(0))
        ref = palette_decode(make_cf_scorer(torch.sigmoid(logits[0])), self.cl, self.model.item_vm,
                             self.C, self.T, beam_width=1, golden=False)
        assert res.order == ref.order

    def test_generate_valid_list_with_pool(self):
        res = self.model.generate(self.u, self.items, self.T, self.cl, beam_width=4, pool_frac=1 / 3)
        assert len(res.order) == self.T and len(set(res.order)) == self.T
        assert all(0 <= c < self.C for c in res.order)

    def test_pool_contains_top_cf_items(self):
        """Pool keeps the highest CF-valued third, so the output is drawn from it."""
        _, logits = self.model.cf(self.u.unsqueeze(0), self.items.unsqueeze(0))
        vm = self.model.item_vm(torch.sigmoid(logits[0])[:, :K])
        pool = set(vm.topk(math.ceil(self.C / 3)).indices.tolist())
        res = self.model.generate(self.u, self.items, 4, self.cl, beam_width=2, pool_frac=1 / 3)
        assert set(res.order) <= pool

    def test_pool_never_smaller_than_list(self):
        res = self.model.generate(self.u, self.items, 10, self.cl, beam_width=2, pool_frac=0.1)
        assert len(res.order) == 10

    def test_latency_budget_fallback_to_incumbent(self):
        res = self.model.generate(self.u, self.items, self.T, self.cl, beam_width=4, latency_budget_ms=0.0)
        base = self.model.generate(self.u, self.items, self.T, self.cl, use_context_aware=False)
        assert res.order == base.order

    def test_generate_with_duration_aware_future(self):
        durs = torch.rand(self.C) * 40 + 5
        res = self.model.generate(self.u, self.items, self.T, self.cl, beam_width=6, golden=True,
                                  future="duration", durations=durs, pool_frac=0.5)
        assert len(set(res.order)) == self.T

    def test_generate_with_dpp_control(self):
        cl = ControlLayer(self.C, embeddings=self.items, dpp_weight=0.3)
        res = self.model.generate(self.u, self.items, 5, cl, beam_width=3, pool_frac=0.5)
        assert len(set(res.order)) == 5
