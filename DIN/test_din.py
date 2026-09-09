"""Tests for the DIN implementation.

Run with:
    pytest test_din.py -v
"""

import math

import pytest
import torch
import torch.nn.functional as F

from din import (
    Dice,
    sum_pooling,
    average_pooling,
    LocalActivationUnit,
    activation_weighted_pooling,
    mini_batch_aware_l2_penalty,
    MLPTower,
    DINModel,
    pairwise_auc,
    user_weighted_auc,
    rel_impr,
)

B, H, D = 6, 10, 8


# ---------------------------------------------------------------------------
# Dice
# ---------------------------------------------------------------------------

class TestDice:
    def test_output_shape(self):
        dice = Dice(D)
        x = torch.randn(B, H, D)
        assert dice(x).shape == (B, H, D)

    def test_degenerates_to_prelu_at_zero_stats(self):
        dice = Dice(D, alpha_init=0.3)
        dice.eval()
        dice.running_mean.zero_()
        dice.running_var.zero_()
        x = torch.randn(20, D)
        out = dice(x)
        expected = torch.where(x > 0, x, 0.3 * x)
        assert torch.allclose(out, expected, atol=1e-4)

    def test_training_updates_running_stats(self):
        dice = Dice(D)
        dice.train()
        before = dice.running_mean.clone()
        dice(torch.randn(B, D) * 5 + 3)
        assert not torch.allclose(before, dice.running_mean)

    def test_eval_does_not_update_running_stats(self):
        dice = Dice(D)
        dice.eval()
        before = dice.running_mean.clone()
        dice(torch.randn(B, D))
        assert torch.allclose(before, dice.running_mean)

    def test_gradient(self):
        dice = Dice(D)
        x = torch.randn(B, D, requires_grad=True)
        dice(x).sum().backward()
        assert x.grad is not None
        assert dice.alpha.grad is not None


# ---------------------------------------------------------------------------
# Pooling (Eq. 1)
# ---------------------------------------------------------------------------

class TestPooling:
    def test_sum_pooling_matches_manual(self):
        emb = torch.randn(2, 4, 3)
        mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
        out = sum_pooling(emb, mask)
        assert torch.allclose(out[0], emb[0, :2].sum(dim=0))
        assert torch.allclose(out[1], emb[1].sum(dim=0))

    def test_average_pooling_matches_manual(self):
        emb = torch.randn(1, 4, 3)
        mask = torch.tensor([[True, True, True, False]])
        out = average_pooling(emb, mask)
        assert torch.allclose(out[0], emb[0, :3].mean(dim=0), atol=1e-6)

    def test_average_pooling_handles_all_masked(self):
        emb = torch.randn(1, 4, 3)
        mask = torch.zeros(1, 4, dtype=torch.bool)
        out = average_pooling(emb, mask)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# LocalActivationUnit / activation_weighted_pooling (Eq. 3)
# ---------------------------------------------------------------------------

class TestLocalActivationUnit:
    def test_output_shape(self):
        unit = LocalActivationUnit(D)
        behaviors = torch.randn(B, H, D)
        ad = torch.randn(B, D)
        assert unit(behaviors, ad).shape == (B, H)

    def test_weights_not_normalized(self):
        unit = LocalActivationUnit(D)
        behaviors = torch.randn(B, H, D)
        ad = torch.randn(B, D)
        weights = unit(behaviors, ad)
        # Unnormalized: essentially never sums to exactly 1 for random weights.
        assert not torch.allclose(weights.sum(dim=1), torch.ones(B), atol=1e-3)

    def test_different_ads_give_different_weights(self):
        unit = LocalActivationUnit(D)
        behaviors = torch.randn(1, H, D)
        ad1, ad2 = torch.randn(1, D), torch.randn(1, D)
        assert not torch.allclose(unit(behaviors, ad1), unit(behaviors, ad2))

    def test_gradient(self):
        unit = LocalActivationUnit(D)
        behaviors = torch.randn(B, H, D, requires_grad=True)
        ad = torch.randn(B, D, requires_grad=True)
        unit(behaviors, ad).sum().backward()
        assert behaviors.grad is not None and ad.grad is not None


class TestActivationWeightedPooling:
    def test_matches_manual_weighted_sum(self):
        behaviors = torch.randn(2, 3, 4)
        weights = torch.randn(2, 3)
        mask = torch.ones(2, 3, dtype=torch.bool)
        out = activation_weighted_pooling(behaviors, weights, mask)
        expected = (behaviors * weights.unsqueeze(-1)).sum(dim=1)
        assert torch.allclose(out, expected)

    def test_masked_positions_excluded(self):
        behaviors = torch.randn(1, 3, 4)
        weights = torch.tensor([[1.0, 1.0, 100.0]])
        mask = torch.tensor([[True, True, False]])
        out = activation_weighted_pooling(behaviors, weights, mask)
        expected = behaviors[0, 0] * 1.0 + behaviors[0, 1] * 1.0
        assert torch.allclose(out[0], expected)


# ---------------------------------------------------------------------------
# mini_batch_aware_l2_penalty (Eq. 4-7)
# ---------------------------------------------------------------------------

class TestMiniBatchAwareL2Penalty:
    def test_matches_closed_form(self):
        weight = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
        counts = torch.tensor([2.0, 4.0, 5.0])
        batch_ids = torch.tensor([0, 2, 2])
        penalty = mini_batch_aware_l2_penalty(weight, batch_ids, counts, lam=1.0)
        expected = (1.0 / 2.0) + (9.0 + 16.0) / 5.0  # only rows 0 and 2 touched, each counted once
        assert penalty.item() == pytest.approx(expected)

    def test_untouched_rows_do_not_affect_penalty(self):
        weight = torch.tensor([[1.0], [1000.0]])
        counts = torch.tensor([1.0, 1.0])
        penalty = mini_batch_aware_l2_penalty(weight, torch.tensor([0]), counts, lam=1.0)
        assert penalty.item() == pytest.approx(1.0)

    def test_gradient_only_reaches_touched_rows(self):
        weight = torch.nn.Parameter(torch.tensor([[1.0], [2.0], [3.0]]))
        counts = torch.tensor([1.0, 1.0, 1.0])
        penalty = mini_batch_aware_l2_penalty(weight, torch.tensor([0, 0, 2]), counts, lam=1.0)
        penalty.backward()
        assert weight.grad[0].item() != 0.0
        assert weight.grad[1].item() == 0.0
        assert weight.grad[2].item() != 0.0

    def test_scales_with_lambda(self):
        weight = torch.tensor([[1.0, 1.0]])
        counts = torch.tensor([1.0])
        p1 = mini_batch_aware_l2_penalty(weight, torch.tensor([0]), counts, lam=1.0)
        p2 = mini_batch_aware_l2_penalty(weight, torch.tensor([0]), counts, lam=2.0)
        assert p2.item() == pytest.approx(2 * p1.item())


# ---------------------------------------------------------------------------
# MLPTower
# ---------------------------------------------------------------------------

class TestMLPTower:
    def test_output_shape(self):
        mlp = MLPTower(in_dim=10, hidden_dims=[16, 8])
        assert mlp(torch.randn(B, 10)).shape == (B,)

    def test_gradient(self):
        mlp = MLPTower(in_dim=10, hidden_dims=[16, 8])
        x = torch.randn(B, 10, requires_grad=True)
        mlp(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# DINModel
# ---------------------------------------------------------------------------

class TestDINModel:
    def _make(self, use_local_activation=True, activation="dice"):
        return DINModel(
            profile_cardinalities=[2, 5],
            behavior_vocab_size=100,
            context_cardinalities=[3],
            embed_dim=D,
            mlp_hidden_dims=[16, 8],
            use_local_activation=use_local_activation,
            activation=activation,
        )

    def _inputs(self, b=B, h=H):
        profile_ids = [torch.randint(0, 2, (b,)), torch.randint(0, 5, (b,))]
        behavior_ids = torch.randint(1, 100, (b, h))
        lengths = torch.randint(3, h + 1, (b,))
        mask = torch.arange(h).unsqueeze(0) < lengths.unsqueeze(1)
        ad_id = torch.randint(0, 100, (b,))
        context_ids = [torch.randint(0, 3, (b,))]
        return profile_ids, behavior_ids, mask, ad_id, context_ids

    def test_output_shapes(self):
        model = self._make()
        logits, weights = model(*self._inputs())
        assert logits.shape == (B,)
        assert weights.shape == (B, H)

    def test_base_model_weights_sum_to_one(self):
        model = self._make(use_local_activation=False, activation="prelu")
        _, weights = model(*self._inputs())
        assert torch.allclose(weights.sum(dim=1), torch.ones(B), atol=1e-5)

    def test_din_representation_varies_with_candidate_ad(self):
        # The paper's central claim: DIN's pooled user representation changes
        # per candidate ad, unlike the base model's fixed representation.
        model = self._make(use_local_activation=True)
        model.eval()
        profile_ids, behavior_ids, mask, _, context_ids = self._inputs()
        ad_a = torch.zeros(B, dtype=torch.long) + 1
        ad_b = torch.zeros(B, dtype=torch.long) + 50
        with torch.no_grad():
            _, w_a = model(profile_ids, behavior_ids, mask, ad_a, context_ids)
            _, w_b = model(profile_ids, behavior_ids, mask, ad_b, context_ids)
        assert not torch.allclose(w_a, w_b)

    def test_base_model_representation_is_ad_independent(self):
        model = self._make(use_local_activation=False, activation="prelu")
        model.eval()
        profile_ids, behavior_ids, mask, _, context_ids = self._inputs()
        ad_a = torch.zeros(B, dtype=torch.long) + 1
        ad_b = torch.zeros(B, dtype=torch.long) + 50
        with torch.no_grad():
            _, w_a = model(profile_ids, behavior_ids, mask, ad_a, context_ids)
            _, w_b = model(profile_ids, behavior_ids, mask, ad_b, context_ids)
        assert torch.allclose(w_a, w_b)  # pooling weights don't depend on the ad at all

    def test_padded_positions_do_not_affect_logits(self):
        model = self._make(use_local_activation=True)
        model.eval()
        profile_ids, behavior_ids, mask, ad_id, context_ids = self._inputs()
        perturbed = behavior_ids.clone()
        # Perturb only positions beyond each row's valid length.
        for i in range(B):
            valid_len = mask[i].sum().item()
            if valid_len < H:
                perturbed[i, valid_len:] = (perturbed[i, valid_len:] + 1) % 99 + 1
        with torch.no_grad():
            logits1, _ = model(profile_ids, behavior_ids, mask, ad_id, context_ids)
            logits2, _ = model(profile_ids, perturbed, mask, ad_id, context_ids)
        assert torch.allclose(logits1, logits2, atol=1e-5)

    def test_gradient(self):
        model = self._make()
        logits, _ = model(*self._inputs())
        labels = torch.randint(0, 2, (B,)).float()
        F.binary_cross_entropy_with_logits(logits, labels).backward()
        assert model.goods_embedding.weight.grad is not None

    def test_finite_output(self):
        model = self._make()
        logits, _ = model(*self._inputs())
        assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Evaluation: AUC / RelaImpr (Eq. 10-11)
# ---------------------------------------------------------------------------

class TestPairwiseAUC:
    def test_perfect_separation_is_one(self):
        scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
        labels = torch.tensor([1, 1, 0, 0])
        assert pairwise_auc(scores, labels) == pytest.approx(1.0)

    def test_worst_separation_is_zero(self):
        scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
        labels = torch.tensor([1, 1, 0, 0])
        assert pairwise_auc(scores, labels) == pytest.approx(0.0)

    def test_ties_give_half_credit(self):
        scores = torch.tensor([0.5, 0.5])
        labels = torch.tensor([1, 0])
        assert pairwise_auc(scores, labels) == pytest.approx(0.5)

    def test_missing_class_is_nan(self):
        assert math.isnan(pairwise_auc(torch.tensor([0.1, 0.2]), torch.tensor([1, 1])))


class TestUserWeightedAUC:
    def test_weights_by_impression_count(self):
        # user A: 1 impression pair (perfect, AUC=1.0); user B: 3 impressions (perfect too, AUC=1.0)
        user_ids = ["A", "A", "B", "B", "B", "B"]
        scores = torch.tensor([0.9, 0.1, 0.9, 0.8, 0.2, 0.1])
        labels = torch.tensor([1, 0, 1, 1, 0, 0])
        auc = user_weighted_auc(user_ids, scores, labels)
        assert auc == pytest.approx(1.0)

    def test_skips_users_with_single_class(self):
        user_ids = ["A", "A", "B", "B"]
        scores = torch.tensor([0.9, 0.1, 0.5, 0.5])
        labels = torch.tensor([1, 0, 1, 1])  # user B has no negatives -> undefined AUC, skipped
        auc = user_weighted_auc(user_ids, scores, labels)
        assert auc == pytest.approx(1.0)  # only user A contributes

    def test_all_undefined_gives_nan(self):
        user_ids = ["A"]
        scores = torch.tensor([0.5])
        labels = torch.tensor([1])
        assert math.isnan(user_weighted_auc(user_ids, scores, labels))


class TestRelImpr:
    def test_zero_when_equal_to_base(self):
        assert rel_impr(0.7, 0.7) == pytest.approx(0.0)

    def test_positive_when_better(self):
        assert rel_impr(0.8, 0.7) > 0.0

    def test_negative_when_worse(self):
        assert rel_impr(0.6, 0.7) < 0.0

    def test_matches_known_value(self):
        # From the paper's own numbers (Table 3, DIN vs BaseModel on Amazon): AUC 0.7337 vs 0.7300 -> RelaImpr 1.61%
        assert rel_impr(0.7337, 0.7300) * 100 == pytest.approx(1.61, abs=0.05)
