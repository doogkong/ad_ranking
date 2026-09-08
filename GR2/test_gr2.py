"""Tests for the GR2 implementation.

Run with:
    pytest test_gr2.py -v
"""

import pytest
import torch

from gr2 import (
    RQKMeansTokenizer,
    build_targeted_prompt,
    build_rejection_prompt,
    rejection_sample,
    sft_loss,
    opd_loss,
    auc_reward,
    ndcg_reward,
    conditional_reward,
    group_relative_advantage,
    dynamic_sampling_filter,
    dapo_loss,
    compressor_reward,
)


# ---------------------------------------------------------------------------
# RQKMeansTokenizer
# ---------------------------------------------------------------------------

class TestRQKMeansTokenizer:
    def test_encode_shape(self):
        tok = RQKMeansTokenizer(num_levels=3, codebook_size=8, dim=6, n_iter=5)
        catalog = torch.randn(80, 6)
        tok.fit(catalog)
        codes = tok.encode(catalog)
        assert codes.shape == (80, 3)

    def test_encode_requires_fit(self):
        tok = RQKMeansTokenizer(num_levels=2, codebook_size=4, dim=6)
        with pytest.raises(RuntimeError):
            tok.encode(torch.randn(10, 6))

    def test_codes_within_codebook_range(self):
        tok = RQKMeansTokenizer(num_levels=2, codebook_size=5, dim=4, n_iter=5)
        catalog = torch.randn(40, 4)
        tok.fit(catalog)
        codes = tok.encode(catalog)
        assert codes.min() >= 0
        assert codes.max() < 5

    def test_uniqueness_bounds(self):
        tok = RQKMeansTokenizer(num_levels=3, codebook_size=16, dim=8, n_iter=10)
        catalog = torch.randn(200, 8)
        tok.fit(catalog)
        codes = tok.encode(catalog)
        u = tok.uniqueness(codes)
        assert 0.0 <= u <= 1.0

    def test_uniqueness_all_unique(self):
        codes = torch.arange(10).unsqueeze(1)
        assert RQKMeansTokenizer.uniqueness(codes) == 1.0

    def test_uniqueness_all_duplicate(self):
        codes = torch.zeros(10, 2, dtype=torch.long)
        assert RQKMeansTokenizer.uniqueness(codes) == pytest.approx(1 / 10)

    def test_residual_reduces_quantization_error(self):
        # More levels should quantize the catalog at least as well (lower
        # total residual energy after the final level).
        catalog = torch.randn(100, 8)

        def total_residual(num_levels):
            tok = RQKMeansTokenizer(num_levels=num_levels, codebook_size=8, dim=8, n_iter=15, seed=1)
            tok.fit(catalog)
            residual = catalog.clone()
            for centroids in tok.codebooks:
                codes = tok._nearest(residual, centroids)
                residual = residual - centroids[codes]
            return residual.pow(2).sum().item()

        assert total_residual(3) <= total_residual(1) + 1e-4


# ---------------------------------------------------------------------------
# Prompt builders + rejection sampling
# ---------------------------------------------------------------------------

class TestPromptsAndRejectionSampling:
    def test_targeted_prompt_reveals_target(self):
        p = build_targeted_prompt(["a", "b"], ["c", "d"], "c")
        assert p["target"] == "c"
        assert p["history"] == ["a", "b"]

    def test_rejection_prompt_has_no_target_field(self):
        p = build_rejection_prompt(["a", "b"], ["c", "d"])
        assert "target" not in p

    def test_rejection_sample_accepts_on_match(self):
        def teacher(prompt):
            return "trace", 2

        trace, idx, accepted = rejection_sample(["h"], ["c0", "c1", "c2"], target_index=2, teacher_fn=teacher)
        assert accepted
        assert idx == 2

    def test_rejection_sample_rejects_after_max_attempts(self):
        def teacher(prompt):
            return "trace", 0

        trace, idx, accepted = rejection_sample(["h"], ["c0", "c1"], target_index=1, teacher_fn=teacher, max_attempts=3)
        assert not accepted

    def test_rejection_sample_respects_max_attempts_count(self):
        calls = {"n": 0}

        def teacher(prompt):
            calls["n"] += 1
            return "trace", 0

        rejection_sample(["h"], ["c0", "c1"], target_index=1, teacher_fn=teacher, max_attempts=5)
        assert calls["n"] == 5


# ---------------------------------------------------------------------------
# SFT / OPD losses
# ---------------------------------------------------------------------------

class TestSFTLoss:
    def test_scalar_output(self):
        lp = torch.randn(4, 6)
        mask = torch.ones(4, 6, dtype=torch.bool)
        loss = sft_loss(lp, lp, mask, mask, lambda_r=0.5, lambda_o=1.0)
        assert loss.dim() == 0

    def test_masking_excludes_padding(self):
        lp = torch.zeros(1, 4)
        lp[0, 2:] = 10.0  # large log-prob in the padded (masked-out) region
        mask = torch.tensor([[True, True, False, False]])
        loss = sft_loss(lp, lp, mask, mask, lambda_r=1.0, lambda_o=0.0)
        assert loss.item() == pytest.approx(0.0)

    def test_gradient(self):
        lp = torch.randn(2, 5, requires_grad=True)
        mask = torch.ones(2, 5, dtype=torch.bool)
        sft_loss(lp, lp, mask, mask, lambda_r=0.5, lambda_o=1.0).backward()
        assert lp.grad is not None

    def test_lambda_weighting(self):
        r_lp = torch.ones(1, 4)
        o_lp = torch.ones(1, 4)
        mask = torch.ones(1, 4, dtype=torch.bool)
        loss_hi_r = sft_loss(r_lp, o_lp, mask, mask, lambda_r=10.0, lambda_o=0.0)
        loss_hi_o = sft_loss(r_lp, o_lp, mask, mask, lambda_r=0.0, lambda_o=10.0)
        assert torch.allclose(loss_hi_r, loss_hi_o)  # symmetric since r_lp == o_lp here
        assert loss_hi_r.item() < 0  # -lambda * sum(positive logprobs)


class TestOPDLoss:
    def _inputs(self, G=4, T=6):
        logp_student = torch.randn(G, T, requires_grad=True)
        logp_student_old = logp_student.detach().clone()
        logp_teacher = torch.randn(G, T)
        advantages = torch.randn(G)
        mask = torch.ones(G, T, dtype=torch.bool)
        return logp_student, logp_student_old, logp_teacher, advantages, mask

    def test_scalar_output(self):
        loss = opd_loss(*self._inputs())
        assert loss.dim() == 0

    def test_gradient(self):
        logp_student, logp_student_old, logp_teacher, advantages, mask = self._inputs()
        loss = opd_loss(logp_student, logp_student_old, logp_teacher, advantages, mask)
        loss.backward()
        assert logp_student.grad is not None

    def test_kl_term_zero_when_student_equals_teacher(self):
        # If logp_student == logp_teacher, logratio=0 -> kl_hat = exp(0)-0-1 = 0.
        G, T = 3, 5
        logp = torch.randn(G, T)
        mask = torch.ones(G, T, dtype=torch.bool)
        advantages = torch.zeros(G)  # zero out the policy-gradient term
        loss = opd_loss(logp, logp.clone(), logp.clone(), advantages, mask, beta=1.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_kl_term_nonnegative(self):
        G, T = 4, 8
        logp_student = torch.randn(G, T)
        logp_teacher = torch.randn(G, T)
        mask = torch.ones(G, T, dtype=torch.bool)
        advantages = torch.zeros(G)
        loss = opd_loss(logp_student, logp_student.clone(), logp_teacher, advantages, mask, beta=1.0)
        assert loss.item() >= -1e-6  # pure KL term, must be ~non-negative


# ---------------------------------------------------------------------------
# Ranking rewards
# ---------------------------------------------------------------------------

class TestAUCReward:
    def test_perfect_ranking(self):
        ranks = torch.tensor([1, 2, 3, 4])
        labels = torch.tensor([1, 1, 0, 0])
        assert auc_reward(ranks, labels) == pytest.approx(1.0)

    def test_worst_ranking(self):
        ranks = torch.tensor([1, 2, 3, 4])
        labels = torch.tensor([0, 0, 1, 1])
        assert auc_reward(ranks, labels) == pytest.approx(0.0)

    def test_partial_ranking(self):
        ranks = torch.tensor([1, 2, 3, 4])
        labels = torch.tensor([0, 1, 1, 0])
        # pos={2,3}(ranks), neg={1,4}: pairs (2<1)=F,(2<4)=T,(3<1)=F,(3<4)=T -> 2/4
        assert auc_reward(ranks, labels) == pytest.approx(0.5)

    def test_raises_without_both_classes(self):
        ranks = torch.tensor([1, 2, 3])
        labels = torch.tensor([1, 1, 1])
        with pytest.raises(ValueError):
            auc_reward(ranks, labels)

    def test_permutation_invariant_within_class(self):
        ranks_a = torch.tensor([1, 2, 3, 4])
        ranks_b = torch.tensor([2, 1, 4, 3])  # swap within each label class
        labels = torch.tensor([1, 1, 0, 0])
        assert auc_reward(ranks_a, labels) == pytest.approx(auc_reward(ranks_b, labels))


class TestNDCGReward:
    def test_perfect_ranking_is_one(self):
        ranks = torch.tensor([1, 2, 3])
        grades = torch.tensor([2, 1, 0])
        assert ndcg_reward(ranks, grades) == pytest.approx(1.0)

    def test_reversed_ranking_is_below_one(self):
        ranks = torch.tensor([3, 2, 1])
        grades = torch.tensor([2, 1, 0])
        assert ndcg_reward(ranks, grades) < 1.0

    def test_all_zero_grades_returns_zero(self):
        ranks = torch.tensor([1, 2, 3])
        grades = torch.tensor([0, 0, 0])
        assert ndcg_reward(ranks, grades) == 0.0

    def test_bounded_in_unit_interval(self):
        ranks = torch.tensor([2, 3, 1, 4])
        grades = torch.tensor([1, 2, 0, 1])
        r = ndcg_reward(ranks, grades)
        assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# Conditional (de-hacked) reward
# ---------------------------------------------------------------------------

class TestConditionalReward:
    def test_invalid_format_returns_format_term_only(self):
        ranks = torch.tensor([1, 2, 3])
        labels = torch.tensor([1, 0, 0])
        r = conditional_reward(ranks, labels, r_fmt=1.0, alpha=0.3, valid_format=False)
        assert r == pytest.approx(0.3)

    def test_identity_with_suboptimal_order_zeroes_rank_reward(self):
        # Identity order is [1,0,0]-labeled with positive NOT at rank 1 -> suboptimal.
        ranks = torch.tensor([1, 2, 3])
        labels = torch.tensor([0, 1, 0])
        r = conditional_reward(ranks, labels, r_fmt=1.0, alpha=0.2, valid_format=True)
        assert r == pytest.approx(0.2)

    def test_identity_with_optimal_order_keeps_rank_reward(self):
        ranks = torch.tensor([1, 2, 3])
        labels = torch.tensor([1, 0, 0])  # identity already puts the positive first -> optimal
        r = conditional_reward(ranks, labels, r_fmt=1.0, alpha=0.2, valid_format=True)
        assert r == pytest.approx(auc_reward(ranks, labels) + 0.2)

    def test_non_identity_valid_format_uses_full_reward(self):
        ranks = torch.tensor([2, 1, 3])
        labels = torch.tensor([1, 0, 0])
        r = conditional_reward(ranks, labels, r_fmt=1.0, alpha=0.5, valid_format=True)
        assert r == pytest.approx(auc_reward(ranks, labels) + 0.5)


# ---------------------------------------------------------------------------
# DAPO
# ---------------------------------------------------------------------------

class TestGroupRelativeAdvantage:
    def test_zero_mean_unit_std(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        adv = group_relative_advantage(rewards)
        assert adv.mean().item() == pytest.approx(0.0, abs=1e-6)
        assert adv.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-5)

    def test_zero_variance_group_returns_zeros(self):
        rewards = torch.tensor([0.5, 0.5, 0.5])
        adv = group_relative_advantage(rewards)
        assert torch.all(adv == 0.0)


class TestDynamicSamplingFilter:
    def test_drops_all_correct_and_all_incorrect(self):
        rewards = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
        keep = dynamic_sampling_filter(rewards)
        assert keep.tolist() == [False, False, True]

    def test_keeps_mixed_groups(self):
        rewards = torch.tensor([[1.0, 0.0], [0.3, 0.7]])
        keep = dynamic_sampling_filter(rewards)
        assert keep.all()


class TestDAPOLoss:
    def test_scalar_output(self):
        ratios = torch.ones(3, 5)
        advantages = torch.tensor([1.0, -1.0, 0.5])
        mask = torch.ones(3, 5, dtype=torch.bool)
        loss = dapo_loss(ratios, advantages, mask)
        assert loss.dim() == 0

    def test_matches_unclipped_when_ratio_is_one(self):
        ratios = torch.ones(2, 4)
        advantages = torch.tensor([2.0, -3.0])
        mask = torch.ones(2, 4, dtype=torch.bool)
        loss = dapo_loss(ratios, advantages, mask)
        expected = -(advantages.unsqueeze(-1).expand(2, 4)).sum() / 8
        assert loss.item() == pytest.approx(expected.item())

    def test_padding_excluded_via_mask(self):
        ratios = torch.ones(1, 4)
        advantages = torch.tensor([1.0])
        mask = torch.tensor([[True, True, False, False]])
        loss = dapo_loss(ratios, advantages, mask)
        assert loss.item() == pytest.approx(-1.0)

    def test_clipping_bounds_large_positive_ratio(self):
        ratios = torch.tensor([[10.0]])
        advantages = torch.tensor([1.0])
        mask = torch.tensor([[True]])
        loss = dapo_loss(ratios, advantages, mask, eps_lo=0.2, eps_hi=0.28)
        assert loss.item() == pytest.approx(-1.28)


# ---------------------------------------------------------------------------
# Context-compressor reward
# ---------------------------------------------------------------------------

class TestCompressorReward:
    def test_solvable_weights_ranking_quality_more(self):
        r_high_rank = compressor_reward(True, info_preservation=1, ranking_quality=10, original_len=100, compressed_len=50)
        r_high_info = compressor_reward(True, info_preservation=10, ranking_quality=1, original_len=100, compressed_len=50)
        assert r_high_rank > r_high_info

    def test_unsolvable_weights_info_preservation_more(self):
        r_high_rank = compressor_reward(False, info_preservation=1, ranking_quality=10, original_len=100, compressed_len=50)
        r_high_info = compressor_reward(False, info_preservation=10, ranking_quality=1, original_len=100, compressed_len=50)
        assert r_high_info > r_high_rank

    def test_more_compression_increases_comp_term(self):
        r_small_comp = compressor_reward(True, 5, 5, original_len=100, compressed_len=90)
        r_big_comp = compressor_reward(True, 5, 5, original_len=100, compressed_len=10)
        assert r_big_comp > r_small_comp

    def test_ellipsis_penalty_scales_reward(self):
        full = compressor_reward(True, 5, 5, 100, 50, ellipsis_penalty=1.0)
        penalized = compressor_reward(True, 5, 5, 100, 50, ellipsis_penalty=0.5)
        assert penalized == pytest.approx(full * 0.5)

    def test_no_compression_gives_zero_comp_term(self):
        r = compressor_reward(True, 10, 10, original_len=100, compressed_len=100, alpha=1.0)
        assert r == pytest.approx(0.0)
