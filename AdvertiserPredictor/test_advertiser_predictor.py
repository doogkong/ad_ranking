"""Tests for the AdvertiserPredictor implementation.

Run with:
    pytest test_advertiser_predictor.py -v
"""

from datetime import date

import pytest
import torch
import torch.nn as nn

from advertiser_predictor import (
    select_active_users,
    construct_next_advertiser_label,
    select_users_for_incremental_inference,
    build_advertiser_prediction_prompt,
    parse_structured_output,
    rank_match_reward,
    length_penalty,
    grpo_reward,
    freeze_all_but_sid_embeddings,
    unfreeze_all,
    blend_candidates,
    advertiser_concentration,
    rank_of_advertiser,
    recall_at_k,
    mean_recall_at_k,
)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

class TestSelectActiveUsers:
    def test_filters_by_lookback_window(self):
        users = [
            {"id": "u1", "last_conversion_date": date(2026, 4, 1)},
            {"id": "u2", "last_conversion_date": date(2026, 1, 1)},
        ]
        active = select_active_users(users, as_of_date=date(2026, 5, 1), lookback_days=90)
        assert [u["id"] for u in active] == ["u1"]

    def test_boundary_is_inclusive(self):
        users = [{"id": "u1", "last_conversion_date": date(2026, 2, 1)}]
        active = select_active_users(users, as_of_date=date(2026, 5, 2), lookback_days=90)
        assert len(active) == 1

    def test_empty_users(self):
        assert select_active_users([], as_of_date=date(2026, 5, 1)) == []


class TestConstructNextAdvertiserLabel:
    def test_returns_first_conversion_in_window(self):
        conversions = [(date(2026, 5, 3), "AdvB"), (date(2026, 5, 2), "AdvA")]
        label = construct_next_advertiser_label(conversions, date(2026, 5, 2), date(2026, 5, 8))
        assert label == "AdvA"

    def test_excludes_outside_window(self):
        conversions = [(date(2026, 5, 1), "AdvA"), (date(2026, 5, 20), "AdvB")]
        label = construct_next_advertiser_label(conversions, date(2026, 5, 2), date(2026, 5, 8))
        assert label is None

    def test_no_conversions_returns_none(self):
        assert construct_next_advertiser_label([], date(2026, 5, 2), date(2026, 5, 8)) is None


class TestSelectUsersForIncrementalInference:
    def test_intersection_only(self):
        result = select_users_for_incremental_inference(["u1", "u2", "u3"], ["u2", "u4"])
        assert result == ["u2"]

    def test_preserves_eligible_order(self):
        result = select_users_for_incremental_inference(["u3", "u1", "u2"], ["u1", "u2"])
        assert result == ["u1", "u2"]

    def test_no_overlap(self):
        assert select_users_for_incremental_inference(["u1"], ["u2"]) == []


# ---------------------------------------------------------------------------
# Prompt design + output parsing
# ---------------------------------------------------------------------------

class TestBuildAdvertiserPredictionPrompt:
    def _base_kwargs(self):
        return dict(
            profile={"age": 30, "gender": "F", "state": "NY"},
            behavior={"onsite_searches": ["shoes"]},
            active_advertisers=["Nike"],
            preset_advertiser_pool=["Amazon"],
        )

    def test_sft_requests_single_advertiser(self):
        prompt = build_advertiser_prediction_prompt(**self._base_kwargs(), stage="sft")
        assert "exactly one advertiser" in prompt

    def test_grpo_requests_structured_block(self):
        prompt = build_advertiser_prediction_prompt(**self._base_kwargs(), stage="grpo", num_advertisers=20, num_interests=5)
        assert "EXACTLY 20 ranked advertiser names" in prompt
        assert "<advertiser_names>" in prompt
        assert "<interests>" in prompt

    def test_zero_interests_omits_interest_tag(self):
        prompt = build_advertiser_prediction_prompt(**self._base_kwargs(), stage="grpo", num_advertisers=5, num_interests=0)
        assert "<interests>" not in prompt

    def test_includes_profile_and_behavior_fields(self):
        prompt = build_advertiser_prediction_prompt(**self._base_kwargs(), stage="sft")
        assert "age=30" in prompt
        assert "shoes" in prompt
        assert "Nike" in prompt
        assert "Amazon" in prompt

    def test_includes_sid_sequences_when_given(self):
        prompt = build_advertiser_prediction_prompt(**self._base_kwargs(), stage="grpo", sid_sequences=[[1, 2, 3]])
        assert "Semantic ID" in prompt


class TestParseStructuredOutput:
    def test_parses_both_fields(self):
        text = "<answer><interests>[a|b]</interests><advertiser_names>X|Y|Z</advertiser_names></answer>"
        parsed = parse_structured_output(text)
        assert parsed["interests"] == ["a", "b"]
        assert parsed["advertisers"] == ["X", "Y", "Z"]

    def test_missing_tag_gives_empty_list(self):
        text = "<answer><advertiser_names>X|Y</advertiser_names></answer>"
        parsed = parse_structured_output(text)
        assert parsed["interests"] == []
        assert parsed["advertisers"] == ["X", "Y"]

    def test_strips_whitespace_and_brackets(self):
        text = "<advertiser_names> X | Y </advertiser_names>"
        assert parse_structured_output(text)["advertisers"] == ["X", "Y"]

    def test_no_match_returns_empty(self):
        assert parse_structured_output("no tags here") == {"advertisers": [], "interests": []}


# ---------------------------------------------------------------------------
# GRPO reward (Eq. 1-5)
# ---------------------------------------------------------------------------

class TestRankMatchReward:
    def test_rank_one_gets_base_plus_bonus(self):
        assert rank_match_reward(1) == pytest.approx(0.1 * 19 + 2.0)

    def test_rank_four_still_gets_bonus(self):
        assert rank_match_reward(4) == pytest.approx(0.1 * 16 + 2.0)

    def test_rank_five_loses_bonus(self):
        assert rank_match_reward(5) == pytest.approx(0.1 * 15)

    def test_rank_twenty_is_zero_bonus_and_low_base(self):
        assert rank_match_reward(20) == pytest.approx(0.0)

    def test_absent_advertiser_is_zero(self):
        assert rank_match_reward(None) == 0.0


class TestLengthPenalty:
    def test_zero_at_target(self):
        assert length_penalty(20, 20) == 0.0

    def test_small_deviation(self):
        assert length_penalty(18, 20) == pytest.approx(0.1 * 2 + 1.0)

    def test_capped_at_large_deviation(self):
        assert length_penalty(0, 20) == pytest.approx(1.0 + 1.0)  # capped 0.1*20=2.0 -> min(2.0,1.0)=1.0


class TestGrpoReward:
    def test_matches_eq1_composition(self):
        reward = grpo_reward(rank=2, num_advertisers_predicted=20, num_interests_predicted=5)
        expected = rank_match_reward(2) - length_penalty(20, 20) - length_penalty(5, 5)
        assert reward == pytest.approx(expected)

    def test_penalizes_wrong_count(self):
        good = grpo_reward(rank=1, num_advertisers_predicted=20, num_interests_predicted=5)
        bad = grpo_reward(rank=1, num_advertisers_predicted=10, num_interests_predicted=5)
        assert good > bad

    def test_absent_ground_truth_is_negative_or_zero(self):
        reward = grpo_reward(rank=None, num_advertisers_predicted=20, num_interests_predicted=5)
        assert reward <= 0.0


# ---------------------------------------------------------------------------
# Semantic-ID two-phase pretraining
# ---------------------------------------------------------------------------

class TestSidTwoPhasePretraining:
    def _toy_model(self, vocab_size=20, d_model=8):
        embedding = nn.Embedding(vocab_size, d_model)
        model = nn.Sequential(embedding, nn.Linear(d_model, d_model))
        return model, embedding

    def test_phase1_freezes_non_embedding_params(self):
        model, embedding = self._toy_model()
        freeze_all_but_sid_embeddings(model, embedding, sid_token_ids=[15, 16, 17])
        assert not model[1].weight.requires_grad
        assert embedding.weight.requires_grad

    def test_phase1_masks_gradient_for_non_sid_rows(self):
        model, embedding = self._toy_model(vocab_size=20, d_model=8)
        freeze_all_but_sid_embeddings(model, embedding, sid_token_ids=[15, 16, 17])
        x = torch.tensor([0, 1, 15, 16])  # mix of non-SID and SID rows
        model(x).sum().backward()
        assert embedding.weight.grad[0].abs().sum().item() == 0.0
        assert embedding.weight.grad[15].abs().sum().item() > 0.0

    def test_unfreeze_all_restores_grad(self):
        model, embedding = self._toy_model()
        freeze_all_but_sid_embeddings(model, embedding, sid_token_ids=[15])
        unfreeze_all(model)
        assert all(p.requires_grad for p in model.parameters())

    def test_removing_hook_stops_masking(self):
        model, embedding = self._toy_model(vocab_size=20, d_model=8)
        hook = freeze_all_but_sid_embeddings(model, embedding, sid_token_ids=[15])
        hook.remove()
        unfreeze_all(model)
        x = torch.tensor([0, 1, 15])
        model(x).sum().backward()
        assert embedding.weight.grad[0].abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# Candidate blending
# ---------------------------------------------------------------------------

class TestBlendCandidates:
    def test_appends_new_llm_candidates(self):
        blended = blend_candidates(["A", "B"], ["C", "D"], llm_quota=2)
        assert blended == ["A", "B", "C", "D"]

    def test_dedupes_against_main(self):
        blended = blend_candidates(["A", "B"], ["B", "C"], llm_quota=2)
        assert blended == ["A", "B", "C"]

    def test_respects_quota(self):
        blended = blend_candidates(["A"], ["B", "C", "D"], llm_quota=1)
        assert blended == ["A", "B"]

    def test_zero_quota_adds_nothing(self):
        blended = blend_candidates(["A"], ["B", "C"], llm_quota=0)
        assert blended == ["A"]


class TestAdvertiserConcentration:
    def test_uniform_list_is_low_concentration(self):
        assert advertiser_concentration(["A", "B", "C", "D"]) == pytest.approx(0.25)

    def test_single_dominant_advertiser(self):
        assert advertiser_concentration(["A", "A", "A", "B"]) == pytest.approx(0.75)

    def test_empty_is_zero(self):
        assert advertiser_concentration([]) == 0.0


# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------

class TestRankOfAdvertiser:
    def test_found(self):
        assert rank_of_advertiser(["A", "B", "C"], "B") == 2

    def test_not_found(self):
        assert rank_of_advertiser(["A", "B"], "Z") is None


class TestRecallAtK:
    def test_hit(self):
        assert recall_at_k(["A", "B", "C"], "B", k=2) == 1.0

    def test_miss_outside_k(self):
        assert recall_at_k(["A", "B", "C"], "C", k=2) == 0.0


class TestMeanRecallAtK:
    def test_mean_over_pairs(self):
        pairs = [(["A", "B"], "A"), (["C", "D"], "Z")]
        assert mean_recall_at_k(pairs, k=2) == pytest.approx(0.5)

    def test_empty_pairs_is_zero(self):
        assert mean_recall_at_k([], k=5) == 0.0
