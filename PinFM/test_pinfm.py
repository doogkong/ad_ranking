"""Tests for PinFM implementation.

Run with:
    pytest test_pinfm.py -v
"""

import pytest
import torch

from pinfm import (
    UserActivityEmbedding,
    TargetItemEncoder,
    OutputProjection,
    CausalSelfAttentionLayer,
    CausalTransformer,
    info_nce_loss,
    next_token_loss,
    multi_token_loss,
    future_token_loss,
    deduplicate_sequences,
    compute_savings,
    PinFMDCAT,
    late_fusion_representation,
    candidate_id_randomization,
    item_age_dependent_dropout,
    quantize_dequantize_embedding,
    quantization_error,
)

VOCAB = 200
NUM_ACTIONS = 5
NUM_SURFACES = 3
D = 32
H = 4
L = 2
B = 3
M = 16


# ---------------------------------------------------------------------------
# UserActivityEmbedding / TargetItemEncoder / OutputProjection
# ---------------------------------------------------------------------------

class TestUserActivityEmbedding:
    def _make(self):
        return UserActivityEmbedding(VOCAB, NUM_ACTIONS, NUM_SURFACES, D)

    def test_output_shape_full(self):
        emb = self._make()
        item = torch.randint(0, VOCAB, (B, M))
        action = torch.randint(0, NUM_ACTIONS, (B, M))
        surface = torch.randint(0, NUM_SURFACES, (B, M))
        out = emb(item, action, surface)
        assert out.shape == (B, M, D)

    def test_output_shape_item_only(self):
        # Candidate items have no action/surface embedding.
        emb = self._make()
        item = torch.randint(0, VOCAB, (B, 1))
        out = emb(item)
        assert out.shape == (B, 1, D)

    def test_output_is_l2_normalized(self):
        emb = self._make()
        item = torch.randint(0, VOCAB, (B, M))
        out = emb(item)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_action_changes_output(self):
        emb = self._make()
        item = torch.randint(0, VOCAB, (1, 1))
        out_no_action = emb(item)
        out_with_action = emb(item, torch.zeros(1, 1, dtype=torch.long))
        assert not torch.allclose(out_no_action, out_with_action)

    def test_gradient(self):
        emb = self._make()
        item = torch.randint(0, VOCAB, (B, M))
        out = emb(item)
        out.sum().backward()
        assert emb.item_emb.weight.grad is not None


class TestTargetItemEncoderAndOutputProjection:
    def test_target_encoder_l2_normalized(self):
        enc = TargetItemEncoder(D)
        out = enc(torch.randn(B, M, D))
        assert torch.allclose(out.norm(dim=-1), torch.ones(B, M), atol=1e-5)

    def test_output_projection_l2_normalized(self):
        proj = OutputProjection(D)
        out = proj(torch.randn(B, M, D))
        assert torch.allclose(out.norm(dim=-1), torch.ones(B, M), atol=1e-5)


# ---------------------------------------------------------------------------
# CausalSelfAttentionLayer / CausalTransformer
# ---------------------------------------------------------------------------

class TestCausalTransformer:
    def _make(self, n_layers=2):
        return CausalTransformer(D, n_layers=n_layers, n_heads=H)

    def test_output_shape(self):
        model = self._make()
        x = torch.randn(B, M, D)
        out = model(x)
        assert out.shape == (B, M, D)

    def test_kv_cache_shapes(self):
        model = self._make(n_layers=3)
        x = torch.randn(B, M, D)
        out, cache = model(x, return_kv_cache=True)
        assert out.shape == (B, M, D)
        assert len(cache) == 3
        head_dim = D // H
        for K, V in cache:
            assert K.shape == (B, H, M, head_dim)
            assert V.shape == (B, H, M, head_dim)

    def test_causal_masking(self):
        model = self._make()
        model.eval()
        x1 = torch.randn(1, M, D)
        x2 = x1.clone()
        x2[:, -1, :] = torch.randn(D)
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
        assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)
        assert not torch.allclose(out1[:, -1, :], out2[:, -1, :])

    def test_gradient(self):
        model = self._make()
        x = torch.randn(B, M, D, requires_grad=True)
        model(x).sum().backward()
        assert x.grad is not None

    def test_forward_cross_matches_full_forward(self):
        # Cross-attending a candidate against a cached context must equal a
        # full forward pass over [context, candidate], read off at the last
        # position (both see the same information: full context + self).
        model = self._make(n_layers=2)
        model.eval()
        n_ctx = M - 1
        x = torch.randn(1, M, D)
        with torch.no_grad():
            full_out = model(x)
        naive = full_out[:, -1:, :]

        with torch.no_grad():
            _, cache = model(x[:, :n_ctx, :], return_kv_cache=True)
            cross_out = model.forward_cross(x[:, n_ctx:, :], cache)
        assert torch.allclose(naive, cross_out, atol=1e-4)

    def test_forward_cross_output_shape(self):
        model = self._make()
        x_ctx = torch.randn(B, M, D)
        _, cache = model(x_ctx, return_kv_cache=True)
        cand = torch.randn(B, 1, D)
        out = model.forward_cross(cand, cache)
        assert out.shape == (B, 1, D)


# ---------------------------------------------------------------------------
# InfoNCE-based pretraining losses
# ---------------------------------------------------------------------------

class TestInfoNCELoss:
    def test_scalar_output(self):
        anchor = torch.randn(5, D)
        positive = torch.randn(5, D)
        negatives = torch.randn(5, 8, D)
        loss = info_nce_loss(anchor, positive, negatives, temperature=torch.tensor(0.1))
        assert loss.dim() == 0

    def test_lower_when_positive_matches_anchor(self):
        torch.manual_seed(0)
        anchor = torch.randn(20, D)
        matching_positive = anchor.clone()
        random_positive = torch.randn(20, D)
        negatives = torch.randn(20, 8, D)
        temp = torch.tensor(0.1)
        loss_match = info_nce_loss(anchor, matching_positive, negatives, temp)
        loss_random = info_nce_loss(anchor, random_positive, negatives, temp)
        assert loss_match.item() < loss_random.item()


class TestNextTokenLoss:
    def _seqs(self):
        H_ = torch.randn(B, M, D, requires_grad=True)
        Z_ = torch.randn(B, M, D)
        is_pos = torch.ones(B, M, dtype=torch.bool)
        return H_, Z_, is_pos

    def test_scalar_and_finite(self):
        H_, Z_, is_pos = self._seqs()
        loss = next_token_loss(H_, Z_, is_pos, temperature=0.1)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_gradient_flows_to_H(self):
        H_, Z_, is_pos = self._seqs()
        loss = next_token_loss(H_, Z_, is_pos, temperature=0.1)
        loss.backward()
        assert H_.grad is not None

    def test_no_positives_returns_zero(self):
        H_, Z_, _ = self._seqs()
        is_pos = torch.zeros(B, M, dtype=torch.bool)
        loss = next_token_loss(H_, Z_, is_pos, temperature=0.1)
        assert loss.item() == 0.0


class TestMultiTokenLoss:
    def test_scalar_and_finite(self):
        H_ = torch.randn(B, M, D, requires_grad=True)
        Z_ = torch.randn(B, M, D)
        is_pos = torch.ones(B, M, dtype=torch.bool)
        loss = multi_token_loss(H_, Z_, is_pos, window=4, temperature=0.1)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert H_.grad is not None

    def test_window_larger_than_seq_does_not_crash(self):
        H_ = torch.randn(B, M, D)
        Z_ = torch.randn(B, M, D)
        is_pos = torch.ones(B, M, dtype=torch.bool)
        loss = multi_token_loss(H_, Z_, is_pos, window=M + 10, temperature=0.1)
        assert torch.isfinite(loss)


class TestFutureTokenLoss:
    def test_scalar_and_finite(self):
        H_ = torch.randn(B, M, D, requires_grad=True)
        Z_ = torch.randn(B, M, D)
        is_pos = torch.ones(B, M, dtype=torch.bool)
        loss = future_token_loss(H_, Z_, is_pos, l_d=10, window=4, temperature=0.1)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert H_.grad is not None

    def test_l_d_at_end_returns_zero(self):
        H_ = torch.randn(B, M, D)
        Z_ = torch.randn(B, M, D)
        is_pos = torch.ones(B, M, dtype=torch.bool)
        loss = future_token_loss(H_, Z_, is_pos, l_d=M - 1, window=4, temperature=0.1)
        assert loss.item() == 0.0

    def test_l_d_out_of_range_returns_zero(self):
        H_ = torch.randn(B, M, D)
        Z_ = torch.randn(B, M, D)
        is_pos = torch.ones(B, M, dtype=torch.bool)
        loss = future_token_loss(H_, Z_, is_pos, l_d=M + 5, window=4, temperature=0.1)
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# Deduplication / DCAT
# ---------------------------------------------------------------------------

class TestDeduplicateSequences:
    def test_no_duplicates(self):
        seqs = torch.randint(0, VOCAB, (5, 10))
        unique, inverse = deduplicate_sequences(seqs)
        assert unique.shape[0] == 5
        assert torch.equal(unique[inverse], seqs)

    def test_all_duplicates(self):
        row = torch.randint(0, VOCAB, (1, 10))
        seqs = row.repeat(6, 1)
        unique, inverse = deduplicate_sequences(seqs)
        assert unique.shape[0] == 1
        assert torch.equal(unique[inverse], seqs)

    def test_partial_duplicates_roundtrip(self):
        base = torch.randint(0, VOCAB, (3, 10))
        seqs = base.repeat_interleave(torch.tensor([2, 1, 3]), dim=0)
        unique, inverse = deduplicate_sequences(seqs)
        assert unique.shape[0] == 3
        assert torch.equal(unique[inverse], seqs)


class TestComputeSavings:
    def test_full_dedup(self):
        assert compute_savings(100, 1) == pytest.approx(0.99)

    def test_no_dedup(self):
        assert compute_savings(100, 100) == pytest.approx(0.0)


class TestPinFMDCAT:
    def _make(self):
        return PinFMDCAT(VOCAB, NUM_ACTIONS, NUM_SURFACES, D, n_layers=2, n_heads=H)

    def _repeated_batch(self, n_users=2, n_cand=4, seq_len=8):
        base_item = torch.randint(0, VOCAB, (n_users, seq_len))
        base_action = torch.randint(0, NUM_ACTIONS, (n_users, seq_len))
        base_surface = torch.randint(0, NUM_SURFACES, (n_users, seq_len))
        item = base_item.repeat_interleave(n_cand, dim=0)
        action = base_action.repeat_interleave(n_cand, dim=0)
        surface = base_surface.repeat_interleave(n_cand, dim=0)
        cand_ids = torch.randint(0, VOCAB, (n_users * n_cand,))
        return item, action, surface, cand_ids

    def test_output_shape(self):
        model = self._make()
        item, action, surface, cand_ids = self._repeated_batch()
        repr_, num_unique = model(item, action, surface, cand_ids)
        assert repr_.shape == (8, D)

    def test_dedup_count_matches_unique_users(self):
        model = self._make()
        item, action, surface, cand_ids = self._repeated_batch(n_users=3, n_cand=5)
        _, num_unique = model(item, action, surface, cand_ids)
        assert num_unique == 3

    def test_is_l2_normalized(self):
        model = self._make()
        item, action, surface, cand_ids = self._repeated_batch()
        repr_, _ = model(item, action, surface, cand_ids)
        assert torch.allclose(repr_.norm(dim=-1), torch.ones(8), atol=1e-5)

    def test_gradient(self):
        model = self._make()
        item, action, surface, cand_ids = self._repeated_batch()
        repr_, _ = model(item, action, surface, cand_ids)
        repr_.sum().backward()
        assert model.user_emb.item_emb.weight.grad is not None

    def test_shared_context_different_candidates_differ(self):
        # Same user context, different candidate ids -> different reprs
        # (early fusion is target-aware).
        model = self._make()
        model.eval()
        item, action, surface, _ = self._repeated_batch(n_users=1, n_cand=2)
        cand_ids = torch.tensor([1, 2])
        with torch.no_grad():
            repr_, num_unique = model(item, action, surface, cand_ids)
        assert num_unique == 1
        assert not torch.allclose(repr_[0], repr_[1])

    def test_no_duplicates_still_works(self):
        model = self._make()
        item = torch.randint(0, VOCAB, (5, 8))
        action = torch.randint(0, NUM_ACTIONS, (5, 8))
        surface = torch.randint(0, NUM_SURFACES, (5, 8))
        cand_ids = torch.randint(0, VOCAB, (5,))
        repr_, num_unique = model(item, action, surface, cand_ids)
        assert num_unique == 5
        assert repr_.shape == (5, D)


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------

class TestLateFusionRepresentation:
    def test_mean_mode(self):
        H_ = torch.randn(B, M, D)
        out = late_fusion_representation(H_, mode="mean")
        assert out.shape == (B, D)
        assert torch.allclose(out, H_.mean(dim=1))

    def test_last_mode(self):
        H_ = torch.randn(B, M, D)
        out = late_fusion_representation(H_, mode="last")
        assert torch.equal(out, H_[:, -1, :])

    def test_invalid_mode_raises(self):
        H_ = torch.randn(B, M, D)
        with pytest.raises(ValueError):
            late_fusion_representation(H_, mode="bogus")


# ---------------------------------------------------------------------------
# Cold-start handling
# ---------------------------------------------------------------------------

class TestCandidateIdRandomization:
    def test_approximate_fraction_changed(self):
        torch.manual_seed(0)
        ids = torch.randint(0, VOCAB, (5000,))
        out = candidate_id_randomization(ids, VOCAB, p=0.2)
        changed = (out != ids).float().mean().item()
        assert 0.15 < changed < 0.25

    def test_zero_probability_no_change(self):
        ids = torch.randint(0, VOCAB, (100,))
        out = candidate_id_randomization(ids, VOCAB, p=0.0)
        assert torch.equal(out, ids)

    def test_output_shape_and_dtype(self):
        ids = torch.randint(0, VOCAB, (50,))
        out = candidate_id_randomization(ids, VOCAB, p=0.5)
        assert out.shape == ids.shape
        assert out.dtype == ids.dtype


class TestItemAgeDependentDropout:
    def test_eval_mode_is_noop(self):
        emb = torch.randn(4, D)
        age = torch.tensor([2.0, 10.0, 40.0, 3.0])
        out = item_age_dependent_dropout(emb, age, training=False)
        assert torch.equal(out, emb)

    def test_old_items_less_affected_in_expectation(self):
        torch.manual_seed(0)
        emb = torch.ones(2000, D)
        age = torch.cat([torch.full((1000,), 2.0), torch.full((1000,), 40.0)])
        out = item_age_dependent_dropout(emb, age, training=True)
        fresh_zero_frac = (out[:1000] == 0).float().mean().item()
        old_zero_frac = (out[1000:] == 0).float().mean().item()
        assert fresh_zero_frac > old_zero_frac

    def test_output_shape(self):
        emb = torch.randn(4, D)
        age = torch.tensor([2.0, 10.0, 40.0, 3.0])
        out = item_age_dependent_dropout(emb, age, training=True)
        assert out.shape == emb.shape


# ---------------------------------------------------------------------------
# Embedding quantization simulation
# ---------------------------------------------------------------------------

class TestQuantization:
    def test_output_shape(self):
        x = torch.randn(10, 32)
        out = quantize_dequantize_embedding(x, num_bits=8)
        assert out.shape == x.shape

    def test_int8_more_accurate_than_int4(self):
        torch.manual_seed(0)
        x = torch.randn(500, 32)
        err8 = quantization_error(x, num_bits=8)
        err4 = quantization_error(x, num_bits=4)
        assert err8 < err4

    def test_error_is_small_fraction(self):
        torch.manual_seed(0)
        x = torch.randn(500, 32)
        err8 = quantization_error(x, num_bits=8)
        assert 0.0 <= err8 < 0.05

    def test_constant_row_quantizes_exactly(self):
        x = torch.full((3, 16), 2.5)
        out = quantize_dequantize_embedding(x, num_bits=4)
        assert torch.allclose(out, x, atol=1e-4)
