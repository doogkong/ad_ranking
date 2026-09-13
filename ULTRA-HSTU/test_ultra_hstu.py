"""Tests for ULTRA-HSTU implementation.

Run with:
    pytest test_ultra_hstu.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from ultra_hstu import (
    build_uih_sequence,
    HeterogeneousActionEncoder,
    semi_local_attention_mask,
    attention_complexity,
    UltraHSTULayer,
    UltraHSTU,
    RankingHead,
)

B = 2     # batch size
L = 32    # sequence length
D = 32    # model dim
H = 4     # attention heads
DQK = 8   # per-head query/key dim
DV = 8    # per-head value dim
K1 = 8    # local window
K2 = 4    # global window


# ---------------------------------------------------------------------------
# build_uih_sequence / HeterogeneousActionEncoder
# ---------------------------------------------------------------------------

class TestBuildUihSequence:
    def test_output_shape(self):
        item_emb = torch.randn(B, L, D)
        action_emb = torch.randn(B, L, D)
        x = build_uih_sequence(item_emb, action_emb)
        assert x.shape == (B, L, D)

    def test_is_additive(self):
        item_emb = torch.randn(B, L, D)
        action_emb = torch.randn(B, L, D)
        x = build_uih_sequence(item_emb, action_emb)
        assert torch.allclose(x, item_emb + action_emb)

    def test_candidate_action_masked_to_zero(self):
        item_emb = torch.randn(B, L, D)
        action_emb = torch.randn(B, L, D)
        is_candidate = torch.zeros(B, L, dtype=torch.bool)
        is_candidate[:, -3:] = True
        x = build_uih_sequence(item_emb, action_emb, is_candidate)
        assert torch.allclose(x[:, -3:, :], item_emb[:, -3:, :])
        assert torch.allclose(x[:, :-3, :], (item_emb + action_emb)[:, :-3, :])


class TestHeterogeneousActionEncoder:
    def test_output_shape(self):
        enc = HeterogeneousActionEncoder(D, num_action_types=5, num_intensity_buckets=10, num_context_buckets=24)
        out = enc(
            action_type=torch.randint(0, 5, (B, L)),
            intensity_bucket=torch.randint(0, 10, (B, L)),
            context_bucket=torch.randint(0, 24, (B, L)),
        )
        assert out.shape == (B, L, D)

    def test_gradient(self):
        enc = HeterogeneousActionEncoder(D, num_action_types=5, num_intensity_buckets=10, num_context_buckets=24)
        out = enc(
            action_type=torch.randint(0, 5, (B, L)),
            intensity_bucket=torch.randint(0, 10, (B, L)),
            context_bucket=torch.randint(0, 24, (B, L)),
        )
        out.sum().backward()
        assert enc.action_type_emb.weight.grad is not None
        assert enc.intensity_emb.weight.grad is not None
        assert enc.context_emb.weight.grad is not None

    def test_different_signals_change_output(self):
        enc = HeterogeneousActionEncoder(D, num_action_types=5, num_intensity_buckets=10, num_context_buckets=24)
        action_type = torch.zeros(1, 1, dtype=torch.long)
        context_bucket = torch.zeros(1, 1, dtype=torch.long)
        out_low = enc(action_type, torch.zeros(1, 1, dtype=torch.long), context_bucket)
        out_high = enc(action_type, torch.full((1, 1), 9, dtype=torch.long), context_bucket)
        assert not torch.allclose(out_low, out_high)


# ---------------------------------------------------------------------------
# semi_local_attention_mask / attention_complexity
# ---------------------------------------------------------------------------

class TestSemiLocalAttentionMask:
    def test_output_shape_and_dtype(self):
        mask = semi_local_attention_mask(L, K1, K2)
        assert mask.shape == (L, L)
        assert mask.dtype == torch.bool

    def test_causal(self):
        mask = semi_local_attention_mask(L, K1, K2)
        upper_triangle = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        assert not (mask & upper_triangle).any()

    def test_self_attention_always_allowed(self):
        mask = semi_local_attention_mask(L, K1, K2)
        assert mask.diagonal().all()

    def test_local_window_respected(self):
        # A query far past both windows should only attend within [q-K1+1, q]
        # among non-global positions.
        mask = semi_local_attention_mask(L, local_window=K1, global_window=K2)
        q = L - 1
        allowed = mask[q].nonzero(as_tuple=True)[0]
        for j in allowed.tolist():
            assert j < K2 or (q - j) < K1

    def test_global_anchor_columns_fully_visible_downstream(self):
        # Every position at or after a global anchor column can see it.
        mask = semi_local_attention_mask(L, local_window=K1, global_window=K2)
        for j in range(K2):
            column = mask[j:, j]
            assert column.all()

    def test_larger_local_window_yields_more_true_entries(self):
        small = semi_local_attention_mask(L, local_window=2, global_window=K2).sum()
        large = semi_local_attention_mask(L, local_window=16, global_window=K2).sum()
        assert large > small

    def test_zero_windows_is_diagonal_only(self):
        mask = semi_local_attention_mask(L, local_window=1, global_window=0)
        assert torch.equal(mask, torch.eye(L, dtype=torch.bool))

    def test_large_windows_recovers_full_causal(self):
        mask = semi_local_attention_mask(L, local_window=L, global_window=L)
        causal = ~torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        assert torch.equal(mask, causal)


class TestAttentionComplexity:
    def test_matches_mask_nnz(self):
        sla_nnz, full_nnz, ratio = attention_complexity(L, K1, K2)
        assert sla_nnz == semi_local_attention_mask(L, K1, K2).sum().item()
        assert full_nnz == L * (L + 1) // 2
        assert ratio == pytest.approx(sla_nnz / full_nnz)

    def test_sla_grows_linearly_full_grows_quadratically(self):
        nnz_1, full_1, _ = attention_complexity(256, K1, K2)
        nnz_2, full_2, _ = attention_complexity(512, K1, K2)
        # Doubling L roughly doubles SLA's nnz (linear) but ~quadruples full's.
        assert nnz_2 / nnz_1 < 2.5
        assert full_2 / full_1 == pytest.approx(4.0, abs=0.1)


# ---------------------------------------------------------------------------
# UltraHSTULayer
# ---------------------------------------------------------------------------

class TestUltraHSTULayer:
    def _make(self):
        return UltraHSTULayer(d_model=D, num_heads=H, dqk=DQK, dv=DV)

    def test_output_shape(self):
        layer = self._make()
        z = torch.randn(B, L, D)
        mask = semi_local_attention_mask(L, K1, K2)
        out = layer(z, mask)
        assert out.shape == (B, L, D)

    def test_finite_output(self):
        layer = self._make()
        z = torch.randn(B, L, D)
        mask = semi_local_attention_mask(L, K1, K2)
        out = layer(z, mask)
        assert torch.isfinite(out).all()

    def test_causal_masking(self):
        layer = self._make()
        layer.eval()
        mask = semi_local_attention_mask(L, K1, K2)
        z1 = torch.randn(1, L, D)
        z2 = z1.clone()
        z2[:, -1, :] = torch.randn(D)
        with torch.no_grad():
            out1 = layer(z1, mask)
            out2 = layer(z2, mask)
        assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-6)
        assert not torch.allclose(out1[:, -1, :], out2[:, -1, :])

    def test_sparse_mask_isolates_far_past(self):
        # A query outside both the local and global windows of an edit must
        # be unaffected by that edit.
        layer = self._make()
        layer.eval()
        mask = semi_local_attention_mask(L, local_window=4, global_window=2)
        z1 = torch.randn(1, L, D)
        z2 = z1.clone()
        edit_pos = 10  # not a global anchor (>=2), and outside local window of later queries
        z2[:, edit_pos, :] = torch.randn(D)
        with torch.no_grad():
            out1 = layer(z1, mask)
            out2 = layer(z2, mask)
        far_query = L - 1  # local window is only 4, so position 10 is out of range
        assert torch.allclose(out1[:, far_query, :], out2[:, far_query, :], atol=1e-6)

    def test_gradient(self):
        layer = self._make()
        z = torch.randn(B, L, D, requires_grad=True)
        mask = semi_local_attention_mask(L, K1, K2)
        layer(z, mask).sum().backward()
        assert z.grad is not None
        assert layer.f1.weight.grad is not None
        assert layer.f2.weight.grad is not None


# ---------------------------------------------------------------------------
# UltraHSTU (encoder: SLA + Attention Truncation)
# ---------------------------------------------------------------------------

class TestUltraHSTU:
    def _make(self, **kwargs):
        defaults = dict(
            d_model=D, num_heads=H, dqk=DQK, dv=DV,
            num_layers_full=3, local_window=K1, global_window=K2,
        )
        defaults.update(kwargs)
        return UltraHSTU(**defaults)

    def test_sla_only_output_shape(self):
        model = self._make()
        h_full, h_trunc = model(torch.randn(B, L, D))
        assert h_full.shape == (B, L, D)
        assert h_trunc is None

    def test_finite_output(self):
        model = self._make()
        h_full, _ = model(torch.randn(B, L, D))
        assert torch.isfinite(h_full).all()

    def test_requires_truncate_len_with_truncated_layers(self):
        with pytest.raises(ValueError):
            self._make(num_layers_truncated=2)

    def test_attention_truncation_output_shapes(self):
        model = self._make(num_layers_truncated=2, truncate_len=8)
        h_full, h_trunc = model(torch.randn(B, L, D))
        assert h_full.shape == (B, L, D)
        assert h_trunc.shape == (B, 8, D)

    def test_truncate_len_clamped_to_seq_len(self):
        model = self._make(num_layers_truncated=1, truncate_len=1000)
        h_full, h_trunc = model(torch.randn(B, L, D))
        assert h_trunc.shape == (B, L, D)

    def test_truncated_segment_is_the_latest_positions(self):
        # Perturbing the earliest token (outside the truncated window and
        # outside the local/global SLA reach of late positions) must not
        # change the truncated-stage output.
        model = self._make(num_layers_truncated=2, truncate_len=4, local_window=2, global_window=0)
        model.eval()
        x1 = torch.randn(1, L, D)
        x2 = x1.clone()
        x2[:, 0, :] = torch.randn(D)
        with torch.no_grad():
            _, trunc1 = model(x1)
            _, trunc2 = model(x2)
        assert torch.allclose(trunc1, trunc2, atol=1e-5)

    def test_final_representation_uses_truncated_stage_when_present(self):
        model = self._make(num_layers_truncated=1, truncate_len=4)
        h_full, h_trunc = model(torch.randn(B, L, D))
        rep = model.final_representation(h_full, h_trunc)
        assert torch.equal(rep, h_trunc[:, -1, :])

    def test_final_representation_falls_back_to_full_stage(self):
        model = self._make()
        h_full, h_trunc = model(torch.randn(B, L, D))
        rep = model.final_representation(h_full, h_trunc)
        assert torch.equal(rep, h_full[:, -1, :])

    def test_gradient_through_truncation(self):
        model = self._make(num_layers_truncated=2, truncate_len=8)
        x = torch.randn(B, L, D, requires_grad=True)
        h_full, h_trunc = model(x)
        (h_full.sum() + h_trunc.sum()).backward()
        assert x.grad is not None

    def test_num_layers_scaling_increases_params(self):
        p_small = sum(p.numel() for p in self._make(num_layers_full=2).parameters())
        p_large = sum(p.numel() for p in self._make(num_layers_full=4).parameters())
        assert p_large > p_small

    def test_batch_size_one(self):
        model = self._make()
        h_full, _ = model(torch.randn(1, L, D))
        assert h_full.shape == (1, L, D)


# ---------------------------------------------------------------------------
# RankingHead
# ---------------------------------------------------------------------------

class TestRankingHead:
    def test_output_shape_multi_task(self):
        head = RankingHead(D, hidden_dims=[D], num_tasks=2)
        assert head(torch.randn(B, D)).shape == (B, 2)

    def test_gradient(self):
        head = RankingHead(D, hidden_dims=[D], num_tasks=2)
        x = torch.randn(B, D, requires_grad=True)
        head(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_pipeline_backward(self):
        item_emb = torch.randn(B, L, D)
        action_emb = torch.randn(B, L, D)
        x = build_uih_sequence(item_emb, action_emb)

        model = UltraHSTU(
            d_model=D, num_heads=H, dqk=DQK, dv=DV,
            num_layers_full=2, num_layers_truncated=2,
            local_window=K1, global_window=K2, truncate_len=8,
        )
        head = RankingHead(D, hidden_dims=[D], num_tasks=2)

        h_full, h_trunc = model(x)
        logits = head(model.final_representation(h_full, h_trunc))
        loss = F.binary_cross_entropy_with_logits(logits, torch.zeros(B, 2))
        loss.backward()

        assert logits.shape == (B, 2)
        assert torch.isfinite(loss)
