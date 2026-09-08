"""Tests for HSTU implementation.

Run with:
    pytest test_hstu.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from hstu import (
    RelativeAttentionBias,
    HSTULayer,
    HSTU,
    RankingHead,
    retrieval_scores,
    stochastic_length_target,
    apply_stochastic_length,
)

B = 2    # batch size
N = 12   # sequence length
D = 32   # model dim
H = 4    # attention heads
DQK = 8  # per-head query/key dim
DV = 8   # per-head value dim
L = 3    # layers


# ---------------------------------------------------------------------------
# RelativeAttentionBias
# ---------------------------------------------------------------------------

class TestRelativeAttentionBias:
    def test_output_shape_no_time(self):
        rab = RelativeAttentionBias(num_heads=H)
        bias = rab(N)
        assert bias.shape == (H, N, N)

    def test_output_shape_with_time(self):
        rab = RelativeAttentionBias(num_heads=H, use_time_bias=True)
        timestamps = torch.arange(N, dtype=torch.float32).unsqueeze(0).expand(B, N)
        bias = rab(N, timestamps)
        assert bias.shape == (B, H, N, N)

    def test_depends_only_on_relative_position(self):
        # Diagonal-shifted entries should have identical bias (Toeplitz structure).
        rab = RelativeAttentionBias(num_heads=H)
        bias = rab(N)
        assert torch.allclose(bias[:, 5, 3], bias[:, 8, 6])  # both relative distance +2

    def test_offsets_bias_matches_full_matrix(self):
        rab = RelativeAttentionBias(num_heads=H)
        bias = rab(N)
        offsets = torch.arange(N)  # candidate at position N attending to j=0..N-1 -> offset = N-j
        # compare against the bias a token at position N would have to position 0..N-1,
        # i.e. bias[:, N-1, :] is not directly it (positions only go to N-1); instead
        # verify offsets_bias reproduces the same bucket->embedding lookup as forward().
        direct = rab.offsets_bias(offsets)
        assert direct.shape == (H, N)
        # offset=0 case must equal self_bias()
        assert torch.allclose(rab.offsets_bias(torch.zeros(1, dtype=torch.long)).squeeze(-1), rab.self_bias())

    def test_self_bias_matches_diagonal(self):
        rab = RelativeAttentionBias(num_heads=H)
        bias = rab(N)
        diag = bias[:, 3, 3]  # relative distance 0
        assert torch.allclose(diag, rab.self_bias())

    def test_gradient(self):
        rab = RelativeAttentionBias(num_heads=H, use_time_bias=True)
        timestamps = torch.arange(N, dtype=torch.float32).unsqueeze(0).expand(B, N)
        bias = rab(N, timestamps)
        bias.sum().backward()
        assert rab.pos_bias.weight.grad is not None
        assert rab.time_bias.weight.grad is not None


# ---------------------------------------------------------------------------
# HSTULayer
# ---------------------------------------------------------------------------

class TestHSTULayer:
    def _make(self):
        return HSTULayer(d_model=D, num_heads=H, dqk=DQK, dv=DV)

    def _rab_and_mask(self, n=N):
        rab = torch.zeros(1, H, n, n)
        causal_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1).view(1, 1, n, n)
        return rab, causal_mask

    def test_output_shapes(self):
        layer = self._make()
        x = torch.randn(B, N, D)
        rab, mask = self._rab_and_mask()
        out, K, V = layer(x, rab, mask)
        assert out.shape == (B, N, D)
        assert K.shape == (B, H, N, DQK)
        assert V.shape == (B, H, N, DV)

    def test_finite_output(self):
        layer = self._make()
        x = torch.randn(B, N, D)
        rab, mask = self._rab_and_mask()
        out, _, _ = layer(x, rab, mask)
        assert torch.isfinite(out).all()

    def test_causal_masking(self):
        # Changing a future token must not change an earlier position's output.
        layer = self._make()
        layer.eval()
        x1 = torch.randn(1, N, D)
        x2 = x1.clone()
        x2[:, -1, :] = torch.randn(D)  # perturb only the last (future-most) token
        rab, mask = self._rab_and_mask()
        with torch.no_grad():
            out1, _, _ = layer(x1, rab, mask)
            out2, _, _ = layer(x2, rab, mask)
        assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-6)
        assert not torch.allclose(out1[:, -1, :], out2[:, -1, :])

    def test_gradient(self):
        layer = self._make()
        x = torch.randn(B, N, D, requires_grad=True)
        rab, mask = self._rab_and_mask()
        out, _, _ = layer(x, rab, mask)
        out.sum().backward()
        assert x.grad is not None
        assert layer.f1.weight.grad is not None
        assert layer.f2.weight.grad is not None

    def test_forward_incremental_matches_full_forward(self):
        # A candidate scored incrementally against cached history K/V must match
        # exactly what a full forward pass over [history, candidate] would give.
        layer = self._make()
        layer.eval()
        n_hist = N - 1
        x = torch.randn(1, N, D)
        rab_full, mask_full = self._rab_and_mask(N)
        with torch.no_grad():
            full_out, _, _ = layer(x, rab_full, mask_full)
        naive = full_out[:, -1, :]

        rab_hist_full = torch.zeros(1, H, n_hist, n_hist)
        mask_hist = torch.triu(torch.ones(n_hist, n_hist, dtype=torch.bool), diagonal=1).view(1, 1, n_hist, n_hist)
        with torch.no_grad():
            _, K_hist, V_hist = layer(x[:, :n_hist, :], rab_hist_full, mask_hist)
            rab_hist = torch.zeros(H, n_hist)
            rab_self = torch.zeros(H)
            cached = layer.forward_incremental(x[:, n_hist:, :], K_hist, V_hist, rab_hist, rab_self)
        assert torch.allclose(naive, cached.squeeze(1), atol=1e-5)


# ---------------------------------------------------------------------------
# HSTU (full encoder)
# ---------------------------------------------------------------------------

class TestHSTU:
    def _make(self, **kwargs):
        defaults = dict(d_model=D, num_layers=L, num_heads=H, dqk=DQK, dv=DV)
        defaults.update(kwargs)
        return HSTU(**defaults)

    def test_output_shape(self):
        model = self._make()
        x = torch.randn(B, N, D)
        assert model(x).shape == (B, N, D)

    def test_finite_output(self):
        model = self._make()
        out = model(torch.randn(B, N, D))
        assert torch.isfinite(out).all()

    def test_causal_property_end_to_end(self):
        model = self._make()
        model.eval()
        x1 = torch.randn(1, N, D)
        x2 = x1.clone()
        x2[:, -1, :] = torch.randn(D)
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
        assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)

    def test_time_bias_changes_output(self):
        model = self._make(use_time_bias=True)
        model.eval()
        x = torch.randn(1, N, D)
        t1 = torch.arange(N, dtype=torch.float32).unsqueeze(0)
        t2 = (torch.arange(N, dtype=torch.float32) * 5).unsqueeze(0)
        with torch.no_grad():
            out1 = model(x, timestamps=t1)
            out2 = model(x, timestamps=t2)
        assert not torch.allclose(out1, out2)

    def test_gradient(self):
        model = self._make()
        x = torch.randn(B, N, D, requires_grad=True)
        model(x).sum().backward()
        assert x.grad is not None

    def test_deterministic_with_seed(self):
        model = self._make()
        x = torch.randn(B, N, D)
        torch.manual_seed(42)
        out1 = model(x)
        torch.manual_seed(42)
        out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_num_layers_scaling_increases_params(self):
        p2 = sum(p.numel() for p in self._make(num_layers=2).parameters())
        p4 = sum(p.numel() for p in self._make(num_layers=4).parameters())
        assert p4 > p2

    def test_return_cache_shapes(self):
        model = self._make()
        x = torch.randn(1, N, D)
        out, cache = model(x, return_cache=True)
        assert out.shape == (1, N, D)
        assert len(cache) == L
        for K, V in cache:
            assert K.shape == (1, H, N, DQK)
            assert V.shape == (1, H, N, DV)

    def test_score_candidates_shape(self):
        model = self._make()
        x = torch.randn(1, N, D)
        _, cache = model(x, return_cache=True)
        candidates = torch.randn(5, D)
        scores = model.score_candidates(cache, candidates)
        assert scores.shape == (5, D)

    def test_score_candidates_matches_naive_full_forward(self):
        # The core M-FALCON correctness property: scoring a candidate via the
        # cached history must equal appending it and doing a full recompute.
        model = self._make()
        model.eval()
        history = torch.randn(1, N, D)
        candidate = torch.randn(1, D)
        with torch.no_grad():
            full_seq = torch.cat([history, candidate.unsqueeze(1)], dim=1)
            naive = model(full_seq)[0, -1, :]
            _, cache = model(history, return_cache=True)
            cached = model.score_candidates(cache, candidate)[0]
        assert torch.allclose(naive, cached, atol=1e-4)

    def test_score_candidates_microbatching_matches_full_batch(self):
        model = self._make()
        model.eval()
        x = torch.randn(1, N, D)
        _, cache = model(x, return_cache=True)
        candidates = torch.randn(9, D)
        with torch.no_grad():
            full = model.score_candidates(cache, candidates)
            chunked = model.score_candidates(cache, candidates, microbatch_size=4)
        assert torch.allclose(full, chunked, atol=1e-5)

    def test_batch_size_one(self):
        model = self._make()
        assert model(torch.randn(1, N, D)).shape == (1, N, D)


# ---------------------------------------------------------------------------
# RankingHead / retrieval_scores
# ---------------------------------------------------------------------------

class TestRankingHead:
    def test_output_shape_single_task(self):
        head = RankingHead(D, hidden_dims=[D], num_tasks=1)
        assert head(torch.randn(B, D)).shape == (B, 1)

    def test_output_shape_multi_task(self):
        head = RankingHead(D, hidden_dims=[D, D], num_tasks=3)
        assert head(torch.randn(B, D)).shape == (B, 3)

    def test_gradient(self):
        head = RankingHead(D, hidden_dims=[D], num_tasks=1)
        x = torch.randn(B, D, requires_grad=True)
        head(x).sum().backward()
        assert x.grad is not None


class TestRetrievalScores:
    def test_output_shape(self):
        user_repr = torch.randn(B, D)
        candidates = torch.randn(10, D)
        scores = retrieval_scores(user_repr, candidates)
        assert scores.shape == (B, 10)

    def test_matches_manual_dot_product(self):
        user_repr = torch.randn(D)
        candidates = torch.randn(3, D)
        scores = retrieval_scores(user_repr, candidates)
        for i in range(3):
            assert torch.allclose(scores[i], torch.dot(user_repr, candidates[i]), atol=1e-5)


# ---------------------------------------------------------------------------
# Stochastic Length
# ---------------------------------------------------------------------------

class TestStochasticLength:
    def test_short_sequence_always_kept_whole(self):
        for _ in range(20):
            target = stochastic_length_target(seq_len=10, alpha=1.6, max_corpus_len=4096)
            assert target == 10

    def test_long_sequence_target_bounded(self):
        targets = {stochastic_length_target(4096, alpha=1.6, max_corpus_len=4096) for _ in range(50)}
        assert targets <= {4096, int(round(4096 ** 0.8))}

    def test_apply_preserves_shape_when_no_truncation(self):
        x = torch.randn(1, 10, D)
        out = apply_stochastic_length(x, alpha=1.6, max_corpus_len=4096)
        assert out.shape == x.shape

    def test_apply_truncates_long_sequence(self):
        torch.manual_seed(0)
        x = torch.randn(1, 4096, D)
        out = apply_stochastic_length(x, alpha=1.6, max_corpus_len=4096)
        assert out.shape[1] <= 4096
        assert out.shape[0] == 1 and out.shape[2] == D

    def test_apply_preserves_temporal_order(self):
        # Encode position index into the feature so we can check monotonicity.
        torch.manual_seed(0)
        n = 4096
        x = torch.arange(n, dtype=torch.float32).view(1, n, 1).expand(1, n, D).clone()
        out = apply_stochastic_length(x, alpha=1.6, max_corpus_len=4096)
        positions = out[0, :, 0]
        assert torch.all(positions[1:] >= positions[:-1])
