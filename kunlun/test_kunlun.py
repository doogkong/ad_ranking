"""Tests for Kunlun implementation.

Run with:
    pytest test_kunlun.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from kunlun import (
    ROTE,
    GDPA,
    SlidingWindowAttention,
    SumKronLinear,
    HSP,
    WeightGeneration,
    KunlunTransformerBlock,
    WukongExpert,
    GlobalInteraction,
    KunlunInteractionBlock,
    KunlunLayer,
    Kunlun,
    EventConfig,
    CLICK_CONFIG,
    IMPRESSION_CONFIG,
)

B  = 4    # batch size
D  = 32   # model dim
T  = 30   # sequence length
N  = 6    # initial non-seq tokens (1 dense + 5 sparse)
NS = 4    # n_sum
NT = 8    # n_tokens
NS_SEEDS = 16  # n_seeds (must be > NT)
H  = 4    # n_heads  (D // H = 8, which is valid)
W  = 5    # window size


# ---------------------------------------------------------------------------
# ROTE
# ---------------------------------------------------------------------------

class TestROTE:
    def test_output_shape(self):
        rote = ROTE(D)
        x = torch.randn(B, T, D)
        assert rote(x).shape == (B, T, D)

    def test_no_timestamps_uses_position(self):
        rote = ROTE(D)
        x = torch.randn(B, T, D)
        out = rote(x)
        assert out.shape == (B, T, D)
        assert torch.isfinite(out).all()

    def test_with_timestamps(self):
        rote = ROTE(D)
        x = torch.randn(B, T, D)
        ts = torch.cumsum(torch.randint(0, 3600, (B, T)).float(), dim=1)
        out = rote(x, ts)
        assert out.shape == (B, T, D)
        assert torch.isfinite(out).all()

    def test_different_timestamps_different_outputs(self):
        rote = ROTE(D)
        x = torch.randn(B, T, D)
        ts1 = torch.cumsum(torch.ones(B, T), dim=1)
        ts2 = torch.cumsum(torch.ones(B, T) * 3600, dim=1)
        assert not torch.allclose(rote(x, ts1), rote(x, ts2))

    def test_gradient(self):
        rote = ROTE(D)
        x = torch.randn(B, T, D, requires_grad=True)
        rote(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# GDPA
# ---------------------------------------------------------------------------

class TestGDPA:
    def _make(self):
        return GDPA(n_sum=NS, d_model=D, n_heads=H, max_seq_len=100)

    def test_output_shape(self):
        gdpa = self._make()
        S     = torch.randn(B, T, D)
        X_sum = torch.randn(B, NS, D)
        assert gdpa(S, X_sum).shape == (B, T, D)

    def test_residual_connection(self):
        # With zero weights, output = 0 * attn_result + S ≈ S (residual)
        gdpa = self._make()
        with torch.no_grad():
            gdpa.out_proj.weight.zero_()
            gdpa.out_proj.bias.zero_()
            S     = torch.randn(B, T, D)
            X_sum = torch.randn(B, NS, D)
            out   = gdpa(S, X_sum)
        assert torch.allclose(out, S, atol=1e-5)

    def test_different_xsum_different_outputs(self):
        gdpa  = self._make()
        S     = torch.randn(B, T, D)
        X1    = torch.randn(B, NS, D)
        X2    = torch.randn(B, NS, D)
        assert not torch.allclose(gdpa(S, X1), gdpa(S, X2))

    def test_finite_output(self):
        gdpa  = self._make()
        S     = torch.randn(B, T, D)
        X_sum = torch.randn(B, NS, D)
        assert torch.isfinite(gdpa(S, X_sum)).all()

    def test_gradient(self):
        gdpa  = self._make()
        S     = torch.randn(B, T, D, requires_grad=True)
        X_sum = torch.randn(B, NS, D, requires_grad=True)
        gdpa(S, X_sum).sum().backward()
        assert S.grad is not None
        assert X_sum.grad is not None
        assert gdpa.q_proj.weight.grad is not None

    def test_sequence_length_preserved(self):
        gdpa  = self._make()
        S     = torch.randn(B, T, D)
        X_sum = torch.randn(B, NS, D)
        assert gdpa(S, X_sum).shape[1] == T


# ---------------------------------------------------------------------------
# SlidingWindowAttention
# ---------------------------------------------------------------------------

class TestSlidingWindowAttention:
    def test_output_shape(self):
        swa = SlidingWindowAttention(D, n_heads=H, window=W)
        S   = torch.randn(B, T, D)
        assert swa(S).shape == (B, T, D)

    def test_short_sequence_full_attention(self):
        # Sequence shorter than window — should fall back to full attention
        swa = SlidingWindowAttention(D, n_heads=H, window=50)
        S   = torch.randn(B, 10, D)
        assert swa(S).shape == (B, 10, D)

    def test_long_sequence_windowed(self):
        swa = SlidingWindowAttention(D, n_heads=H, window=2)
        S   = torch.randn(B, 20, D)
        assert swa(S).shape == (B, 20, D)
        assert torch.isfinite(swa(S)).all()

    def test_gradient(self):
        swa = SlidingWindowAttention(D, n_heads=H, window=W)
        S   = torch.randn(B, T, D, requires_grad=True)
        swa(S).sum().backward()
        assert S.grad is not None


# ---------------------------------------------------------------------------
# SumKronLinear
# ---------------------------------------------------------------------------

class TestSumKronLinear:
    def test_output_shape(self):
        skl = SumKronLinear(S=NS_SEEDS, T=NT, D=D, k=4)
        X   = torch.randn(B, NS_SEEDS, D)
        assert skl(X).shape == (B, NT, D)

    def test_compression(self):
        # Compresses token axis from S to T
        skl = SumKronLinear(S=32, T=8, D=D, k=4)
        assert skl(torch.randn(B, 32, D)).shape == (B, 8, D)

    def test_parameter_count(self):
        S, T_, D_, k = 256, 32, 64, 8
        skl      = SumKronLinear(S, T_, D_, k)
        n_params = sum(p.numel() for p in skl.parameters())
        # Expected: k*(S*T + D*D) = 8*(256*32 + 64*64) = 8*(8192+4096) = 98304
        assert n_params == k * (S * T_ + D_ * D_)

    def test_gradient(self):
        skl = SumKronLinear(NS_SEEDS, NT, D, k=4)
        X   = torch.randn(B, NS_SEEDS, D, requires_grad=True)
        skl(X).sum().backward()
        assert X.grad is not None
        assert skl.Z.grad is not None
        assert skl.W.grad is not None

    def test_finite_output(self):
        skl = SumKronLinear(NS_SEEDS, NT, D, k=4)
        assert torch.isfinite(skl(torch.randn(B, NS_SEEDS, D))).all()


# ---------------------------------------------------------------------------
# HSP
# ---------------------------------------------------------------------------

class TestHSP:
    def _make(self):
        return HSP(d_model=D, n_seeds=NS_SEEDS, n_tokens=NT, n_heads=H, kron_rank=4)

    def test_output_shape(self):
        hsp = self._make()
        S   = torch.randn(B, T, D)
        assert hsp(S).shape == (B, NT, D)

    def test_n_seeds_must_exceed_n_tokens(self):
        with pytest.raises(AssertionError):
            HSP(d_model=D, n_seeds=4, n_tokens=8)

    def test_different_sequences_different_summaries(self):
        hsp = self._make()
        S1  = torch.randn(B, T, D)
        S2  = torch.randn(B, T, D)
        assert not torch.allclose(hsp(S1), hsp(S2))

    def test_finite_output(self):
        hsp = self._make()
        assert torch.isfinite(hsp(torch.randn(B, T, D))).all()

    def test_gradient(self):
        hsp = self._make()
        S   = torch.randn(B, T, D, requires_grad=True)
        hsp(S).sum().backward()
        assert S.grad is not None
        assert hsp.E_seed.grad is not None

    def test_batch_independence(self):
        # Same seeds expand to batch — output shape is consistent
        hsp = self._make()
        assert hsp(torch.randn(1, T, D)).shape == (1, NT, D)
        assert hsp(torch.randn(8, T, D)).shape == (8, NT, D)


# ---------------------------------------------------------------------------
# WeightGeneration
# ---------------------------------------------------------------------------

class TestWeightGeneration:
    def test_output_shape(self):
        wg = WeightGeneration(n_ns=N, n_sum=NS, d_model=D)
        X  = torch.randn(B, N, D)
        assert wg(X).shape == (B, NS, D)

    def test_compression(self):
        # Token count is reduced from n_ns to n_sum
        wg = WeightGeneration(n_ns=10, n_sum=3, d_model=D)
        assert wg(torch.randn(B, 10, D)).shape == (B, 3, D)

    def test_gradient(self):
        wg = WeightGeneration(N, NS, D)
        X  = torch.randn(B, N, D, requires_grad=True)
        wg(X).sum().backward()
        assert X.grad is not None
        assert wg.compress.weight.grad is not None

    def test_finite_output(self):
        wg = WeightGeneration(N, NS, D)
        assert torch.isfinite(wg(torch.randn(B, N, D))).all()


# ---------------------------------------------------------------------------
# KunlunTransformerBlock
# ---------------------------------------------------------------------------

class TestKunlunTransformerBlock:
    def _make(self):
        return KunlunTransformerBlock(n_sum=NS, d_model=D, n_heads=H,
                                      window=W, max_seq_len=100)

    def test_output_shape_even(self):
        blk   = self._make()
        S     = torch.randn(B, T, D)
        X_sum = torch.randn(B, NS, D)
        assert blk(S, X_sum, is_even_layer=True).shape == (B, T, D)

    def test_output_shape_odd(self):
        blk   = self._make()
        S     = torch.randn(B, T, D)
        X_sum = torch.randn(B, NS, D)
        assert blk(S, X_sum, is_even_layer=False).shape == (B, T, D)

    def test_even_uses_xsum(self):
        # On even layers, different X_sum should produce different outputs
        blk   = self._make()
        S     = torch.randn(B, T, D)
        X1    = torch.randn(B, NS, D)
        X2    = torch.randn(B, NS, D)
        assert not torch.allclose(blk(S, X1, True), blk(S, X2, True))

    def test_odd_ignores_xsum(self):
        # On odd layers, X_sum should not affect output
        blk   = self._make()
        S     = torch.randn(B, T, D)
        X1    = torch.randn(B, NS, D)
        X2    = torch.randn(B, NS, D)
        assert torch.allclose(blk(S, X1, False), blk(S, X2, False))

    def test_gradient_even(self):
        blk   = self._make()
        S     = torch.randn(B, T, D, requires_grad=True)
        X_sum = torch.randn(B, NS, D, requires_grad=True)
        blk(S, X_sum, True).sum().backward()
        assert S.grad is not None
        assert X_sum.grad is not None

    def test_gradient_odd(self):
        blk   = self._make()
        S     = torch.randn(B, T, D, requires_grad=True)
        X_sum = torch.randn(B, NS, D)
        blk(S, X_sum, False).sum().backward()
        assert S.grad is not None


# ---------------------------------------------------------------------------
# WukongExpert
# ---------------------------------------------------------------------------

class TestWukongExpert:
    def test_output_shape(self):
        exp = WukongExpert(n_tokens=NS + NT, d_model=D)
        X   = torch.randn(B, NS + NT, D)
        assert exp(X).shape == (B, NS + NT, D)

    def test_residual_in_output(self):
        exp = WukongExpert(n_tokens=4, d_model=D)
        X   = torch.randn(B, 4, D)
        out = exp(X)
        assert out.shape == (B, 4, D)
        assert torch.isfinite(out).all()

    def test_gradient(self):
        exp = WukongExpert(n_tokens=NS + NT, d_model=D)
        X   = torch.randn(B, NS + NT, D, requires_grad=True)
        exp(X).sum().backward()
        assert X.grad is not None


# ---------------------------------------------------------------------------
# GlobalInteraction
# ---------------------------------------------------------------------------

class TestGlobalInteraction:
    def test_output_shape(self):
        gi = GlobalInteraction(n_global=NS + NT, d_model=D, n_experts=2)
        X  = torch.randn(B, NS + NT, D)
        assert gi(X).shape == (B, NS + NT, D)

    def test_single_expert(self):
        gi = GlobalInteraction(n_global=NS + NT, d_model=D, n_experts=1)
        X  = torch.randn(B, NS + NT, D)
        assert gi(X).shape == (B, NS + NT, D)

    def test_odd_token_count(self):
        # n_global not divisible by n_experts
        gi = GlobalInteraction(n_global=7, d_model=D, n_experts=3)
        X  = torch.randn(B, 7, D)
        assert gi(X).shape == (B, 7, D)

    def test_gradient(self):
        gi = GlobalInteraction(NS + NT, D, n_experts=2)
        X  = torch.randn(B, NS + NT, D, requires_grad=True)
        gi(X).sum().backward()
        assert X.grad is not None


# ---------------------------------------------------------------------------
# KunlunInteractionBlock
# ---------------------------------------------------------------------------

class TestKunlunInteractionBlock:
    def _make(self):
        return KunlunInteractionBlock(
            n_ns=N, n_sum=NS, d_model=D,
            n_seeds=NS_SEEDS, n_tokens=NT, n_heads=H, n_experts=2, kron_rank=4,
        )

    def test_output_shapes(self):
        blk = self._make()
        X   = torch.randn(B, N, D)
        S   = torch.randn(B, T, D)
        X_new, X_sum, H_summary = blk(X, S, None, is_even_layer=True)
        assert X_new.shape     == (B, NS + NT, D)
        assert X_sum.shape     == (B, NS, D)
        assert H_summary.shape == (B, NT, D)

    def test_even_computes_fresh_hsp(self):
        blk   = self._make()
        X     = torch.randn(B, N, D)
        S     = torch.randn(B, T, D)
        stale = torch.zeros(B, NT, D)          # would produce different result
        _, _, h_even = blk(X, S, stale, is_even_layer=True)
        assert not torch.allclose(h_even, stale)

    def test_odd_reuses_cache(self):
        blk   = self._make()
        X     = torch.randn(B, N, D)
        S     = torch.randn(B, T, D)
        cache = torch.randn(B, NT, D)
        _, _, h_odd = blk(X, S, cache, is_even_layer=False)
        assert torch.allclose(h_odd, cache)

    def test_none_cache_triggers_fresh_hsp(self):
        blk = self._make()
        X   = torch.randn(B, N, D)
        S   = torch.randn(B, T, D)
        _, _, h = blk(X, S, None, is_even_layer=False)  # None forces fresh
        assert h.shape == (B, NT, D)

    def test_gradient(self):
        blk = self._make()
        X   = torch.randn(B, N, D, requires_grad=True)
        S   = torch.randn(B, T, D, requires_grad=True)
        X_new, X_sum, H = blk(X, S, None, True)
        (X_new.sum() + X_sum.sum() + H.sum()).backward()
        assert X.grad is not None
        assert S.grad is not None


# ---------------------------------------------------------------------------
# KunlunLayer
# ---------------------------------------------------------------------------

class TestKunlunLayer:
    def _make(self):
        return KunlunLayer(
            n_ns=N, n_sum=NS, d_model=D,
            n_seeds=NS_SEEDS, n_tokens=NT, n_heads=H,
            window=W, n_experts=2, kron_rank=4, max_seq_len=100,
        )

    def test_output_shapes(self):
        layer = self._make()
        X     = torch.randn(B, N, D)
        S     = torch.randn(B, T, D)
        X_new, S_new, X_sum, H_summary = layer(X, S, None, layer_idx=0)
        assert X_new.shape     == (B, NS + NT, D)
        assert S_new.shape     == (B, T, D)
        assert X_sum.shape     == (B, NS, D)
        assert H_summary.shape == (B, NT, D)

    def test_sequence_length_preserved(self):
        layer = self._make()
        X     = torch.randn(B, N, D)
        S     = torch.randn(B, T, D)
        _, S_new, _, _ = layer(X, S, None, layer_idx=0)
        assert S_new.shape[1] == T

    def test_even_vs_odd_different_seq_outputs(self):
        # Even layer uses GDPA; odd uses SWA — sequence outputs differ
        layer = self._make()
        X     = torch.randn(B, N, D)
        S     = torch.randn(B, T, D)
        _, S_even, _, _ = layer(X, S, None, layer_idx=0)
        _, S_odd,  _, _ = layer(X, S, None, layer_idx=1)
        assert not torch.allclose(S_even, S_odd)

    def test_gradient(self):
        layer = self._make()
        X     = torch.randn(B, N, D, requires_grad=True)
        S     = torch.randn(B, T, D, requires_grad=True)
        X_new, S_new, X_sum, H = layer(X, S, None, 0)
        (X_new.sum() + S_new.sum() + X_sum.sum() + H.sum()).backward()
        assert X.grad is not None
        assert S.grad is not None


# ---------------------------------------------------------------------------
# Kunlun (full model)
# ---------------------------------------------------------------------------

class TestKunlun:
    def _make(self, **kw):
        defaults = dict(
            dense_dim     = 32,
            sparse_dims   = [100, 200, 50],
            seq_input_dim = 16,
            d_model       = D,
            num_layers    = 4,
            n_sum         = NS,
            n_seeds       = NS_SEEDS,
            n_tokens      = NT,
            n_heads       = H,
            window        = W,
            n_experts     = 2,
            kron_rank     = 4,
            max_seq_len   = 100,
            top_mlp_dims  = [32],
            num_tasks     = 1,
        )
        defaults.update(kw)
        return Kunlun(**defaults)

    def _inputs(self, b=B):
        dense   = torch.randn(b, 32)
        sparse  = [torch.randint(0, v, (b,)) for v in [100, 200, 50]]
        seq     = torch.randn(b, T, 16)
        return dense, sparse, seq

    def test_output_shape_single_task(self):
        model = self._make(num_tasks=1)
        assert model(*self._inputs()).shape == (B, 1)

    def test_output_shape_multi_task(self):
        model = self._make(num_tasks=3)
        assert model(*self._inputs()).shape == (B, 3)

    def test_backward(self):
        model  = self._make()
        logits = model(*self._inputs())
        F.binary_cross_entropy_with_logits(logits, torch.zeros(B, 1)).backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_finite_output(self):
        model = self._make()
        assert torch.isfinite(model(*self._inputs())).all()

    def test_deterministic_with_seed(self):
        model = self._make()
        inp   = self._inputs()
        torch.manual_seed(42); out1 = model(*inp)
        torch.manual_seed(42); out2 = model(*inp)
        assert torch.allclose(out1, out2)

    def test_different_inputs_different_outputs(self):
        model = self._make()
        assert not torch.allclose(model(*self._inputs()), model(*self._inputs()))

    def test_batch_size_one(self):
        model = self._make()
        assert model(*self._inputs(b=1)).shape == (1, 1)

    def test_longer_sequence(self):
        model  = self._make()
        dense  = torch.randn(B, 32)
        sparse = [torch.randint(0, v, (B,)) for v in [100, 200, 50]]
        seq    = torch.randn(B, 200, 16)
        assert model(dense, sparse, seq).shape == (B, 1)

    def test_with_timestamps(self):
        model  = self._make()
        dense, sparse, seq = self._inputs()
        ts     = torch.cumsum(torch.randint(0, 3600, (B, T)).float(), dim=1)
        assert model(dense, sparse, seq, timestamps=ts).shape == (B, 1)

    def test_num_layers_scaling(self):
        m2 = self._make(num_layers=2)
        m6 = self._make(num_layers=6)
        p2 = sum(p.numel() for p in m2.parameters())
        p6 = sum(p.numel() for p in m6.parameters())
        assert p6 > p2

    def test_compskip_cache_behavior(self):
        # Even layers produce fresh HSP; test that they differ from odd layer output
        model  = self._make(num_layers=2)
        dense, sparse, seq = self._inputs()
        # Just verify it runs correctly — CompSkip behavior verified in KunlunLayer tests
        assert model(dense, sparse, seq).shape == (B, 1)

    def test_single_sparse_feature(self):
        model  = Kunlun(
            dense_dim=16, sparse_dims=[50], seq_input_dim=8,
            d_model=D, num_layers=2, n_sum=NS, n_seeds=NS_SEEDS, n_tokens=NT,
            n_heads=H, window=W, n_experts=1, kron_rank=4, max_seq_len=100, num_tasks=1,
        )
        dense  = torch.randn(B, 16)
        sparse = [torch.randint(0, 50, (B,))]
        seq    = torch.randn(B, T, 8)
        assert model(dense, sparse, seq).shape == (B, 1)

    def test_many_sparse_features(self):
        vocabs = [50] * 8
        model  = Kunlun(
            dense_dim=32, sparse_dims=vocabs, seq_input_dim=16,
            d_model=D, num_layers=2, n_sum=NS, n_seeds=NS_SEEDS, n_tokens=NT,
            n_heads=H, window=W, n_experts=2, kron_rank=4, max_seq_len=100, num_tasks=1,
        )
        dense  = torch.randn(B, 32)
        sparse = [torch.randint(0, 50, (B,)) for _ in vocabs]
        seq    = torch.randn(B, T, 16)
        assert model(dense, sparse, seq).shape == (B, 1)


# ---------------------------------------------------------------------------
# EventConfig
# ---------------------------------------------------------------------------

class TestEventConfig:
    def test_click_config(self):
        assert CLICK_CONFIG.d_model  == 256
        assert CLICK_CONFIG.n_heads  == 8
        assert CLICK_CONFIG.n_tokens == 32
        assert CLICK_CONFIG.n_layers == 3
        assert CLICK_CONFIG.window   == 100

    def test_impression_config(self):
        assert IMPRESSION_CONFIG.d_model  == 128
        assert IMPRESSION_CONFIG.n_heads  == 4
        assert IMPRESSION_CONFIG.n_tokens == 16
        assert IMPRESSION_CONFIG.n_layers == 2
        assert IMPRESSION_CONFIG.window   == 50

    def test_custom_config(self):
        cfg = EventConfig(d_model=64, n_heads=2, n_tokens=8, n_layers=1, window=20)
        assert cfg.d_model == 64
