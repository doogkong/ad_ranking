"""Tests for OneTrans implementation.

Run with:
    pytest test_onetrans.py -v
"""

import pytest
import torch
import torch.nn as nn

from onetrans import (
    RMSNorm,
    AutoSplitNSTokenizer,
    SequentialTokenizer,
    MixedCausalAttention,
    MixedFFN,
    OneTransBlock,
    OneTrans,
    _pyramid_query_counts,
)

B = 4   # batch size used throughout


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(32)
        x = torch.randn(B, 8, 32)
        assert norm(x).shape == x.shape

    def test_unit_rms_after_norm(self):
        norm = RMSNorm(64)
        norm.weight.data.fill_(1.0)
        x = torch.randn(B, 10, 64)
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_weight_scales_output(self):
        norm = RMSNorm(16)
        norm.weight.data.fill_(2.0)
        x = torch.randn(B, 4, 16)
        out = norm(x)
        norm.weight.data.fill_(1.0)
        out_unit = norm(x)
        assert torch.allclose(out, 2.0 * out_unit, atol=1e-5)

    def test_gradient_flows(self):
        norm = RMSNorm(32)
        x = torch.randn(B, 5, 32, requires_grad=True)
        norm(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# AutoSplitNSTokenizer
# ---------------------------------------------------------------------------

class TestAutoSplitNSTokenizer:
    def test_output_shape(self):
        tok = AutoSplitNSTokenizer(ns_input_dim=64, d_model=32, L_NS=8)
        x = torch.randn(B, 64)
        out = tok(x)
        assert out.shape == (B, 8, 32)

    def test_different_L_NS(self):
        for L in [4, 8, 16]:
            tok = AutoSplitNSTokenizer(ns_input_dim=32, d_model=16, L_NS=L)
            assert tok(torch.randn(B, 32)).shape == (B, L, 16)

    def test_gradient_flows(self):
        tok = AutoSplitNSTokenizer(ns_input_dim=64, d_model=32, L_NS=8)
        x = torch.randn(B, 64, requires_grad=True)
        tok(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# SequentialTokenizer
# ---------------------------------------------------------------------------

class TestSequentialTokenizer:
    def test_output_shape_single_seq(self):
        tok = SequentialTokenizer(seq_dims=[32], d_model=16)
        seqs = [torch.randn(B, 10, 32)]
        out = tok(seqs)
        assert out.shape == (B, 10, 16)

    def test_output_shape_multi_seq(self):
        # Two sequences L=5 and L=7, plus 1 SEP → L_S = 5 + 1 + 7 = 13
        tok = SequentialTokenizer(seq_dims=[32, 64], d_model=16)
        seqs = [torch.randn(B, 5, 32), torch.randn(B, 7, 64)]
        out = tok(seqs)
        assert out.shape == (B, 13, 16)

    def test_sep_token_count(self):
        # n sequences → n-1 SEP tokens
        tok = SequentialTokenizer(seq_dims=[8, 8, 8], d_model=16)
        seqs = [torch.randn(B, 3, 8) for _ in range(3)]
        out = tok(seqs)
        # 3 seqs × 3 tokens + 2 SEP = 11
        assert out.shape == (B, 11, 16)

    def test_type_ids(self):
        tok = SequentialTokenizer(seq_dims=[32, 32], d_model=16)
        seqs = [torch.randn(B, 5, 32), torch.randn(B, 5, 32)]
        out_with    = tok(seqs, type_ids=[0, 1])
        out_without = tok(seqs)
        # Type embeddings shift outputs
        assert not torch.allclose(out_with[:, :5], out_without[:, :5])

    def test_gradient_flows(self):
        tok = SequentialTokenizer(seq_dims=[32], d_model=16)
        x = torch.randn(B, 5, 32, requires_grad=True)
        tok([x]).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# MixedCausalAttention
# ---------------------------------------------------------------------------

class TestMixedCausalAttention:
    def _make(self, d=32, h=4, L_NS=4):
        return MixedCausalAttention(d_model=d, n_heads=h, L_NS=L_NS)

    def test_output_shape_no_pyramid(self):
        attn = self._make()
        L_S, L_NS = 10, 4
        x = torch.randn(B, L_S + L_NS, 32)
        out = attn(x, L_S=L_S, query_start=0)
        assert out.shape == (B, L_S + L_NS, 32)

    def test_output_shape_pyramid(self):
        # query_start=6 means L_q = L_S - 6 = 4 S-token queries
        attn = self._make()
        L_S, L_NS = 10, 4
        x = torch.randn(B, L_S + L_NS, 32)
        out = attn(x, L_S=L_S, query_start=6)
        assert out.shape == (B, 4 + L_NS, 32)

    def test_causal_no_future_leakage(self):
        # Verify first S-token output does not depend on later S-tokens.
        torch.manual_seed(0)
        attn = self._make(d=16, h=4, L_NS=2)
        L_S = 6
        x = torch.randn(B, L_S + 2, 16)
        # Perturb last S-token
        x2 = x.clone()
        x2[:, L_S - 1, :] += 10.0
        out1 = attn(x, L_S=L_S, query_start=0)
        out2 = attn(x2, L_S=L_S, query_start=0)
        # First S-token (position 0) should be unaffected
        assert torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-5)

    def test_gradient_flows(self):
        attn = self._make()
        x = torch.randn(B, 14, 32, requires_grad=True)
        attn(x, L_S=10, query_start=0).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# MixedFFN
# ---------------------------------------------------------------------------

class TestMixedFFN:
    def test_output_shape(self):
        ffn = MixedFFN(d_model=32, L_NS=4)
        x = torch.randn(B, 14, 32)  # 10 S + 4 NS
        out = ffn(x, L_S_q=10)
        assert out.shape == (B, 14, 32)

    def test_output_shape_pyramid(self):
        ffn = MixedFFN(d_model=32, L_NS=4)
        x = torch.randn(B, 8, 32)   # 4 S-query + 4 NS
        out = ffn(x, L_S_q=4)
        assert out.shape == (B, 8, 32)

    def test_s_and_ns_use_different_weights(self):
        # Same input values at S vs NS positions → different outputs (different weights)
        torch.manual_seed(0)
        ffn = MixedFFN(d_model=16, L_NS=2)
        x = torch.ones(1, 4, 16)  # 2 S-query + 2 NS, all same values
        out = ffn(x, L_S_q=2)
        assert not torch.allclose(out[0, 0], out[0, 2])  # S vs NS output differs

    def test_gradient_flows(self):
        ffn = MixedFFN(d_model=32, L_NS=4)
        x = torch.randn(B, 14, 32, requires_grad=True)
        ffn(x, L_S_q=10).sum().backward()
        assert x.grad is not None
        assert ffn.W_S1.weight.grad is not None
        assert ffn.W_NS1.grad is not None


# ---------------------------------------------------------------------------
# Pyramid query counts
# ---------------------------------------------------------------------------

class TestPyramidQueryCounts:
    def test_length(self):
        counts = _pyramid_query_counts(L_S=32, L_NS=8, n_layers=4)
        assert len(counts) == 4

    def test_first_is_L_S(self):
        counts = _pyramid_query_counts(L_S=32, L_NS=8, n_layers=4, multiple_of=8)
        assert counts[0] == 32

    def test_last_is_L_NS(self):
        counts = _pyramid_query_counts(L_S=32, L_NS=8, n_layers=4)
        assert counts[-1] == 8

    def test_monotone_decreasing(self):
        counts = _pyramid_query_counts(L_S=64, L_NS=8, n_layers=6, multiple_of=8)
        assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))

    def test_all_at_least_L_NS(self):
        counts = _pyramid_query_counts(L_S=32, L_NS=8, n_layers=5)
        assert all(c >= 8 for c in counts)

    def test_single_layer(self):
        counts = _pyramid_query_counts(L_S=32, L_NS=8, n_layers=1)
        assert counts == [8]


# ---------------------------------------------------------------------------
# OneTransBlock
# ---------------------------------------------------------------------------

class TestOneTransBlock:
    def test_output_shape_no_pyramid(self):
        block = OneTransBlock(d_model=32, n_heads=4, L_NS=4)
        x = torch.randn(B, 14, 32)
        out = block(x, L_S=10, query_start=0)
        assert out.shape == (B, 14, 32)

    def test_output_shape_with_pyramid(self):
        block = OneTransBlock(d_model=32, n_heads=4, L_NS=4)
        x = torch.randn(B, 14, 32)
        # query_start=6 → L_q=4 S-queries + 4 NS = 8 tokens out
        out = block(x, L_S=10, query_start=6)
        assert out.shape == (B, 8, 32)

    def test_gradient_flows(self):
        block = OneTransBlock(d_model=16, n_heads=4, L_NS=4)
        x = torch.randn(B, 10, 16, requires_grad=True)
        block(x, L_S=6, query_start=0).sum().backward()
        assert x.grad is not None
        assert block.norm1.weight.grad is not None
        assert block.norm2.weight.grad is not None


# ---------------------------------------------------------------------------
# OneTrans (full model)
# ---------------------------------------------------------------------------

class TestOneTrans:
    def _make_model(self, **kwargs):
        defaults = dict(
            seq_dims=[32, 32],
            ns_input_dim=64,
            d_model=32,
            n_heads=4,
            n_layers=4,
            L_NS=8,
            max_seq_len=64,
            num_tasks=2,
            ffn_expand=2,
            pyramid_mult=8,
        )
        defaults.update(kwargs)
        return OneTrans(**defaults)

    def _make_inputs(self, seq_lens=(5, 7), seq_dim=32, ns_dim=64):
        seqs = [torch.randn(B, L, seq_dim) for L in seq_lens]
        ns   = torch.randn(B, ns_dim)
        return seqs, ns

    def test_output_shape(self):
        model = self._make_model()
        seqs, ns = self._make_inputs()
        logits = model(seqs, ns)
        assert logits.shape == (B, 2)

    def test_single_task(self):
        model = self._make_model(num_tasks=1)
        seqs, ns = self._make_inputs()
        logits = model(seqs, ns)
        assert logits.shape == (B, 1)

    def test_three_tasks(self):
        model = self._make_model(num_tasks=3)
        seqs, ns = self._make_inputs()
        logits = model(seqs, ns)
        assert logits.shape == (B, 3)

    def test_single_sequence(self):
        model = self._make_model(seq_dims=[32])
        seqs = [torch.randn(B, 10, 32)]
        ns   = torch.randn(B, 64)
        logits = model(seqs, ns)
        assert logits.shape == (B, 2)

    def test_type_ids(self):
        model = self._make_model()
        seqs, ns = self._make_inputs()
        logits_typed   = model(seqs, ns, type_ids=[0, 1])
        logits_untyped = model(seqs, ns)
        assert not torch.allclose(logits_typed, logits_untyped)

    def test_backward(self):
        model = self._make_model()
        seqs, ns = self._make_inputs()
        logits = model(seqs, ns)
        logits.sum().backward()
        first_param = next(model.parameters())
        assert first_param.grad is not None

    def test_grad_exists_for_all_parameters(self):
        model = self._make_model()
        seqs, ns = self._make_inputs()
        # Pass type_ids so that type_emb.weight participates in the forward pass.
        model(seqs, ns, type_ids=[0, 1]).sum().backward()
        no_grad = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is None]
        assert no_grad == [], f"Parameters missing gradient: {no_grad}"

    def test_deterministic_with_seed(self):
        torch.manual_seed(42)
        model = self._make_model()
        seqs, ns = self._make_inputs()
        torch.manual_seed(7)
        out1 = model(seqs, ns)
        torch.manual_seed(7)
        out2 = model(seqs, ns)
        assert torch.allclose(out1, out2)

    def test_different_inputs_different_outputs(self):
        model = self._make_model()
        seqs1, ns1 = self._make_inputs()
        seqs2, ns2 = self._make_inputs()
        assert not torch.allclose(model(seqs1, ns1), model(seqs2, ns2))

    def test_variable_sequence_length(self):
        # Model should handle different L_i at inference
        model = self._make_model(max_seq_len=128)
        for lens in [(3, 5), (10, 20), (1, 1)]:
            seqs = [torch.randn(B, L, 32) for L in lens]
            ns   = torch.randn(B, 64)
            logits = model(seqs, ns)
            assert logits.shape == (B, 2)

    def test_many_sequences(self):
        model = self._make_model(seq_dims=[16] * 5, max_seq_len=128)
        seqs = [torch.randn(B, 5, 16) for _ in range(5)]
        ns   = torch.randn(B, 64)
        assert model(seqs, ns).shape == (B, 2)
