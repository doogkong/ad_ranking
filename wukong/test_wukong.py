"""Tests for Wukong implementation.

Run with:
    pytest test_wukong.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from wukong import (
    FeatureGroupEmbedding,
    EmbeddingLayer,
    OptimizedFM,
    FMBlock,
    LCBBlock,
    WukongLayer,
    Wukong,
    wukong_small,
    wukong_medium,
)

B = 4    # batch size
D = 32   # model dim
N = 6    # number of feature groups / tokens


def _groups(dims=None, b=B):
    dims = dims or [16] * N
    return [torch.randn(b, d) for d in dims]


# ---------------------------------------------------------------------------
# FeatureGroupEmbedding
# ---------------------------------------------------------------------------

class TestFeatureGroupEmbedding:
    def test_output_shape(self):
        proj = FeatureGroupEmbedding(64, D)
        assert proj(torch.randn(B, 64)).shape == (B, D)

    def test_different_input_dims(self):
        for dim in [8, 32, 128, 512]:
            proj = FeatureGroupEmbedding(dim, D)
            assert proj(torch.randn(B, dim)).shape == (B, D)

    def test_gradient(self):
        proj = FeatureGroupEmbedding(32, D)
        x = torch.randn(B, 32, requires_grad=True)
        proj(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# EmbeddingLayer
# ---------------------------------------------------------------------------

class TestEmbeddingLayer:
    def test_output_shape(self):
        dims = [16, 32, 64]
        emb = EmbeddingLayer(dims, D)
        out = emb(_groups(dims))
        assert out.shape == (B, 3, D)

    def test_num_tokens(self):
        dims = [8] * 10
        emb = EmbeddingLayer(dims, D)
        assert emb.n == 10
        assert emb(_groups(dims)).shape == (B, 10, D)

    def test_gradient_flows(self):
        dims = [16, 32]
        emb = EmbeddingLayer(dims, D)
        inputs = [torch.randn(B, d, requires_grad=True) for d in dims]
        emb(inputs).sum().backward()
        for x in inputs:
            assert x.grad is not None

    def test_different_groups_different_projections(self):
        # Same raw values through different group MLPs → different tokens
        emb = EmbeddingLayer([16, 16], D)
        x = torch.randn(B, 16)
        out = emb([x, x])
        assert not torch.allclose(out[:, 0], out[:, 1])


# ---------------------------------------------------------------------------
# OptimizedFM
# ---------------------------------------------------------------------------

class TestOptimizedFM:
    def test_output_shape(self):
        fm = OptimizedFM(n_tokens=N, d_model=D, k=8)
        X = torch.randn(B, N, D)
        assert fm(X).shape == (B, N, 8)

    def test_different_k(self):
        for k in [4, 8, 16, 32]:
            fm = OptimizedFM(N, D, k)
            assert fm(torch.randn(B, N, D)).shape == (B, N, k)

    def test_attentive_projection_varies_with_input(self):
        fm = OptimizedFM(N, D, k=8)
        X1, X2 = torch.randn(B, N, D), torch.randn(B, N, D)
        # Different inputs should produce different FM outputs (attentive Y)
        assert not torch.allclose(fm(X1), fm(X2))

    def test_gradient(self):
        fm = OptimizedFM(N, D, k=8)
        X = torch.randn(B, N, D, requires_grad=True)
        fm(X).sum().backward()
        assert X.grad is not None


# ---------------------------------------------------------------------------
# FMBlock
# ---------------------------------------------------------------------------

class TestFMBlock:
    def test_output_shape(self):
        fmb = FMBlock(n_tokens=N, d_model=D, n_F=8, k=8, mlp_dims=[64])
        X = torch.randn(B, N, D)
        assert fmb(X).shape == (B, 8, D)

    def test_different_n_F(self):
        for nf in [4, 8, 16]:
            fmb = FMBlock(N, D, n_F=nf, k=8)
            assert fmb(torch.randn(B, N, D)).shape == (B, nf, D)

    def test_gradient(self):
        fmb = FMBlock(N, D, n_F=8, k=8)
        X = torch.randn(B, N, D, requires_grad=True)
        fmb(X).sum().backward()
        assert X.grad is not None
        assert fmb.fm.attn_proj.weight.grad is not None


# ---------------------------------------------------------------------------
# LCBBlock
# ---------------------------------------------------------------------------

class TestLCBBlock:
    def test_output_shape(self):
        lcb = LCBBlock(n_in=N, n_L=8)
        X = torch.randn(B, N, D)
        assert lcb(X).shape == (B, 8, D)

    def test_preserves_d_model(self):
        for n_L in [4, 6, 10]:
            lcb = LCBBlock(N, n_L)
            out = lcb(torch.randn(B, N, D))
            assert out.shape == (B, n_L, D)

    def test_linear_recombination(self):
        # LCB is a linear transform over tokens — scaling input should scale output
        lcb = LCBBlock(N, 4)
        X = torch.randn(B, N, D)
        assert torch.allclose(lcb(2 * X), 2 * lcb(X), atol=1e-5)

    def test_gradient(self):
        lcb = LCBBlock(N, 8)
        X = torch.randn(B, N, D, requires_grad=True)
        lcb(X).sum().backward()
        assert X.grad is not None
        assert lcb.W.weight.grad is not None


# ---------------------------------------------------------------------------
# WukongLayer
# ---------------------------------------------------------------------------

class TestWukongLayer:
    def _make(self, n_in=N, n_F=8, n_L=8):
        return WukongLayer(n_in=n_in, d_model=D, n_F=n_F, n_L=n_L, k=8, mlp_dims=[64])

    def test_output_shape(self):
        layer = self._make()
        X = torch.randn(B, N, D)
        out = layer(X)
        assert out.shape == (B, 16, D)   # n_F + n_L = 8 + 8

    def test_output_shape_first_layer_different_n(self):
        # First layer: n_in (=N) often ≠ n_F + n_L
        layer = self._make(n_in=N, n_F=10, n_L=6)
        out = layer(torch.randn(B, N, D))
        assert out.shape == (B, 16, D)

    def test_subsequent_layer_same_shape(self):
        # Layers 2+ have n_in = n_F + n_L = 16
        layer = self._make(n_in=16)
        out = layer(torch.randn(B, 16, D))
        assert out.shape == (B, 16, D)

    def test_residual_stabilises_output(self):
        # With random init, output should be reasonable (not explode)
        torch.manual_seed(0)
        layer = self._make()
        X = torch.randn(B, N, D)
        out = layer(X)
        assert out.abs().max().item() < 100.0

    def test_gradient(self):
        layer = self._make()
        X = torch.randn(B, N, D, requires_grad=True)
        layer(X).sum().backward()
        assert X.grad is not None

    def test_layer_norm_applied(self):
        # LN normalises: output per-token rms should be close to 1
        layer = self._make()
        X = torch.randn(B, N, D) * 100   # large input
        out = layer(X)
        # With LN weight=1, mean≈0, var≈1 across d_model
        std = out.std(dim=-1)
        assert std.mean().item() < 5.0   # not exploding


# ---------------------------------------------------------------------------
# Wukong (full model)
# ---------------------------------------------------------------------------

class TestWukong:
    def _make(self, feature_dims=None, **kwargs):
        dims = feature_dims or [16] * N
        defaults = dict(
            d_model=D,
            num_layers=3,
            n_F=8,
            n_L=8,
            k=8,
            mlp_dims=[64],
            top_mlp_dims=[32],
            num_tasks=2,
        )
        defaults.update(kwargs)
        return Wukong(dims, **defaults)

    def test_output_shape(self):
        model = self._make()
        logits = model(_groups())
        assert logits.shape == (B, 2)

    def test_single_task(self):
        model = self._make(num_tasks=1)
        logits = model(_groups())
        assert logits.shape == (B, 1)

    def test_three_tasks(self):
        model = self._make(num_tasks=3)
        assert model(_groups()).shape == (B, 3)

    def test_single_feature_group(self):
        model = self._make(feature_dims=[64], num_tasks=1)
        out = model([torch.randn(B, 64)])
        assert out.shape == (B, 1)

    def test_many_feature_groups(self):
        dims = [32] * 20
        model = self._make(feature_dims=dims)
        assert model(_groups(dims)).shape == (B, 2)

    def test_backward(self):
        model = self._make()
        logits = model(_groups())
        F.binary_cross_entropy_with_logits(logits, torch.zeros(B, 2)).backward()
        assert next(model.parameters()).grad is not None

    def test_grad_all_parameters(self):
        model = self._make()
        model(_groups()).sum().backward()
        no_grad = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is None]
        assert no_grad == [], f"Missing grad: {no_grad}"

    def test_deterministic_with_seed(self):
        model = self._make()
        g = _groups()
        torch.manual_seed(1)
        out1 = model(g)
        torch.manual_seed(1)
        out2 = model(g)
        assert torch.allclose(out1, out2)

    def test_different_inputs_different_outputs(self):
        model = self._make()
        assert not torch.allclose(model(_groups()), model(_groups()))

    def test_num_layers_scaling(self):
        # More layers → more parameters (FMB grows the stack)
        dims = [16] * 4
        m2 = Wukong(dims, d_model=D, num_layers=2, n_F=8, n_L=8, k=8)
        m4 = Wukong(dims, d_model=D, num_layers=4, n_F=8, n_L=8, k=8)
        params2 = sum(p.numel() for p in m2.parameters())
        params4 = sum(p.numel() for p in m4.parameters())
        assert params4 > params2

    def test_interaction_order_growth(self):
        # Deeper model should differ more from shallow on high-interaction input
        # (existence test: both produce valid outputs)
        dims = [16] * 4
        shallow = Wukong(dims, d_model=D, num_layers=1, n_F=8, n_L=8, k=8, num_tasks=1)
        deep    = Wukong(dims, d_model=D, num_layers=6, n_F=8, n_L=8, k=8, num_tasks=1)
        g = _groups(dims)
        assert shallow(g).shape == (B, 1)
        assert deep(g).shape == (B, 1)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

class TestFactories:
    def test_wukong_small(self):
        model = wukong_small([32, 32, 32])
        out = model([torch.randn(B, 32)] * 3)
        assert out.shape == (B, 1)

    def test_wukong_medium(self):
        model = wukong_medium([32, 32])
        out = model([torch.randn(B, 32)] * 2)
        assert out.shape == (B, 1)
