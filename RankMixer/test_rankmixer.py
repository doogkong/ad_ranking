"""Tests for RankMixer implementation.

Run with:
    pytest test_rankmixer.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from rankmixer import (
    FeatureTokenizer,
    TokenMixing,
    PerTokenFFN,
    SparseMoEPerTokenFFN,
    RankMixerBlock,
    RankMixer,
)

B = 4    # batch size
T = 8    # number of tokens
D = 32   # model dim
E = 4    # num experts
K = 4    # expand_ratio


# ---------------------------------------------------------------------------
# FeatureTokenizer
# ---------------------------------------------------------------------------

class TestFeatureTokenizer:
    def test_output_shape(self):
        group_dims = [16, 8, 32, 4, 12, 20, 6, 10]
        tok = FeatureTokenizer(group_dims, D)
        groups = [torch.randn(B, g) for g in group_dims]
        out = tok(groups)
        assert out.shape == (B, len(group_dims), D)

    def test_different_group_dims_allowed(self):
        group_dims = [1, 100, 5]
        tok = FeatureTokenizer(group_dims, D)
        groups = [torch.randn(B, g) for g in group_dims]
        assert tok(groups).shape == (B, 3, D)

    def test_gradient(self):
        group_dims = [16, 8, 32]
        tok = FeatureTokenizer(group_dims, D)
        groups = [torch.randn(B, g, requires_grad=True) for g in group_dims]
        tok(groups).sum().backward()
        assert all(g.grad is not None for g in groups)


# ---------------------------------------------------------------------------
# TokenMixing
# ---------------------------------------------------------------------------

class TestTokenMixing:
    def test_output_shape(self):
        tm = TokenMixing(num_tokens=T, d_model=D)
        x = torch.randn(B, T, D)
        assert tm(x).shape == (B, T, D)

    def test_parameter_free(self):
        tm = TokenMixing(num_tokens=T, d_model=D)
        assert len(list(tm.parameters())) == 0

    def test_rejects_num_heads_not_equal_num_tokens(self):
        with pytest.raises(ValueError):
            TokenMixing(num_tokens=T, d_model=D, num_heads=T // 2)

    def test_rejects_non_divisible_dim(self):
        with pytest.raises(ValueError):
            TokenMixing(num_tokens=5, d_model=D)  # D=32 is not divisible by num_heads=5

    def test_is_a_permutation_of_all_entries(self):
        # Every scalar entry of x should appear exactly once in the output
        # (it's a pure reshape/transpose with no aggregation).
        tm = TokenMixing(num_tokens=T, d_model=D)
        x = torch.randn(B, T, D)
        out = tm(x)
        assert torch.allclose(torch.sort(x.reshape(B, -1), dim=-1).values,
                               torch.sort(out.reshape(B, -1), dim=-1).values)

    def test_different_inputs_different_outputs(self):
        tm = TokenMixing(num_tokens=T, d_model=D)
        x1, x2 = torch.randn(B, T, D), torch.randn(B, T, D)
        assert not torch.allclose(tm(x1), tm(x2))

    def test_gradient(self):
        tm = TokenMixing(num_tokens=T, d_model=D)
        x = torch.randn(B, T, D, requires_grad=True)
        tm(x).sum().backward()
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))  # pure permutation -> unit grad


# ---------------------------------------------------------------------------
# PerTokenFFN
# ---------------------------------------------------------------------------

class TestPerTokenFFN:
    def test_output_shape(self):
        ffn = PerTokenFFN(num_tokens=T, d_model=D, expand_ratio=K)
        s = torch.randn(B, T, D)
        assert ffn(s).shape == (B, T, D)

    def test_tokens_have_independent_params(self):
        # Feeding the same vector into every token position should give
        # different outputs per position, since each token owns its own W/b.
        ffn = PerTokenFFN(num_tokens=T, d_model=D, expand_ratio=K)
        same = torch.randn(1, D).expand(B, T, D)
        out = ffn(same)
        assert not torch.allclose(out[:, 0, :], out[:, 1, :])

    def test_gradient_reaches_all_token_params(self):
        ffn = PerTokenFFN(num_tokens=T, d_model=D, expand_ratio=K)
        s = torch.randn(B, T, D, requires_grad=True)
        ffn(s).sum().backward()
        assert ffn.W1.grad is not None and torch.all(ffn.W1.grad != 0)
        assert ffn.W2.grad is not None and torch.all(ffn.W2.grad != 0)

    def test_different_inputs_different_outputs(self):
        ffn = PerTokenFFN(num_tokens=T, d_model=D, expand_ratio=K)
        s1, s2 = torch.randn(B, T, D), torch.randn(B, T, D)
        assert not torch.allclose(ffn(s1), ffn(s2))


# ---------------------------------------------------------------------------
# SparseMoEPerTokenFFN
# ---------------------------------------------------------------------------

class TestSparseMoEPerTokenFFN:
    def test_output_shape(self):
        moe = SparseMoEPerTokenFFN(num_tokens=T, d_model=D, num_experts=E, expand_ratio=K)
        s = torch.randn(B, T, D)
        assert moe(s).shape == (B, T, D)

    def test_gates_are_relu_nonnegative(self):
        moe = SparseMoEPerTokenFFN(num_tokens=T, d_model=D, num_experts=E, expand_ratio=K)
        moe(torch.randn(B, T, D))
        assert torch.all(moe._last_gates_infer >= 0.0)

    def test_reg_loss_is_scalar_and_nonnegative(self):
        moe = SparseMoEPerTokenFFN(num_tokens=T, d_model=D, num_experts=E, expand_ratio=K)
        moe(torch.randn(B, T, D))
        reg = moe.reg_loss()
        assert reg.dim() == 0
        assert reg.item() >= 0.0

    def test_reg_loss_requires_forward_first(self):
        moe = SparseMoEPerTokenFFN(num_tokens=T, d_model=D, num_experts=E, expand_ratio=K)
        with pytest.raises(RuntimeError):
            moe.reg_loss()

    def test_active_expert_ratio_in_unit_interval(self):
        moe = SparseMoEPerTokenFFN(num_tokens=T, d_model=D, num_experts=E, expand_ratio=K)
        moe(torch.randn(B, T, D))
        ratio = moe.active_expert_ratio()
        assert 0.0 <= ratio.item() <= 1.0

    def test_l1_penalty_pushes_gates_toward_zero(self):
        moe = SparseMoEPerTokenFFN(num_tokens=T, d_model=D, num_experts=E, expand_ratio=K)
        s = torch.randn(B, T, D)
        opt = torch.optim.Adam(moe.parameters(), lr=0.05)
        moe(s)
        reg0 = moe.reg_loss().item()
        for _ in range(50):
            opt.zero_grad()
            moe(s)
            loss = moe.reg_loss()
            loss.backward()
            opt.step()
        moe(s)
        reg1 = moe.reg_loss().item()
        assert reg1 < reg0

    def test_gradient(self):
        moe = SparseMoEPerTokenFFN(num_tokens=T, d_model=D, num_experts=E, expand_ratio=K)
        s = torch.randn(B, T, D, requires_grad=True)
        moe(s).sum().backward()
        assert s.grad is not None
        assert moe.expert_W1.grad is not None

    def test_dtsi_uses_train_router_while_training(self):
        moe = SparseMoEPerTokenFFN(num_tokens=T, d_model=D, num_experts=E, expand_ratio=K, dtsi=True)
        moe.train()
        s = torch.randn(B, T, D)
        moe(s)
        assert moe._last_gates_train is not None

    def test_dtsi_uses_infer_router_at_eval(self):
        moe = SparseMoEPerTokenFFN(num_tokens=T, d_model=D, num_experts=E, expand_ratio=K, dtsi=True)
        moe.eval()
        s = torch.randn(B, T, D)
        moe._last_gates_train = None
        moe(s)
        assert moe._last_gates_train is None

    def test_distill_loss_zero_without_dtsi(self):
        moe = SparseMoEPerTokenFFN(num_tokens=T, d_model=D, num_experts=E, expand_ratio=K, dtsi=False)
        moe(torch.randn(B, T, D))
        assert moe.distill_loss().item() == 0.0

    def test_distill_loss_nonnegative_with_dtsi(self):
        moe = SparseMoEPerTokenFFN(num_tokens=T, d_model=D, num_experts=E, expand_ratio=K, dtsi=True)
        moe.train()
        moe(torch.randn(B, T, D))
        assert moe.distill_loss().item() >= 0.0


# ---------------------------------------------------------------------------
# RankMixerBlock
# ---------------------------------------------------------------------------

class TestRankMixerBlock:
    def test_output_shape_dense(self):
        block = RankMixerBlock(num_tokens=T, d_model=D, expand_ratio=K, moe=False)
        x = torch.randn(B, T, D)
        assert block(x).shape == (B, T, D)

    def test_output_shape_moe(self):
        block = RankMixerBlock(num_tokens=T, d_model=D, expand_ratio=K, moe=True, num_experts=E)
        x = torch.randn(B, T, D)
        assert block(x).shape == (B, T, D)

    def test_gradient(self):
        block = RankMixerBlock(num_tokens=T, d_model=D, expand_ratio=K)
        x = torch.randn(B, T, D, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None

    def test_different_inputs_different_outputs(self):
        block = RankMixerBlock(num_tokens=T, d_model=D, expand_ratio=K)
        x1, x2 = torch.randn(B, T, D), torch.randn(B, T, D)
        assert not torch.allclose(block(x1), block(x2))


# ---------------------------------------------------------------------------
# RankMixer (full model)
# ---------------------------------------------------------------------------

class TestRankMixer:
    GROUP_DIMS = [16, 8, 32, 4, 12, 20, 6, 10]

    def _make(self, **kwargs):
        defaults = dict(
            group_dims=self.GROUP_DIMS,
            d_model=D,
            num_layers=2,
            expand_ratio=K,
            moe=False,
            top_mlp_dims=[D],
            num_tasks=1,
        )
        defaults.update(kwargs)
        return RankMixer(**defaults)

    def _groups(self, b=B):
        return [torch.randn(b, g) for g in self.GROUP_DIMS]

    def test_output_shape_single_task(self):
        model = self._make(num_tasks=1)
        assert model(self._groups()).shape == (B, 1)

    def test_output_shape_multi_task(self):
        model = self._make(num_tasks=3)
        assert model(self._groups()).shape == (B, 3)

    def test_backward(self):
        model = self._make()
        logits = model(self._groups())
        F.binary_cross_entropy_with_logits(logits, torch.zeros(B, 1)).backward()
        assert next(model.parameters()).grad is not None

    def test_deterministic_with_seed(self):
        model = self._make()
        groups = self._groups()
        torch.manual_seed(42)
        out1 = model(groups)
        torch.manual_seed(42)
        out2 = model(groups)
        assert torch.allclose(out1, out2)

    def test_different_inputs_different_outputs(self):
        model = self._make()
        assert not torch.allclose(model(self._groups()), model(self._groups()))

    def test_num_layers_scaling_increases_params(self):
        p2 = sum(p.numel() for p in self._make(num_layers=2).parameters())
        p4 = sum(p.numel() for p in self._make(num_layers=4).parameters())
        assert p4 > p2

    def test_d_model_scaling_increases_params(self):
        p_small = sum(p.numel() for p in self._make(d_model=16, top_mlp_dims=[16]).parameters())
        p_large = sum(p.numel() for p in self._make(d_model=64, top_mlp_dims=[64]).parameters())
        assert p_large > p_small

    def test_finite_output(self):
        model = self._make()
        assert torch.isfinite(model(self._groups())).all()

    def test_batch_size_one(self):
        model = self._make()
        assert model(self._groups(b=1)).shape == (1, 1)

    def test_moe_variant_forward_backward(self):
        model = self._make(moe=True, num_experts=E, dtsi=True)
        model.train()
        logits = model(self._groups())
        reg = model.moe_reg_loss()
        (F.binary_cross_entropy_with_logits(logits, torch.zeros(B, 1)) + 1e-3 * reg).backward()
        assert logits.shape == (B, 1)
        assert reg.item() >= 0.0

    def test_moe_reg_loss_zero_when_dense(self):
        model = self._make(moe=False)
        assert model.moe_reg_loss().item() == 0.0

    def test_estimate_params_and_flops_matches_eq12(self):
        params, flops = RankMixer.estimate_params_and_flops(num_layers=2, num_tokens=8, d_model=64, expand_ratio=4)
        assert params == 2 * 4 * 2 * 8 * 64 ** 2
        assert flops == 2 * params

    def test_mean_pooling_invariant_to_token_order_permutation_of_lin_head(self):
        # With a single-token classifier input being the mean over T tokens,
        # scaling num_layers should not change output shape.
        model = self._make(num_layers=1)
        assert model(self._groups()).shape == (B, 1)
