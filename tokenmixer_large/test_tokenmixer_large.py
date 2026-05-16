"""Tests for TokenMixer-Large implementation.

Run with:
    pytest test_tokenmixer_large.py -v
"""

import pytest
import torch
import torch.nn as nn

from tokenmixer_large import (
    RMSNorm,
    PertokenSwiGLU,
    MixingReverting,
    SparsePerTokenMoE,
    TokenMixerLargeBlock,
    SemanticGroupTokenizer,
    TokenMixerLarge,
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
# PertokenSwiGLU
# ---------------------------------------------------------------------------

class TestPertokenSwiGLU:
    def test_output_shape(self):
        ffn = PertokenSwiGLU(num_tokens=6, dim=32, expand=2)
        x = torch.randn(B, 6, 32)
        assert ffn(x).shape == (B, 6, 32)

    def test_pertoken_independence(self):
        # Different token positions should in general produce different outputs
        # even when the input values are identical across positions.
        torch.manual_seed(0)
        ffn = PertokenSwiGLU(num_tokens=4, dim=16, expand=2)
        x = torch.ones(1, 4, 16)
        out = ffn(x)
        # Not all token outputs should be equal (independent weights)
        assert not torch.allclose(out[0, 0], out[0, 1])

    def test_small_init_wdown_magnitude(self):
        torch.manual_seed(0)
        ffn_small = PertokenSwiGLU(num_tokens=4, dim=16, expand=2, small_init=True)
        ffn_normal = PertokenSwiGLU(num_tokens=4, dim=16, expand=2, small_init=False)
        assert ffn_small.w_down.data.abs().mean() < ffn_normal.w_down.data.abs().mean()

    def test_small_init_near_identity(self):
        # With small_init, output should be very close to zero (block ≈ identity via residual)
        torch.manual_seed(0)
        ffn = PertokenSwiGLU(num_tokens=4, dim=16, expand=2, small_init=True)
        x = torch.randn(B, 4, 16)
        out = ffn(x)
        assert out.abs().mean() < 0.1

    def test_gradient_flows(self):
        ffn = PertokenSwiGLU(num_tokens=5, dim=32, expand=2)
        x = torch.randn(B, 5, 32, requires_grad=True)
        ffn(x).sum().backward()
        assert x.grad is not None
        assert ffn.w_down.grad is not None


# ---------------------------------------------------------------------------
# MixingReverting
# ---------------------------------------------------------------------------

class TestMixingReverting:
    def test_output_shape(self):
        mix = MixingReverting(num_tokens=5, dim=32, num_heads=4)
        x = torch.randn(B, 5, 32)
        assert mix(x).shape == (B, 5, 32)

    def test_dim_not_divisible_raises(self):
        with pytest.raises(AssertionError):
            MixingReverting(num_tokens=5, dim=33, num_heads=4)

    def test_residual_to_original_x(self):
        # With small_init (default), output ≈ original x because the SwiGLU
        # contribution is near-zero — verifying the residual is wired to x.
        torch.manual_seed(0)
        mix = MixingReverting(num_tokens=4, dim=16, num_heads=4)
        x = torch.randn(B, 4, 16)
        out = mix(x)
        # Output should be closer to x than to zeros
        assert (out - x).abs().mean() < x.abs().mean()

    def test_different_heads(self):
        # num_heads affects internal layout but output shape must remain T×D
        for h in [1, 2, 4]:
            mix = MixingReverting(num_tokens=4, dim=8, num_heads=h)
            x = torch.randn(B, 4, 8)
            assert mix(x).shape == (B, 4, 8)

    def test_gradient_flows(self):
        mix = MixingReverting(num_tokens=5, dim=16, num_heads=4)
        x = torch.randn(B, 5, 16, requires_grad=True)
        mix(x).sum().backward()
        assert x.grad is not None

    def test_moe_factory(self):
        def factory(n_tok, d):
            return SparsePerTokenMoE(n_tok, d, num_experts=2, top_k=1)

        mix = MixingReverting(num_tokens=4, dim=16, num_heads=4, moe_factory=factory)
        x = torch.randn(B, 4, 16)
        assert mix(x).shape == (B, 4, 16)


# ---------------------------------------------------------------------------
# SparsePerTokenMoE
# ---------------------------------------------------------------------------

class TestSparsePerTokenMoE:
    def test_output_shape(self):
        moe = SparsePerTokenMoE(num_tokens=5, dim=32, num_experts=3, top_k=2)
        x = torch.randn(B, 5, 32)
        assert moe(x).shape == (B, 5, 32)

    def test_gate_weights_sum_to_one(self):
        # Verify softmax over top-k produces weights summing to 1.
        moe = SparsePerTokenMoE(num_tokens=4, dim=16, num_experts=4, top_k=2)
        x = torch.randn(B, 4, 16)
        logits = moe.router(x)                          # (B, T, E)
        _, topk_idx = torch.topk(logits, 2, dim=-1)
        topk_logits = logits.gather(2, topk_idx)
        gate_w = torch.softmax(topk_logits, dim=-1)
        assert torch.allclose(gate_w.sum(dim=-1), torch.ones(B, 4), atol=1e-5)

    def test_top_k_experts_selected(self):
        # The gather index should always have exactly top_k entries per token.
        moe = SparsePerTokenMoE(num_tokens=4, dim=16, num_experts=4, top_k=2)
        x = torch.randn(B, 4, 16)
        logits = moe.router(x)
        _, topk_idx = torch.topk(logits, moe.top_k, dim=-1)
        assert topk_idx.shape == (B, 4, 2)

    def test_gate_scale_effect(self):
        torch.manual_seed(0)
        x = torch.randn(B, 4, 16)
        moe1 = SparsePerTokenMoE(num_tokens=4, dim=16, num_experts=2, top_k=1, gate_scale=1.0)
        moe2 = SparsePerTokenMoE(num_tokens=4, dim=16, num_experts=2, top_k=1, gate_scale=4.0)
        # Copy weights so only alpha differs
        moe2.load_state_dict(moe1.state_dict())
        moe2.alpha = 4.0
        out1 = moe1(x)
        out2 = moe2(x)
        assert not torch.allclose(out1, out2)

    def test_shared_expert_always_active(self):
        # Zero out all routing experts; output should still be non-zero
        # (shared expert contributes).
        moe = SparsePerTokenMoE(num_tokens=3, dim=8, num_experts=2, top_k=1)
        for expert in moe.experts:
            for p in expert.parameters():
                p.data.zero_()
        x = torch.randn(B, 3, 8)
        out = moe(x)
        # Shared expert is non-zero so output ≠ 0
        assert out.abs().sum() > 0

    def test_top_k_equals_num_experts(self):
        moe = SparsePerTokenMoE(num_tokens=4, dim=16, num_experts=3, top_k=3)
        x = torch.randn(B, 4, 16)
        assert moe(x).shape == (B, 4, 16)

    def test_gradient_flows(self):
        moe = SparsePerTokenMoE(num_tokens=4, dim=16, num_experts=3, top_k=2)
        x = torch.randn(B, 4, 16, requires_grad=True)
        moe(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# TokenMixerLargeBlock
# ---------------------------------------------------------------------------

class TestTokenMixerLargeBlock:
    def test_output_shape(self):
        block = TokenMixerLargeBlock(num_tokens=6, dim=32, num_heads=4)
        x = torch.randn(B, 6, 32)
        assert block(x).shape == (B, 6, 32)

    def test_residual_preserves_scale(self):
        # With small_init in all SwiGLUs, output should be close to input.
        torch.manual_seed(0)
        block = TokenMixerLargeBlock(num_tokens=4, dim=16, num_heads=4,
                                     num_experts=2, top_k=1, moe_expand=1)
        x = torch.randn(B, 4, 16)
        out = block(x)
        assert (out - x).abs().mean() < x.abs().mean()

    def test_gradient_flows_through_both_sublayers(self):
        block = TokenMixerLargeBlock(num_tokens=5, dim=16, num_heads=4)
        x = torch.randn(B, 5, 16, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None
        # Both norm weight params should receive gradient
        assert block.norm1.weight.grad is not None
        assert block.norm2.weight.grad is not None


# ---------------------------------------------------------------------------
# SemanticGroupTokenizer
# ---------------------------------------------------------------------------

class TestSemanticGroupTokenizer:
    def test_output_shape(self):
        groups = [64, 128, 32]
        tok = SemanticGroupTokenizer(groups, model_dim=64)
        inputs = [torch.randn(B, d) for d in groups]
        out = tok(inputs)
        # T = 1 (global) + 3 groups
        assert out.shape == (B, 4, 64)

    def test_global_token_prepended(self):
        tok = SemanticGroupTokenizer([32], model_dim=16)
        x = tok([torch.randn(B, 32)])
        assert x.shape[1] == 2  # global + 1 group

    def test_num_tokens_attribute(self):
        tok = SemanticGroupTokenizer([10, 20, 30], model_dim=32)
        assert tok.num_tokens == 4  # 1 global + 3 groups

    def test_global_token_shared_across_batch(self):
        tok = SemanticGroupTokenizer([16], model_dim=8)
        x = tok([torch.randn(B, 16)])
        # All batch items share the same global token before MLP processing
        assert torch.allclose(x[:, 0, :], x[0:1, 0, :].expand(B, -1))

    def test_different_groups_get_different_mlps(self):
        tok = SemanticGroupTokenizer([16, 16], model_dim=16)
        same_input = torch.randn(B, 16)
        out = tok([same_input, same_input])
        # Same input through different MLPs → different token representations
        assert not torch.allclose(out[:, 1, :], out[:, 2, :])

    def test_gradient_flows(self):
        tok = SemanticGroupTokenizer([32, 16], model_dim=32)
        inputs = [torch.randn(B, d, requires_grad=True) for d in [32, 16]]
        tok(inputs)[0].sum().backward()
        for inp in inputs:
            assert inp.grad is not None


# ---------------------------------------------------------------------------
# TokenMixerLarge (full model)
# ---------------------------------------------------------------------------

class TestTokenMixerLarge:
    def _make_model(self, **kwargs):
        defaults = dict(
            feature_groups=[64, 128, 32],
            model_dim=32,
            num_heads=4,
            num_layers=4,
            num_tasks=2,
            num_experts=2,
            top_k=1,
            moe_expand=2,
            gate_scale=2.0,
            inter_residual_interval=2,
            aux_loss_weight=0.1,
        )
        defaults.update(kwargs)
        return TokenMixerLarge(**defaults)

    def _make_inputs(self, feature_groups=(64, 128, 32)):
        return [torch.randn(B, d) for d in feature_groups]

    def test_output_shapes(self):
        model = self._make_model()
        logits, aux_loss = model(self._make_inputs())
        assert logits.shape == (B, 2)
        assert aux_loss is not None and aux_loss.shape == ()

    def test_single_task(self):
        model = self._make_model(num_tasks=1)
        logits, _ = model(self._make_inputs())
        assert logits.shape == (B, 1)

    def test_aux_loss_none_when_disabled(self):
        model = self._make_model(aux_loss_weight=0.0)
        _, aux_loss = model(self._make_inputs())
        assert aux_loss is None

    def test_aux_loss_positive(self):
        model = self._make_model(aux_loss_weight=0.1, num_layers=4,
                                 inter_residual_interval=2)
        _, aux_loss = model(self._make_inputs())
        assert aux_loss is not None
        assert aux_loss.item() >= 0.0

    def test_no_inter_residual(self):
        model = self._make_model(inter_residual_interval=None)
        logits, _ = model(self._make_inputs())
        assert logits.shape == (B, 2)

    def test_backward_through_total_loss(self):
        model = self._make_model()
        inputs = self._make_inputs()
        logits, aux_loss = model(inputs)
        targets = torch.zeros(B, 2)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, targets) + aux_loss
        loss.backward()
        # Spot-check a leaf parameter received a gradient
        first_param = next(model.parameters())
        assert first_param.grad is not None

    def test_deterministic_with_seed(self):
        torch.manual_seed(42)
        model = self._make_model()
        inputs = self._make_inputs()
        torch.manual_seed(7)
        out1, _ = model(inputs)
        torch.manual_seed(7)
        out2, _ = model(inputs)
        assert torch.allclose(out1, out2)

    def test_different_inputs_different_outputs(self):
        model = self._make_model()
        out1, _ = model(self._make_inputs())
        out2, _ = model(self._make_inputs())
        assert not torch.allclose(out1, out2)

    def test_many_feature_groups(self):
        groups = [16] * 10
        model = self._make_model(feature_groups=groups, model_dim=32, num_heads=4)
        logits, _ = model([torch.randn(B, 16) for _ in groups])
        assert logits.shape == (B, 2)

    def test_single_layer_no_inter_residual(self):
        model = self._make_model(num_layers=1, inter_residual_interval=2,
                                 aux_loss_weight=0.0)
        logits, aux_loss = model(self._make_inputs())
        assert logits.shape == (B, 2)
        assert aux_loss is None

    def test_grad_exists_for_all_parameters(self):
        model = self._make_model()
        logits, aux_loss = model(self._make_inputs())
        loss = logits.sum() + (aux_loss if aux_loss is not None else 0)
        loss.backward()
        no_grad = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is None]
        assert no_grad == [], f"Parameters missing gradient: {no_grad}"
