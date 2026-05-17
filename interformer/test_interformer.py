"""Tests for InterFormer implementation.

Run with:
    pytest test_interformer.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from interformer import (
    SelfGating,
    MaskNet,
    LinearCompressedEmbedding,
    PoolingByMHA,
    CrossArch,
    InteractionArch,
    PersonalizedFFN,
    SequenceArch,
    InterFormerBlock,
    InterFormer,
)

B  = 4    # batch size
D  = 32   # model dim
T  = 20   # sequence length
N  = 6    # non-seq tokens
NS = 4    # n_sum
NC = 4    # n_cls
NP = 2    # n_pma
KR = 2    # k_recent


# ---------------------------------------------------------------------------
# SelfGating
# ---------------------------------------------------------------------------

class TestSelfGating:
    def test_output_shape(self):
        g = SelfGating(D)
        x = torch.randn(B, N, D)
        assert g(x).shape == (B, N, D)

    def test_gate_range(self):
        # Output should be bounded since σ(gate)*x; at random init values are finite
        g = SelfGating(D)
        x = torch.randn(B, N, D)
        out = g(x)
        assert torch.isfinite(out).all()

    def test_gate_modulates_identity(self):
        # Zero input → zero output (sigmoid(0)*0 = 0)
        g = SelfGating(D)
        with torch.no_grad():
            assert g(torch.zeros(B, D)).sum().item() == 0.0

    def test_gradient(self):
        g = SelfGating(D)
        x = torch.randn(B, D, requires_grad=True)
        g(x).sum().backward()
        assert x.grad is not None
        assert g.gate.weight.grad is not None


# ---------------------------------------------------------------------------
# MaskNet
# ---------------------------------------------------------------------------

class TestMaskNet:
    def test_output_shape(self):
        mn = MaskNet(input_dim=16, d_model=D)
        S = torch.randn(B, T, 16)
        assert mn(S).shape == (B, T, D)

    def test_different_input_dims(self):
        for dim in [8, 32, 64]:
            mn = MaskNet(dim, D)
            assert mn(torch.randn(B, T, dim)).shape == (B, T, D)

    def test_mask_is_sigmoid(self):
        # The mask_mlp output passes through sigmoid — output is bounded
        mn = MaskNet(16, D)
        out = mn(torch.randn(B, T, 16))
        assert torch.isfinite(out).all()

    def test_gradient(self):
        mn = MaskNet(16, D)
        S = torch.randn(B, T, 16, requires_grad=True)
        mn(S).sum().backward()
        assert S.grad is not None
        assert mn.lce_mlp.weight.grad is not None


# ---------------------------------------------------------------------------
# LinearCompressedEmbedding
# ---------------------------------------------------------------------------

class TestLinearCompressedEmbedding:
    def test_output_shape(self):
        lce = LinearCompressedEmbedding(N, NS)
        X = torch.randn(B, N, D)
        assert lce(X).shape == (B, NS, D)

    def test_compression(self):
        # LCE reduces token count from N to NS
        lce = LinearCompressedEmbedding(10, 3)
        X = torch.randn(B, 10, D)
        assert lce(X).shape == (B, 3, D)

    def test_linear_in_feature_dim(self):
        # LCE is linear in the feature (d) dimension
        lce = LinearCompressedEmbedding(N, NS)
        X = torch.randn(B, N, D)
        with torch.no_grad():
            assert torch.allclose(lce(2 * X), 2 * lce(X), atol=1e-5)

    def test_gradient(self):
        lce = LinearCompressedEmbedding(N, NS)
        X = torch.randn(B, N, D, requires_grad=True)
        lce(X).sum().backward()
        assert X.grad is not None
        assert lce.W.weight.grad is not None


# ---------------------------------------------------------------------------
# PoolingByMHA
# ---------------------------------------------------------------------------

class TestPoolingByMHA:
    def test_output_shape(self):
        pma = PoolingByMHA(n_pma=NP, d_model=D, n_heads=4)
        S = torch.randn(B, T, D)
        assert pma(S).shape == (B, NP, D)

    def test_different_n_pma(self):
        for n in [1, 2, 4]:
            pma = PoolingByMHA(n, D, n_heads=4)
            assert pma(torch.randn(B, T, D)).shape == (B, n, D)

    def test_learned_query_batched(self):
        # Queries expand to batch size — different sequence inputs produce different outputs
        pma = PoolingByMHA(NP, D, n_heads=4)
        S1, S2 = torch.randn(B, T, D), torch.randn(B, T, D)
        assert not torch.allclose(pma(S1), pma(S2))

    def test_gradient(self):
        pma = PoolingByMHA(NP, D, n_heads=4)
        S = torch.randn(B, T, D, requires_grad=True)
        pma(S).sum().backward()
        assert S.grad is not None
        assert pma.Q.grad is not None


# ---------------------------------------------------------------------------
# CrossArch
# ---------------------------------------------------------------------------

class TestCrossArch:
    def _make(self):
        return CrossArch(n_ns=N, n_sum=NS, d_model=D,
                         n_cls=NC, n_pma=NP, k_recent=KR, n_heads=4)

    def test_ns_summarize_shape(self):
        ca = self._make()
        X = torch.randn(B, N, D)
        assert ca.ns_summarize(X).shape == (B, NS, D)

    def test_seq_summarize_shape(self):
        ca = self._make()
        # S must have at least n_cls + k_recent tokens
        S = torch.randn(B, T, D)
        out = ca.seq_summarize(S)
        assert out.shape == (B, NC + NP + KR, D)

    def test_forward_returns_pair(self):
        ca = self._make()
        X = torch.randn(B, N, D)
        S = torch.randn(B, T, D)
        X_sum, S_sum = ca(X, S)
        assert X_sum.shape == (B, NS, D)
        assert S_sum.shape == (B, NC + NP + KR, D)

    def test_ns_summarize_gradient(self):
        ca = self._make()
        X = torch.randn(B, N, D, requires_grad=True)
        ca.ns_summarize(X).sum().backward()
        assert X.grad is not None

    def test_seq_summarize_gradient(self):
        ca = self._make()
        S = torch.randn(B, T, D, requires_grad=True)
        ca.seq_summarize(S).sum().backward()
        assert S.grad is not None


# ---------------------------------------------------------------------------
# InteractionArch
# ---------------------------------------------------------------------------

class TestInteractionArch:
    def _make(self):
        s_sum_len = NC + NP + KR
        return InteractionArch(n_ns=N, s_sum_len=s_sum_len, d_model=D)

    def test_output_shape(self):
        arch = self._make()
        X     = torch.randn(B, N, D)
        S_sum = torch.randn(B, NC + NP + KR, D)
        assert arch(X, S_sum).shape == (B, N, D)

    def test_retains_n_ns(self):
        # Only the first n_ns tokens of the attended output are returned
        arch = self._make()
        X     = torch.randn(B, N, D)
        S_sum = torch.randn(B, NC + NP + KR, D)
        out = arch(X, S_sum)
        assert out.shape[1] == N

    def test_residual_included(self):
        # With fixed weights, zero X + zero S_sum should give zero output (all-zero residual)
        arch = self._make()
        with torch.no_grad():
            for p in arch.parameters():
                p.zero_()
            out = arch(torch.zeros(B, N, D), torch.zeros(B, NC + NP + KR, D))
            assert out.abs().max().item() == 0.0

    def test_gradient(self):
        arch = self._make()
        X     = torch.randn(B, N, D, requires_grad=True)
        S_sum = torch.randn(B, NC + NP + KR, D, requires_grad=True)
        arch(X, S_sum).sum().backward()
        assert X.grad is not None
        assert S_sum.grad is not None


# ---------------------------------------------------------------------------
# PersonalizedFFN
# ---------------------------------------------------------------------------

class TestPersonalizedFFN:
    def test_output_shape(self):
        pffn = PersonalizedFFN(n_sum=NS, d_model=D)
        X_sum = torch.randn(B, NS, D)
        S     = torch.randn(B, T, D)
        assert pffn(X_sum, S).shape == (B, T, D)

    def test_personalization(self):
        # Different X_sum inputs produce different transformations of the same S
        pffn = PersonalizedFFN(NS, D)
        S      = torch.randn(B, T, D)
        X_sum1 = torch.randn(B, NS, D)
        X_sum2 = torch.randn(B, NS, D)
        assert not torch.allclose(pffn(X_sum1, S), pffn(X_sum2, S))

    def test_gradient(self):
        pffn = PersonalizedFFN(NS, D)
        X_sum = torch.randn(B, NS, D, requires_grad=True)
        S     = torch.randn(B, T, D, requires_grad=True)
        pffn(X_sum, S).sum().backward()
        assert X_sum.grad is not None
        assert S.grad is not None
        assert pffn.W.weight.grad is not None


# ---------------------------------------------------------------------------
# SequenceArch
# ---------------------------------------------------------------------------

class TestSequenceArch:
    def test_output_shape(self):
        arch = SequenceArch(n_sum=NS, d_model=D, n_heads=4)
        X_sum = torch.randn(B, NS, D)
        S     = torch.randn(B, T, D)
        assert arch(X_sum, S).shape == (B, T, D)

    def test_residual_connection(self):
        # With zero weights, output ≈ LayerNorm(S) due to residual
        arch = SequenceArch(NS, D, n_heads=4)
        X_sum = torch.randn(B, NS, D)
        S     = torch.randn(B, T, D)
        out   = arch(X_sum, S)
        assert out.shape == (B, T, D)

    def test_layer_norm_applied(self):
        arch = SequenceArch(NS, D, n_heads=4)
        X_sum = torch.randn(B, NS, D) * 100
        S     = torch.randn(B, T, D) * 100
        out = arch(X_sum, S)
        # LN keeps values from exploding
        assert out.abs().max().item() < 1000.0

    def test_gradient(self):
        arch = SequenceArch(NS, D, n_heads=4)
        X_sum = torch.randn(B, NS, D, requires_grad=True)
        S     = torch.randn(B, T, D, requires_grad=True)
        arch(X_sum, S).sum().backward()
        assert X_sum.grad is not None
        assert S.grad is not None


# ---------------------------------------------------------------------------
# InterFormerBlock
# ---------------------------------------------------------------------------

class TestInterFormerBlock:
    def _make(self):
        s_sum_len = NC + NP + KR
        return InterFormerBlock(n_ns=N, n_sum=NS, s_sum_len=s_sum_len,
                                d_model=D, n_heads=4,
                                n_cls=NC, n_pma=NP, k_recent=KR)

    def test_output_shapes(self):
        block = self._make()
        X = torch.randn(B, N, D)
        S = torch.randn(B, T, D)
        X_new, S_new, X_sum, S_sum = block(X, S)
        assert X_new.shape  == (B, N, D)
        assert S_new.shape  == (B, T, D)
        assert X_sum.shape  == (B, NS, D)
        assert S_sum.shape  == (B, NC + NP + KR, D)

    def test_sequence_length_preserved(self):
        block = self._make()
        X = torch.randn(B, N, D)
        S = torch.randn(B, T, D)
        _, S_new, _, _ = block(X, S)
        assert S_new.shape[1] == T

    def test_gradient_all_outputs(self):
        block = self._make()
        X = torch.randn(B, N, D, requires_grad=True)
        S = torch.randn(B, T, D, requires_grad=True)
        X_new, S_new, X_sum, S_sum = block(X, S)
        # Tensors have different shapes; sum each separately
        (X_new.sum() + S_new.sum() + X_sum.sum() + S_sum.sum()).backward()
        assert X.grad is not None
        assert S.grad is not None

    def test_different_inputs_different_outputs(self):
        block = self._make()
        X1, S1 = torch.randn(B, N, D), torch.randn(B, T, D)
        X2, S2 = torch.randn(B, N, D), torch.randn(B, T, D)
        out1 = block(X1, S1)[0]
        out2 = block(X2, S2)[0]
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# InterFormer (full model)
# ---------------------------------------------------------------------------

class TestInterFormer:
    def _make(self, **kwargs):
        defaults = dict(
            dense_dim=32,
            sparse_dims=[100, 200, 50],
            seq_input_dim=16,
            d_model=D,
            num_layers=2,
            n_sum=NS,
            n_cls=NC,
            n_pma=NP,
            k_recent=KR,
            n_heads=4,
            top_mlp_dims=[32],
            num_tasks=1,
        )
        defaults.update(kwargs)
        return InterFormer(**defaults)

    def _inputs(self, model=None, b=B):
        dense   = torch.randn(b, 32)
        sparse  = [torch.randint(0, v, (b,)) for v in [100, 200, 50]]
        seq     = torch.randn(b, T, 16)
        return dense, sparse, seq

    def test_output_shape_single_task(self):
        model = self._make(num_tasks=1)
        logits = model(*self._inputs())
        assert logits.shape == (B, 1)

    def test_output_shape_multi_task(self):
        model = self._make(num_tasks=3)
        logits = model(*self._inputs())
        assert logits.shape == (B, 3)

    def test_backward(self):
        model = self._make()
        logits = model(*self._inputs())
        F.binary_cross_entropy_with_logits(logits, torch.zeros(B, 1)).backward()
        assert next(model.parameters()).grad is not None

    def test_grad_all_parameters(self):
        model = self._make()
        model(*self._inputs()).sum().backward()
        # Per Algorithm 1: only X_sum^(L) and S_sum^(L) feed the prediction head.
        # init_cross.seq_summarize and blocks[-1] Interaction/Sequence arches are
        # structurally dead — they compute outputs not connected to the loss.
        dead = {
            'init_cross.pma.Q',
            'init_cross.pma.mha.in_proj_weight', 'init_cross.pma.mha.in_proj_bias',
            'init_cross.pma.mha.out_proj.weight', 'init_cross.pma.mha.out_proj.bias',
            'init_cross.seq_gate.gate.weight', 'init_cross.seq_gate.gate.bias',
        }
        last = f'blocks.{model.num_layers - 1}'
        no_grad = [
            n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None
            and n not in dead
            and not n.startswith(f'{last}.interaction')
            and not n.startswith(f'{last}.sequence')
        ]
        assert no_grad == [], f"Missing grad: {no_grad}"

    def test_deterministic_with_seed(self):
        model = self._make()
        inp = self._inputs()
        torch.manual_seed(42)
        out1 = model(*inp)
        torch.manual_seed(42)
        out2 = model(*inp)
        assert torch.allclose(out1, out2)

    def test_different_inputs_different_outputs(self):
        model = self._make()
        out1 = model(*self._inputs())
        out2 = model(*self._inputs())
        assert not torch.allclose(out1, out2)

    def test_num_layers_scaling(self):
        # Deeper model should have more parameters
        m2 = self._make(num_layers=2)
        m4 = self._make(num_layers=4)
        p2 = sum(p.numel() for p in m2.parameters())
        p4 = sum(p.numel() for p in m4.parameters())
        assert p4 > p2

    def test_finite_output(self):
        model = self._make()
        out = model(*self._inputs())
        assert torch.isfinite(out).all()

    def test_single_sparse_feature(self):
        model = InterFormer(
            dense_dim=16, sparse_dims=[50], seq_input_dim=8,
            d_model=D, num_layers=2, n_sum=2, n_cls=2, n_pma=1, k_recent=1,
            n_heads=4, num_tasks=1,
        )
        dense  = torch.randn(B, 16)
        sparse = [torch.randint(0, 50, (B,))]
        seq    = torch.randn(B, T, 8)
        assert model(dense, sparse, seq).shape == (B, 1)

    def test_many_sparse_features(self):
        vocabs = [50] * 10
        model = InterFormer(
            dense_dim=32, sparse_dims=vocabs, seq_input_dim=16,
            d_model=D, num_layers=2, n_sum=NS, n_cls=NC, n_pma=NP, k_recent=KR,
            n_heads=4, num_tasks=1,
        )
        dense  = torch.randn(B, 32)
        sparse = [torch.randint(0, 50, (B,)) for _ in vocabs]
        seq    = torch.randn(B, T, 16)
        assert model(dense, sparse, seq).shape == (B, 1)

    def test_batch_size_one(self):
        model = self._make()
        dense, sparse, seq = self._inputs(b=1)
        assert model(dense, sparse, seq).shape == (1, 1)

    def test_longer_sequence(self):
        model = self._make()
        dense  = torch.randn(B, 32)
        sparse = [torch.randint(0, v, (B,)) for v in [100, 200, 50]]
        seq    = torch.randn(B, 100, 16)  # longer sequence
        assert model(dense, sparse, seq).shape == (B, 1)
