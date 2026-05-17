"""Tests for Meta Lattice implementation.

Run with:
    pytest test_meta_lattice.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from meta_lattice import (
    SwishRMSNorm,
    lattice_filter,
    LatticeZipper,
    CategoricalProcessor,
    DenseProcessor,
    SequenceProcessor,
    MixingNetwork,
    ExtendedContextStorage,
    TransformerBlock,
    DWFBlock,
    TaskModule,
    LatticeKTAP,
    LatticeNetwork,
)

B  = 4    # batch size
D  = 32   # model dim
T  = 20   # sequence length
H  = 4    # n_heads


# ---------------------------------------------------------------------------
# SwishRMSNorm
# ---------------------------------------------------------------------------

class TestSwishRMSNorm:
    def test_output_shape(self):
        m = SwishRMSNorm(D)
        assert m(torch.randn(B, D)).shape == (B, D)

    def test_3d_input(self):
        m = SwishRMSNorm(D)
        assert m(torch.randn(B, T, D)).shape == (B, T, D)

    def test_finite_output(self):
        m = SwishRMSNorm(D)
        assert torch.isfinite(m(torch.randn(B, D))).all()

    def test_gradient(self):
        m = SwishRMSNorm(D)
        x = torch.randn(B, D, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None

    def test_different_from_identity(self):
        m = SwishRMSNorm(D)
        x = torch.randn(B, D)
        assert not torch.allclose(m(x), x)

    def test_large_values_stay_bounded(self):
        # SwishRMSNorm normalizes so large inputs don't explode
        m = SwishRMSNorm(D)
        x = torch.randn(B, D) * 1000
        assert torch.isfinite(m(x)).all()


# ---------------------------------------------------------------------------
# LatticeFilter
# ---------------------------------------------------------------------------

class TestLatticeFilter:
    def test_returns_target_count(self):
        scores  = torch.rand(10, 3)
        result  = lattice_filter(scores, target_count=4)
        assert len(result) == 4

    def test_valid_indices(self):
        scores = torch.rand(10, 3)
        for idx in lattice_filter(scores, 5):
            assert 0 <= idx < 10

    def test_sorted_output(self):
        scores = torch.rand(10, 3)
        result = lattice_filter(scores, 5)
        assert result == sorted(result)

    def test_select_all(self):
        scores = torch.rand(5, 2)
        result = lattice_filter(scores, 5)
        assert len(result) == 5

    def test_select_more_than_available(self):
        # Cannot select more features than exist
        scores = torch.rand(3, 2)
        result = lattice_filter(scores, 10)
        assert len(result) <= 3

    def test_pareto_dominated_excluded_first(self):
        # Feature 0 strictly dominates feature 1 on both tasks
        scores = torch.tensor([[0.9, 0.9], [0.1, 0.1], [0.5, 0.5]])
        result = lattice_filter(scores, 1)
        # Feature 0 is on the Pareto frontier and should be selected first
        assert 0 in result

    def test_single_task(self):
        scores = torch.rand(8, 1)
        result = lattice_filter(scores, 3)
        assert len(result) == 3

    def test_deterministic_with_seed(self):
        scores = torch.rand(20, 5)
        r1 = lattice_filter(scores, 7, seed=0)
        r2 = lattice_filter(scores, 7, seed=0)
        assert r1 == r2

    def test_different_seeds_may_differ(self):
        scores = torch.rand(20, 5)
        r1 = lattice_filter(scores, 7, seed=0)
        r2 = lattice_filter(scores, 7, seed=99)
        # Not guaranteed to differ, but with 20 features and large frontier likely
        # Just check both are valid
        assert len(r1) == 7 and len(r2) == 7


# ---------------------------------------------------------------------------
# LatticeZipper
# ---------------------------------------------------------------------------

class TestLatticeZipper:
    def _make(self, n_windows=3, in_dim=D, n_tasks=2):
        return LatticeZipper(n_windows, in_dim, n_tasks)

    def test_train_output_shape(self):
        z = self._make()
        x = torch.randn(B, D)
        w = torch.randint(0, 3, (B,))
        z.train()
        assert z(x, w).shape == (B, 2)

    def test_eval_oracle_head(self):
        z = self._make()
        x = torch.randn(B, D)
        z.eval()
        out = z(x)
        assert out.shape == (B, 2)

    def test_none_window_idx_uses_oracle(self):
        z = self._make()
        z.train()
        x = torch.randn(B, D)
        # None window_idx during training still uses oracle
        out = z(x, None)
        assert out.shape == (B, 2)

    def test_train_routes_to_different_heads(self):
        # Force all to window 0 vs all to window 2 — should differ
        z = self._make(n_windows=3)
        x = torch.randn(B, D)
        z.train()
        w0 = torch.zeros(B, dtype=torch.long)
        w2 = torch.full((B,), 2, dtype=torch.long)
        out0 = z(x, w0)
        out2 = z(x, w2)
        assert not torch.allclose(out0, out2)

    def test_single_window_no_routing(self):
        z = LatticeZipper(1, D, 1)
        z.train()
        x = torch.randn(B, D)
        w = torch.zeros(B, dtype=torch.long)
        assert z(x, w).shape == (B, 1)

    def test_gradient_flows_to_assigned_head(self):
        z = self._make(n_windows=3)
        z.train()
        x = torch.randn(B, D, requires_grad=True)
        w = torch.zeros(B, dtype=torch.long)  # all window 0
        z(x, w).sum().backward()
        assert x.grad is not None
        assert z.heads[0].weight.grad is not None

    def test_oracle_is_last_head(self):
        z = self._make(n_windows=3)
        assert z.oracle_idx == 2


# ---------------------------------------------------------------------------
# CategoricalProcessor
# ---------------------------------------------------------------------------

class TestCategoricalProcessor:
    def test_output_shape(self):
        cp = CategoricalProcessor([100, 200, 50], D)
        feats = [torch.randint(0, v, (B,)) for v in [100, 200, 50]]
        assert cp(feats).shape == (B, 3, D)

    def test_single_feature(self):
        cp = CategoricalProcessor([100], D)
        assert cp([torch.randint(0, 100, (B,))]).shape == (B, 1, D)

    def test_gradient(self):
        cp = CategoricalProcessor([50], D)
        feats = [torch.randint(0, 50, (B,))]
        cp(feats).sum().backward()
        assert cp.embeddings[0].weight.grad is not None


# ---------------------------------------------------------------------------
# DenseProcessor
# ---------------------------------------------------------------------------

class TestDenseProcessor:
    def test_output_shape(self):
        dp = DenseProcessor([8, 16], D)
        feats = [torch.randn(B, 8), torch.randn(B, 16)]
        assert dp(feats).shape == (B, 2, D)

    def test_single_feature(self):
        dp = DenseProcessor([32], D)
        assert dp([torch.randn(B, 32)]).shape == (B, 1, D)

    def test_gradient(self):
        dp = DenseProcessor([8], D)
        x  = torch.randn(B, 8, requires_grad=True)
        dp([x]).sum().backward()
        assert x.grad is not None

    def test_bias_less(self):
        dp = DenseProcessor([8], D)
        for proj in dp.projections:
            assert proj.bias is None


# ---------------------------------------------------------------------------
# SequenceProcessor
# ---------------------------------------------------------------------------

class TestSequenceProcessor:
    def test_output_shape(self):
        sp = SequenceProcessor(32, D, n_heads=H)
        S  = torch.randn(B, T, 32)
        assert sp(S).shape == (B, T, D)

    def test_gradient(self):
        sp = SequenceProcessor(32, D, n_heads=H)
        S  = torch.randn(B, T, 32, requires_grad=True)
        sp(S).sum().backward()
        assert S.grad is not None

    def test_finite_output(self):
        sp = SequenceProcessor(16, D, n_heads=H)
        assert torch.isfinite(sp(torch.randn(B, T, 16))).all()


# ---------------------------------------------------------------------------
# MixingNetwork
# ---------------------------------------------------------------------------

class TestMixingNetwork:
    def test_output_shape(self):
        mn  = MixingNetwork(5, D)
        O_c = torch.randn(B, 3, D)
        O_d = torch.randn(B, 2, D)
        assert mn(O_c, O_d).shape == (B, 5, D)

    def test_gradient(self):
        mn  = MixingNetwork(4, D)
        O_c = torch.randn(B, 2, D, requires_grad=True)
        O_d = torch.randn(B, 2, D, requires_grad=True)
        mn(O_c, O_d).sum().backward()
        assert O_c.grad is not None
        assert O_d.grad is not None

    def test_finite_output(self):
        mn = MixingNetwork(3, D)
        assert torch.isfinite(mn(torch.randn(B, 2, D), torch.randn(B, 1, D))).all()


# ---------------------------------------------------------------------------
# ExtendedContextStorage
# ---------------------------------------------------------------------------

class TestExtendedContextStorage:
    def test_empty_returns_none(self):
        ecs = ExtendedContextStorage(D, max_layers=4)
        assert ecs.get_residual() is None

    def test_residual_shape_after_push(self):
        ecs = ExtendedContextStorage(D, max_layers=4)
        ecs.push(torch.randn(B, 3, D))
        res = ecs.get_residual()
        assert res is not None
        assert res.shape == (B, D)

    def test_reset_clears_store(self):
        ecs = ExtendedContextStorage(D, max_layers=4)
        ecs.push(torch.randn(B, 3, D))
        ecs.reset()
        assert ecs.get_residual() is None

    def test_multiple_pushes_accumulate(self):
        ecs = ExtendedContextStorage(D, max_layers=4)
        for _ in range(3):
            ecs.push(torch.randn(B, 5, D))
        res = ecs.get_residual()
        assert res is not None
        assert res.shape == (B, D)

    def test_gradient_through_residual(self):
        ecs = ExtendedContextStorage(D, max_layers=4)
        x   = torch.randn(B, 3, D, requires_grad=True)
        ecs.push(x)
        res = ecs.get_residual()
        res.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TestTransformerBlock:
    def test_output_shape(self):
        tb = TransformerBlock(D, n_heads=H, n_domains=2)
        S  = torch.randn(B, T, D)
        assert tb(S).shape == (B, T, D)

    def test_with_domain_ids(self):
        tb  = TransformerBlock(D, n_heads=H, n_domains=3)
        S   = torch.randn(B, T, D)
        did = torch.randint(0, 3, (B,))
        assert tb(S, did).shape == (B, T, D)

    def test_domain_ids_affect_output(self):
        tb = TransformerBlock(D, n_heads=H, n_domains=2)
        S  = torch.randn(B, T, D)
        d0 = torch.zeros(B, dtype=torch.long)
        d1 = torch.ones(B, dtype=torch.long)
        assert not torch.allclose(tb(S, d0), tb(S, d1))

    def test_gradient(self):
        tb  = TransformerBlock(D, n_heads=H, n_domains=2)
        S   = torch.randn(B, T, D, requires_grad=True)
        did = torch.randint(0, 2, (B,))
        tb(S, did).sum().backward()
        assert S.grad is not None

    def test_finite_output(self):
        tb = TransformerBlock(D, n_heads=H, n_domains=1)
        assert torch.isfinite(tb(torch.randn(B, T, D))).all()


# ---------------------------------------------------------------------------
# DWFBlock
# ---------------------------------------------------------------------------

class TestDWFBlock:
    def _make(self, n_seq=5, n_cd=6, n_out=8):
        return DWFBlock(n_seq, n_cd, D, n_out=n_out, k=8)

    def test_output_shape(self):
        dwf  = self._make(n_seq=5, n_cd=6, n_out=8)
        O_s  = torch.randn(B, 5, D)
        O_cd = torch.randn(B, 6, D)
        assert dwf(O_s, O_cd).shape == (B, 8, D)

    def test_gradient(self):
        dwf  = self._make()
        O_s  = torch.randn(B, 5, D, requires_grad=True)
        O_cd = torch.randn(B, 6, D, requires_grad=True)
        dwf(O_s, O_cd).sum().backward()
        assert O_s.grad is not None
        assert O_cd.grad is not None

    def test_finite_output(self):
        dwf  = self._make()
        assert torch.isfinite(
            dwf(torch.randn(B, 5, D), torch.randn(B, 6, D))
        ).all()

    def test_different_inputs_different_outputs(self):
        dwf  = self._make()
        O_s1 = torch.randn(B, 5, D)
        O_s2 = torch.randn(B, 5, D)
        O_cd = torch.randn(B, 6, D)
        assert not torch.allclose(dwf(O_s1, O_cd), dwf(O_s2, O_cd))


# ---------------------------------------------------------------------------
# TaskModule
# ---------------------------------------------------------------------------

class TestTaskModule:
    def test_output_shape(self):
        tm = TaskModule(64, [32], n_tasks=1)
        assert tm(torch.randn(B, 64)).shape == (B, 1)

    def test_multi_task(self):
        tm = TaskModule(64, [32], n_tasks=3)
        assert tm(torch.randn(B, 64)).shape == (B, 3)

    def test_gradient(self):
        tm = TaskModule(32, [16], 1)
        x  = torch.randn(B, 32, requires_grad=True)
        tm(x).sum().backward()
        assert x.grad is not None

    def test_bias_less(self):
        tm = TaskModule(32, [16], 1)
        for module in tm.mlp:
            if isinstance(module, torch.nn.Linear):
                assert module.bias is None


# ---------------------------------------------------------------------------
# LatticeKTAP
# ---------------------------------------------------------------------------

class TestLatticeKTAP:
    def _make(self):
        return LatticeKTAP(teacher_dim=64, student_dim=D, cache_size=100)

    def test_query_miss_returns_zeros(self):
        ktap = self._make()
        keys = torch.arange(B, dtype=torch.int64)
        embs, logits = ktap.query(keys)
        assert embs.shape == (B, D)
        assert torch.allclose(embs, torch.zeros_like(embs))
        assert logits is None

    def test_store_and_query_hit(self):
        ktap = self._make()
        keys   = torch.arange(B, dtype=torch.int64)
        embs   = torch.randn(B, 64)
        logits = torch.randn(B, 2)
        ktap.store(keys, embs, logits)
        out_embs, out_logits = ktap.query(keys)
        assert out_embs.shape   == (B, D)
        assert out_logits is not None
        assert out_logits.shape == (B, 2)

    def test_projection_applied(self):
        ktap  = self._make()
        keys  = torch.arange(B, dtype=torch.int64)
        embs  = torch.randn(B, 64)
        logits = torch.randn(B, 1)
        ktap.store(keys, embs, logits)
        out, _ = ktap.query(keys)
        # Projection changes the shape from teacher_dim to student_dim
        assert out.shape == (B, D)

    def test_distillation_loss(self):
        ktap   = self._make()
        s_log  = torch.randn(B, 3)
        t_log  = torch.randn(B, 3)
        loss   = ktap.distillation_loss(s_log, t_log)
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_cache_eviction(self):
        ktap = LatticeKTAP(64, D, cache_size=3)
        for i in range(10):
            k = torch.tensor([i], dtype=torch.int64)
            e = torch.randn(1, 64)
            l = torch.randn(1, 1)
            ktap.store(k, e, l)
        assert len(ktap._cache) <= 3


# ---------------------------------------------------------------------------
# LatticeNetwork (full model)
# ---------------------------------------------------------------------------

class TestLatticeNetwork:
    def _make(self, **kw):
        defaults = dict(
            vocab_sizes   = [100, 200, 50],
            dense_dims    = [8, 16],
            seq_input_dim = 32,
            d_model       = D,
            n_layers      = 3,
            n_domains     = 2,
            n_out_tokens  = 8,
            fm_rank       = 8,
            n_heads       = H,
            task_hidden   = [32],
            n_tasks       = 2,
            n_windows     = 3,
        )
        defaults.update(kw)
        return LatticeNetwork(**defaults)

    def _inputs(self, b=B):
        return (
            [torch.randint(0, v, (b,)) for v in [100, 200, 50]],
            [torch.randn(b, d) for d in [8, 16]],
            torch.randn(b, T, 32),
        )

    def test_train_output_shape(self):
        model = self._make()
        model.train()
        cf, df, sf = self._inputs()
        w  = torch.randint(0, 3, (B,))
        did = torch.randint(0, 2, (B,))
        assert model(cf, df, sf, did, w).shape == (B, 2)

    def test_eval_output_shape(self):
        model = self._make()
        model.eval()
        cf, df, sf = self._inputs()
        with torch.no_grad():
            assert model(cf, df, sf).shape == (B, 2)

    def test_single_task_single_window(self):
        model = self._make(n_tasks=1, n_windows=1)
        cf, df, sf = self._inputs()
        assert model(cf, df, sf).shape == (B, 1)

    def test_multi_task_no_zipper(self):
        model = self._make(n_tasks=3, n_windows=1)
        cf, df, sf = self._inputs()
        assert model(cf, df, sf).shape == (B, 3)

    def test_backward(self):
        model = self._make()
        model.train()
        cf, df, sf = self._inputs()
        w   = torch.randint(0, 3, (B,))
        out = model(cf, df, sf, window_idx=w)
        F.binary_cross_entropy_with_logits(out, torch.zeros_like(out)).backward()
        assert any(p.grad is not None for p in model.parameters())

    def test_finite_output(self):
        model = self._make()
        cf, df, sf = self._inputs()
        assert torch.isfinite(model(cf, df, sf)).all()

    def test_deterministic_eval(self):
        model = self._make()
        model.eval()
        cf, df, sf = self._inputs()
        torch.manual_seed(0); o1 = model(cf, df, sf)
        torch.manual_seed(0); o2 = model(cf, df, sf)
        assert torch.allclose(o1, o2)

    def test_batch_size_one(self):
        model = self._make()
        cf, df, sf = self._inputs(b=1)
        assert model(cf, df, sf).shape == (1, 2)

    def test_different_inputs_different_outputs(self):
        model = self._make()
        cf1, df1, sf1 = self._inputs()
        cf2, df2, sf2 = self._inputs()
        assert not torch.allclose(model(cf1, df1, sf1), model(cf2, df2, sf2))

    def test_deeper_model_more_params(self):
        m2 = self._make(n_layers=2)
        m4 = self._make(n_layers=4)
        p2 = sum(p.numel() for p in m2.parameters())
        p4 = sum(p.numel() for p in m4.parameters())
        assert p4 > p2

    def test_with_domain_ids(self):
        model = self._make()
        cf, df, sf = self._inputs()
        did = torch.randint(0, 2, (B,))
        assert model(cf, df, sf, domain_ids=did).shape == (B, 2)

    def test_domain_ids_affect_output(self):
        model = self._make()
        cf, df, sf = self._inputs()
        d0 = torch.zeros(B, dtype=torch.long)
        d1 = torch.ones(B, dtype=torch.long)
        assert not torch.allclose(model(cf, df, sf, d0), model(cf, df, sf, d1))

    def test_train_vs_eval_zipper_differ(self):
        # Training routes to window heads; eval uses oracle — can differ
        model = self._make()
        cf, df, sf = self._inputs()
        w = torch.zeros(B, dtype=torch.long)   # force window 0
        model.train(); out_train = model(cf, df, sf, window_idx=w)
        model.eval();  out_eval  = model(cf, df, sf, window_idx=w)
        # Training uses head 0, eval uses head 2 (oracle) — they differ
        assert not torch.allclose(out_train, out_eval)

    def test_ktap_integration(self):
        model = self._make(ktap_dim=64)
        cf, df, sf = self._inputs()
        # Pre-populate KTAP cache
        keys  = torch.arange(B, dtype=torch.int64)
        embs  = torch.randn(B, 64)
        logits = torch.randn(B, 2)
        model.ktap.store(keys, embs, logits)
        out = model(cf, df, sf, ktap_keys=keys)
        assert out.shape == (B, 2)
        assert torch.isfinite(out).all()

    def test_longer_sequence(self):
        model = self._make()
        cf    = [torch.randint(0, v, (B,)) for v in [100, 200, 50]]
        df    = [torch.randn(B, d) for d in [8, 16]]
        sf    = torch.randn(B, 200, 32)        # longer sequence
        assert model(cf, df, sf).shape == (B, 2)
