"""Tests for llm_ranking.py — LLM-enriched Foundation-Expert ranking model."""

import pytest
import torch
import torch.nn.functional as F

from llm_ranking import (
    ConsistencyRegularizer,
    LLMEnrichedExpertModel,
    LLMEnrichedFoundationModel,
    LLMFeatureEncoder,
    SemanticItemEmbedding,
)


# ---------------------------------------------------------------------------
# LLMFeatureEncoder
# ---------------------------------------------------------------------------

class TestLLMFeatureEncoder:
    @pytest.fixture
    def encoder(self):
        torch.manual_seed(0)
        return LLMFeatureEncoder(n_categories=16, d_llm=24, d_out=32)

    def test_output_shape_2d(self, encoder):
        B = 4
        out = encoder(torch.randn(B, 16), F.normalize(torch.randn(B, 24), dim=-1))
        assert out.shape == (B, 32)

    def test_output_shape_3d(self, encoder):
        B, M = 4, 6
        out = encoder(
            torch.randn(B, M, 16),
            F.normalize(torch.randn(B, M, 24), dim=-1),
        )
        assert out.shape == (B, M, 32)

    def test_backward(self, encoder):
        B = 4
        encoder(torch.randn(B, 16), F.normalize(torch.randn(B, 24), dim=-1)).sum().backward()

    def test_logits_go_through_sigmoid(self, encoder):
        # Large positive vs large negative logits should give different outputs
        captions = torch.zeros(2, 24)
        out_pos = encoder(torch.full((2, 16), 100.0), captions)
        out_neg = encoder(torch.full((2, 16), -100.0), captions)
        assert not torch.allclose(out_pos, out_neg)

    def test_batch_size_one(self, encoder):
        out = encoder(torch.randn(1, 16), F.normalize(torch.randn(1, 24), dim=-1))
        assert out.shape == (1, 32)


# ---------------------------------------------------------------------------
# SemanticItemEmbedding
# ---------------------------------------------------------------------------

class TestSemanticItemEmbedding:
    @pytest.fixture
    def emb(self):
        torch.manual_seed(0)
        return SemanticItemEmbedding(
            prod_dim=12, ctx_dim=6, act_dim=4,
            n_categories=16, d_llm=12, d_model=24,
        )

    def _inputs(self, B=4, M=6):
        return (
            torch.randn(B, M, 12),                         # prod
            torch.randn(B, M, 6),                          # ctx
            torch.randn(B, M, 16),                         # cat logits
            F.normalize(torch.randn(B, M, 12), dim=-1),    # captions
        )

    def test_output_shape_without_act(self, emb):
        out = emb(*self._inputs(4, 6))
        assert out.shape == (4, 6, 24)

    def test_output_shape_with_act(self, emb):
        prod, ctx, cat, cap = self._inputs(4, 6)
        act = torch.randn(4, 6, 4)
        out = emb(prod, ctx, cat, cap, act)
        assert out.shape == (4, 6, 24)

    def test_act_changes_output(self, emb):
        prod, ctx, cat, cap = self._inputs(2, 3)
        act = torch.randn(2, 3, 4)
        out_no_act = emb(prod, ctx, cat, cap)
        out_with_act = emb(prod, ctx, cat, cap, act)
        assert not torch.allclose(out_no_act, out_with_act)

    def test_backward_with_act(self, emb):
        prod, ctx, cat, cap = self._inputs(2, 3)
        act = torch.randn(2, 3, 4)
        emb(prod, ctx, cat, cap, act).sum().backward()

    def test_backward_without_act(self, emb):
        emb(*self._inputs(2, 3)).sum().backward()


# ---------------------------------------------------------------------------
# ConsistencyRegularizer
# ---------------------------------------------------------------------------

class TestConsistencyRegularizer:
    def test_output_is_scalar(self):
        reg = ConsistencyRegularizer()
        loss = reg(torch.randn(4, 6, 24), torch.randn(4, 6, 24))
        assert loss.shape == ()

    def test_zero_loss_for_identical_tensors(self):
        reg = ConsistencyRegularizer()
        tae = torch.randn(4, 6, 24)
        assert reg(tae, tae.clone()).item() == pytest.approx(0.0, abs=1e-5)

    def test_shadow_is_detached(self):
        reg = ConsistencyRegularizer()
        primary = torch.randn(3, 5, 12, requires_grad=True)
        shadow = torch.randn(3, 5, 12, requires_grad=True)
        reg(primary, shadow).backward()
        assert primary.grad is not None
        assert shadow.grad is None   # detached inside ConsistencyRegularizer

    def test_weighted_uniform_matches_unweighted(self):
        reg = ConsistencyRegularizer()
        P, M, d = 4, 6, 8
        primary = torch.randn(P, M, d)
        shadow = torch.zeros(P, M, d)
        loss_no_w = reg(primary, shadow)
        loss_w1   = reg(primary, shadow, weights=torch.ones(P))
        assert loss_w1.item() == pytest.approx(loss_no_w.item(), rel=1e-4)

    def test_higher_weight_increases_loss(self):
        reg = ConsistencyRegularizer()
        primary = torch.randn(2, 4, 8)
        shadow = torch.zeros(2, 4, 8)
        loss_low  = reg(primary, shadow, weights=torch.tensor([0.1, 0.1]))
        loss_high = reg(primary, shadow, weights=torch.tensor([10.0, 10.0]))
        assert loss_high.item() > loss_low.item()

    def test_sum_reduction_equals_mean_times_p(self):
        reg_mean = ConsistencyRegularizer(reduction='mean')
        reg_sum  = ConsistencyRegularizer(reduction='sum')
        P, M, d = 4, 6, 8
        primary = torch.randn(P, M, d)
        shadow = torch.zeros(P, M, d)
        assert reg_sum(primary, shadow).item() == pytest.approx(
            reg_mean(primary, shadow).item() * P, rel=1e-4
        )


# ---------------------------------------------------------------------------
# LLMEnrichedFoundationModel
# ---------------------------------------------------------------------------

class TestLLMEnrichedFoundationModel:
    @pytest.fixture
    def fm(self):
        torch.manual_seed(0)
        return LLMEnrichedFoundationModel(
            prod_dim=12, ctx_dim=6, act_dim=4, aux_dim=8,
            n_categories=16, d_llm=12,
            d_model=24, n_layers=2, n_heads=4,
            n_main_tasks=3, n_aux_tasks=2,
            consistency_weight=0.1, semantic_weight=0.05,
        )

    @pytest.fixture
    def data(self):
        B, N, M = 4, 8, 6
        return dict(
            B=B, N=N, M=M,
            hist_prod=torch.randn(B, N, 12),
            hist_ctx=torch.randn(B, N, 6),
            hist_act=torch.randn(B, N, 4),
            hist_cat=torch.randn(B, N, 16),
            hist_cap=F.normalize(torch.randn(B, N, 12), dim=-1),
            cand_prod=torch.randn(B, M, 12),
            cand_ctx=torch.randn(B, M, 6),
            cand_cat=torch.randn(B, M, 16),
            cand_cap=F.normalize(torch.randn(B, M, 12), dim=-1),
            aux=torch.randn(B, M, 8),
        )

    def _forward(self, fm, d):
        return fm(
            d['hist_prod'], d['hist_ctx'], d['hist_act'], d['hist_cat'], d['hist_cap'],
            d['cand_prod'], d['cand_ctx'], d['cand_cat'], d['cand_cap'],
            d['aux'],
        )

    def test_forward_output_shapes(self, fm, data):
        B, M = data['B'], data['M']
        tae, main, aux, sem = self._forward(fm, data)
        assert tae.shape  == (B, M, 24)
        assert main.shape == (B, M, 3)
        assert aux.shape  == (B, M, 2)
        assert sem.shape  == (B, M, 16)

    def test_compute_loss_without_consistency(self, fm, data):
        B, M = data['B'], data['M']
        tae, main, aux, sem = self._forward(fm, data)
        loss, bd = fm.compute_loss(
            main, torch.zeros_like(main),
            aux, torch.zeros_like(aux),
            sem, torch.randn(B, M, 16),
        )
        assert loss.item() > 0
        assert set(bd.keys()) == {'main', 'aux', 'semantic', 'consistency'}
        assert bd['consistency'].item() == pytest.approx(0.0, abs=1e-6)

    def test_compute_loss_with_consistency(self, fm, data):
        B, M = data['B'], data['M']
        P = 2
        tae, main, aux, sem = self._forward(fm, data)
        tae_shadow = tae[:P] + 0.1 * torch.randn(P, M, 24)
        _, bd = fm.compute_loss(
            main, torch.zeros_like(main),
            aux, torch.zeros_like(aux),
            sem, torch.randn(B, M, 16),
            tae_primary=tae[:P],
            tae_shadow=tae_shadow,
            pair_weights=torch.tensor([0.9, 0.7]),
        )
        assert bd['consistency'].item() > 0.0

    def test_consistency_zero_for_identical_pairs(self, fm, data):
        B, M = data['B'], data['M']
        P = 2
        tae, main, aux, sem = self._forward(fm, data)
        _, bd = fm.compute_loss(
            main, torch.zeros_like(main),
            aux, torch.zeros_like(aux),
            sem, torch.randn(B, M, 16),
            tae_primary=tae[:P],
            tae_shadow=tae[:P].clone(),
        )
        assert bd['consistency'].item() == pytest.approx(0.0, abs=1e-5)

    def test_backward(self, fm, data):
        B, M = data['B'], data['M']
        P = 2
        tae, main, aux, sem = self._forward(fm, data)
        tae_shadow = tae[:P].detach() + 0.05 * torch.randn(P, M, 24)
        loss, _ = fm.compute_loss(
            main, torch.zeros_like(main),
            aux, torch.zeros_like(aux),
            sem, torch.randn(B, M, 16),
            tae_primary=tae[:P],
            tae_shadow=tae_shadow,
        )
        loss.backward()

    def test_semantic_loss_in_breakdown(self, fm, data):
        B, M = data['B'], data['M']
        tae, main, aux, sem = self._forward(fm, data)
        _, bd = fm.compute_loss(
            main, torch.zeros_like(main),
            aux, torch.zeros_like(aux),
            sem, torch.randn(B, M, 16),
        )
        assert bd['semantic'].item() > 0.0

    def test_param_count_positive(self, fm):
        assert sum(p.numel() for p in fm.parameters()) > 0


# ---------------------------------------------------------------------------
# LLMEnrichedExpertModel
# ---------------------------------------------------------------------------

class TestLLMEnrichedExpertModel:
    @pytest.fixture
    def expert(self):
        torch.manual_seed(0)
        return LLMEnrichedExpertModel(
            fm_dim=24, prod_dim=12, ctx_dim=6, act_dim=4, surf_dim=6,
            n_categories=16, d_llm=12,
            d_expert=12, n_layers=1, n_heads=4, n_tasks=2,
        )

    @pytest.fixture
    def data(self):
        B, M, T = 4, 6, 8
        return dict(
            B=B, M=M, T=T,
            tae=torch.randn(B, M, 24),
            short_prod=torch.randn(B, T, 12),
            short_ctx=torch.randn(B, T, 6),
            short_act=torch.randn(B, T, 4),
            short_cat=torch.randn(B, T, 16),
            short_cap=F.normalize(torch.randn(B, T, 12), dim=-1),
            cand_cat=torch.randn(B, M, 16),
            cand_cap=F.normalize(torch.randn(B, M, 12), dim=-1),
            surf=torch.randn(B, M, 6),
        )

    def _forward(self, expert, d):
        return expert(
            d['tae'].detach(),
            d['short_prod'], d['short_ctx'], d['short_act'],
            d['short_cat'], d['short_cap'],
            d['cand_cat'], d['cand_cap'],
            d['surf'],
        )

    def test_output_shape(self, expert, data):
        B, M = data['B'], data['M']
        logits = self._forward(expert, data)
        assert logits.shape == (B, M, 2)

    def test_backward(self, expert, data):
        logits = self._forward(expert, data)
        F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()

    def test_expert_params_have_gradients_after_backward(self, expert, data):
        logits = self._forward(expert, data)
        F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
        for p in expert.parameters():
            assert p.grad is not None

    def test_tae_gradient_does_not_flow_when_detached(self, expert, data):
        tae = data['tae'].requires_grad_(True)
        logits = expert(
            tae.detach(),
            data['short_prod'], data['short_ctx'], data['short_act'],
            data['short_cat'], data['short_cap'],
            data['cand_cat'], data['cand_cap'],
            data['surf'],
        )
        F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits)).backward()
        assert tae.grad is None   # detached before passing to expert

    def test_expert_smaller_than_fm(self, expert):
        fm = LLMEnrichedFoundationModel(
            prod_dim=12, ctx_dim=6, act_dim=4, aux_dim=8,
            n_categories=16, d_llm=12,
            d_model=24, n_layers=2, n_heads=4,
            n_main_tasks=3, n_aux_tasks=2,
        )
        expert_params = sum(p.numel() for p in expert.parameters())
        fm_params = sum(p.numel() for p in fm.parameters())
        ratio = expert_params / fm_params
        assert 0.1 <= ratio <= 0.5   # target 20-40%

    def test_param_count_positive(self, expert):
        assert sum(p.numel() for p in expert.parameters()) > 0

    def test_batch_size_one(self, expert, data):
        d = {k: v[:1] for k, v in data.items() if isinstance(v, torch.Tensor)}
        d.update({k: v for k, v in data.items() if not isinstance(v, torch.Tensor)})
        logits = expert(
            d['tae'].detach(),
            d['short_prod'], d['short_ctx'], d['short_act'],
            d['short_cat'], d['short_cap'],
            d['cand_cat'], d['cand_cap'],
            d['surf'],
        )
        assert logits.shape == (1, data['M'], 2)
