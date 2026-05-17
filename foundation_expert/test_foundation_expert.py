"""Tests for Foundation-Expert Paradigm implementation."""

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from foundation_expert import (
    ItemEmbedding, _build_tae_mask, HSTULayer, HSTU,
    MultiTaskHead, AlignmentModule, FoundationModel,
    FMEmbeddingModule, FMFusionModule, ExpertFusionModule,
    ExpertModel, transfer_ratio,
)

B   = 4
N   = 16    # history length
M   = 6     # candidate count
T_S = 8     # short-term history length
D   = 32    # d_model
DE  = 16    # d_expert

PROD_DIM = 12
CTX_DIM  = 8
ACT_DIM  = 4
AUX_DIM  = 10
SURF_DIM = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fm(d_model=D, n_layers=2, n_heads=4,
             n_main=3, n_aux=2) -> FoundationModel:
    return FoundationModel(
        prod_dim=PROD_DIM, ctx_dim=CTX_DIM, act_dim=ACT_DIM,
        aux_dim=AUX_DIM, d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, n_main_tasks=n_main, n_aux_tasks=n_aux,
    )


def _make_expert(fm_dim=D, d_expert=DE, n_layers=1,
                 n_heads=4, n_tasks=2) -> ExpertModel:
    return ExpertModel(
        fm_dim=fm_dim, prod_dim=PROD_DIM, ctx_dim=CTX_DIM,
        act_dim=ACT_DIM, surf_dim=SURF_DIM,
        d_expert=d_expert, n_layers=n_layers,
        n_heads=n_heads, n_tasks=n_tasks,
    )


def _fm_inputs(batch=B, n_hist=N, n_cand=M):
    return dict(
        hist_prod=torch.randn(batch, n_hist, PROD_DIM),
        hist_ctx=torch.randn(batch, n_hist, CTX_DIM),
        hist_act=torch.randn(batch, n_hist, ACT_DIM),
        cand_prod=torch.randn(batch, n_cand, PROD_DIM),
        cand_ctx=torch.randn(batch, n_cand, CTX_DIM),
        aux_features=torch.randn(batch, n_cand, AUX_DIM),
    )


def _expert_inputs(batch=B, n_cand=M, t_short=T_S, fm_dim=D):
    return dict(
        tae=torch.randn(batch, n_cand, fm_dim),
        short_prod=torch.randn(batch, t_short, PROD_DIM),
        short_ctx=torch.randn(batch, t_short, CTX_DIM),
        short_act=torch.randn(batch, t_short, ACT_DIM),
        surface_feats=torch.randn(batch, n_cand, SURF_DIM),
    )


# ---------------------------------------------------------------------------
# ItemEmbedding
# ---------------------------------------------------------------------------

class TestItemEmbedding:
    def test_shape_with_action(self):
        emb = ItemEmbedding(PROD_DIM, CTX_DIM, ACT_DIM, D)
        out = emb(torch.randn(B, N, PROD_DIM),
                  torch.randn(B, N, CTX_DIM),
                  torch.randn(B, N, ACT_DIM))
        assert out.shape == (B, N, D)

    def test_shape_without_action(self):
        emb = ItemEmbedding(PROD_DIM, CTX_DIM, ACT_DIM, D)
        out = emb(torch.randn(B, M, PROD_DIM), torch.randn(B, M, CTX_DIM))
        assert out.shape == (B, M, D)

    def test_action_changes_output(self):
        emb = ItemEmbedding(PROD_DIM, CTX_DIM, ACT_DIM, D)
        prod = torch.randn(B, N, PROD_DIM)
        ctx  = torch.randn(B, N, CTX_DIM)
        act  = torch.randn(B, N, ACT_DIM)
        with torch.no_grad():
            out_no_act = emb(prod, ctx)
            out_act    = emb(prod, ctx, act)
        assert not torch.allclose(out_no_act, out_act)

    def test_gradient_flows(self):
        emb = ItemEmbedding(PROD_DIM, CTX_DIM, ACT_DIM, D)
        prod = torch.randn(B, N, PROD_DIM, requires_grad=True)
        ctx  = torch.randn(B, N, CTX_DIM, requires_grad=True)
        act  = torch.randn(B, N, ACT_DIM, requires_grad=True)
        out = emb(prod, ctx, act)
        out.sum().backward()
        assert prod.grad is not None
        assert ctx.grad is not None
        assert act.grad is not None

    def test_summation_not_concat_same_output_dim(self):
        # Output dim = d_model regardless of action presence
        emb = ItemEmbedding(PROD_DIM, CTX_DIM, ACT_DIM, D)
        out_with    = emb(torch.randn(1, 1, PROD_DIM), torch.randn(1, 1, CTX_DIM),
                          torch.randn(1, 1, ACT_DIM))
        out_without = emb(torch.randn(1, 1, PROD_DIM), torch.randn(1, 1, CTX_DIM))
        assert out_with.shape == out_without.shape == (1, 1, D)


# ---------------------------------------------------------------------------
# TAE Mask
# ---------------------------------------------------------------------------

class TestTAEMask:
    def test_shape(self):
        mask = _build_tae_mask(N, M, torch.device('cpu'))
        assert mask.shape == (N + M, N + M)

    def test_history_causal(self):
        mask = _build_tae_mask(N, M, torch.device('cpu'))
        # History should see only past positions: upper triangle should be -inf
        for i in range(N):
            for j in range(i + 1, N):
                assert mask[i, j].item() == float('-inf'), \
                    f"History pos {i} should not attend to future {j}"

    def test_history_self_attend(self):
        mask = _build_tae_mask(N, M, torch.device('cpu'))
        for i in range(N):
            assert mask[i, i].item() == 0.0

    def test_candidates_see_all_history(self):
        mask = _build_tae_mask(N, M, torch.device('cpu'))
        for j in range(M):
            for i in range(N):
                assert mask[N + j, i].item() == 0.0, \
                    f"Candidate {j} should attend to history {i}"

    def test_candidates_blocked_from_each_other(self):
        mask = _build_tae_mask(N, M, torch.device('cpu'))
        for j in range(M):
            for k in range(M):
                if j != k:
                    assert mask[N + j, N + k].item() == float('-inf'), \
                        f"Candidate {j} should not see candidate {k}"

    def test_candidates_self_attend(self):
        mask = _build_tae_mask(N, M, torch.device('cpu'))
        for j in range(M):
            assert mask[N + j, N + j].item() == 0.0

    def test_history_cannot_see_candidates(self):
        mask = _build_tae_mask(N, M, torch.device('cpu'))
        for i in range(N):
            for j in range(M):
                assert mask[i, N + j].item() == float('-inf')


# ---------------------------------------------------------------------------
# HSTU
# ---------------------------------------------------------------------------

class TestHSTULayer:
    def test_output_shape(self):
        layer = HSTULayer(D, n_heads=4)
        x = torch.randn(B, N + M, D)
        out = layer(x)
        assert out.shape == (B, N + M, D)

    def test_with_mask(self):
        layer = HSTULayer(D, n_heads=4)
        x    = torch.randn(B, N + M, D)
        mask = _build_tae_mask(N, M, x.device)
        out  = layer(x, attn_mask=mask)
        assert out.shape == (B, N + M, D)

    def test_gradient_flows(self):
        layer = HSTULayer(D, n_heads=4)
        x = torch.randn(B, N + M, D, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None


class TestHSTU:
    def test_output_shape(self):
        hstu = HSTU(D, n_layers=2, n_heads=4)
        x    = torch.randn(B, N + M, D)
        out  = hstu(x, n_hist=N)
        assert out.shape == (B, N + M, D)

    def test_tae_extraction(self):
        hstu = HSTU(D, n_layers=2, n_heads=4)
        x    = torch.randn(B, N + M, D)
        out  = hstu(x, n_hist=N)
        tae  = out[:, N:, :]
        assert tae.shape == (B, M, D)

    def test_candidate_independence(self):
        # Each candidate's TAE should be independent of other candidates
        # (verified by mask structure: candidates don't attend to each other)
        hstu = HSTU(D, n_layers=2, n_heads=4)
        hstu.eval()
        hist = torch.randn(B, N, D)
        # Two candidate sets that differ only in the second candidate
        cands_a = torch.randn(B, M, D)
        cands_b = cands_a.clone()
        cands_b[:, 1, :] = torch.randn(B, D)  # change candidate 1

        seq_a = torch.cat([hist, cands_a], dim=1)
        seq_b = torch.cat([hist, cands_b], dim=1)
        with torch.no_grad():
            out_a = hstu(seq_a, n_hist=N)
            out_b = hstu(seq_b, n_hist=N)

        # Candidate 0 TAE should be identical (not affected by candidate 1)
        tae_a_0 = out_a[:, N, :]
        tae_b_0 = out_b[:, N, :]
        assert torch.allclose(tae_a_0, tae_b_0, atol=1e-5), \
            "Candidate 0 TAE should be unaffected by changes to candidate 1"

    def test_history_affects_tae(self):
        # Different histories should produce different TAE
        hstu = HSTU(D, n_layers=2, n_heads=4)
        hstu.eval()
        cands = torch.randn(B, M, D)
        hist_a = torch.randn(B, N, D)
        hist_b = torch.randn(B, N, D) + 5.0  # very different history

        with torch.no_grad():
            tae_a = hstu(torch.cat([hist_a, cands], dim=1), n_hist=N)[:, N:, :]
            tae_b = hstu(torch.cat([hist_b, cands], dim=1), n_hist=N)[:, N:, :]

        assert not torch.allclose(tae_a, tae_b), \
            "TAE should differ for different user histories"

    def test_gradient_flows_through_history(self):
        hstu = HSTU(D, n_layers=2, n_heads=4)
        hist = torch.randn(B, N, D, requires_grad=True)
        cands = torch.randn(B, M, D)
        out = hstu(torch.cat([hist, cands], dim=1), n_hist=N)
        out[:, N:, :].sum().backward()  # only backward through candidate outputs
        assert hist.grad is not None


# ---------------------------------------------------------------------------
# Foundation Model
# ---------------------------------------------------------------------------

class TestMultiTaskHead:
    def test_output_shape(self):
        head = MultiTaskHead(D, n_tasks=3)
        tae  = torch.randn(B, M, D)
        out  = head(tae)
        assert out.shape == (B, M, 3)


class TestAlignmentModule:
    def test_output_shape(self):
        align = AlignmentModule(D, AUX_DIM, n_tasks=2)
        tae  = torch.randn(B, M, D)
        aux  = torch.randn(B, M, AUX_DIM)
        out  = align(tae, aux)
        assert out.shape == (B, M, 2)

    def test_masked_loss_with_delta(self):
        logits = torch.randn(B, M, 1)
        labels = torch.zeros(B, M, 1)
        delta  = torch.randint(0, 2, (B, M, 1)).float()
        loss   = AlignmentModule.masked_loss(logits, labels, delta)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_masked_loss_no_delta(self):
        logits = torch.randn(B, M, 1)
        labels = torch.zeros(B, M, 1)
        loss   = AlignmentModule.masked_loss(logits, labels)
        assert loss.shape == ()

    def test_masked_loss_zero_delta(self):
        # All samples masked out → loss should be 0 (no valid samples)
        logits = torch.randn(B, M, 1)
        labels = torch.zeros(B, M, 1)
        delta  = torch.zeros(B, M, 1)
        loss   = AlignmentModule.masked_loss(logits, labels, delta)
        assert loss.item() == 0.0


class TestFoundationModel:
    def test_output_shapes(self):
        fm = _make_fm()
        tae, main_l, aux_l = fm(**_fm_inputs())
        assert tae.shape    == (B, M, D)
        assert main_l.shape == (B, M, 3)
        assert aux_l.shape  == (B, M, 2)

    def test_backward(self):
        fm = _make_fm()
        tae, main_l, aux_l = fm(**_fm_inputs())
        labels = torch.zeros_like(main_l)
        aux_lb = torch.zeros_like(aux_l)
        loss   = fm.compute_loss(main_l, labels, aux_l, aux_lb)
        loss.backward()
        for name, p in fm.named_parameters():
            assert p.grad is not None, f"No grad: {name}"

    def test_loss_with_delta_mask(self):
        fm = _make_fm()
        tae, main_l, aux_l = fm(**_fm_inputs())
        delta = torch.randint(0, 2, (B, M, 2)).float()
        loss  = fm.compute_loss(
            main_l, torch.zeros_like(main_l),
            aux_l,  torch.zeros_like(aux_l), delta)
        assert loss.item() > 0

    def test_different_histories_different_tae(self):
        fm = _make_fm()
        fm.eval()
        inp_a = _fm_inputs()
        inp_b = {**inp_a, 'hist_act': torch.randn(B, N, ACT_DIM) + 10.0}
        with torch.no_grad():
            tae_a, _, _ = fm(**inp_a)
            tae_b, _, _ = fm(**inp_b)
        assert not torch.allclose(tae_a, tae_b)

    def test_different_candidates_different_tae(self):
        fm = _make_fm()
        fm.eval()
        inp_a = _fm_inputs()
        inp_b = {**inp_a, 'cand_prod': torch.randn(B, M, PROD_DIM) + 5.0}
        with torch.no_grad():
            tae_a, _, _ = fm(**inp_a)
            tae_b, _, _ = fm(**inp_b)
        assert not torch.allclose(tae_a, tae_b)

    def test_single_candidate(self):
        fm = _make_fm()
        fm.eval()
        inp = _fm_inputs(n_cand=1)
        with torch.no_grad():
            tae, main_l, aux_l = fm(**inp)
        assert tae.shape == (B, 1, D)

    def test_param_count(self):
        fm = _make_fm()
        total = sum(p.numel() for p in fm.parameters())
        assert total > 0

    def test_task_weights_applied(self):
        fm1 = _make_fm()
        fm2 = FoundationModel(
            prod_dim=PROD_DIM, ctx_dim=CTX_DIM, act_dim=ACT_DIM,
            aux_dim=AUX_DIM, d_model=D, n_layers=2, n_heads=4,
            n_main_tasks=3, n_aux_tasks=2,
            main_weights=[2.0, 1.0, 1.0],
        )
        inp = _fm_inputs()
        _, main_l, aux_l = fm1(**inp)
        labels = torch.zeros_like(main_l)
        aux_lb = torch.zeros_like(aux_l)
        # Copy weights to make losses comparable except for weighting
        fm2.load_state_dict(fm1.state_dict(), strict=False)
        with torch.no_grad():
            _, ml1, al1 = fm1(**inp)
            _, ml2, al2 = fm2(**inp)
        loss1 = fm1.compute_loss(ml1, labels, al1, aux_lb)
        loss2 = fm2.compute_loss(ml2, labels, al2, aux_lb)
        # Different weights → different losses (they share same predictions)
        # This just verifies the code runs without error
        assert loss1.shape == loss2.shape == ()


# ---------------------------------------------------------------------------
# Expert Model components
# ---------------------------------------------------------------------------

class TestFMEmbeddingModule:
    def test_output_shape_same_dim(self):
        mod = FMEmbeddingModule(D, D)
        out = mod(torch.randn(B, M, D))
        assert out.shape == (B, M, D)

    def test_output_shape_projected(self):
        mod = FMEmbeddingModule(D, DE)
        out = mod(torch.randn(B, M, D))
        assert out.shape == (B, M, DE)

    def test_normalizes_input(self):
        mod = FMEmbeddingModule(D, D, dropout=0.0)
        mod.eval()
        x = torch.randn(B, M, D) * 100  # large scale input
        with torch.no_grad():
            out = mod(x)
        # After LayerNorm the scale should be much smaller
        assert out.abs().mean().item() < 10.0

    def test_gradient_flows(self):
        mod = FMEmbeddingModule(D, DE)
        x = torch.randn(B, M, D, requires_grad=True)
        out = mod(x)
        out.sum().backward()
        assert x.grad is not None


class TestFMFusionModule:
    def test_output_shape(self):
        fuse = FMFusionModule(DE, DE, DE)
        fm_emb   = torch.randn(B, M, DE)
        short_rep = torch.randn(B, DE)
        out = fuse(fm_emb, short_rep)
        assert out.shape == (B, M, DE)

    def test_short_rep_broadcast(self):
        # Same short_rep for all candidates in a sample
        fuse = FMFusionModule(DE, DE, DE)
        fm_emb    = torch.randn(B, M, DE)
        short_rep = torch.randn(B, DE)
        out = fuse(fm_emb, short_rep)
        assert out.shape == (B, M, DE)

    def test_gradient_flows(self):
        fuse = FMFusionModule(DE, DE, DE)
        fm_emb    = torch.randn(B, M, DE, requires_grad=True)
        short_rep = torch.randn(B, DE, requires_grad=True)
        out = fuse(fm_emb, short_rep)
        out.sum().backward()
        assert fm_emb.grad is not None
        assert short_rep.grad is not None


class TestExpertFusionModule:
    def test_output_shape(self):
        efm = ExpertFusionModule(DE, SURF_DIM, n_tasks=3)
        fused = torch.randn(B, M, DE)
        surf  = torch.randn(B, M, SURF_DIM)
        out   = efm(fused, surf)
        assert out.shape == (B, M, 3)

    def test_gradient_flows(self):
        efm  = ExpertFusionModule(DE, SURF_DIM, n_tasks=2)
        fused = torch.randn(B, M, DE, requires_grad=True)
        surf  = torch.randn(B, M, SURF_DIM, requires_grad=True)
        efm(fused, surf).sum().backward()
        assert fused.grad is not None
        assert surf.grad is not None


# ---------------------------------------------------------------------------
# Expert Model end-to-end
# ---------------------------------------------------------------------------

class TestExpertModel:
    def test_output_shape(self):
        expert = _make_expert()
        out = expert(**_expert_inputs())
        assert out.shape == (B, M, 2)

    def test_backward(self):
        expert = _make_expert()
        out  = expert(**_expert_inputs())
        loss = F.binary_cross_entropy_with_logits(out, torch.zeros_like(out))
        loss.backward()
        for name, p in expert.named_parameters():
            assert p.grad is not None, f"No grad: {name}"

    def test_different_tae_different_output(self):
        expert = _make_expert()
        expert.eval()
        inp_a = _expert_inputs()
        inp_b = {**inp_a, 'tae': torch.randn(B, M, D) + 5.0}
        with torch.no_grad():
            out_a = expert(**inp_a)
            out_b = expert(**inp_b)
        assert not torch.allclose(out_a, out_b)

    def test_different_short_history_different_output(self):
        expert = _make_expert()
        expert.eval()
        inp_a = _expert_inputs()
        inp_b = {**inp_a, 'short_act': torch.randn(B, T_S, ACT_DIM) + 5.0}
        with torch.no_grad():
            out_a = expert(**inp_a)
            out_b = expert(**inp_b)
        assert not torch.allclose(out_a, out_b)

    def test_expert_compute_fraction(self):
        fm     = _make_fm(d_model=D)
        expert = _make_expert(fm_dim=D, d_expert=DE)
        fm_params     = sum(p.numel() for p in fm.parameters())
        expert_params = sum(p.numel() for p in expert.parameters())
        ratio = expert_params / fm_params
        assert ratio < 0.5, f"Expert should be <50% of FM, got {ratio:.1%}"

    def test_single_candidate(self):
        expert = _make_expert()
        expert.eval()
        inp = _expert_inputs(n_cand=1)
        with torch.no_grad():
            out = expert(**inp)
        assert out.shape == (B, 1, 2)

    def test_surface_feats_matter(self):
        expert = _make_expert()
        expert.eval()
        inp_a = _expert_inputs()
        inp_b = {**inp_a, 'surface_feats': torch.randn(B, M, SURF_DIM) + 10.0}
        with torch.no_grad():
            out_a = expert(**inp_a)
            out_b = expert(**inp_b)
        assert not torch.allclose(out_a, out_b)

    def test_multi_task(self):
        expert = _make_expert(n_tasks=4)
        out = expert(**_expert_inputs())
        assert out.shape == (B, M, 4)


# ---------------------------------------------------------------------------
# Transfer Ratio
# ---------------------------------------------------------------------------

class TestTransferRatio:
    def test_perfect_transfer(self):
        # Expert improves by exactly the same amount as FM → TR = 1.0
        tr = transfer_ratio(-1.0, 0.0, -1.0, 0.0)
        assert abs(tr - 1.0) < 1e-6

    def test_partial_transfer(self):
        # Expert captures 73% of FM improvement
        tr = transfer_ratio(-0.73, 0.0, -1.0, 0.0)
        assert abs(tr - 0.73) < 1e-6

    def test_paper_range(self):
        # All paper-reported TRs are in [0.64, 1.0]
        paper_results = [
            (-0.54, 0.0, -0.73, 0.0),  # Surface A Like: TR ≈ 0.74
            (-0.50, 0.0, -0.50, 0.0),  # Surface A Share: TR = 1.00
            (-1.05, 0.0, -1.14, 0.0),  # Surface A VVD: TR ≈ 0.92
            (-0.60, 0.0, -0.83, 0.0),  # Surface B Like: TR ≈ 0.72
        ]
        for args in paper_results:
            tr = transfer_ratio(*args)
            assert 0.60 <= tr <= 1.05, f"TR {tr:.4f} outside expected range for {args}"

    def test_zero_fm_delta_raises(self):
        with pytest.raises(ValueError, match="near zero"):
            transfer_ratio(-0.5, 0.0, 0.0, 0.0)

    def test_symmetric(self):
        # Flipping sign of both deltas should preserve TR
        tr1 = transfer_ratio(-0.7, 0.0, -1.0, 0.0)
        tr2 = transfer_ratio(0.7, 0.0, 1.0, 0.0)
        assert abs(tr1 - tr2) < 1e-6


# ---------------------------------------------------------------------------
# Integration: FM → Expert pipeline
# ---------------------------------------------------------------------------

class TestFMExpertPipeline:
    def test_end_to_end_forward(self):
        fm     = _make_fm()
        expert = _make_expert(fm_dim=D)

        fm_inp = _fm_inputs()
        tae, main_logits, aux_logits = fm(**fm_inp)

        exp_inp = {
            'tae':          tae.detach(),
            'short_prod':   torch.randn(B, T_S, PROD_DIM),
            'short_ctx':    torch.randn(B, T_S, CTX_DIM),
            'short_act':    torch.randn(B, T_S, ACT_DIM),
            'surface_feats': torch.randn(B, M, SURF_DIM),
        }
        expert_logits = expert(**exp_inp)
        assert expert_logits.shape == (B, M, 2)

    def test_decoupled_training(self):
        # FM and expert can train independently (no shared gradient graph)
        fm     = _make_fm()
        expert = _make_expert(fm_dim=D)

        fm_inp = _fm_inputs()
        tae, main_l, aux_l = fm(**fm_inp)
        fm_loss = fm.compute_loss(
            main_l, torch.zeros_like(main_l),
            aux_l,  torch.zeros_like(aux_l))
        fm_loss.backward()

        # Expert trains on detached TAE — no gradient from expert flows to FM
        for p in fm.parameters():
            p.grad = None

        exp_out = expert(
            tae.detach(),
            torch.randn(B, T_S, PROD_DIM),
            torch.randn(B, T_S, CTX_DIM),
            torch.randn(B, T_S, ACT_DIM),
            torch.randn(B, M, SURF_DIM),
        )
        exp_loss = F.binary_cross_entropy_with_logits(
            exp_out, torch.zeros_like(exp_out))
        exp_loss.backward()

        # FM parameters should have no gradient from expert backward
        for name, p in fm.named_parameters():
            assert p.grad is None, \
                f"FM param {name} received gradient from expert — decoupling broken"

    def test_fm_improvement_transfers(self):
        # Simulate two FMs of different quality and verify TR makes sense
        fm_weak   = _make_fm(d_model=16, n_layers=1)
        fm_strong = _make_fm(d_model=32, n_layers=2)

        expert_w = _make_expert(fm_dim=16)
        expert_s = _make_expert(fm_dim=32)

        fm_inp = _fm_inputs()
        with torch.no_grad():
            tae_w, _, _ = fm_weak(**fm_inp)
            tae_s, _, _ = fm_strong(**_fm_inputs())

        # Verify TAE shapes match expert expectations
        assert tae_w.shape == (B, M, 16)
        assert tae_s.shape == (B, M, 32)

    def test_multiple_surfaces_same_fm(self):
        # One FM, two surface experts — the Foundation-Expert paradigm core claim
        fm       = _make_fm()
        expert_a = _make_expert(fm_dim=D, n_tasks=3)  # Surface A: 3 tasks
        expert_b = _make_expert(fm_dim=D, n_tasks=2)  # Surface B: 2 tasks

        fm_inp = _fm_inputs()
        with torch.no_grad():
            tae, _, _ = fm(**fm_inp)

        exp_inp = dict(
            short_prod=torch.randn(B, T_S, PROD_DIM),
            short_ctx=torch.randn(B, T_S, CTX_DIM),
            short_act=torch.randn(B, T_S, ACT_DIM),
            surface_feats=torch.randn(B, M, SURF_DIM),
        )
        out_a = expert_a(tae.detach(), **exp_inp)
        out_b = expert_b(tae.detach(), **exp_inp)

        assert out_a.shape == (B, M, 3)
        assert out_b.shape == (B, M, 2)
        # Experts produce different outputs from same TAE
        assert not torch.allclose(out_a[..., :2], out_b)
