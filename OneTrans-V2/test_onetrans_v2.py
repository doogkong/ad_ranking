"""Tests for the OneTrans-V2 implementation.

Run with:
    pytest test_onetrans_v2.py -v
"""

import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from onetrans_v2 import (
    Config,
    RMSNorm,
    MixedLinear,
    SparseMoE,
    rotate_time,
    compute_anchors,
    StageBatch,
    build_stage_mask,
    TransBlock,
    OneTransV2Backbone,
    OneTransV2,
    distillation_loss,
    two_stage_topk,
    hit_rate_at_m,
    mup_hidden_lr,
    mup_param_groups,
    build_optimizers,
    sid_to_items,
    make_batch,
    N_RETR_TOKENS,
)


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def model(cfg):
    torch.manual_seed(0)
    return OneTransV2(cfg).eval()


def slice_batch(batch, b=None, exposures=None, prefix=None):
    """Select exposures and/or truncate the behavior sequence."""
    out = dict(batch)
    for k in ("t_req", "ctx", "pre_feat", "fine_feat", "z_oc", "z_disc", "z_ad", "z_aov", "sid", "y_pre", "y_fine"):
        if exposures is not None:
            out[k] = out[k][:, exposures]
    if prefix is not None:
        for k in ("seq_items", "seq_actions", "seq_times"):
            out[k] = out[k][:, :prefix]
    return out


# ---------------------------------------------------------------------------
# RMSNorm / MixedLinear / MoE
# ---------------------------------------------------------------------------

class TestBlocks:
    def test_rmsnorm_unit_rms(self):
        y = RMSNorm(16)(torch.randn(4, 16) * 5)
        assert torch.allclose(y.pow(2).mean(-1), torch.ones(4), atol=1e-3)

    def test_mixed_linear_slots_independent(self):
        ml = MixedLinear(8, 6, n_slots=3)
        x = torch.randn(2, 2, 8)
        y = ml.forward_slots(x, torch.tensor([0, 2]))
        assert torch.allclose(y[:, 0], x[:, 0] @ ml.slot[0], atol=1e-5)
        assert torch.allclose(y[:, 1], x[:, 1] @ ml.slot[2], atol=1e-5)

    def test_mixed_linear_init_fan_in_variance(self):
        ml = MixedLinear(400, 300, n_slots=2)
        assert ml.shared.var().item() == pytest.approx(1 / 400, rel=0.15)

    def test_moe_shape_and_router_weights(self):
        moe = SparseMoE(16, n_routed=6, top_k=2, d_expert=8, routed_scale=2.5)
        x = torch.randn(3, 5, 16)
        assert moe(x).shape == x.shape
        w, idx = moe.route(x.reshape(-1, 16))
        assert torch.allclose(w.sum(-1), torch.ones(15), atol=1e-5)      # renormalized to 1
        assert idx.shape == (15, 2)

    def test_moe_sigmoid_router_is_independent_per_expert(self):
        moe = SparseMoE(8, 4, 2, 4)
        s = torch.sigmoid(moe.router(torch.randn(10, 8)))
        assert not torch.allclose(s.sum(-1), torch.ones(10))          # not a softmax simplex

    def test_moe_matches_dense_reference(self):
        moe = SparseMoE(8, 4, 2, 6, routed_scale=1.5)
        x = torch.randn(7, 8)
        w, idx = moe.route(x)
        ref = moe.shared(x).clone()
        for n in range(7):
            for j in range(2):
                ref[n] += 1.5 * w[n, j] * moe.experts[idx[n, j]](x[n])
        assert torch.allclose(moe(x), ref, atol=1e-5)

    def test_moe_gradients_flow_to_router(self):
        moe = SparseMoE(8, 4, 2, 4)
        moe(torch.randn(6, 8)).sum().backward()
        assert moe.router.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Time rotary encoding (Eq. 11)
# ---------------------------------------------------------------------------

class TestTimeRotary:
    P = (10.0, 100.0)

    def test_relative_time_only(self):
        q, k = torch.randn(4), torch.randn(4)
        te, ti = torch.tensor(37.0), torch.tensor(12.0)
        a = (rotate_time(q, te, self.P) * rotate_time(k, ti, self.P)).sum()
        shift = 555.0
        b = (rotate_time(q, te + shift, self.P) * rotate_time(k, ti + shift, self.P)).sum()
        assert a.item() == pytest.approx(b.item(), abs=1e-3)

    def test_matches_rotation_by_difference(self):
        q, k = torch.randn(4), torch.randn(4)
        te, ti = torch.tensor(50.0), torch.tensor(20.0)
        lhs = (rotate_time(q, te, self.P) * rotate_time(k, ti, self.P)).sum()
        rhs = (q * rotate_time(k, ti - te, self.P)).sum()
        assert lhs.item() == pytest.approx(rhs.item(), abs=1e-4)

    def test_preserves_norm_and_zero_is_identity(self):
        x = torch.randn(3, 4)
        assert torch.allclose(rotate_time(x, torch.tensor(123.4), self.P).norm(dim=-1), x.norm(dim=-1), atol=1e-5)
        assert torch.allclose(rotate_time(x, torch.tensor(0.0), self.P), x)

    def test_full_period_is_identity(self):
        x = torch.randn(2, 2)
        assert torch.allclose(rotate_time(x, torch.tensor(10.0), (10.0,)), x, atol=1e-4)


# ---------------------------------------------------------------------------
# Anchors and the stage visibility mask (Sec 4.1)
# ---------------------------------------------------------------------------

class TestAnchorsAndMask:
    def test_anchors(self):
        bt = torch.tensor([[1.0, 2.0, 3.0, 5.0]])
        rt = torch.tensor([[0.5, 2.0, 4.0, 9.0]])
        assert compute_anchors(bt, rt).tolist() == [[-1, 1, 2, 3]]

    def make_stage(self, anchors=(2, 3), per=3, L=5):
        E = len(anchors)
        M = E * per
        return StageBatch(
            x=torch.zeros(1, M, 4),
            slot=torch.arange(per).repeat(E),
            group=torch.arange(E).repeat_interleave(per),
            anchor=torch.tensor(anchors).repeat_interleave(per).view(1, M),
            t_req=torch.zeros(1, M),
        ), L

    def test_mask_structure(self):
        stage, L = self.make_stage()
        m = build_stage_mask(stage, L)[0]                      # [M, L+M]
        assert m.shape == (6, 11)
        # exposure 0 (anchor 2): sees S[0..2], not S[3..4]
        assert m[0, :L].tolist() == [True, True, True, False, False]
        # exposure 1 (anchor 3)
        assert m[3, :L].tolist() == [True, True, True, True, False]
        # within-group causality: token 1 sees token 0 and itself, not token 2
        assert m[1, L:L + 3].tolist() == [True, True, False]
        # no cross-exposure visibility
        assert not m[0, L + 3:].any() and not m[3, L:L + 3].any()

    def test_every_token_sees_itself(self):
        stage, L = self.make_stage(anchors=(-1, -1))
        m = build_stage_mask(stage, L)[0]
        assert m[torch.arange(6), L + torch.arange(6)].all()
        assert not m[:, :L].any()                               # anchor -1: no behavior visible

    def test_stage_groups_isolated(self):
        """Different stages of the same exposure cannot see each other."""
        M, L = 4, 3
        stage = StageBatch(torch.zeros(1, M, 4), torch.arange(M), torch.tensor([0, 0, 1, 2]),
                           torch.full((1, M), 2), torch.zeros(1, M))
        m = build_stage_mask(stage, L)[0, :, L:]
        assert not m[2, :2].any() and not m[3, :3].any() and not m[0, 2:].any()


# ---------------------------------------------------------------------------
# TransBlock / backbone
# ---------------------------------------------------------------------------

class TestBackbone:
    def test_residual_multiplier(self, cfg):
        assert TransBlock(cfg, 8).gamma == pytest.approx(1 / math.sqrt(16))

    def test_behavior_stream_is_causal(self, cfg):
        blk = TransBlock(cfg, 2).eval()
        x, t = torch.randn(1, 6, cfg.d_model), torch.arange(6.0).view(1, 6)
        y1, _ = blk.forward_behavior(x, t)
        x2 = x.clone(); x2[:, 4:] = torch.randn(1, 2, cfg.d_model)
        y2, _ = blk.forward_behavior(x2, t)
        assert torch.allclose(y1[:, :4], y2[:, :4], atol=1e-5)
        assert not torch.allclose(y1[:, 4:], y2[:, 4:], atol=1e-5)

    def test_behavior_stream_ignores_absolute_time(self, cfg):
        """Behavior self-attention is left unchanged by the time encoding."""
        blk = TransBlock(cfg, 2).eval()
        x, t = torch.randn(1, 6, cfg.d_model), torch.rand(1, 6) * 100
        a, _ = blk.forward_behavior(x, t)
        b, _ = blk.forward_behavior(x, t + 12345.0)
        assert torch.allclose(a, b, atol=1e-5)

    def test_gqa_cache_shape(self, cfg):
        bb = OneTransV2Backbone(cfg)
        _, caches = bb.encode_context(torch.randn(2, 7, cfg.d_model), torch.arange(7.0).expand(2, 7))
        assert len(caches) == cfg.n_layers
        assert caches[0][0].shape == (2, cfg.n_kv_heads, 7, cfg.head_dim)
        assert cfg.n_kv_heads < cfg.n_heads

    def test_stage_output_shape(self, cfg):
        bb = OneTransV2Backbone(cfg).eval()
        M = cfg.per_exposure
        stage = StageBatch(torch.randn(2, M, cfg.d_model), torch.arange(M), torch.zeros(M, dtype=torch.long),
                           torch.full((2, M), 4), torch.full((2, M), 10.0))
        assert bb(torch.randn(2, 6, cfg.d_model), torch.arange(6.0).expand(2, 6), stage).shape == (2, M, cfg.d_model)

    def test_time_dims_fit_head(self):
        with pytest.raises(AssertionError):
            TransBlock(Config(head_dim=2, time_periods=(1.0, 2.0)), 2)


# ---------------------------------------------------------------------------
# Sequence-Native Training: causal views, isolation
# ---------------------------------------------------------------------------

class TestSNT:
    def setup_method(self):
        self.cfg = Config()
        torch.manual_seed(0)
        self.model = OneTransV2(self.cfg).eval()
        self.batch = make_batch(self.cfg, B=2, L=12, E=3)

    def fwd(self, batch):
        with torch.no_grad():
            return self.model(batch)

    def test_output_shapes(self):
        out = self.fwd(self.batch)
        c = self.cfg
        assert out["oc"].shape == (2, 3, c.n_oc) and out["aov"].shape == (2, 3, c.n_aov)
        assert out["sid"].shape == (2, 3, c.sid_levels, c.sid_vocab)
        assert out["pre"].shape == (2, 3, c.n_pre_targets) and out["fine"].shape == (2, 3, c.n_fine_targets)

    def test_exposure_sees_only_its_causal_prefix(self):
        """Core SNT property: an exposure in a multi-exposure sequence encoded ONCE equals that
        exposure alone on the truncated prefix S[:anchor+1]."""
        out = self.fwd(self.batch)
        anchors = compute_anchors(self.batch["seq_times"], self.batch["t_req"])
        for e in range(3):
            a = int(anchors[:, e].min())
            if not torch.equal(anchors[:, e], torch.full_like(anchors[:, e], a)):
                continue
            single = self.fwd(slice_batch(self.batch, exposures=[e], prefix=a + 1))
            for key in ("oc", "aov", "pre", "fine", "sid"):
                assert torch.allclose(out[key][:, e], single[key][:, 0], atol=1e-4), (key, e)

    def test_future_behavior_does_not_leak(self):
        b = {k: v.clone() for k, v in self.batch.items()}
        anchors = compute_anchors(b["seq_times"], b["t_req"])
        a0 = int(anchors[:, 0].min())
        ref = self.fwd(b)
        b["seq_items"][:, a0 + 1:] = (b["seq_items"][:, a0 + 1:] + 7) % self.cfg.item_vocab
        new = self.fwd(b)
        # exposure 0 cannot see anything after its anchor
        assert torch.allclose(ref["pre"][:, 0], new["pre"][:, 0], atol=1e-5)
        assert torch.allclose(ref["fine"][:, 0], new["fine"][:, 0], atol=1e-5)

    def test_other_exposures_do_not_interfere(self):
        b = {k: v.clone() for k, v in self.batch.items()}
        ref = self.fwd(b)
        b["fine_feat"][:, 1] = torch.randn_like(b["fine_feat"][:, 1])
        b["ctx"][:, 2] = torch.randn_like(b["ctx"][:, 2])
        new = self.fwd(b)
        assert torch.allclose(ref["fine"][:, 0], new["fine"][:, 0], atol=1e-5)
        assert torch.allclose(ref["oc"][:, 1], new["oc"][:, 1], atol=1e-5)

    def test_stages_are_isolated(self):
        """Fine-rank/retrieval tokens cannot influence pre-rank, and vice versa."""
        b = {k: v.clone() for k, v in self.batch.items()}
        ref = self.fwd(b)
        b["fine_feat"] = torch.randn_like(b["fine_feat"])
        b["ctx"] = torch.randn_like(b["ctx"]); b["sid"] = (b["sid"] + 1) % self.cfg.sid_vocab
        new = self.fwd(b)
        assert torch.allclose(ref["pre"], new["pre"], atol=1e-5)
        b2 = {k: v.clone() for k, v in self.batch.items()}
        b2["pre_feat"] = torch.randn_like(b2["pre_feat"])
        new2 = self.fwd(b2)
        assert torch.allclose(ref["fine"], new2["fine"], atol=1e-5)
        assert torch.allclose(ref["sid"], new2["sid"], atol=1e-5)

    def test_sid_logits_causal_within_retrieval_stage(self):
        """SID_1 logits may depend on SID_0 input but SID_0 logits must not depend on SID_0/1 inputs."""
        b = {k: v.clone() for k, v in self.batch.items()}
        ref = self.fwd(b)
        b["sid"][..., 0] = (b["sid"][..., 0] + 3) % self.cfg.sid_vocab
        new = self.fwd(b)
        assert torch.allclose(ref["sid"][:, :, 0], new["sid"][:, :, 0], atol=1e-5)
        assert not torch.allclose(ref["sid"][:, :, 1], new["sid"][:, :, 1], atol=1e-5)

    def test_time_shift_invariance(self):
        """Request-relative time encoding: shifting all timestamps leaves outputs unchanged."""
        b = {k: v.clone() for k, v in self.batch.items()}
        ref = self.fwd(b)
        b["seq_times"] = b["seq_times"] + 5e4
        b["t_req"] = b["t_req"] + 5e4
        new = self.fwd(b)
        for key in ("pre", "fine", "oc"):
            assert torch.allclose(ref[key], new[key], atol=1e-3)

    def test_request_age_matters(self):
        """Yet the age of a behavior relative to the request changes the output."""
        b = {k: v.clone() for k, v in self.batch.items()}
        ref = self.fwd(b)
        b["t_req"] = b["t_req"] + 3000.0           # same anchors (all behaviors older), different ages
        new = self.fwd(b)
        assert not torch.allclose(ref["fine"], new["fine"], atol=1e-4)

    def test_token_specific_parameters(self):
        """Swapping slot weights changes outputs; shared behavior stream is unaffected."""
        out = self.fwd(self.batch)
        with torch.no_grad():
            for blk in self.model.backbone.blocks:
                blk.qkvg.slot[self.cfg.stage_offset["P"]].add_(torch.randn_like(blk.qkvg.slot[0]))
        new = self.fwd(self.batch)
        assert not torch.allclose(out["pre"], new["pre"], atol=1e-5)
        assert torch.allclose(out["fine"], new["fine"], atol=1e-5)   # different slot untouched


# ---------------------------------------------------------------------------
# Losses: DCGR, ranking, KD (Eq. 6-9)
# ---------------------------------------------------------------------------

class TestLosses:
    def test_distillation_matches_formula(self):
        o_t, o_s = torch.randn(5, 2), torch.randn(5, 2)
        mu_t, mu_s, tau = torch.tensor([0.3, -0.2]), torch.tensor([0.1, 0.4]), 0.5
        pt, ps = torch.sigmoid((o_t - mu_t) / tau), torch.sigmoid((o_s - mu_s) / tau)
        ref = (-(pt * ps.log() + (1 - pt) * (1 - ps).log())).sum(-1).mean()
        assert distillation_loss(o_t, o_s, mu_t, mu_s, tau).item() == pytest.approx(ref.item(), rel=1e-5)

    def test_distillation_detaches_teacher(self):
        o_t = torch.randn(4, 2, requires_grad=True)
        o_s = torch.randn(4, 2, requires_grad=True)
        z = torch.zeros(2)
        distillation_loss(o_t, o_s, z, z).backward()
        assert o_t.grad is None and o_s.grad is not None

    def test_distillation_minimized_when_student_matches_teacher(self):
        o = torch.randn(6, 2)
        z = torch.zeros(2)
        same = distillation_loss(o, o.clone(), z, z)
        diff = distillation_loss(o, torch.randn(6, 2) * 3, z, z)
        assert same < diff

    def test_distillation_shift_invariance_through_centering(self):
        o_t, o_s = torch.randn(6, 2), torch.randn(6, 2)
        z = torch.zeros(2)
        a = distillation_loss(o_t, o_s, z, z)
        b = distillation_loss(o_t + 5, o_s - 3, torch.full((2,), 5.0), torch.full((2,), -3.0))
        assert a.item() == pytest.approx(b.item(), abs=1e-5)

    def test_total_loss_is_weighted_sum(self, cfg):
        torch.manual_seed(0)
        m = OneTransV2(cfg).eval()
        batch = make_batch(cfg)
        l = m.loss(batch, m(batch))
        total = cfg.lam_r * l["retrieval"] + cfg.lam_p * l["pre"] + cfg.lam_f * l["fine"] + cfg.lam_kd * l["kd"]
        assert l["loss"].item() == pytest.approx(total.item(), rel=1e-5)

    def test_paper_default_loss_weights(self):
        c = Config()
        assert (c.lam_r, c.lam_p, c.lam_f, c.lam_kd, c.tau) == (0.1, 1.0, 1.0, 1.0, 0.5)

    def test_kd_gradient_only_moves_student(self, cfg):
        """L_kd must not change the fine-rank (teacher) parameters' gradient."""
        torch.manual_seed(0)
        m = OneTransV2(cfg).train()
        batch = make_batch(cfg)
        out = m(batch)
        l_kd = cfg.lam_kd * m.loss(batch, out)["kd"]
        m.zero_grad()
        l_kd.backward()
        assert m.pre_head.weight.grad.abs().sum() > 0
        assert m.fine_head.weight.grad is None or m.fine_head.weight.grad.abs().sum() == 0

    def test_kd_ema_updates_in_training_only(self, cfg):
        torch.manual_seed(0)
        m = OneTransV2(cfg)
        batch = make_batch(cfg)
        m.eval(); m.loss(batch, m(batch))
        assert m.kd_mu_t.abs().sum() == 0
        m.train(); m.loss(batch, m(batch))
        assert m.kd_mu_t.abs().sum() > 0

    def test_all_losses_backprop_and_update(self, cfg):
        torch.manual_seed(0)
        m = OneTransV2(cfg).train()
        batch = make_batch(cfg)
        sparse, dense = build_optimizers(m)
        before = m.pre_head.weight.detach().clone()
        for _ in range(3):
            l = m.loss(batch, m(batch, null_prob=0.2))
            sparse.zero_grad(); dense.zero_grad()
            l["loss"].backward()
            sparse.step(); dense.step()
        assert not torch.allclose(before, m.pre_head.weight)
        assert torch.isfinite(l["loss"])

    def test_overfits_tiny_batch(self, cfg):
        torch.manual_seed(0)
        m = OneTransV2(cfg).train()
        batch = make_batch(cfg, B=2, L=8, E=2)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        first = None
        for _ in range(60):
            l = m.loss(batch, m(batch))
            first = first if first is not None else l["retrieval"].item()
            opt.zero_grad(); l["loss"].backward(); opt.step()
        assert l["retrieval"].item() < 0.5 * first

    def test_null_token_masks_aov_loss(self, cfg):
        torch.manual_seed(0)
        m = OneTransV2(cfg).eval()
        batch = make_batch(cfg)
        out = m(batch)
        out_all_null = dict(out, null=torch.ones_like(out["null"]))
        bad = dict(batch, z_aov=(batch["z_aov"] + 1) % cfg.n_aov)
        a = m.loss(batch, out_all_null)["retrieval"]
        b = m.loss(bad, out_all_null)["retrieval"]
        # aov term is fully masked -> retrieval loss unchanged when only aov labels change
        assert a.item() == pytest.approx(b.item(), abs=1e-6)

    def test_null_prob_replaces_decision_tokens(self, cfg):
        torch.manual_seed(0)
        m = OneTransV2(cfg).train()
        batch = make_batch(cfg)
        out = m(batch, null_prob=1.0)
        assert out["null"].all()
        m.eval()
        assert not m(batch, null_prob=1.0)["null"].any()     # only active in training


# ---------------------------------------------------------------------------
# Serving: shared context, pre-rank, fine-rank
# ---------------------------------------------------------------------------

class TestServing:
    def setup_method(self):
        self.cfg = Config()
        torch.manual_seed(0)
        self.model = OneTransV2(self.cfg).eval()
        self.batch = make_batch(self.cfg, B=1, L=10, E=1)
        self.anchor = int(compute_anchors(self.batch["seq_times"], self.batch["t_req"])[0, 0])
        self.t_req = float(self.batch["t_req"][0, 0])
        b = self.batch
        self.caches = self.model.encode_user(b["seq_items"], b["seq_actions"], b["seq_times"])

    def test_cached_prerank_matches_joint_forward(self):
        with torch.no_grad():
            out = self.model(self.batch)
        pre = self.model.prerank(self.caches, self.batch["pre_feat"][:, 0], self.anchor, self.t_req)
        assert torch.allclose(pre, out["pre"][:, 0], atol=1e-4)

    def test_cached_finerank_matches_joint_forward(self):
        with torch.no_grad():
            out = self.model(self.batch)
        fine = self.model.finerank(self.caches, self.batch["fine_feat"][:, 0], self.anchor, self.t_req)
        assert torch.allclose(fine, out["fine"][:, 0], atol=1e-4)

    def test_candidates_scored_independently(self):
        feats = torch.randn(6, self.cfg.pre_dim)
        batched = self.model.prerank(self.caches, feats, self.anchor, self.t_req)
        single = self.model.prerank(self.caches, feats[2:3], self.anchor, self.t_req)
        assert torch.allclose(batched[2:3], single, atol=1e-5)

    def test_one_encoding_serves_all_stages(self):
        """The same cache object feeds retrieval, pre-rank and fine-rank."""
        ret = self.model.retrieve(self.caches, self.batch["ctx"][:, 0], self.anchor, self.t_req, k=3)
        pre = self.model.prerank(self.caches, torch.randn(4, self.cfg.pre_dim), self.anchor, self.t_req)
        fine = self.model.finerank(self.caches, torch.randn(2, self.cfg.fine_dim), self.anchor, self.t_req)
        assert len(ret) == 3 and pre.shape == (4, 2) and fine.shape == (2, 2)


# ---------------------------------------------------------------------------
# DCGR decoding (Eq. 2-5)
# ---------------------------------------------------------------------------

class TestDCGR:
    def setup_method(self):
        self.cfg = Config(sid_vocab=8)
        torch.manual_seed(1)
        self.model = OneTransV2(self.cfg).eval()
        b = make_batch(self.cfg, B=1, L=10, E=1, seed=1)
        self.ctx = b["ctx"][:, 0]
        self.anchor = int(compute_anchors(b["seq_times"], b["t_req"])[0, 0])
        self.t_req = float(b["t_req"][0, 0])
        self.caches = self.model.encode_user(b["seq_items"], b["seq_actions"], b["seq_times"])

    def retrieve(self, **kw):
        return self.model.retrieve(self.caches, self.ctx, self.anchor, self.t_req, **kw)

    _exact_cache = None

    def exact_scores(self):
        if TestDCGR._exact_cache is None:      # model/seed are deterministic: compute once
            TestDCGR._exact_cache = self._exact_scores()
        return TestDCGR._exact_cache

    def _exact_scores(self):
        """Brute-force log P(z|H) + sum_l log P(SID_l | z, SID_<l) through the full model."""
        c, m = self.cfg, self.model
        out = {}
        with torch.no_grad():
            def hid(x, cb=1):
                st = m._single_stage(x, 0, self.anchor, self.t_req)
                return m.backbone.run_stage(st, self.caches)
            h0 = hid(m.retrieval_tokens(self.ctx, 1))[:, 0]
            lp = [F.log_softmax(hd(h0), -1)[0] for hd in (m.oc_head, m.disc_head, m.ad_head)]
            for z in itertools.product(range(c.n_oc), range(c.n_disc), range(c.n_ad), range(c.n_aov)):
                if (z[3] == 0) != (z[0] == 0):
                    continue
                t = lambda v: torch.tensor([v])
                h1 = hid(m.retrieval_tokens(self.ctx, 2, zp=(t(z[0]), t(z[1]), t(z[2]))))[:, 1]
                base = lp[0][z[0]] + lp[1][z[1]] + lp[2][z[2]] + F.log_softmax(m.aov_head(h1), -1)[0, z[3]]
                for s in itertools.product(range(c.sid_vocab), repeat=3):
                    ss = [t(v) for v in s]
                    tot = base.clone()
                    for lvl in range(3):
                        h = hid(m.retrieval_tokens(self.ctx, 3 + lvl, zp=(t(z[0]), t(z[1]), t(z[2])),
                                                   z_aov=t(z[3]), sid_prefix=ss[:lvl]))[:, -1]
                        tot = tot + F.log_softmax(m.sid_heads[lvl](h), -1)[0, s[lvl]]
                    out[(z, s)] = tot.item()
        return out

    def test_returns_k_distinct_sorted_sids_within_vocab(self):
        res = self.retrieve(k=6)
        assert len(res) == 6
        assert len({(r.sid, r.decision) for r in res}) == 6
        assert [r.score for r in res] == sorted((r.score for r in res), reverse=True)
        assert all(0 <= x < self.cfg.sid_vocab and len(r.sid) == 3 for r in res for x in r.sid)

    def test_scores_are_true_joint_log_probs(self):
        """Reported score == log P(z|H) + sum log P(SID_l | ...) computed by brute force."""
        res = self.retrieve(k=4)
        exact = self.exact_scores()
        for r in res:
            assert r.score == pytest.approx(exact[(r.decision, r.sid)], abs=1e-3)

    def test_top1_is_global_argmax_at_full_beam(self):
        exact = self.exact_scores()
        best_key = max(exact, key=exact.get)
        res = self.retrieve(k=400)
        assert (res[0].decision, res[0].sid) == best_key

    def test_decisions_respect_validity(self):
        """A spending level exists iff the purchase level is non-zero."""
        for r in self.retrieve(k=30):
            z_oc, _, _, z_aov = r.decision
            assert (z_aov == 0) == (z_oc == 0)

    def test_beta_zero_is_unsteered_even_with_phi(self):
        phi = {"ad": torch.tensor([0.0, 5.0])}
        a = self.retrieve(k=5)
        b = self.retrieve(k=5, beta=0.0, phi=phi)
        assert [(r.sid, r.decision) for r in a] == [(r.sid, r.decision) for r in b]

    def test_offset_steers_decisions_toward_target(self):
        phi = {"ad": torch.tensor([0.0, 1.0])}
        share = lambda res: sum(r.decision[2] for r in res) / len(res)
        shares = [share(self.retrieve(k=20, beta=b, phi=phi)) for b in (0.0, 2.0, 50.0)]
        assert shares[0] <= shares[1] <= shares[2]
        assert shares[2] == 1.0                     # strong offset: every retrieved prefix is sponsored

    def test_offset_is_soft_not_a_hard_rule(self):
        """Small beta only shifts scores by beta*phi(z); it does not forbid other decisions."""
        phi = {"ad": torch.tensor([0.0, 0.01])}
        res = self.retrieve(k=20, beta=1.0, phi=phi)
        assert len(res) == 20
        base = {(r.sid, r.decision): r.score for r in self.retrieve(k=20)}
        for r in res:
            if (r.sid, r.decision) in base:
                assert r.score == pytest.approx(base[(r.sid, r.decision)] + 0.01 * r.decision[2], abs=1e-4)

    def test_offset_does_not_change_item_term(self):
        """beta*phi only shifts the decision prefix score: for a fixed (z, SID) score difference == beta*phi(z)."""
        phi = {"oc": torch.tensor([0.0, 1.0, 2.0])}
        base = {(r.sid, r.decision): r.score for r in self.retrieve(k=60)}
        steered = {(r.sid, r.decision): r.score for r in self.retrieve(k=60, beta=0.7, phi=phi)}
        common = set(base) & set(steered)
        assert common
        for key in common:
            assert steered[key] - base[key] == pytest.approx(0.7 * phi["oc"][key[1][0]].item(), abs=1e-4)

    def test_cfg_weight_one_equals_plain_decoding(self):
        a = self.retrieve(k=5)
        b = self.retrieve(k=5, cfg_weight=1.0)
        assert [(r.sid, r.decision) for r in a] == [(r.sid, r.decision) for r in b]
        assert [r.score for r in a] == pytest.approx([r.score for r in b], abs=1e-4)

    def test_cfg_weight_changes_sid_scoring(self):
        a = self.retrieve(k=8, cfg_weight=1.0)
        b = self.retrieve(k=8, cfg_weight=3.0)
        assert [r.score for r in a] != pytest.approx([r.score for r in b])

    def test_cfg_weight_zero_ignores_decision_in_sid_term(self):
        """w=0: SID scoring uses only the null-conditioned distribution."""
        res = self.retrieve(k=6, cfg_weight=0.0)
        assert len(res) == 6 and all(math.isfinite(r.score) for r in res)

    def test_sid_to_items_returns_all_items_sharing_a_code(self):
        res = self.retrieve(k=3)
        index = {res[0].sid: [11, 12], res[1].sid: [13]}
        assert sid_to_items(res, index) == [11, 12, 13]


class TestTopKAndMetrics:
    def test_two_stage_equals_global_topk(self):
        torch.manual_seed(0)
        for _ in range(20):
            s = torch.randn(7, 11)
            v, beam, tok = two_stage_topk(s, 5)
            gv, gi = s.flatten().topk(5)
            assert torch.allclose(v, gv)
            assert torch.equal(beam * 11 + tok, gi)

    def test_two_stage_handles_small_pools(self):
        v, beam, tok = two_stage_topk(torch.randn(2, 3), 10)
        assert v.numel() == 6
        v, _, _ = two_stage_topk(torch.randn(1, 4), 2)
        assert v.numel() == 2

    def test_hit_rate(self):
        top = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
        assert hit_rate_at_m(top, (1, 2, 3), 1) == 1.0
        assert hit_rate_at_m(top, (4, 5, 6), 1) == 0.0
        assert hit_rate_at_m(top, (4, 5, 6), 10) == 1.0
        assert hit_rate_at_m(top, (0, 0, 0), 10) == 0.0


# ---------------------------------------------------------------------------
# muP / Depth-muP (Table 1), optimizers
# ---------------------------------------------------------------------------

class TestMuP:
    def test_width_rule(self):
        assert mup_hidden_lr(1e-3, width=256, depth=4, base_width=64, base_depth=4) == pytest.approx(1e-3 / 4)

    def test_depth_rule(self):
        assert mup_hidden_lr(1e-3, 64, 16, 64, 4) == pytest.approx(1e-3 / 2)       # 1/sqrt(m_N), m_N = 4

    def test_rules_compose(self):
        assert mup_hidden_lr(1e-3, 128, 16, 64, 4) == pytest.approx(1e-3 / (2 * 2))

    def test_base_model_unchanged(self):
        assert mup_hidden_lr(2e-3, 64, 4, 64, 4) == pytest.approx(2e-3)

    def test_param_groups_partition(self):
        cfg = Config(d_model=128, n_layers=4)
        m = OneTransV2(cfg)
        groups = mup_param_groups(m, 1e-3, base_width=64, base_depth=1)
        hidden, other = groups
        assert hidden["lr"] == pytest.approx(1e-3 / (2 * 2))
        assert other["lr"] == 1e-3
        ids = [id(p) for g in groups for p in g["params"]]
        assert len(ids) == len(set(ids))
        emb = {id(p) for p in m.embed.parameters()}
        assert emb.isdisjoint(ids)                      # embeddings go to the sparse optimizer
        assert all(p.ndim >= 2 for p in hidden["params"])

    def test_optimizer_hyperparameters(self):
        sparse, dense = build_optimizers(OneTransV2(Config()))
        assert isinstance(sparse, torch.optim.Adagrad)
        assert sparse.defaults["lr"] == 0.1 and sparse.defaults["initial_accumulator_value"] == 1.0
        assert isinstance(dense, torch.optim.AdamW)
        assert dense.defaults["betas"] == (0.0, 0.99999) and dense.defaults["weight_decay"] == 0.01

    def test_hidden_weight_activations_width_invariant(self):
        """Fan-in init keeps W x at Theta(1) as width grows."""
        scales = []
        for d in (64, 256, 1024):
            ml = MixedLinear(d, d, 1)
            scales.append((torch.randn(32, d) @ ml.shared).pow(2).mean().item())
        assert max(scales) / min(scales) < 1.3


class TestConfig:
    def test_stage_layout(self):
        c = Config(n_fine_tokens=4)
        assert c.stage_sizes == {"R": N_RETR_TOKENS, "P": 1, "F": 4}
        assert c.stage_offset == {"R": 0, "P": 5, "F": 6}
        assert c.per_exposure == 10

    def test_paper_decision_space(self):
        c = Config()
        assert (c.n_oc, c.n_ad, c.sid_levels) == (3, 2, 3)      # 3 purchase levels, 2 supply types, 3-level SID
