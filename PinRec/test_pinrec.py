"""Tests for PinRec implementation.

Run with:
    pytest test_pinrec.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from pinrec import (
    ItemEmbedder,
    TemporalEncoder,
    PinRecInputEncoder,
    CausalTransformer,
    OutcomeEmbedding,
    OutcomeConditionedHead,
    CountMinSketch,
    sampled_softmax_loss,
    pretraining_next_item_loss,
    finetuning_next_item_loss,
    generate_unconditional,
    generate_outcome_conditioned,
    allocate_budget,
    compress_embeddings,
)

D = 32
H = 4
B = 3
M = 12
RAW_DIM = 48
NUM_ACTIONS = 5
NUM_SURFACES = 3
NUM_OUTCOMES = 4


# ---------------------------------------------------------------------------
# ItemEmbedder / TemporalEncoder / PinRecInputEncoder
# ---------------------------------------------------------------------------

class TestItemEmbedder:
    def test_output_shape_and_l2_norm(self):
        emb = ItemEmbedder(RAW_DIM, D)
        out = emb(torch.randn(B, M, RAW_DIM))
        assert out.shape == (B, M, D)
        assert torch.allclose(out.norm(dim=-1), torch.ones(B, M), atol=1e-5)

    def test_gradient(self):
        emb = ItemEmbedder(RAW_DIM, D)
        x = torch.randn(B, M, RAW_DIM, requires_grad=True)
        emb(x).sum().backward()
        assert x.grad is not None

    def test_generalizes_to_unseen_feature_vectors(self):
        # Since this operates on raw features (not an id lookup table), any
        # new feature vector can be embedded -- the zero-shot property.
        emb = ItemEmbedder(RAW_DIM, D)
        unseen = torch.randn(1, RAW_DIM) * 100  # far outside training distribution
        out = emb(unseen)
        assert torch.isfinite(out).all()


class TestTemporalEncoder:
    def test_output_shape_absolute_only(self):
        enc = TemporalEncoder(D)
        out = enc(torch.rand(B, M) * 365)
        assert out.shape == (B, M, D)

    def test_output_shape_with_relative(self):
        enc = TemporalEncoder(D)
        abs_t = torch.rand(B, M) * 365
        prev_t = abs_t - torch.rand(B, M)
        out = enc(abs_t, prev_t)
        assert out.shape == (B, M, D)

    def test_periodicity(self):
        enc = TemporalEncoder(D, periods=(7.0,))
        t0 = torch.tensor([0.0])
        t1 = torch.tensor([7.0])
        out0 = enc(t0)
        out1 = enc(t1)
        assert torch.allclose(out0, out1, atol=1e-4)

    def test_relative_time_changes_output(self):
        enc = TemporalEncoder(D)
        abs_t = torch.tensor([10.0])
        out_no_rel = enc(abs_t)
        out_rel = enc(abs_t, torch.tensor([5.0]))
        assert not torch.allclose(out_no_rel, out_rel)


class TestPinRecInputEncoder:
    def _make(self):
        return PinRecInputEncoder(D, NUM_ACTIONS, NUM_SURFACES)

    def test_output_shape(self):
        enc = self._make()
        item_repr = torch.randn(B, M, D)
        action_ids = torch.randint(0, NUM_ACTIONS, (B, M))
        surface_ids = torch.randint(0, NUM_SURFACES, (B, M))
        abs_time = torch.rand(B, M) * 100
        out = enc(item_repr, action_ids, surface_ids, abs_time)
        assert out.shape == (B, M, D)

    def test_action_changes_output(self):
        enc = self._make()
        item_repr = torch.randn(1, 1, D)
        surface_ids = torch.zeros(1, 1, dtype=torch.long)
        abs_time = torch.zeros(1, 1)
        out_a0 = enc(item_repr, torch.zeros(1, 1, dtype=torch.long), surface_ids, abs_time)
        out_a1 = enc(item_repr, torch.ones(1, 1, dtype=torch.long), surface_ids, abs_time)
        assert not torch.allclose(out_a0, out_a1)


# ---------------------------------------------------------------------------
# CausalTransformer
# ---------------------------------------------------------------------------

class TestCausalTransformer:
    def _make(self, n_layers=2):
        return CausalTransformer(D, n_layers=n_layers, n_heads=H)

    def test_output_shape(self):
        model = self._make()
        out = model(torch.randn(B, M, D))
        assert out.shape == (B, M, D)

    def test_causal_masking(self):
        model = self._make()
        model.eval()
        x1 = torch.randn(1, M, D)
        x2 = x1.clone()
        x2[:, -1, :] = torch.randn(D)
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
        assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)
        assert not torch.allclose(out1[:, -1, :], out2[:, -1, :])

    def test_gradient(self):
        model = self._make()
        x = torch.randn(B, M, D, requires_grad=True)
        model(x).sum().backward()
        assert x.grad is not None

    def test_finite_output(self):
        model = self._make()
        out = model(torch.randn(B, M, D))
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# OutcomeEmbedding / OutcomeConditionedHead
# ---------------------------------------------------------------------------

class TestOutcomeConditionedHead:
    def test_unconditioned_output_shape_and_l2_norm(self):
        head = OutcomeConditionedHead(D)
        out = head(torch.randn(B, D))
        assert out.shape == (B, D)
        assert torch.allclose(out.norm(dim=-1), torch.ones(B), atol=1e-5)

    def test_conditioned_output_shape(self):
        head = OutcomeConditionedHead(D)
        outcome_emb = OutcomeEmbedding(NUM_OUTCOMES, D)
        h = torch.randn(B, D)
        cond = [outcome_emb(torch.zeros(B, dtype=torch.long))]
        out = head(h, cond)
        assert out.shape == (B, D)
        assert torch.allclose(out.norm(dim=-1), torch.ones(B), atol=1e-5)

    def test_different_conditions_give_different_outputs(self):
        head = OutcomeConditionedHead(D)
        outcome_emb = OutcomeEmbedding(NUM_OUTCOMES, D)
        h = torch.randn(1, D)
        out0 = head(h, [outcome_emb(torch.zeros(1, dtype=torch.long))])
        out1 = head(h, [outcome_emb(torch.ones(1, dtype=torch.long))])
        assert not torch.allclose(out0, out1)

    def test_multiple_conditions_summed(self):
        head = OutcomeConditionedHead(D)
        outcome_emb = OutcomeEmbedding(NUM_OUTCOMES, D)
        h = torch.randn(1, D)
        c1 = outcome_emb(torch.zeros(1, dtype=torch.long))
        c2 = outcome_emb(torch.ones(1, dtype=torch.long))
        out_single = head(h, [c1])
        out_multi = head(h, [c1, c2])
        assert not torch.allclose(out_single, out_multi)

    def test_gradient(self):
        head = OutcomeConditionedHead(D)
        h = torch.randn(B, D, requires_grad=True)
        head(h).sum().backward()
        assert h.grad is not None


# ---------------------------------------------------------------------------
# CountMinSketch
# ---------------------------------------------------------------------------

class TestCountMinSketch:
    def test_estimate_at_least_true_count(self):
        cms = CountMinSketch(width=64, depth=4)
        ids = torch.tensor([1, 1, 1, 2, 3])
        cms.update(ids)
        est = cms.estimate(torch.tensor([1, 2, 3]))
        assert est[0].item() >= 3
        assert est[1].item() >= 1
        assert est[2].item() >= 1

    def test_unseen_id_has_zero_or_more(self):
        cms = CountMinSketch(width=256, depth=4)
        cms.update(torch.tensor([1, 2, 3]))
        est = cms.estimate(torch.tensor([999]))
        assert est.item() >= 0

    def test_log_bias_increases_with_frequency(self):
        cms = CountMinSketch(width=512, depth=4)
        cms.update(torch.full((100,), 1))
        cms.update(torch.full((1,), 2))
        bias_popular = cms.log_bias(torch.tensor([1])).item()
        bias_rare = cms.log_bias(torch.tensor([2])).item()
        assert bias_popular > bias_rare

    def test_output_shape_matches_input(self):
        cms = CountMinSketch()
        ids = torch.randint(0, 100, (4, 5))
        cms.update(ids)
        est = cms.estimate(ids)
        assert est.shape == ids.shape


# ---------------------------------------------------------------------------
# Sampled softmax loss / pretraining / fine-tuning losses
# ---------------------------------------------------------------------------

class TestSampledSoftmaxLoss:
    def _cms(self, ids):
        cms = CountMinSketch()
        cms.update(ids)
        return cms

    def test_scalar_output(self):
        anchor = torch.randn(5, D)
        positive = torch.randn(5, D)
        positive_ids = torch.randint(0, 100, (5,))
        pool = torch.randn(10, D)
        pool_ids = torch.randint(0, 100, (10,))
        cms = self._cms(pool_ids)
        loss = sampled_softmax_loss(anchor, positive, positive_ids, pool, pool_ids, cms)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_lower_when_anchor_matches_positive(self):
        torch.manual_seed(0)
        anchor = F.normalize(torch.randn(20, D), dim=-1)
        matching_positive = anchor.clone()
        random_positive = F.normalize(torch.randn(20, D), dim=-1)
        positive_ids = torch.randint(0, 100, (20,))
        pool = F.normalize(torch.randn(30, D), dim=-1)
        pool_ids = torch.randint(0, 100, (30,))
        cms = self._cms(pool_ids)
        loss_match = sampled_softmax_loss(anchor, matching_positive, positive_ids, pool, pool_ids, cms, lam=5.0)
        loss_random = sampled_softmax_loss(anchor, random_positive, positive_ids, pool, pool_ids, cms, lam=5.0)
        assert loss_match.item() < loss_random.item()


class TestPretrainingAndFinetuningLosses:
    def _cms(self, *id_tensors):
        cms = CountMinSketch()
        for ids in id_tensors:
            cms.update(ids)
        return cms

    def test_pretraining_loss_scalar_and_finite(self):
        H_ = torch.randn(B, M, D, requires_grad=True)
        Z_ = torch.randn(B, M, D)
        item_ids = torch.randint(0, 500, (B, M))
        cms = self._cms(item_ids)
        loss = pretraining_next_item_loss(H_, Z_, item_ids, cms)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert H_.grad is not None

    def test_finetuning_loss_scalar_and_finite(self):
        H_ = torch.randn(B, M, D, requires_grad=True)
        Z_ = torch.randn(B, M, D)
        item_ids = torch.randint(0, 500, (B, M))
        impression_embeds = torch.randn(B, 4, D)
        impression_ids = torch.randint(0, 500, (B, 4))
        cms = self._cms(item_ids, impression_ids)
        loss = finetuning_next_item_loss(H_, Z_, item_ids, impression_embeds, impression_ids, cms)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert H_.grad is not None

    def test_finetuning_and_pretraining_differ(self):
        torch.manual_seed(0)
        H_ = torch.randn(B, M, D)
        Z_ = torch.randn(B, M, D)
        item_ids = torch.randint(0, 500, (B, M))
        impression_embeds = torch.randn(B, 4, D)
        impression_ids = torch.randint(0, 500, (B, 4))
        cms = self._cms(item_ids, impression_ids)
        l_pre = pretraining_next_item_loss(H_, Z_, item_ids, cms)
        l_ft = finetuning_next_item_loss(H_, Z_, item_ids, impression_embeds, impression_ids, cms)
        assert l_pre.item() != l_ft.item()


# ---------------------------------------------------------------------------
# Autoregressive generation
# ---------------------------------------------------------------------------

class TestGeneration:
    def _backbone_and_head(self):
        return CausalTransformer(D, n_layers=2, n_heads=H), OutcomeConditionedHead(D)

    def test_unconditional_output_shape(self):
        backbone, head = self._backbone_and_head()
        context = torch.randn(B, M, D)
        out = generate_unconditional(backbone, head, context, num_steps=3)
        assert out.shape == (B, 3, D)

    def test_unconditional_is_l2_normalized(self):
        backbone, head = self._backbone_and_head()
        context = torch.randn(B, M, D)
        out = generate_unconditional(backbone, head, context, num_steps=2)
        assert torch.allclose(out.norm(dim=-1), torch.ones(B, 2), atol=1e-5)

    def test_outcome_conditioned_output_shapes(self):
        backbone, head = self._backbone_and_head()
        outcome_emb = OutcomeEmbedding(NUM_OUTCOMES, D)
        context = torch.randn(B, M, D)
        K = 3
        condition_sets = [[outcome_emb(torch.full((B,), k, dtype=torch.long))] for k in range(K)]
        out = generate_outcome_conditioned(backbone, head, context, condition_sets, num_steps=4)
        assert set(out.keys()) == {0, 1, 2}
        for v in out.values():
            assert v.shape == (B, 4, D)

    def test_outcome_conditioned_different_outcomes_differ(self):
        backbone, head = self._backbone_and_head()
        outcome_emb = OutcomeEmbedding(NUM_OUTCOMES, D)
        context = torch.randn(1, M, D)
        condition_sets = [[outcome_emb(torch.tensor([0]))], [outcome_emb(torch.tensor([1]))]]
        out = generate_outcome_conditioned(
            backbone, head, context, condition_sets, num_steps=2,
            generator=torch.Generator().manual_seed(0),
        )
        # First step's embeddings for the two outcomes must differ, since
        # they see the same context but different conditions.
        assert not torch.allclose(out[0][:, 0, :], out[1][:, 0, :])


# ---------------------------------------------------------------------------
# Budget allocation / embedding compression
# ---------------------------------------------------------------------------

class TestAllocateBudget:
    def test_sums_to_total(self):
        alloc = allocate_budget(100, {0: 0.5, 1: 0.3, 2: 0.2})
        assert sum(alloc.values()) == 100

    def test_rounding_drift_corrected(self):
        alloc = allocate_budget(10, {0: 1 / 3, 1: 1 / 3, 2: 1 / 3})
        assert sum(alloc.values()) == 10

    def test_invalid_fractions_raise(self):
        with pytest.raises(ValueError):
            allocate_budget(100, {0: 0.5, 1: 0.6})

    def test_proportional_allocation(self):
        alloc = allocate_budget(100, {0: 0.7, 1: 0.3})
        assert alloc[0] == 70
        assert alloc[1] == 30


class TestCompressEmbeddings:
    def test_no_merging_below_threshold(self):
        embeds = F.normalize(torch.eye(4), dim=-1)  # orthogonal, similarity 0
        budgets = torch.ones(4)
        comp_e, comp_b = compress_embeddings(embeds, budgets, threshold=0.9)
        assert comp_e.shape[0] == 4
        assert torch.equal(comp_b, budgets)

    def test_identical_embeddings_merge(self):
        e = F.normalize(torch.randn(1, D), dim=-1)
        embeds = e.repeat(3, 1)
        budgets = torch.ones(3)
        comp_e, comp_b = compress_embeddings(embeds, budgets, threshold=0.9)
        assert comp_e.shape[0] == 1
        assert comp_b.item() == 3.0

    def test_budget_conservation(self):
        torch.manual_seed(0)
        embeds = F.normalize(torch.randn(10, D), dim=-1)
        budgets = torch.arange(1, 11, dtype=torch.float32)
        comp_e, comp_b = compress_embeddings(embeds, budgets, threshold=0.9)
        assert comp_b.sum().item() == pytest.approx(budgets.sum().item())

    def test_partial_merge(self):
        base = F.normalize(torch.randn(1, D), dim=-1)
        near_dup = F.normalize(base + 0.001 * torch.randn(1, D), dim=-1)
        far = F.normalize(-base, dim=-1)  # opposite direction, low similarity
        embeds = torch.cat([base, near_dup, far], dim=0)
        budgets = torch.ones(3)
        comp_e, comp_b = compress_embeddings(embeds, budgets, threshold=0.9)
        assert comp_e.shape[0] == 2
        assert comp_b.sum().item() == pytest.approx(3.0)
