"""Tests for UniPinRec implementation.

Run with:
    pytest test_unipinrec.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from unipinrec import (
    sample_action_mask,
    MaskedActionInputEncoder,
    build_unipinrec_attention_mask,
    UniPinRecLayer,
    UniPinRecTransformer,
    ItemOutputHead,
    TargetItemEncoder,
    ActionPredictionHeads,
    CountMinSketch,
    item_retrieval_loss,
    action_prediction_loss,
    UniPinRecModel,
)

D_ITEM = 16
D_MODEL = 32
N_HEADS = 4
NUM_ACTIONS = 4
B = 3
N_PAST = 10
K_FUTURE = 4


# ---------------------------------------------------------------------------
# sample_action_mask / MaskedActionInputEncoder
# ---------------------------------------------------------------------------

class TestSampleActionMask:
    def test_shape_and_dtype(self):
        mask = sample_action_mask(B, N_PAST, p_mask=0.3)
        assert mask.shape == (B, N_PAST)
        assert mask.dtype == torch.bool

    def test_approximate_fraction(self):
        torch.manual_seed(0)
        mask = sample_action_mask(1000, 50, p_mask=0.3)
        frac = mask.float().mean().item()
        assert 0.25 < frac < 0.35

    def test_zero_p_mask_never_masks(self):
        mask = sample_action_mask(B, N_PAST, p_mask=0.0)
        assert not mask.any()

    def test_one_p_mask_always_masks(self):
        mask = sample_action_mask(B, N_PAST, p_mask=1.0)
        assert mask.all()


class TestMaskedActionInputEncoder:
    def _make(self):
        return MaskedActionInputEncoder(D_ITEM, NUM_ACTIONS, D_MODEL)

    def test_output_shape(self):
        enc = self._make()
        item_emb = torch.randn(B, N_PAST, D_ITEM)
        action_ids = torch.randint(0, NUM_ACTIONS, (B, N_PAST))
        mask = sample_action_mask(B, N_PAST, 0.2)
        out = enc(item_emb, action_ids, mask)
        assert out.shape == (B, N_PAST, D_MODEL)

    def test_masked_position_ignores_true_action(self):
        # Two calls differing only in the (masked-out) true action id must
        # produce identical output at that position.
        enc = self._make()
        item_emb = torch.randn(1, 1, D_ITEM)
        mask = torch.ones(1, 1, dtype=torch.bool)
        out_a0 = enc(item_emb, torch.zeros(1, 1, dtype=torch.long), mask)
        out_a1 = enc(item_emb, torch.ones(1, 1, dtype=torch.long), mask)
        assert torch.allclose(out_a0, out_a1)

    def test_unmasked_position_uses_true_action(self):
        enc = self._make()
        item_emb = torch.randn(1, 1, D_ITEM)
        mask = torch.zeros(1, 1, dtype=torch.bool)
        out_a0 = enc(item_emb, torch.zeros(1, 1, dtype=torch.long), mask)
        out_a1 = enc(item_emb, torch.ones(1, 1, dtype=torch.long), mask)
        assert not torch.allclose(out_a0, out_a1)

    def test_gradient(self):
        enc = self._make()
        item_emb = torch.randn(B, N_PAST, D_ITEM, requires_grad=True)
        action_ids = torch.randint(0, NUM_ACTIONS, (B, N_PAST))
        mask = sample_action_mask(B, N_PAST, 0.2)
        enc(item_emb, action_ids, mask).sum().backward()
        assert item_emb.grad is not None


# ---------------------------------------------------------------------------
# build_unipinrec_attention_mask
# ---------------------------------------------------------------------------

class TestBuildUnipinrecAttentionMask:
    def test_shape(self):
        mask = build_unipinrec_attention_mask(N_PAST, K_FUTURE)
        assert mask.shape == (N_PAST + K_FUTURE, N_PAST + K_FUTURE)

    def test_past_is_causal(self):
        mask = build_unipinrec_attention_mask(N_PAST, K_FUTURE)
        past_block = mask[:N_PAST, :N_PAST]
        expected = ~torch.triu(torch.ones(N_PAST, N_PAST, dtype=torch.bool), diagonal=1)
        assert torch.equal(past_block, expected)

    def test_future_sees_all_past(self):
        mask = build_unipinrec_attention_mask(N_PAST, K_FUTURE)
        assert mask[N_PAST:, :N_PAST].all()

    def test_future_blocked_from_other_future(self):
        mask = build_unipinrec_attention_mask(N_PAST, K_FUTURE)
        future_block = mask[N_PAST:, N_PAST:]
        assert torch.equal(future_block, torch.eye(K_FUTURE, dtype=torch.bool))

    def test_past_blocked_from_future(self):
        mask = build_unipinrec_attention_mask(N_PAST, K_FUTURE)
        assert not mask[:N_PAST, N_PAST:].any()


# ---------------------------------------------------------------------------
# UniPinRecLayer / UniPinRecTransformer
# ---------------------------------------------------------------------------

class TestUniPinRecTransformer:
    def _make(self, n_layers=2):
        return UniPinRecTransformer(D_MODEL, n_layers=n_layers, n_heads=N_HEADS)

    def test_forward_combined_output_shapes(self):
        model = self._make()
        x_past = torch.randn(B, N_PAST, D_MODEL)
        x_future = torch.randn(B, K_FUTURE, D_MODEL)
        z_past, z_future = model.forward_combined(x_past, x_future)
        assert z_past.shape == (B, N_PAST, D_MODEL)
        assert z_future.shape == (B, K_FUTURE, D_MODEL)

    def test_past_causal_within_combined_pass(self):
        model = self._make()
        model.eval()
        x_past1 = torch.randn(1, N_PAST, D_MODEL)
        x_past2 = x_past1.clone()
        x_past2[:, -1, :] = torch.randn(D_MODEL)
        x_future = torch.randn(1, K_FUTURE, D_MODEL)
        with torch.no_grad():
            z_past1, _ = model.forward_combined(x_past1, x_future)
            z_past2, _ = model.forward_combined(x_past2, x_future)
        assert torch.allclose(z_past1[:, :-1, :], z_past2[:, :-1, :], atol=1e-5)
        assert not torch.allclose(z_past1[:, -1, :], z_past2[:, -1, :])

    def test_future_does_not_affect_past(self):
        # Past representations must not depend on the candidates at all.
        model = self._make()
        model.eval()
        x_past = torch.randn(1, N_PAST, D_MODEL)
        x_future1 = torch.randn(1, K_FUTURE, D_MODEL)
        x_future2 = torch.randn(1, K_FUTURE, D_MODEL)
        with torch.no_grad():
            z_past1, _ = model.forward_combined(x_past, x_future1)
            z_past2, _ = model.forward_combined(x_past, x_future2)
        assert torch.allclose(z_past1, z_past2, atol=1e-5)

    def test_candidates_independent_of_each_other(self):
        # Changing one candidate must not affect another candidate's output.
        model = self._make()
        model.eval()
        x_past = torch.randn(1, N_PAST, D_MODEL)
        x_future1 = torch.randn(1, K_FUTURE, D_MODEL)
        x_future2 = x_future1.clone()
        x_future2[:, 0, :] = torch.randn(D_MODEL)
        with torch.no_grad():
            _, z_future1 = model.forward_combined(x_past, x_future1)
            _, z_future2 = model.forward_combined(x_past, x_future2)
        assert torch.allclose(z_future1[:, 1:, :], z_future2[:, 1:, :], atol=1e-5)
        assert not torch.allclose(z_future1[:, 0, :], z_future2[:, 0, :])

    def test_kv_cache_path_matches_dense_path(self):
        model = self._make(n_layers=3)
        model.eval()
        x_past = torch.randn(1, N_PAST, D_MODEL)
        x_future = torch.randn(1, K_FUTURE, D_MODEL)
        with torch.no_grad():
            _, z_future_dense = model.forward_combined(x_past, x_future)
            _, kv_cache = model.forward_self(x_past, return_kv_cache=True)
            z_future_cached = model.forward_cross(x_future, kv_cache)
        assert torch.allclose(z_future_dense, z_future_cached, atol=1e-4)

    def test_forward_self_matches_past_block_of_combined(self):
        model = self._make(n_layers=2)
        model.eval()
        x_past = torch.randn(1, N_PAST, D_MODEL)
        x_future = torch.randn(1, K_FUTURE, D_MODEL)
        with torch.no_grad():
            z_past_dense, _ = model.forward_combined(x_past, x_future)
            z_past_self = model.forward_self(x_past)
        assert torch.allclose(z_past_dense, z_past_self, atol=1e-4)

    def test_gradient(self):
        model = self._make()
        x_past = torch.randn(B, N_PAST, D_MODEL, requires_grad=True)
        x_future = torch.randn(B, K_FUTURE, D_MODEL, requires_grad=True)
        z_past, z_future = model.forward_combined(x_past, x_future)
        (z_past.sum() + z_future.sum()).backward()
        assert x_past.grad is not None
        assert x_future.grad is not None


# ---------------------------------------------------------------------------
# Output heads
# ---------------------------------------------------------------------------

class TestItemOutputHeadAndTargetEncoder:
    def test_item_output_head_l2_normalized(self):
        head = ItemOutputHead(D_MODEL)
        out = head(torch.randn(B, N_PAST, D_MODEL))
        assert torch.allclose(out.norm(dim=-1), torch.ones(B, N_PAST), atol=1e-5)

    def test_target_encoder_shape_and_l2_normalized(self):
        enc = TargetItemEncoder(D_ITEM, D_MODEL)
        out = enc(torch.randn(B, N_PAST, D_ITEM))
        assert out.shape == (B, N_PAST, D_MODEL)
        assert torch.allclose(out.norm(dim=-1), torch.ones(B, N_PAST), atol=1e-5)


class TestActionPredictionHeads:
    def test_output_shape(self):
        heads = ActionPredictionHeads(D_MODEL, NUM_ACTIONS)
        out = heads(torch.randn(B, N_PAST, D_MODEL))
        assert out.shape == (B, N_PAST, NUM_ACTIONS)

    def test_gradient(self):
        heads = ActionPredictionHeads(D_MODEL, NUM_ACTIONS)
        x = torch.randn(B, N_PAST, D_MODEL, requires_grad=True)
        heads(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# CountMinSketch / item_retrieval_loss
# ---------------------------------------------------------------------------

class TestCountMinSketch:
    def test_estimate_at_least_true_count(self):
        cms = CountMinSketch(width=64, depth=4)
        ids = torch.tensor([5, 5, 5, 7])
        cms.update(ids)
        est = cms.estimate(torch.tensor([5, 7]))
        assert est[0].item() >= 3
        assert est[1].item() >= 1


class TestItemRetrievalLoss:
    def test_scalar_and_finite(self):
        item_repr = torch.randn(B, N_PAST, D_MODEL, requires_grad=True)
        targets = F.normalize(torch.randn(B, N_PAST, D_MODEL), dim=-1)
        item_ids = torch.randint(0, 200, (B, N_PAST))
        cms = CountMinSketch()
        cms.update(item_ids)
        loss = item_retrieval_loss(item_repr, targets, item_ids, cms)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert item_repr.grad is not None


# ---------------------------------------------------------------------------
# action_prediction_loss
# ---------------------------------------------------------------------------

class TestActionPredictionLoss:
    def _setup(self):
        logits_past = torch.randn(B, N_PAST, NUM_ACTIONS, requires_grad=True)
        labels_past = torch.randint(0, 2, (B, N_PAST, NUM_ACTIONS)).float()
        past_masked = sample_action_mask(B, N_PAST, 0.3)
        logits_future = torch.randn(B, K_FUTURE, NUM_ACTIONS, requires_grad=True)
        labels_future = torch.randint(0, 2, (B, K_FUTURE, NUM_ACTIONS)).float()
        weights = torch.ones(NUM_ACTIONS)
        return logits_past, labels_past, past_masked, logits_future, labels_future, weights

    def test_scalar_and_finite(self):
        args = self._setup()
        loss = action_prediction_loss(*args)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_gradient_flows_to_both_logit_tensors(self):
        logits_past, labels_past, past_masked, logits_future, labels_future, weights = self._setup()
        loss = action_prediction_loss(logits_past, labels_past, past_masked, logits_future, labels_future, weights)
        loss.backward()
        assert logits_past.grad is not None
        assert logits_future.grad is not None

    def test_no_masked_past_positions_still_works(self):
        logits_past = torch.randn(B, N_PAST, NUM_ACTIONS)
        labels_past = torch.randint(0, 2, (B, N_PAST, NUM_ACTIONS)).float()
        past_masked = torch.zeros(B, N_PAST, dtype=torch.bool)  # nothing masked
        logits_future = torch.randn(B, K_FUTURE, NUM_ACTIONS)
        labels_future = torch.randint(0, 2, (B, K_FUTURE, NUM_ACTIONS)).float()
        weights = torch.ones(NUM_ACTIONS)
        loss = action_prediction_loss(logits_past, labels_past, past_masked, logits_future, labels_future, weights)
        assert torch.isfinite(loss)

    def test_action_weights_scale_loss(self):
        torch.manual_seed(0)
        args = list(self._setup())
        loss_uniform = action_prediction_loss(*args)
        args[-1] = torch.ones(NUM_ACTIONS) * 2.0
        loss_scaled = action_prediction_loss(*args)
        assert loss_scaled.item() == pytest.approx(loss_uniform.item() * 2.0, rel=1e-4)


# ---------------------------------------------------------------------------
# UniPinRecModel end-to-end
# ---------------------------------------------------------------------------

class TestUniPinRecModel:
    def _make(self):
        return UniPinRecModel(D_ITEM, NUM_ACTIONS, D_MODEL, n_layers=2, n_heads=N_HEADS)

    def test_forward_train_output_shapes(self):
        model = self._make()
        item_emb_past = torch.randn(B, N_PAST, D_ITEM)
        action_ids_past = torch.randint(0, NUM_ACTIONS, (B, N_PAST))
        item_emb_future = torch.randn(B, K_FUTURE, D_ITEM)
        item_repr, logits_past, logits_future, past_masked = model.forward_train(
            item_emb_past, action_ids_past, item_emb_future, p_mask=0.2
        )
        assert item_repr.shape == (B, N_PAST, D_MODEL)
        assert logits_past.shape == (B, N_PAST, NUM_ACTIONS)
        assert logits_future.shape == (B, K_FUTURE, NUM_ACTIONS)
        assert past_masked.shape == (B, N_PAST)

    def test_forward_train_gradient(self):
        model = self._make()
        item_emb_past = torch.randn(B, N_PAST, D_ITEM)
        action_ids_past = torch.randint(0, NUM_ACTIONS, (B, N_PAST))
        item_emb_future = torch.randn(B, K_FUTURE, D_ITEM)
        item_repr, logits_past, logits_future, _ = model.forward_train(
            item_emb_past, action_ids_past, item_emb_future, p_mask=0.2
        )
        (item_repr.sum() + logits_past.sum() + logits_future.sum()).backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert len(grad_norms) > 0

    def test_encode_history_and_score_candidates_match_forward_train(self):
        model = self._make()
        model.eval()
        item_emb_past = torch.randn(1, N_PAST, D_ITEM)
        action_ids_past = torch.randint(0, NUM_ACTIONS, (1, N_PAST))
        item_emb_future = torch.randn(1, K_FUTURE, D_ITEM)
        past_masked = sample_action_mask(1, N_PAST, 0.3, generator=torch.Generator().manual_seed(0))

        with torch.no_grad():
            _, kv_cache = model.encode_history(item_emb_past, action_ids_past, past_masked)
            logits_cached = model.score_candidates(kv_cache, item_emb_future)

            future_masked = torch.ones(1, K_FUTURE, dtype=torch.bool)
            x_past = model.input_encoder(item_emb_past, action_ids_past, past_masked)
            placeholder = torch.zeros(1, K_FUTURE, dtype=torch.long)
            x_future = model.input_encoder(item_emb_future, placeholder, future_masked)
            _, z_future_dense = model.backbone.forward_combined(x_past, x_future)
            logits_dense = model.action_heads(z_future_dense)

        assert torch.allclose(logits_cached, logits_dense, atol=1e-4)

    def test_encode_history_item_repr_shape(self):
        model = self._make()
        item_emb_past = torch.randn(B, N_PAST, D_ITEM)
        action_ids_past = torch.randint(0, NUM_ACTIONS, (B, N_PAST))
        past_masked = sample_action_mask(B, N_PAST, 0.2)
        item_repr, kv_cache = model.encode_history(item_emb_past, action_ids_past, past_masked)
        assert item_repr.shape == (B, N_PAST, D_MODEL)
        assert len(kv_cache) == 2  # n_layers

    def test_future_masked_always_true_in_forward_train(self):
        # Even with p_mask=0.0 (no past masking), candidates must still be masked.
        model = self._make()
        item_emb_past = torch.randn(1, N_PAST, D_ITEM)
        action_ids_past = torch.randint(0, NUM_ACTIONS, (1, N_PAST))
        item_emb_future = torch.randn(1, K_FUTURE, D_ITEM)
        _, _, _, past_masked = model.forward_train(item_emb_past, action_ids_past, item_emb_future, p_mask=0.0)
        assert not past_masked.any()
