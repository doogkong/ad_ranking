"""Tests for the LoopFM implementation.

Run with:
    pytest test_loopfm.py -v
"""

import pytest
import torch

from loopfm import (
    extract_embedding,
    stop_gradient,
    MatryoshkaAutoencoder,
    quantize_int4,
    build_user_sequence,
    group_by_key,
    MeanPoolSequenceEncoder,
    SumPoolSequenceEncoder,
    AttentionPoolSequenceEncoder,
    loopfm_training_loss,
    normalized_entropy,
    transfer_ratio,
)

B, L, D = 4, 8, 16


# ---------------------------------------------------------------------------
# Stage 1 — Extraction
# ---------------------------------------------------------------------------

class TestExtractEmbedding:
    def test_output_shape(self):
        acts = [torch.randn(B, 8), torch.randn(B, 4), torch.randn(B, 2)]
        e = extract_embedding(acts)
        assert e.shape == (B, 14)

    def test_preserves_values_in_order(self):
        a = torch.ones(B, 2)
        b = torch.zeros(B, 3)
        e = extract_embedding([a, b])
        assert torch.all(e[:, :2] == 1.0)
        assert torch.all(e[:, 2:] == 0.0)


class TestStopGradient:
    def test_no_grad_fn(self):
        x = torch.randn(4, requires_grad=True)
        y = stop_gradient(x)
        assert y.grad_fn is None

    def test_blocks_gradient_flow(self):
        x = torch.randn(4, requires_grad=True)
        w = torch.randn(4, requires_grad=True)
        y = stop_gradient(x) * w  # w keeps the graph alive so backward() is well-defined
        y.sum().backward()
        assert x.grad is None
        assert w.grad is not None


# ---------------------------------------------------------------------------
# Stage 2 — Matryoshka Autoencoder + INT4 quantization
# ---------------------------------------------------------------------------

class TestMatryoshkaAutoencoder:
    def test_output_shapes(self):
        ae = MatryoshkaAutoencoder(raw_dim=D, latent_dim=8, prefix_dims=[2, 4, 8])
        e = torch.randn(B, D)
        z, recons = ae(e)
        assert z.shape == (B, 8)
        assert set(recons.keys()) == {2, 4, 8}
        for r in recons.values():
            assert r.shape == (B, D)

    def test_latent_bounded_by_tanh(self):
        ae = MatryoshkaAutoencoder(raw_dim=D, latent_dim=8)
        z = ae.encode(torch.randn(100, D) * 10)
        assert torch.all(z >= -1.0) and torch.all(z <= 1.0)

    def test_default_prefix_dims_is_latent_dim(self):
        ae = MatryoshkaAutoencoder(raw_dim=D, latent_dim=8)
        assert ae.prefix_dims == [8]

    def test_rejects_prefix_larger_than_latent(self):
        with pytest.raises(ValueError):
            MatryoshkaAutoencoder(raw_dim=D, latent_dim=8, prefix_dims=[4, 16])

    def test_decode_pads_short_prefix(self):
        ae = MatryoshkaAutoencoder(raw_dim=D, latent_dim=8)
        z = torch.randn(B, 8)
        full = ae.decode(z)
        truncated = ae.decode(z[:, :3])
        assert full.shape == truncated.shape == (B, D)
        assert not torch.allclose(full, truncated)

    def test_mrae_loss_gradient(self):
        ae = MatryoshkaAutoencoder(raw_dim=D, latent_dim=8, prefix_dims=[4, 8])
        e = torch.randn(B, D)
        z, recons = ae(e)
        loss = ae.mrae_loss(e, recons)
        loss.backward()
        assert ae.encoder[0].weight.grad is not None
        assert ae.decoder.weight.grad is not None

    def test_mrae_loss_decreases_with_training(self):
        torch.manual_seed(0)
        ae = MatryoshkaAutoencoder(raw_dim=D, latent_dim=8, prefix_dims=[4, 8])
        e = torch.randn(32, D)
        opt = torch.optim.Adam(ae.parameters(), lr=0.05)
        losses = []
        for _ in range(200):
            opt.zero_grad()
            z, recons = ae(e)
            loss = ae.mrae_loss(e, recons)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]

    def test_stop_gradient_keeps_upstream_untouched(self):
        upstream = torch.randn(B, D, requires_grad=True)
        ae = MatryoshkaAutoencoder(raw_dim=D, latent_dim=8)
        z, recons = ae(stop_gradient(upstream))
        ae.mrae_loss(upstream.detach(), recons).backward()
        assert upstream.grad is None


class TestQuantizeInt4:
    def test_output_range(self):
        z = torch.linspace(-1, 1, 100)
        zq = quantize_int4(z)
        assert torch.all(zq >= -1.0) and torch.all(zq <= 7 / 8)

    def test_at_most_16_levels(self):
        z = torch.rand(1000) * 2 - 1
        zq = quantize_int4(z)
        assert zq.unique().numel() <= 16

    def test_idempotent(self):
        z = torch.rand(50) * 2 - 1
        once = quantize_int4(z)
        twice = quantize_int4(once)
        assert torch.allclose(once, twice)

    def test_zero_maps_to_zero(self):
        assert quantize_int4(torch.tensor([0.0])).item() == 0.0


# ---------------------------------------------------------------------------
# Stage 3 — Structuring
# ---------------------------------------------------------------------------

class TestBuildUserSequence:
    def test_excludes_current_and_future(self):
        history = [(1.0, torch.zeros(4)), (5.0, torch.ones(4)), (10.0, torch.ones(4) * 2)]
        seq = build_user_sequence(history, t_cur=5.0, max_len=10)
        assert seq.shape[0] == 1  # only t=1.0 qualifies (t < t_cur strictly)

    def test_chronological_order_preserved(self):
        history = [(3.0, torch.tensor([3.0])), (1.0, torch.tensor([1.0])), (2.0, torch.tensor([2.0]))]
        seq = build_user_sequence(history, t_cur=10.0, max_len=10)
        assert seq.squeeze(-1).tolist() == [1.0, 2.0, 3.0]

    def test_truncates_to_most_recent_max_len(self):
        history = [(float(t), torch.tensor([float(t)])) for t in range(20)]
        seq = build_user_sequence(history, t_cur=20.0, max_len=5)
        assert seq.squeeze(-1).tolist() == [15.0, 16.0, 17.0, 18.0, 19.0]

    def test_empty_history_returns_empty_tensor(self):
        seq = build_user_sequence([], t_cur=5.0, max_len=10)
        assert seq.shape[0] == 0

    def test_all_history_after_t_cur_returns_empty(self):
        history = [(10.0, torch.zeros(4)), (20.0, torch.zeros(4))]
        seq = build_user_sequence(history, t_cur=5.0, max_len=10)
        assert seq.shape[0] == 0


class TestGroupByKey:
    def test_groups_correctly(self):
        records = [
            {"key": "u1", "timestamp": 1.0, "embedding": torch.zeros(2)},
            {"key": "u2", "timestamp": 2.0, "embedding": torch.zeros(2)},
            {"key": "u1", "timestamp": 3.0, "embedding": torch.zeros(2)},
        ]
        grouped = group_by_key(records)
        assert set(grouped.keys()) == {"u1", "u2"}
        assert len(grouped["u1"]) == 2
        assert len(grouped["u2"]) == 1

    def test_empty_records(self):
        assert group_by_key([]) == {}


# ---------------------------------------------------------------------------
# Stage 4 — VM-side sequence encoders + combined loss
# ---------------------------------------------------------------------------

class TestMeanPoolSequenceEncoder:
    def test_matches_manual_mean(self):
        enc = MeanPoolSequenceEncoder()
        seq = torch.randn(2, 4, 3)
        mask = torch.ones(2, 4, dtype=torch.bool)
        out = enc(seq, mask)
        assert torch.allclose(out, seq.mean(dim=1), atol=1e-6)

    def test_respects_mask(self):
        enc = MeanPoolSequenceEncoder()
        seq = torch.zeros(1, 4, 2)
        seq[0, :2] = 10.0
        mask = torch.tensor([[True, True, False, False]])
        out = enc(seq, mask)
        assert torch.allclose(out, torch.tensor([[10.0, 10.0]]))

    def test_all_masked_gives_zero_not_nan(self):
        enc = MeanPoolSequenceEncoder()
        seq = torch.randn(1, 4, 2)
        mask = torch.zeros(1, 4, dtype=torch.bool)
        out = enc(seq, mask)
        assert torch.isfinite(out).all()
        assert torch.allclose(out, torch.zeros_like(out))


class TestSumPoolSequenceEncoder:
    def test_matches_manual_sum(self):
        enc = SumPoolSequenceEncoder()
        seq = torch.randn(2, 4, 3)
        mask = torch.ones(2, 4, dtype=torch.bool)
        out = enc(seq, mask)
        assert torch.allclose(out, seq.sum(dim=1), atol=1e-6)

    def test_respects_mask(self):
        enc = SumPoolSequenceEncoder()
        seq = torch.ones(1, 4, 2)
        mask = torch.tensor([[True, True, False, False]])
        out = enc(seq, mask)
        assert torch.allclose(out, torch.tensor([[2.0, 2.0]]))


class TestAttentionPoolSequenceEncoder:
    def test_output_shape(self):
        enc = AttentionPoolSequenceEncoder(d_model=D)
        seq = torch.randn(B, L, D)
        mask = torch.ones(B, L, dtype=torch.bool)
        query = torch.randn(B, D)
        out = enc(seq, mask, query)
        assert out.shape == (B, D)

    def test_masked_entries_get_zero_weight(self):
        enc = AttentionPoolSequenceEncoder(d_model=D)
        enc.eval()
        seq = torch.randn(1, L, D)
        query = torch.randn(1, D)
        mask_full = torch.ones(1, L, dtype=torch.bool)
        mask_partial = mask_full.clone()
        mask_partial[0, -3:] = False

        with torch.no_grad():
            out_full_masked_input = seq.clone()
            out_full_masked_input[0, -3:] = torch.randn(3, D) * 1000  # perturb masked-out entries
            out1 = enc(seq, mask_partial, query)
            out2 = enc(out_full_masked_input, mask_partial, query)
        assert torch.allclose(out1, out2, atol=1e-4)

    def test_different_queries_give_different_output(self):
        enc = AttentionPoolSequenceEncoder(d_model=D)
        seq = torch.randn(1, L, D)
        mask = torch.ones(1, L, dtype=torch.bool)
        q1, q2 = torch.randn(1, D), torch.randn(1, D)
        out1 = enc(seq, mask, q1)
        out2 = enc(seq, mask, q2)
        assert not torch.allclose(out1, out2)

    def test_gradient(self):
        enc = AttentionPoolSequenceEncoder(d_model=D)
        seq = torch.randn(B, L, D, requires_grad=True)
        mask = torch.ones(B, L, dtype=torch.bool)
        query = torch.randn(B, D)
        enc(seq, mask, query).sum().backward()
        assert seq.grad is not None


class TestLoopFMTrainingLoss:
    def test_scalar_output(self):
        vm_logits = torch.randn(B, requires_grad=True)
        fm_logits = torch.randn(B)
        labels = torch.randint(0, 2, (B,)).float()
        loss = loopfm_training_loss(vm_logits, fm_logits, labels, kd_weight=1.0)
        assert loss.dim() == 0

    def test_gradient_flows_to_vm_not_fm(self):
        vm_logits = torch.randn(B, requires_grad=True)
        fm_logits = torch.randn(B, requires_grad=True)
        labels = torch.randint(0, 2, (B,)).float()
        loss = loopfm_training_loss(vm_logits, fm_logits, labels, kd_weight=1.0)
        loss.backward()
        assert vm_logits.grad is not None
        assert fm_logits.grad is None

    def test_kd_weight_zero_matches_pure_task_loss(self):
        import torch.nn.functional as F
        vm_logits = torch.randn(B)
        fm_logits = torch.randn(B)
        labels = torch.randint(0, 2, (B,)).float()
        loss = loopfm_training_loss(vm_logits, fm_logits, labels, kd_weight=0.0)
        expected = F.binary_cross_entropy_with_logits(vm_logits, labels)
        assert torch.allclose(loss, expected)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

class TestNormalizedEntropy:
    def test_near_perfect_predictions_give_low_ne(self):
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0] * 25)
        probs = labels * 0.99 + (1 - labels) * 0.01
        ne = normalized_entropy(labels, probs)
        assert ne < 0.05

    def test_predicting_base_rate_gives_ne_near_one(self):
        labels = torch.randint(0, 2, (2000,)).float()
        p_bar = labels.mean().item()
        probs = torch.full_like(labels, p_bar)
        ne = normalized_entropy(labels, probs)
        assert ne == pytest.approx(1.0, abs=0.05)

    def test_worse_than_base_rate_gives_ne_above_one(self):
        labels = torch.cat([torch.ones(900), torch.zeros(100)])
        probs = torch.cat([torch.full((900,), 0.1), torch.full((100,), 0.9)])  # confidently wrong
        ne = normalized_entropy(labels, probs)
        assert ne > 1.0


class TestTransferRatio:
    def test_basic_computation(self):
        tr = transfer_ratio(ne_vm_before=0.50, ne_vm_after=0.48, ne_fm_before=0.40, ne_fm_after=0.36)
        assert tr == pytest.approx((0.50 - 0.48) / (0.40 - 0.36))

    def test_zero_fm_improvement_raises(self):
        with pytest.raises(ValueError):
            transfer_ratio(ne_vm_before=0.5, ne_vm_after=0.48, ne_fm_before=0.4, ne_fm_after=0.4)

    def test_negative_transfer_is_negative_ratio(self):
        tr = transfer_ratio(ne_vm_before=0.50, ne_vm_after=0.52, ne_fm_before=0.40, ne_fm_after=0.36)
        assert tr < 0
