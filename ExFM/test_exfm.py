"""Tests for the ExFM implementation.

Run with:
    pytest test_exfm.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from exfm import (
    h_loss,
    DataAugmentationService,
    grad_scale,
    label_scaling,
    VMWithAuxiliaryHead,
    ah_loss,
    StudentAdapter,
    sa_loss,
    VMWithStudentAdapter,
    ah_sa_loss,
    train_step_with_student_adapter,
    normalized_entropy,
)

B, D, H = 8, 8, 16


# ---------------------------------------------------------------------------
# h_loss
# ---------------------------------------------------------------------------

class TestHLoss:
    def test_matches_bce_with_logits(self):
        y_hat = torch.randn(B)
        y = torch.randint(0, 2, (B,)).float()
        assert torch.allclose(h_loss(y_hat, y), F.binary_cross_entropy_with_logits(y_hat, y))

    def test_accepts_soft_labels(self):
        y_hat = torch.randn(B)
        y_soft = torch.rand(B)
        loss = h_loss(y_hat, y_soft)
        assert torch.isfinite(loss)

    def test_gradient(self):
        y_hat = torch.randn(B, requires_grad=True)
        y = torch.randint(0, 2, (B,)).float()
        h_loss(y_hat, y).backward()
        assert y_hat.grad is not None

    def test_lower_when_prediction_matches_label(self):
        y = torch.tensor([1.0, 0.0, 1.0, 0.0])
        good = h_loss(torch.tensor([5.0, -5.0, 5.0, -5.0]), y)
        bad = h_loss(torch.tensor([-5.0, 5.0, -5.0, 5.0]), y)
        assert good.item() < bad.item()


# ---------------------------------------------------------------------------
# DataAugmentationService
# ---------------------------------------------------------------------------

class TestDataAugmentationService:
    def _make(self):
        calls = {"n": 0}

        def fm_predict(features):
            calls["n"] += 1
            return 0.5

        return DataAugmentationService(fm_predict), calls

    def test_amortizes_fm_calls_across_overlapping_vms(self):
        das, calls = self._make()
        vm_examples = {
            "vm1": [{"example_id": i, "features": torch.zeros(2), "label": torch.tensor(0.0)} for i in [1, 2]],
            "vm2": [{"example_id": i, "features": torch.zeros(2), "label": torch.tensor(0.0)} for i in [2, 3]],
        }
        shared = das.build_shared_dataset(vm_examples)
        das.annotate_with_fm_predictions(shared)
        assert len(shared) == 3  # examples 1, 2, 3 — 2 is deduplicated
        assert das.fm_calls == 3  # not 4 (sum of per-VM example counts)
        assert calls["n"] == 3

    def test_shared_dataset_tracks_vm_membership(self):
        das, _ = self._make()
        vm_examples = {
            "vm1": [{"example_id": 1, "features": torch.zeros(2), "label": torch.tensor(0.0)}],
            "vm2": [{"example_id": 1, "features": torch.zeros(2), "label": torch.tensor(0.0)}],
        }
        shared = das.build_shared_dataset(vm_examples)
        assert set(shared[1]["vm_names"]) == {"vm1", "vm2"}

    def test_vm_training_data_only_includes_own_examples(self):
        das, _ = self._make()
        vm_examples = {
            "vm1": [{"example_id": 1, "features": torch.zeros(2), "label": torch.tensor(1.0)}],
            "vm2": [{"example_id": 2, "features": torch.zeros(2), "label": torch.tensor(0.0)}],
        }
        shared = das.build_shared_dataset(vm_examples)
        das.annotate_with_fm_predictions(shared)
        vm1_data = das.vm_training_data(shared, "vm1")
        assert len(vm1_data) == 1
        assert vm1_data[0]["example_id"] == 1
        assert "fm_prediction" in vm1_data[0]

    def test_fm_prediction_is_annotated_correctly(self):
        das, _ = self._make()
        vm_examples = {"vm1": [{"example_id": 1, "features": torch.zeros(2), "label": torch.tensor(1.0)}]}
        shared = das.build_shared_dataset(vm_examples)
        das.annotate_with_fm_predictions(shared)
        assert das.vm_training_data(shared, "vm1")[0]["fm_prediction"] == 0.5


# ---------------------------------------------------------------------------
# grad_scale
# ---------------------------------------------------------------------------

class TestGradScale:
    def test_forward_is_identity(self):
        x = torch.randn(B, D)
        assert torch.equal(grad_scale(x, 5.0), x)

    def test_backward_scales_gradient(self):
        x = torch.randn(B, D, requires_grad=True)
        grad_scale(x, 3.0).sum().backward()
        assert torch.allclose(x.grad, torch.full_like(x, 3.0))

    def test_beta_one_is_transparent(self):
        x = torch.randn(B, D, requires_grad=True)
        grad_scale(x, 1.0).sum().backward()
        assert torch.allclose(x.grad, torch.ones_like(x))


# ---------------------------------------------------------------------------
# label_scaling
# ---------------------------------------------------------------------------

class TestLabelScaling:
    def test_scales_up(self):
        y_f = torch.tensor([0.1, 0.2])
        assert torch.allclose(label_scaling(y_f, alpha=2.0), torch.tensor([0.2, 0.4]))

    def test_clips_at_one(self):
        y_f = torch.tensor([0.6, 0.9])
        scaled = label_scaling(y_f, alpha=3.0)
        assert torch.all(scaled <= 1.0)

    def test_alpha_one_is_identity(self):
        y_f = torch.rand(10)
        assert torch.allclose(label_scaling(y_f, alpha=1.0), y_f)


# ---------------------------------------------------------------------------
# VMWithAuxiliaryHead / ah_loss
# ---------------------------------------------------------------------------

class TestVMWithAuxiliaryHead:
    def _make(self):
        backbone = nn.Sequential(nn.Linear(D, H), nn.ReLU())
        return VMWithAuxiliaryHead(backbone, backbone_out_dim=H)

    def test_output_shapes(self):
        vm = self._make()
        y_s, y_d = vm(torch.randn(B, D))
        assert y_s.shape == (B,)
        assert y_d.shape == (B,)

    def test_gradient_scaling_changes_backbone_gradient_magnitude(self):
        vm = self._make()
        x = torch.randn(B, D)

        def backbone_grad_norm(beta):
            vm.zero_grad()
            _, y_d = vm(x, beta=beta)
            y_d.sum().backward()
            return vm.backbone[0].weight.grad.norm().item()

        norm_beta1 = backbone_grad_norm(1.0)
        norm_beta10 = backbone_grad_norm(10.0)
        assert norm_beta10 == pytest.approx(norm_beta1 * 10, rel=1e-4)

    def test_serving_head_gradient_unaffected_by_beta(self):
        vm = self._make()
        x = torch.randn(B, D)

        def serving_grad(beta):
            vm.zero_grad()
            y_s, _ = vm(x, beta=beta)
            y_s.sum().backward()
            return vm.serving_head.net[0].weight.grad.clone()

        g1 = serving_grad(1.0)
        g10 = serving_grad(10.0)
        assert torch.allclose(g1, g10)


class TestAhLoss:
    def test_matches_manual_composition(self):
        y_s = torch.randn(B)
        y = torch.randint(0, 2, (B,)).float()
        y_d = torch.randn(B)
        y_f = torch.rand(B) * 0.3
        loss = ah_loss(y_s, y, y_d, y_f, alpha=2.0, w=5.0)
        expected = h_loss(y_s, y) + 5.0 * h_loss(y_d, label_scaling(y_f, 2.0))
        assert torch.allclose(loss, expected)

    def test_gradient(self):
        y_s = torch.randn(B, requires_grad=True)
        y_d = torch.randn(B, requires_grad=True)
        y = torch.randint(0, 2, (B,)).float()
        y_f = torch.rand(B) * 0.3
        ah_loss(y_s, y, y_d, y_f).backward()
        assert y_s.grad is not None and y_d.grad is not None


# ---------------------------------------------------------------------------
# StudentAdapter / sa_loss
# ---------------------------------------------------------------------------

class TestStudentAdapter:
    def test_output_shape(self):
        sa = StudentAdapter()
        y_f = torch.rand(B)
        assert sa(y_f).shape == (B,)

    def test_gradient(self):
        sa = StudentAdapter()
        y_f = torch.rand(B)
        y = torch.randint(0, 2, (B,)).float()
        sa_loss(sa(y_f), y).backward()
        assert sa.mlp.net[0].weight.grad is not None

    def test_learns_to_predict_label_from_fm_score(self):
        torch.manual_seed(0)
        sa = StudentAdapter(hidden_dims=[16])
        opt = torch.optim.Adam(sa.parameters(), lr=0.05)
        y_f = torch.rand(200)
        y = (y_f > 0.5).float()  # a simple, learnable relationship
        losses = []
        for _ in range(300):
            opt.zero_grad()
            loss = sa_loss(sa(y_f), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# VMWithStudentAdapter / ah_sa_loss / Algorithm 1
# ---------------------------------------------------------------------------

class TestVMWithStudentAdapter:
    def _make(self):
        backbone = nn.Sequential(nn.Linear(D, H), nn.ReLU())
        return VMWithStudentAdapter(backbone, backbone_out_dim=H)

    def test_output_shapes(self):
        vm = self._make()
        y_s, y_d_fm, y_d_sa = vm(torch.randn(B, D))
        assert y_s.shape == y_d_fm.shape == y_d_sa.shape == (B,)

    def test_ah_sa_loss_scalar(self):
        vm = self._make()
        y_s, y_d_fm, y_d_sa = vm(torch.randn(B, D))
        y = torch.randint(0, 2, (B,)).float()
        y_f = torch.rand(B) * 0.3
        y_sa_sg = torch.rand(B)
        loss = ah_sa_loss(y_s, y, y_d_fm, y_f, y_d_sa, y_sa_sg)
        assert loss.dim() == 0


class TestTrainStepWithStudentAdapter:
    def _setup(self):
        torch.manual_seed(0)
        vm = VMWithStudentAdapter(nn.Sequential(nn.Linear(D, H), nn.ReLU()), backbone_out_dim=H)
        sa = StudentAdapter()
        vm_opt = torch.optim.Adam(vm.parameters(), lr=1e-2)
        sa_opt = torch.optim.Adam(sa.parameters(), lr=1e-2)
        features = torch.randn(32, D)
        y = torch.randint(0, 2, (32,)).float()
        y_f = torch.rand(32) * 0.3
        return vm, sa, vm_opt, sa_opt, features, y, y_f

    def test_returns_loss_dict(self):
        vm, sa, vm_opt, sa_opt, features, y, y_f = self._setup()
        stats = train_step_with_student_adapter(vm, sa, vm_opt, sa_opt, features, y, y_f)
        assert set(stats.keys()) == {"l_sta", "l_ah_sa"}
        assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())

    def test_losses_decrease_over_training(self):
        vm, sa, vm_opt, sa_opt, features, y, y_f = self._setup()
        first = train_step_with_student_adapter(vm, sa, vm_opt, sa_opt, features, y, y_f, alpha=2.0, w=3.0, beta=2.0)
        for _ in range(100):
            last = train_step_with_student_adapter(vm, sa, vm_opt, sa_opt, features, y, y_f, alpha=2.0, w=3.0, beta=2.0)
        assert last["l_ah_sa"] < first["l_ah_sa"]

    def test_sa_updated_before_vm_uses_it(self):
        vm, sa, vm_opt, sa_opt, features, y, y_f = self._setup()
        sa_params_before = [p.clone() for p in sa.parameters()]
        train_step_with_student_adapter(vm, sa, vm_opt, sa_opt, features, y, y_f)
        assert any(not torch.allclose(a, b) for a, b in zip(sa_params_before, sa.parameters()))

    def test_vm_step_does_not_update_sa_a_second_time(self):
        # SA should only be updated once per call (step 1 of Algorithm 1),
        # not again via gradients flowing back through the stop-gradiented target.
        vm, sa, vm_opt, sa_opt, features, y, y_f = self._setup()
        train_step_with_student_adapter(vm, sa, vm_opt, sa_opt, features, y, y_f)
        sa_params_after_first = [p.clone() for p in sa.parameters()]
        # Manually redo just the VM optimization step without touching SA's optimizer.
        with torch.no_grad():
            y_sa_prob_sg = torch.sigmoid(sa(y_f))
        vm_opt.zero_grad()
        y_s, y_d_fm, y_d_sa = vm(features, beta=1.0)
        loss = ah_sa_loss(y_s, y, y_d_fm, y_f, y_d_sa, y_sa_prob_sg)
        loss.backward()
        vm_opt.step()
        assert all(torch.allclose(a, b) for a, b in zip(sa_params_after_first, sa.parameters()))


# ---------------------------------------------------------------------------
# normalized_entropy
# ---------------------------------------------------------------------------

class TestNormalizedEntropy:
    def test_near_perfect_predictions_give_low_ne(self):
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0] * 25)
        probs = labels * 0.99 + (1 - labels) * 0.01
        assert normalized_entropy(labels, probs) < 0.05

    def test_predicting_base_rate_gives_ne_near_one(self):
        labels = torch.randint(0, 2, (2000,)).float()
        p_bar = labels.mean().item()
        probs = torch.full_like(labels, p_bar)
        assert normalized_entropy(labels, probs) == pytest.approx(1.0, abs=0.05)
