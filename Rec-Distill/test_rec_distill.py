"""Tests for the Rec-Distill implementation.

Run with:
    pytest test_rec_distill.py -v
"""

import pytest
import torch
import torch.nn as nn

from rec_distill import (
    teacher_gain,
    distillation_transferability,
    distill_gain,
    TeacherSignalCache,
    HybridTrainingScheduler,
    kd_loss_temperature,
    kd_loss_ce,
    student_objective,
    mse_grad_factor,
    ce_grad_factor,
    DecoupledTowerStudent,
    main_tower_loss,
    auxiliary_tower_loss,
    DebiasParams,
    sampling_aware_debias,
    cross_debias_distill_loss,
)


# ---------------------------------------------------------------------------
# Gain decomposition (Eq. 1, 4-5)
# ---------------------------------------------------------------------------

class TestGainDecomposition:
    def test_teacher_gain(self):
        assert teacher_gain(0.88, 0.86) == pytest.approx(0.02)

    def test_transferability_matches_definition(self):
        eta = distillation_transferability(p_teacher=0.90, p_student_raw=0.80, p_student_distill=0.88)
        assert eta == pytest.approx((0.88 - 0.80) / (0.90 - 0.80))

    def test_transferability_full_recovery_is_one(self):
        eta = distillation_transferability(p_teacher=0.90, p_student_raw=0.80, p_student_distill=0.90)
        assert eta == pytest.approx(1.0)

    def test_transferability_raises_when_no_teacher_gain(self):
        with pytest.raises(ValueError):
            distillation_transferability(p_teacher=0.8, p_student_raw=0.8, p_student_distill=0.85)

    def test_identity_holds(self):
        p_t, p_s_raw, p_s_distill = 0.8879, 0.8624, 0.8818
        tg = teacher_gain(p_t, p_s_raw)
        eta = distillation_transferability(p_t, p_s_raw, p_s_distill)
        dg = distill_gain(p_s_raw, p_s_distill)
        assert dg == pytest.approx(tg * eta, abs=1e-9)


# ---------------------------------------------------------------------------
# TeacherSignalCache / HybridTrainingScheduler
# ---------------------------------------------------------------------------

class TestTeacherSignalCache:
    def test_write_then_read(self):
        cache = TeacherSignalCache()
        cache.write("ex1", 0.42)
        assert cache.read("ex1") == pytest.approx(0.42)

    def test_one_write_serves_multiple_reads(self):
        cache = TeacherSignalCache()
        cache.write("ex1", 0.5)
        cache.read("ex1")
        cache.read("ex1")
        cache.read("ex1")
        assert cache.write_count == 1
        assert cache.read_count == 3

    def test_read_batch_preserves_order(self):
        cache = TeacherSignalCache()
        for i in range(4):
            cache.write(f"ex{i}", float(i))
        batch = cache.read_batch(["ex3", "ex1", "ex0"])
        assert batch.tolist() == [3.0, 1.0, 0.0]

    def test_missing_key_raises(self):
        cache = TeacherSignalCache()
        with pytest.raises(KeyError):
            cache.read("nonexistent")


class TestHybridTrainingScheduler:
    def test_phase_boundaries(self):
        sched = HybridTrainingScheduler(batch_phase_steps=100)
        assert sched.phase(99) == "batch"
        assert sched.phase(100) == "streaming"

    def test_batch_phase_always_distilled(self):
        sched = HybridTrainingScheduler(batch_phase_steps=100, distill_from_step=50)
        assert sched.is_distillation_active(0)
        assert sched.is_distillation_active(99)

    def test_persistent_distillation_from_start_of_streaming(self):
        sched = HybridTrainingScheduler(batch_phase_steps=100, distill_from_step=0)
        assert sched.is_distillation_active(100)
        assert sched.is_distillation_active(10_000)

    def test_mid_stage_added_distillation(self):
        sched = HybridTrainingScheduler(batch_phase_steps=100, distill_from_step=50)
        assert not sched.is_distillation_active(120)   # 20 steps into streaming, < 50
        assert sched.is_distillation_active(150)        # 50 steps into streaming


# ---------------------------------------------------------------------------
# Black-box distillation loss (Eq. 2, 8-10)
# ---------------------------------------------------------------------------

class TestKDLoss:
    def test_kd_loss_temperature_scalar(self):
        z_t, z_s = torch.randn(10), torch.randn(10)
        loss = kd_loss_temperature(z_t, z_s, tau=2.0)
        assert loss.dim() == 0

    def test_zero_when_student_matches_teacher(self):
        z = torch.randn(10)
        assert kd_loss_temperature(z, z.clone(), tau=1.0).item() == pytest.approx(0.0, abs=1e-5)
        assert kd_loss_ce(z, z.clone()).item() > 0.0  # CE against itself isn't 0 (it's the entropy)

    def test_tau_equals_one_gradient_matches_ce(self):
        # Eq. 2 at tau=1 has the same GRADIENT w.r.t. the student as Eq. 8's
        # plain CE (they differ only by a student-independent constant).
        z_t = torch.randn(20)
        z_s_a = torch.randn(20, requires_grad=True)
        z_s_b = z_s_a.detach().clone().requires_grad_(True)

        kd_loss_temperature(z_t, z_s_a, tau=1.0).backward()
        kd_loss_ce(z_t, z_s_b).backward()
        assert torch.allclose(z_s_a.grad, z_s_b.grad, atol=1e-5)

    def test_gradient(self):
        z_t, z_s = torch.randn(10), torch.randn(10, requires_grad=True)
        kd_loss_ce(z_t, z_s).backward()
        assert z_s.grad is not None


class TestStudentObjective:
    def test_matches_manual_composition(self):
        task, distill = torch.tensor(1.0), torch.tensor(0.5)
        assert student_objective(task, distill, alpha=0.3) == pytest.approx(1.15)

    def test_alpha_zero_ignores_distillation(self):
        task, distill = torch.tensor(1.0), torch.tensor(100.0)
        assert student_objective(task, distill, alpha=0.0) == pytest.approx(1.0)


class TestGradFactors:
    def test_ce_factor_matches_error(self):
        assert ce_grad_factor(torch.tensor(0.7), torch.tensor(0.3)).item() == pytest.approx(0.4)

    def test_mse_factor_vanishes_near_saturation(self):
        p_s = torch.tensor(0.9999)
        p_t = torch.tensor(0.0)
        assert mse_grad_factor(p_s, p_t).item() < 1e-3

    def test_ce_factor_stays_informative_near_saturation(self):
        p_s = torch.tensor(0.9999)
        p_t = torch.tensor(0.0)
        assert abs(ce_grad_factor(p_s, p_t).item()) > 0.9

    def test_both_zero_at_perfect_match(self):
        p_s = p_t = torch.tensor(0.5)
        assert mse_grad_factor(p_s, p_t).item() == pytest.approx(0.0)
        assert ce_grad_factor(p_s, p_t).item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# DecoupledTowerStudent (Eq. 6-7)
# ---------------------------------------------------------------------------

class TestDecoupledTowerStudent:
    def _make(self):
        backbone = nn.Sequential(nn.Linear(6, 8), nn.ReLU())
        return DecoupledTowerStudent(backbone, backbone_out_dim=8)

    def test_output_shapes(self):
        model = self._make()
        main_logit, aux_logit = model(torch.randn(4, 6))
        assert main_logit.shape == (4,)
        assert aux_logit.shape == (4,)

    def test_main_tower_isolated_from_distillation_gradient(self):
        model = self._make()
        features = torch.randn(4, 6)
        y = torch.randint(0, 2, (4,)).float()
        _, aux_logit = model(features)
        distill = kd_loss_ce(torch.randn(4), aux_logit)
        aux_loss = auxiliary_tower_loss(aux_logit, y, distill, alpha=0.5)
        model.zero_grad()
        aux_loss.backward()
        assert all(p.grad is None or torch.all(p.grad == 0) for p in model.main_tower.parameters())

    def test_backbone_receives_gradient_from_main_tower_alone(self):
        model = self._make()
        main_logit, _ = model(torch.randn(4, 6))
        y = torch.randint(0, 2, (4,)).float()
        model.zero_grad()
        main_tower_loss(main_logit, y).backward()
        backbone_grad = model.backbone[0].weight.grad
        assert backbone_grad is not None and backbone_grad.abs().sum().item() > 0

    def test_backbone_receives_gradient_from_aux_tower_alone(self):
        model = self._make()
        _, aux_logit = model(torch.randn(4, 6))
        y = torch.randint(0, 2, (4,)).float()
        distill = kd_loss_ce(torch.randn(4), aux_logit)
        model.zero_grad()
        auxiliary_tower_loss(aux_logit, y, distill, alpha=0.5).backward()
        backbone_grad = model.backbone[0].weight.grad
        assert backbone_grad is not None and backbone_grad.abs().sum().item() > 0

    def test_main_tower_loss_ignores_distillation_signal(self):
        y = torch.tensor([1.0, 0.0])
        logit = torch.tensor([0.5, -0.5], requires_grad=True)
        loss_with = main_tower_loss(logit, y)
        # main_tower_loss takes no distillation argument at all -- by construction.
        import inspect
        assert "distill" not in inspect.signature(main_tower_loss).parameters


# ---------------------------------------------------------------------------
# Sampling-aware debias (Eq. 12-14)
# ---------------------------------------------------------------------------

class TestSamplingAwareDebias:
    def test_matches_manual_formula(self):
        z = torch.tensor([0.0])
        params = DebiasParams(r_s=0.2, p_x=0.5, r_plus=0.9, b_s=0.01)
        expected = 1.0 / (1.0 + (0.2 / 0.5) * (1.0 + 1 - 0.9) + 0.01)  # exp(0)=1
        assert sampling_aware_debias(z, params).item() == pytest.approx(expected)

    def test_different_params_give_different_outputs(self):
        z = torch.tensor([0.3, -0.2, 1.5])
        p1 = DebiasParams(r_s=0.1, p_x=0.5, r_plus=0.9, b_s=0.0)
        p2 = DebiasParams(r_s=0.4, p_x=0.3, r_plus=0.7, b_s=0.05)
        assert not torch.allclose(sampling_aware_debias(z, p1), sampling_aware_debias(z, p2))


class TestCrossDebiasDistillLoss:
    def test_scalar_output(self):
        t_raw, s_raw = torch.randn(8), torch.randn(8, requires_grad=True)
        params = DebiasParams(r_s=0.2, p_x=0.5, r_plus=0.9, b_s=0.0)
        loss = cross_debias_distill_loss(t_raw, s_raw, params)
        assert loss.dim() == 0

    def test_gradient_flows_to_student_not_teacher(self):
        t_raw = torch.randn(8, requires_grad=True)
        s_raw = torch.randn(8, requires_grad=True)
        params = DebiasParams(r_s=0.2, p_x=0.5, r_plus=0.9, b_s=0.0)
        cross_debias_distill_loss(t_raw, s_raw, params).backward()
        assert s_raw.grad is not None
        assert t_raw.grad is None  # teacher target is detached

    def test_projection_through_wrong_params_differs(self):
        # Projecting the same teacher logit through two different debias
        # configs should generally land in different corrected spaces --
        # this is exactly why Eq. 13 insists on using the STUDENT's params.
        teacher_raw = torch.tensor([0.5, -0.3, 1.2])
        student_params = DebiasParams(r_s=0.1, p_x=0.5, r_plus=0.9, b_s=0.0)
        teacher_params = DebiasParams(r_s=0.4, p_x=0.3, r_plus=0.7, b_s=0.05)
        t2 = sampling_aware_debias(teacher_raw, teacher_params)
        t2_prime = sampling_aware_debias(teacher_raw, student_params)
        assert not torch.allclose(t2, t2_prime)
