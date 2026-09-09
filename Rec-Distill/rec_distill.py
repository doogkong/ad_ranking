"""Rec-Distill: An Industrial Distillation Pipeline for Large-Scale Recommendation Models.

Reference implementation of the algorithmic components of Rec-Distill, from
*"Rec-Distill: An Industrial Distillation Pipeline for Large-Scale
Recommendation Models"* (ByteDance AML, May 2026), arXiv:2605.29755.

Like GR2, LoopFM, ExFM, and AdvertiserPredictor, Rec-Distill is a training
*pipeline*, not a single encoder architecture. Its problem: naive knowledge
distillation (co-training a teacher and student together) can't scale in
recommendation systems, because industrial recommenders must continuously
learn from non-stationary streaming data — a co-trained teacher that stops
updating quickly goes stale, but co-training also means every student pays
the teacher's training cost, and a single expensive teacher can't cheaply
supervise multiple students across a multi-stage pipeline (retrieval,
pre-ranking, ranking, ...).

Rec-Distill's answer, decomposed into a "1-to-N" *decoupled* teacher-student
paradigm:

  Sec 3.2-3.3   Gain decomposition            -> teacher_gain,
                (Eq. 1, 4-5, 15)                  distillation_transferability,
                                                   distill_gain
  Sec 4.1       Decoupled teacher/student      -> TeacherSignalCache,
                training + caching                HybridTrainingScheduler
  Sec 4.3.2     Black-box distillation loss    -> kd_loss_temperature,
                (Eq. 2-3, 8-11)                    kd_loss_ce, student_objective,
                                                    mse_grad_factor, ce_grad_factor
  Sec 4.3.1     Decoupled-tower student         -> DecoupledTowerStudent,
                architecture (Eq. 6-7)             main_tower_loss, auxiliary_tower_loss
  Sec 4.3.3     Sampling-aware student debias    -> DebiasParams,
                (Eq. 12-14)                        sampling_aware_debias,
                                                    cross_debias_distill_loss

Teacher scaling itself (up to 24B dense parameters via TokenMixer-Large, and
20K-length sequences via LONGER — both implemented elsewhere in this repo,
see `../RankMixer`/`../tokenmixer_large`) and the distributed batch/streaming
infrastructure (Kafka-style intermediate storage, HDFS, Zookeeper-style
snapshot coordination) are out of scope: this module is the LLM/model-
agnostic scaffolding — the reward-free "gain accounting," loss design,
architecture, and debiasing math — around them.
"""

from typing import Callable, Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Sec 3.2-3.3 — Gain decomposition (Eq. 1, 4-5, 15)
# ---------------------------------------------------------------------------

def teacher_gain(p_teacher: float, p_student_raw: float) -> float:
    """Eq. 1/4: DeltaGain_scale = P_T - P_S^raw — the teacher's performance
    advantage over the (non-distilled) raw student, i.e. how much headroom
    scaling up the teacher actually created.
    """
    return p_teacher - p_student_raw


def distillation_transferability(p_teacher: float, p_student_raw: float, p_student_distill: float) -> float:
    """Eq. 5/15: eta = (P_S^distill - P_S^raw) / (P_T - P_S^raw) — the
    fraction of the teacher's advantage over the raw student that the
    distilled student actually recovers. This is the paper's central metric
    for how efficiently a distillation setup converts teacher gain into
    student gain.
    """
    denom = p_teacher - p_student_raw
    if denom == 0:
        raise ValueError(
            "distillation_transferability is undefined when the teacher has no gain over the raw student."
        )
    return (p_student_distill - p_student_raw) / denom


def distill_gain(p_student_raw: float, p_student_distill: float) -> float:
    """DeltaGain_distill = P_S^distill - P_S^raw. By Eq. 1's decomposition,
    this equals `teacher_gain(...) * distillation_transferability(...)` —
    see `test_rec_distill.py::TestGainDecomposition::test_identity_holds`.
    """
    return p_student_distill - p_student_raw


# ---------------------------------------------------------------------------
# Sec 4.1 — Decoupled ("1-to-N") teacher/student training
# ---------------------------------------------------------------------------

class TeacherSignalCache:
    """The "High-Speed Storage" of Fig. 1/2: the teacher writes its logit for
    an example exactly once, during its own forward pass; any number of
    downstream student models can then read that same cached signal without
    ever triggering another teacher inference call. This is the mechanism
    that makes the paper's "1-to-N" paradigm cheap: one teacher can supervise
    many independently-trained students (e.g. one per pipeline stage) without
    their training costs stacking on top of the teacher's.
    """

    def __init__(self) -> None:
        self._store: Dict[object, float] = {}
        self.write_count = 0
        self.read_count = 0

    def write(self, example_id: object, teacher_logit: float) -> None:
        self._store[example_id] = teacher_logit
        self.write_count += 1

    def read(self, example_id: object) -> float:
        self.read_count += 1
        return self._store[example_id]

    def read_batch(self, example_ids) -> Tensor:
        return torch.tensor([self.read(eid) for eid in example_ids])


class HybridTrainingScheduler:
    """Sec 4.4/Fig. 2: the two-phase training schedule — an initial Batch
    Distillation Phase for fast convergence, followed by a Streaming
    Distillation Phase for continuous adaptation to live data drift.

    `distill_from_step` controls how many streaming-phase steps to run
    before distillation supervision turns on. The paper's ablation (Fig. 5)
    shows that even many months in, a student whose streaming distillation
    signal was added mid-stream never catches up to one that had it from the
    very start of streaming — i.e. `distill_from_step=0` (persistent)
    consistently beats `distill_from_step>0` (mid-stage-added).
    """

    def __init__(self, batch_phase_steps: int, distill_from_step: int = 0) -> None:
        self.batch_phase_steps = batch_phase_steps
        self.distill_from_step = distill_from_step

    def phase(self, step: int) -> str:
        return "batch" if step < self.batch_phase_steps else "streaming"

    def is_distillation_active(self, step: int) -> bool:
        if step < self.batch_phase_steps:
            return True  # batch phase: always distilled, for fast initial convergence
        return (step - self.batch_phase_steps) >= self.distill_from_step


# ---------------------------------------------------------------------------
# Sec 4.3.2 — Black-box distillation loss (Eq. 2-3, 8-11)
# ---------------------------------------------------------------------------

def kd_loss_temperature(z_teacher: Tensor, z_student: Tensor, tau: float = 1.0, eps: float = 1e-7) -> Tensor:
    """Eq. 2: L_distill = tau^2 * D_KL(p_T || p_S), the temperature-scaled KL
    divergence between the teacher's and student's temperature-softened
    (sigmoid) output distributions, for a binary target.
    """
    p_t = torch.sigmoid(z_teacher / tau).clamp(eps, 1 - eps)
    p_s = torch.sigmoid(z_student / tau).clamp(eps, 1 - eps)
    kl = p_t * torch.log(p_t / p_s) + (1 - p_t) * torch.log((1 - p_t) / (1 - p_s))
    return (tau ** 2) * kl.mean()


def kd_loss_ce(z_teacher: Tensor, z_student: Tensor, eps: float = 1e-7) -> Tensor:
    """Eq. 8: at tau=1, Eq. 2's KL divergence has the same gradient (w.r.t.
    the student) as plain binary cross-entropy against the teacher's soft
    label p_T — so in practice the black-box distillation loss is just CE
    against a soft target, dropping the (student-independent) entropy term
    that KL includes.
    """
    p_t = torch.sigmoid(z_teacher).clamp(eps, 1 - eps)
    p_s = torch.sigmoid(z_student).clamp(eps, 1 - eps)
    return -(p_t * torch.log(p_s) + (1 - p_t) * torch.log(1 - p_s)).mean()


def student_objective(task_loss: Tensor, distill_loss: Tensor, alpha: float) -> Tensor:
    """Eq. 3, relaxed per Eq. 11: L_S = L_task + alpha * L_distill.

    The paper explicitly abandons the naive constraint that the task-loss and
    distill-loss weights must sum to 1 (Eq. 3's alpha/(1-alpha) framing):
    empirically, L_distill's natural scale is often two orders of magnitude
    smaller than L_task in binary classification, so a shared "sums to 1"
    weighting would leave alpha far too small to matter.
    """
    return task_loss + alpha * distill_loss


def mse_grad_factor(p_s: Tensor, p_t: Tensor) -> Tensor:
    """Eq. 9: the (p_s - p_t) * p_s * (1 - p_s) factor of an MSE-style
    distillation loss's gradient w.r.t. model parameters (excluding the
    model-dependent dz_S/d(theta) term). Vanishes as p_s saturates toward 0
    or 1, regardless of how wrong the prediction actually is.
    """
    return (p_s - p_t) * p_s * (1 - p_s)


def ce_grad_factor(p_s: Tensor, p_t: Tensor) -> Tensor:
    """Eq. 10: the (p_s - p_t) factor of the CE distillation loss's
    gradient. Stays proportional to the raw prediction error even as p_s
    saturates — the reason the paper uses CE rather than MSE for black-box
    distillation, avoiding Eq. 9's vanishing-gradient failure mode.
    """
    return p_s - p_t


# ---------------------------------------------------------------------------
# Sec 4.3.1 — Decoupled-tower student architecture (Eq. 6-7)
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims, out_dim: int = 1) -> None:
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class DecoupledTowerStudent(nn.Module):
    """Fig. 1 (right side)/Sec 4.3.1: a shared backbone feeding two
    independent towers.

    - **Main Task Tower**: trained EXCLUSIVELY on the ground-truth task loss
      (Eq. 6) and is what actually serves online traffic.
    - **Auxiliary Tower**: trained on the combined task + distillation loss
      (Eq. 7) and is where the distillation signal actually lands.

    Both towers' gradients flow back into the shared backbone, so its
    learned representations benefit from both direct supervision and
    distilled knowledge — but a stalled or corrupted distillation pipeline
    can only ever affect the Auxiliary Tower's own parameters, never the
    Main Tower's, since the two towers share no parameters downstream of the
    backbone. This is a deliberate fault-isolation property for production
    serving robustness.
    """

    def __init__(self, backbone: nn.Module, backbone_out_dim: int, tower_hidden_dims=None) -> None:
        super().__init__()
        hidden = tower_hidden_dims or [backbone_out_dim]
        self.backbone = backbone
        self.main_tower = _MLP(backbone_out_dim, hidden)
        self.aux_tower = _MLP(backbone_out_dim, hidden)

    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.backbone(features)
        return self.main_tower(x), self.aux_tower(x)


def main_tower_loss(main_logit: Tensor, y: Tensor) -> Tensor:
    """Eq. 6: L_main = L_task."""
    return F.binary_cross_entropy_with_logits(main_logit, y)


def auxiliary_tower_loss(aux_logit: Tensor, y: Tensor, distill_loss_value: Tensor, alpha: float) -> Tensor:
    """Eq. 7: L_aux = L_task + alpha * L_distill."""
    task = F.binary_cross_entropy_with_logits(aux_logit, y)
    return task + alpha * distill_loss_value


# ---------------------------------------------------------------------------
# Sec 4.3.3 — Sampling-aware student debias (Eq. 12-14)
# ---------------------------------------------------------------------------

class DebiasParams(NamedTuple):
    """The four sampling-strategy parameters of Eq. 12, specific to one side
    (teacher or student) of a distillation setup — each side may be trained
    under a different sampling configuration.

    r_s: negative sampling ratio. p_x: effective retention probability of
    the target positive event. r_plus: nominal positive-retention rate.
    b_s: additive bias term introduced by the sampling strategy.
    """

    r_s: float
    p_x: float
    r_plus: float
    b_s: float


def sampling_aware_debias(z: Tensor, params: DebiasParams) -> Tensor:
    """Eq. 12: converts a raw logit `z` (trained on negative-sampled data)
    into a debiased posterior probability estimate, correcting for that
    side's sampling configuration:

        y_hat = 1 / (1 + (r_s / p_x) * (exp(-z) + 1 - r_plus) + b_s)
    """
    return 1.0 / (1.0 + (params.r_s / params.p_x) * (torch.exp(-z) + 1 - params.r_plus) + params.b_s)


def cross_debias_distill_loss(
    teacher_raw_logit: Tensor,
    student_raw_logit: Tensor,
    student_debias_params: DebiasParams,
    eps: float = 1e-7,
) -> Tensor:
    """Eq. 13-14: the teacher and student are generally trained under
    different sampling configurations, so their own respective debias
    corrections (f_T, f_S) land in different corrected probability spaces —
    directly comparing f_T(teacher_raw) against f_S(student_raw) would
    introduce yet another bias from that mismatch.

    The fix: project the teacher's RAW logit through the STUDENT's debias
    function too (Eq. 13: T2' = f_S(T1)), putting both terms in the same
    (student-side-corrected) space, then distill with a CE-style loss
    (Eq. 14: D(T2', S2)) — exactly mirroring `kd_loss_ce`, but applied to
    debiased probabilities rather than raw logits.
    """
    t2_prime = sampling_aware_debias(teacher_raw_logit, student_debias_params).detach()
    s2 = sampling_aware_debias(student_raw_logit, student_debias_params).clamp(eps, 1 - eps)
    return -(t2_prime * torch.log(s2) + (1 - t2_prime) * torch.log(1 - s2)).mean()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)

    print("--- Sec 3.2-3.3: Gain decomposition (Eq. 1, 4-5) ---")
    p_teacher, p_student_raw, p_student_distill = 0.8879, 0.8624, 0.8818
    tg = teacher_gain(p_teacher, p_student_raw)
    eta = distillation_transferability(p_teacher, p_student_raw, p_student_distill)
    dg = distill_gain(p_student_raw, p_student_distill)
    print(f"teacher_gain={tg:.4f}, eta={eta:.4f}, distill_gain={dg:.4f}, "
          f"identity holds: {abs(dg - tg * eta) < 1e-9}")

    print("\n--- Sec 4.1: Decoupled '1-to-N' training ---")
    cache = TeacherSignalCache()
    for i in range(5):
        cache.write(f"ex{i}", float(i) * 0.1)
    student_a_logits = cache.read_batch([f"ex{i}" for i in range(5)])
    student_b_logits = cache.read_batch([f"ex{i}" for i in range(5)])
    print(f"teacher writes: {cache.write_count}, total reads (2 students): {cache.read_count}")

    scheduler = HybridTrainingScheduler(batch_phase_steps=1000, distill_from_step=0)
    print(f"step 500 phase={scheduler.phase(500)}, distill_active={scheduler.is_distillation_active(500)}")
    print(f"step 1500 phase={scheduler.phase(1500)}, distill_active={scheduler.is_distillation_active(1500)}")

    print("\n--- Sec 4.3.2: Black-box distillation loss (Eq. 2, 8-10) ---")
    z_t, z_s = torch.randn(16), torch.randn(16, requires_grad=True)
    loss_ce = kd_loss_ce(z_t, z_s)
    loss_ce.backward()
    print(f"kd_loss_ce: {loss_ce.item():.4f}, backward OK")

    p_s_saturated, p_t = torch.tensor(0.999), torch.tensor(0.0)
    print(f"near saturation (p_s=0.999, p_t=0): mse_grad_factor={mse_grad_factor(p_s_saturated, p_t).item():.6f} "
          f"(vanishing), ce_grad_factor={ce_grad_factor(p_s_saturated, p_t).item():.6f} (stays informative)")

    print("\n--- Sec 4.3.1: Decoupled-tower student ---")
    backbone = nn.Sequential(nn.Linear(10, 16), nn.ReLU())
    student = DecoupledTowerStudent(backbone, backbone_out_dim=16)
    features = torch.randn(8, 10)
    y = torch.randint(0, 2, (8,)).float()
    main_logit, aux_logit = student(features)
    l_main = main_tower_loss(main_logit, y)
    l_aux = auxiliary_tower_loss(aux_logit, y, kd_loss_ce(z_t[:8], aux_logit), alpha=0.1)
    student.zero_grad()
    l_aux.backward()
    main_tower_grad_from_aux = sum(p.grad.abs().sum().item() for p in student.main_tower.parameters() if p.grad is not None)
    print(f"L_aux backward: main tower gradient contamination = {main_tower_grad_from_aux:.6f} (should be 0.0)")

    print("\n--- Sec 4.3.3: Sampling-aware cross-debias (Eq. 12-14) ---")
    student_params = DebiasParams(r_s=0.1, p_x=0.5, r_plus=0.9, b_s=0.01)
    teacher_params = DebiasParams(r_s=0.3, p_x=0.6, r_plus=0.8, b_s=0.02)
    teacher_raw, student_raw = torch.randn(16), torch.randn(16, requires_grad=True)
    t2 = sampling_aware_debias(teacher_raw, teacher_params)      # teacher's own debias space
    t2_prime = sampling_aware_debias(teacher_raw, student_params)  # projected into student's space
    loss = cross_debias_distill_loss(teacher_raw, student_raw, student_params)
    loss.backward()
    print(f"T2 (teacher-space) vs T2' (student-space) differ: {not torch.allclose(t2, t2_prime)}")
    print(f"cross_debias_distill_loss: {loss.item():.4f}, backward OK")


if __name__ == "__main__":
    _smoke_test()
