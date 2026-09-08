"""ExFM: External Large Foundation Model.

Reference PyTorch implementation of the algorithmic components of ExFM, from
*"External Large Foundation Model: How to Efficiently Serve Trillions of
Parameters for Online Ads Recommendation"* (Meta AI, Jul 2025),
arXiv:2502.17494.

Like GR2 and LoopFM, ExFM is a training *framework* for the standard
industrial two-tier setup — a large foundation model (FM) and one or more
compact vertical models (VMs) that actually serve traffic — not a single
encoder architecture. It targets two challenges prior FM-to-VM knowledge
distillation work overlooks:

  C1  Restricted training/inference budget for the serving model — the
      teacher cannot be co-trained with the student without inflating the
      student's own serving cost.
  C2  Streaming, non-stationary data — a teacher trained on an aggregation of
      many VMs' traffic (to amortize its cost across all of them) carries
      cross-domain bias relative to any one VM, and a delay between teacher
      training and its use as supervision opens a "freshness gap".

ExFM's answer:

  Sec 3.1-3.2  External distillation + Data Augmentation Service (DAS)
               -> DataAugmentationService
               The teacher (FM) is trained fully separately from any VM, and
               its predictions are logged once per shared example and reused
               across every VM that needs it — no VM pays any extra
               training/inference cost for the teacher's existence.
  Sec 3.3      Auxiliary Head (AH), Eq. 1-5
               -> h_loss, label_scaling, grad_scale, VMWithAuxiliaryHead, ah_loss
               A dedicated distillation head (not the serving head) consumes
               the FM's supervision, provably reducing cross-domain bias
               transfer; Gradient Scaling / Label Scaling / Loss Weighting
               tune how strongly that supervision reaches the shared backbone.
  Sec 3.4      Student Adapter (SA), Eq. 6-7, Algorithm 1
               -> StudentAdapter, VMWithStudentAdapter, sa_loss,
                  train_step_with_student_adapter
               A small adapter learns to re-calibrate the FM's (possibly
               stale) prediction against fresh ground truth before it's used
               as a second distillation target, closing the freshness gap.

The paper's DAS section also describes a distributed snapshot-publishing
system (Zeus/SPD, backed by Apache ZooKeeper) for safely rotating which FM
checkpoint is used to generate supervision without downtime. That is a
distributed-systems concern orthogonal to the algorithmic core and is not
implemented here — `DataAugmentationService` implements only DAS's
data-sharing/amortization logic.

A note on Eq. 1's sign convention: the paper writes
h(y_hat, y) = y*log(sigma(y_hat)) + (1-y)*log(1-sigma(y_hat)) — the *log-
likelihood*, which is <= 0. What is actually minimized during training is
its negative (standard binary cross-entropy on logits); `h_loss` below
implements that negative quantity directly, and every downstream loss (Eq.
2-7) is built by composing `h_loss` calls exactly as the paper composes h(.).
Throughout, h_loss's first argument is treated as a raw logit (sigma applied
internally) and its second as an already-computed probability/soft label.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Eq. 1: the base per-example loss
# ---------------------------------------------------------------------------

def h_loss(y_hat_logit: Tensor, y: Tensor) -> Tensor:
    """The quantity minimized wherever the paper writes h(y_hat, y) (Eq. 1):
    standard binary cross-entropy, with sigma applied internally to
    `y_hat_logit`. `y` may be a hard {0,1} label or a soft probability
    (e.g. another model's prediction, for distillation).
    """
    return F.binary_cross_entropy_with_logits(y_hat_logit, y)


# ---------------------------------------------------------------------------
# Sec 3.2 — Data Augmentation Service (DAS)
# ---------------------------------------------------------------------------

class DataAugmentationService:
    """Reference implementation of DAS's data-sharing core (Fig. 3a): join
    multiple VMs' (features, label) streams into one shared dataset keyed by
    example id, call the FM exactly once per *shared* (deduplicated) example
    — instead of once per VM — then let each VM read its own examples back
    out, now annotated with the FM's prediction.

    This is what lets FM inference cost stay proportional to the union of
    traffic across VMs rather than the sum, and is amortized further still
    when VMs have overlapping traffic.
    """

    def __init__(self, fm_predict_fn: Callable[[object], float]) -> None:
        self.fm_predict_fn = fm_predict_fn
        self.fm_calls = 0

    def build_shared_dataset(self, vm_examples: Dict[str, List[dict]]) -> Dict[object, dict]:
        """
        Args:
            vm_examples: {vm_name: [{"example_id", "features", "label"}, ...]}.
        Returns:
            {example_id: {"features", "label", "vm_names": [...]}}, deduplicated
            across VMs that observed the same example.
        """
        shared: Dict[object, dict] = {}
        for vm_name, examples in vm_examples.items():
            for ex in examples:
                entry = shared.setdefault(
                    ex["example_id"], {"features": ex["features"], "label": ex["label"], "vm_names": []}
                )
                entry["vm_names"].append(vm_name)
        return shared

    def annotate_with_fm_predictions(self, shared_dataset: Dict[object, dict]) -> Dict[object, dict]:
        """Calls the FM exactly once per shared example, writing the result
        into that example's entry (in place) for every VM to reuse.
        """
        for entry in shared_dataset.values():
            entry["fm_prediction"] = self.fm_predict_fn(entry["features"])
            self.fm_calls += 1
        return shared_dataset

    def vm_training_data(self, shared_dataset: Dict[object, dict], vm_name: str) -> List[dict]:
        """Each VM picks its own examples back out of the shared, annotated dataset."""
        return [
            {"example_id": eid, "features": e["features"], "label": e["label"], "fm_prediction": e["fm_prediction"]}
            for eid, e in shared_dataset.items()
            if vm_name in e["vm_names"]
        ]


# ---------------------------------------------------------------------------
# Sec 3.3 — Auxiliary Head (AH), Eq. 3-5
# ---------------------------------------------------------------------------

class _GradScale(torch.autograd.Function):
    """Identity in the forward pass; scales the incoming gradient by `beta`
    in the backward pass. This is what implements Gradient Scaling (GS,
    Sec 3.3): the distillation head's forward computation is untouched, but
    the strength of the gradient it sends back into the shared backbone is
    controlled by `beta`, independent of the serving head's gradient.
    """

    @staticmethod
    def forward(ctx, x: Tensor, beta: float) -> Tensor:
        ctx.beta = beta
        return x

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return grad_output * ctx.beta, None


def grad_scale(x: Tensor, beta: float) -> Tensor:
    return _GradScale.apply(x, beta)


def label_scaling(y_f: Tensor, alpha: float) -> Tensor:
    """Label Scaling (LS, Sec 3.3): amplify the FM's soft label by `alpha`,
    clipped to stay a valid probability — countering the FM's predictions
    clustering near 0 on the long-tailed engagement distribution, which
    otherwise makes the distillation target nearly uninformative.
    """
    return (alpha * y_f).clamp(max=1.0)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int = 1) -> None:
        super().__init__()
        dims = [in_dim] + hidden_dims + [out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class VMWithAuxiliaryHead(nn.Module):
    """A VM = backbone + serving head + a separate distillation (auxiliary)
    head (Fig. 4a). The serving head is supervised only by the ground-truth
    label; the auxiliary head is supervised only by the FM's soft label —
    disentangling the two is what Theorem 3.1 shows reduces bias transfer
    from the FM to the VM, relative to a single head trained on both.
    """

    def __init__(self, backbone: nn.Module, backbone_out_dim: int, head_hidden_dims: Optional[List[int]] = None) -> None:
        super().__init__()
        hidden = head_hidden_dims or [backbone_out_dim]
        self.backbone = backbone
        self.serving_head = _MLP(backbone_out_dim, hidden)
        self.distill_head = _MLP(backbone_out_dim, hidden)

    def forward(self, features: Tensor, beta: float = 1.0) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            y_s_logit: serving head's output logit.
            y_d_logit: distillation head's output logit, with Gradient
                Scaling (`beta`) applied to its path back into the backbone.
        """
        x = self.backbone(features)
        y_s = self.serving_head(x)
        y_d = self.distill_head(grad_scale(x, beta))
        return y_s, y_d


def ah_loss(y_s_logit: Tensor, y: Tensor, y_d_logit: Tensor, y_f: Tensor, alpha: float = 1.0, w: float = 1.0) -> Tensor:
    """Eq. 5: L_ah = L_s(y_hat_S, y) + w * L_d(y_hat_D, alpha * y_hat_F)

    L_s (Eq. 3) supervises the serving head with the ground-truth label;
    L_d (Eq. 4) supervises the distillation head with the (label-scaled) FM
    prediction; `w` is the Loss Weighting (LW) hyperparameter.
    """
    l_s = h_loss(y_s_logit, y)
    l_d = h_loss(y_d_logit, label_scaling(y_f, alpha))
    return l_s + w * l_d


# ---------------------------------------------------------------------------
# Sec 3.4 — Student Adapter (SA), Eq. 6-7, Algorithm 1
# ---------------------------------------------------------------------------

class StudentAdapter(nn.Module):
    """y_hat_SA = MLP(y_hat_F) (Fig. 4b): a small adapter that learns to
    re-calibrate the FM's prediction against the VM's *fresh* ground truth,
    closing the Freshness Gap between when the FM was trained and when its
    prediction is actually consumed as supervision (Theorem 3.2).

    Outputs a logit (consistent with `h_loss`'s convention); callers apply
    `torch.sigmoid` to get a probability when reusing this as a soft label.
    """

    def __init__(self, hidden_dims: Optional[List[int]] = None) -> None:
        super().__init__()
        self.mlp = _MLP(in_dim=1, hidden_dims=hidden_dims or [8])

    def forward(self, y_f: Tensor) -> Tensor:
        return self.mlp(y_f.unsqueeze(-1))


def sa_loss(y_sa_logit: Tensor, y: Tensor) -> Tensor:
    """Eq. 6: L_sta(y_hat_SA, y) = h(y_hat_SA, y) — trains SA to predict the
    fresh ground truth from the FM's own (possibly stale) prediction.
    """
    return h_loss(y_sa_logit, y)


class VMWithStudentAdapter(nn.Module):
    """A VM with AH *and* a second distillation head for SA's target (Fig.
    4b): backbone + serving head + one distillation head trained against the
    FM's label-scaled prediction (as in `VMWithAuxiliaryHead`) + a second
    distillation head trained against SA's (stop-gradiented) recalibrated
    prediction.
    """

    def __init__(self, backbone: nn.Module, backbone_out_dim: int, head_hidden_dims: Optional[List[int]] = None) -> None:
        super().__init__()
        hidden = head_hidden_dims or [backbone_out_dim]
        self.backbone = backbone
        self.serving_head = _MLP(backbone_out_dim, hidden)
        self.distill_head_fm = _MLP(backbone_out_dim, hidden)
        self.distill_head_sa = _MLP(backbone_out_dim, hidden)

    def forward(self, features: Tensor, beta: float = 1.0) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.backbone(features)
        y_s = self.serving_head(x)
        xg = grad_scale(x, beta)
        y_d_fm = self.distill_head_fm(xg)
        y_d_sa = self.distill_head_sa(xg)
        return y_s, y_d_fm, y_d_sa


def ah_sa_loss(
    y_s_logit: Tensor,
    y: Tensor,
    y_d_fm_logit: Tensor,
    y_f: Tensor,
    y_d_sa_logit: Tensor,
    y_sa_prob_sg: Tensor,
    alpha: float = 1.0,
    w: float = 1.0,
) -> Tensor:
    """Eq. 5 + Eq. 7, the VM's full training loss when using both AH and SA:

        L_ah(y_hat_S, y, y_hat_D, y_hat_F) + L_sa(y_hat_D2, SG(y_hat_SA))
    """
    l_ah = ah_loss(y_s_logit, y, y_d_fm_logit, y_f, alpha, w)
    l_sa = h_loss(y_d_sa_logit, y_sa_prob_sg)
    return l_ah + l_sa


def train_step_with_student_adapter(
    vm: VMWithStudentAdapter,
    sa: StudentAdapter,
    vm_optimizer: torch.optim.Optimizer,
    sa_optimizer: torch.optim.Optimizer,
    features: Tensor,
    y: Tensor,
    y_f: Tensor,
    alpha: float = 1.0,
    w: float = 1.0,
    beta: float = 1.0,
) -> Dict[str, float]:
    """Algorithm 1: one training iteration for a VM with Student Adapter.

        1. Optimize L_sta(y_hat_SA, y) (Eq. 6) to update SA.
        2. Stop-gradient SA's (recalibrated, sigmoid-ed) output.
        3. Optimize L_ah + L_sa (Eq. 5, 7) to update the VM.
    """
    sa_optimizer.zero_grad()
    y_sa_logit = sa(y_f)
    l_sta = sa_loss(y_sa_logit, y)
    l_sta.backward()
    sa_optimizer.step()

    with torch.no_grad():
        y_sa_prob_sg = torch.sigmoid(sa(y_f))

    vm_optimizer.zero_grad()
    y_s, y_d_fm, y_d_sa = vm(features, beta=beta)
    loss = ah_sa_loss(y_s, y, y_d_fm, y_f, y_d_sa, y_sa_prob_sg, alpha, w)
    loss.backward()
    vm_optimizer.step()

    return {"l_sta": l_sta.item(), "l_ah_sa": loss.item()}


# ---------------------------------------------------------------------------
# Evaluation metric (Sec 4.1.2): Normalized Entropy
# ---------------------------------------------------------------------------

def normalized_entropy(labels: Tensor, probs: Tensor, eps: float = 1e-7) -> float:
    """NE (Normalized Entropy), He et al. 2014: log-loss normalized by the
    entropy of the empirical label rate. Lower is better; the paper
    considers a 0.02% NE improvement significant on internal datasets.
    """
    import math

    probs = probs.clamp(min=eps, max=1 - eps)
    log_loss = F.binary_cross_entropy(probs, labels.float()).item()
    p_bar = labels.float().mean().clamp(min=eps, max=1 - eps).item()
    background_entropy = -(p_bar * math.log(p_bar) + (1 - p_bar) * math.log(1 - p_bar))
    return log_loss / background_entropy


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)

    print("--- Sec 3.2: Data Augmentation Service ---")
    calls = {"n": 0}

    def fm_predict(features):
        calls["n"] += 1
        return float(torch.sigmoid(features.sum()))

    das = DataAugmentationService(fm_predict)
    vm_examples = {
        "vm1": [
            {"example_id": 1, "features": torch.randn(4), "label": torch.tensor(1.0)},
            {"example_id": 2, "features": torch.randn(4), "label": torch.tensor(0.0)},
        ],
        "vm2": [
            {"example_id": 2, "features": torch.randn(4), "label": torch.tensor(0.0)},  # overlaps with vm1
            {"example_id": 3, "features": torch.randn(4), "label": torch.tensor(1.0)},
        ],
    }
    shared = das.build_shared_dataset(vm_examples)
    das.annotate_with_fm_predictions(shared)
    vm1_data = das.vm_training_data(shared, "vm1")
    vm2_data = das.vm_training_data(shared, "vm2")
    total_vm_examples = sum(len(v) for v in vm_examples.values())
    print(f"total (vm, example) pairs: {total_vm_examples}, unique examples: {len(shared)}, "
          f"FM calls: {das.fm_calls} (amortized, not {total_vm_examples})")
    print(f"vm1 examples: {len(vm1_data)}, vm2 examples: {len(vm2_data)}")

    print("\n--- Sec 3.3: Auxiliary Head ---")
    backbone = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
    vm = VMWithAuxiliaryHead(backbone, backbone_out_dim=16)
    features = torch.randn(32, 8)
    y = torch.randint(0, 2, (32,)).float()
    y_f = torch.rand(32) * 0.3  # FM predictions, skewed low (long-tailed engagement)

    y_s, y_d = vm(features, beta=5.0)
    loss = ah_loss(y_s, y, y_d, y_f, alpha=3.0, w=10.0)
    loss.backward()
    print(f"ah_loss: {loss.item():.4f}, backward OK")

    print("\n--- Sec 3.4: Student Adapter (Algorithm 1) ---")
    vm_sa = VMWithStudentAdapter(nn.Sequential(nn.Linear(8, 16), nn.ReLU()), backbone_out_dim=16)
    sa = StudentAdapter()
    vm_opt = torch.optim.Adam(vm_sa.parameters(), lr=1e-2)
    sa_opt = torch.optim.Adam(sa.parameters(), lr=1e-2)

    for step in range(50):
        stats = train_step_with_student_adapter(vm_sa, sa, vm_opt, sa_opt, features, y, y_f, alpha=3.0, w=5.0, beta=5.0)
    print(f"after 50 steps: l_sta={stats['l_sta']:.4f}, l_ah_sa={stats['l_ah_sa']:.4f}")

    print("\n--- Normalized Entropy ---")
    probs = torch.sigmoid(vm_sa(features)[0]).detach()
    print(f"NE: {normalized_entropy(y, probs):.4f}")


if __name__ == "__main__":
    _smoke_test()
