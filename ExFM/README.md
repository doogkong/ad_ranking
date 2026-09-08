# ExFM

PyTorch reference implementation of the algorithmic components of **ExFM: External Large Foundation Model**, from *"External Large Foundation Model: How to Efficiently Serve Trillions of Parameters for Online Ads Recommendation"* (Meta AI, Jul 2025).

Paper: https://arxiv.org/abs/2502.17494

Enables serving a trillion-parameter foundation model's benefit across production vertical models **without increasing their training or inference cost**, with consistent NE gains across retrieval, early-ranking, and later-ranking stages, and across VMs from different domains and tasks.

---

## Summary

Like GR2 and LoopFM, ExFM is a **training framework** for the standard industrial two-tier setup — a large foundation model (FM) and multiple compact vertical models (VMs) that actually serve traffic — not a single encoder architecture. It targets two challenges the paper argues prior FM→VM knowledge-distillation work overlooks:

- **C1 — Restricted training/inference budget.** The serving model (VM) has a hard latency budget; the teacher (FM) cannot be co-trained with it without inflating that cost, and building one dedicated large teacher per VM (there are often many, one per ranking stage/domain) doesn't scale.
- **C2 — Streaming, non-stationary data.** New users/ads join and old ones leave constantly; a single training pass is the norm (multi-pass risks over-fitting). A teacher trained on an *aggregation* of many VMs' traffic (to amortize its own cost) inherits **cross-domain bias** relative to any one VM, and the inevitable delay between teacher training and using its predictions as supervision opens a **freshness gap**.

ExFM's answer has three parts, each implemented here:

| Component | Addresses | This repo |
|---|---|---|
| External distillation + Data Augmentation Service (DAS) | C1: no extra training/inference cost on the VM | `DataAugmentationService` |
| Auxiliary Head (AH) | C2 (cross-domain bias): a dedicated head for the FM's supervision | `VMWithAuxiliaryHead`, `ah_loss`, `grad_scale`, `label_scaling` |
| Student Adapter (SA) | C2 (freshness gap): recalibrates stale FM predictions against fresh labels | `StudentAdapter`, `VMWithStudentAdapter`, `sa_loss`, `train_step_with_student_adapter` |

---

## Key Ideas

### External distillation, not co-distillation (Fig. 1a)

Co-distillation trains the teacher and student together, which means every VM served pays the teacher's training cost. ExFM instead trains the FM **completely separately**; its predictions are logged once and reused as external supervision. This is what lets one FM amortize its cost across many VMs (a "1-to-N foundation model") instead of needing a dedicated teacher per serving model.

### Data Augmentation Service — DAS (Sec 3.2, Fig. 3a)

Rather than calling the FM once per VM's training example, DAS joins every VM's (features, label) stream into one shared dataset keyed by example, calls the FM **once per unique example**, and lets each VM read its own subset back out — annotated with that shared FM prediction. VMs with overlapping traffic amortize the FM's inference cost even further. (The paper's snapshot-publishing system for safely rotating FM checkpoints without downtime, backed by Apache ZooKeeper, is a distributed-systems concern orthogonal to this data-sharing logic and isn't implemented here.)

### Auxiliary Head — AH (Sec 3.3, Eq. 3-5)

Feeding both the ground-truth label and the FM's soft label into one shared serving head entangles their gradients — and if the FM is even slightly biased or miscalibrated (a near-certainty when it's trained on an aggregation of many VMs' data), that bias leaks straight into the head the VM actually serves with. AH fixes this with a **separate distillation head** that consumes only the FM's supervision, leaving the serving head supervised only by the true label — which Theorem 3.1 proves reduces bias transfer, and which the paper's ablations show is what turns a marginal +0.35% AUC gain into +1.11%.

Because ad-engagement labels are heavily long-tailed (`y_hat_F` clusters near 0), plain AH distillation alone barely moves the needle in practice. Two knobs amplify it:

- **Gradient Scaling (GS)**: scales *only* the distillation head's gradient into the shared backbone by `beta`, independent of the serving head's gradient — implemented via a custom autograd function (`grad_scale`), the standard "identity-forward, scaled-backward" trick (as in a gradient-reversal layer, just with an arbitrary positive scale instead of -1).
- **Label Scaling (LS)**: amplifies the FM's soft label by `alpha` (clipped to stay a valid probability) so it carries a meaningful gradient even when it's small.

Loss Weighting (LW, the `w` in Eq. 5) is the third knob — a plain scalar on the distillation loss term.

### Student Adapter — SA (Sec 3.4, Eq. 6-7, Algorithm 1)

There's always a delay between when the FM's checkpoint that generates supervision was trained and when the VM trains on fresh data (Fig. 2 shows inference NE degrading monotonically with that delay). SA is a tiny MLP that learns to map the FM's prediction to a better estimate of the *current* ground truth, trained with its own loss (Eq. 6) and then, with its output stop-gradiented, used as a second distillation target for the VM (Eq. 7). Theorem 3.2 bounds how quickly this lets the VM track a drifting FM relative to plain KD.

---

## Key Components

### 1. `h_loss` (Eq. 1) — a sign-convention note

The paper writes `h(y_hat, y) = y*log(sigma(y_hat)) + (1-y)*log(1-sigma(y_hat))`, the *log-likelihood* (a value <= 0). What's actually minimized during training is its negative — standard binary cross-entropy — which is what `h_loss` implements directly. Every downstream loss (Eq. 2-7: `ah_loss`, `sa_loss`, `ah_sa_loss`) is built by composing `h_loss` calls exactly as the paper composes `h(.)`, with the first argument always treated as a raw logit and the second as an already-computed probability/soft label.

### 2. `DataAugmentationService` (Sec 3.2)

`build_shared_dataset` deduplicates examples across VMs; `annotate_with_fm_predictions` calls the FM exactly once per unique example; `vm_training_data` lets each VM read its own examples back out. `test_amortizes_fm_calls_across_overlapping_vms` verifies FM call count equals the number of *unique* examples, not the sum across VMs.

### 3. `VMWithAuxiliaryHead` / `ah_loss` (Eq. 3-5)

A backbone + two heads: `serving_head` (supervised only by `y`) and `distill_head` (supervised only by the label-scaled FM prediction, via `grad_scale`-controlled gradient flow into the backbone).

### 4. `StudentAdapter` / `VMWithStudentAdapter` / `train_step_with_student_adapter` (Eq. 6-7, Algorithm 1)

`train_step_with_student_adapter` implements Algorithm 1 verbatim: (1) update SA against fresh labels, (2) stop-gradient its output, (3) update the VM against both the FM's own prediction and SA's recalibrated one.

### 5. `normalized_entropy` (Sec 4.1.2)

The paper's primary internal metric (He et al., 2014): log-loss normalized by the entropy of the empirical label rate. A 0.02% NE improvement is considered significant on internal datasets.

### What's not implemented

The theoretical bias/regression bounds (Theorem 3.1, 3.2) are population-level results that motivate AH and SA's designs, not algorithms to run; and DAS's distributed snapshot-publishing system (Zeus/SPD, Apache ZooKeeper-backed) is systems infrastructure outside this module's scope.

---

## Usage

```python
from exfm import (
    DataAugmentationService, VMWithAuxiliaryHead, ah_loss,
    StudentAdapter, VMWithStudentAdapter, train_step_with_student_adapter,
)

# Sec 3.2 — amortize FM inference across VMs with overlapping traffic
das = DataAugmentationService(fm_predict_fn=fm_model.predict)
shared = das.build_shared_dataset({"vm1": vm1_examples, "vm2": vm2_examples})
das.annotate_with_fm_predictions(shared)
vm1_train = das.vm_training_data(shared, "vm1")

# Sec 3.3 — Auxiliary Head with Gradient/Label Scaling
vm = VMWithAuxiliaryHead(backbone, backbone_out_dim=256)
y_s, y_d = vm(features, beta=20.0)          # GS
loss = ah_loss(y_s, labels, y_d, fm_predictions, alpha=5.0, w=10.0)  # LS, LW
loss.backward()

# Sec 3.4 — Student Adapter (Algorithm 1)
vm_sa, sa = VMWithStudentAdapter(backbone, backbone_out_dim=256), StudentAdapter()
vm_opt, sa_opt = torch.optim.Adam(vm_sa.parameters()), torch.optim.Adam(sa.parameters())
for features, labels, fm_predictions in data_stream:
    train_step_with_student_adapter(vm_sa, sa, vm_opt, sa_opt, features, labels, fm_predictions,
                                    alpha=5.0, w=10.0, beta=20.0)
```

---

## Files

```
ExFM/
├── exfm.py         # full implementation + smoke test
│   ├── h_loss                          # Eq. 1
│   ├── DataAugmentationService         # Sec 3.2
│   ├── grad_scale, label_scaling       # Sec 3.3 (GS, LS)
│   ├── VMWithAuxiliaryHead, ah_loss    # Sec 3.3, Eq. 3-5
│   ├── StudentAdapter, sa_loss         # Sec 3.4, Eq. 6
│   ├── VMWithStudentAdapter, ah_sa_loss  # Sec 3.4, Eq. 7
│   ├── train_step_with_student_adapter   # Algorithm 1
│   └── normalized_entropy              # Sec 4.1.2 metric
├── test_exfm.py    # pytest test suite (30 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 exfm.py
```

Expected output (abridged):

```
--- Sec 3.2: Data Augmentation Service ---
total (vm, example) pairs: 4, unique examples: 3, FM calls: 3 (amortized, not 4)
vm1 examples: 2, vm2 examples: 2

--- Sec 3.3: Auxiliary Head ---
ah_loss: 7.8139, backward OK

--- Sec 3.4: Student Adapter (Algorithm 1) ---
after 50 steps: l_sta=0.6853, l_ah_sa=3.6170

--- Normalized Entropy ---
NE: 0.2895
```

### Test suite

```bash
python3 -m pytest test_exfm.py -v
```

Expected output:

```
collected 30 items
...
30 passed in ~1s
```

Useful variants:

```bash
python3 -m pytest test_exfm.py::TestTrainStepWithStudentAdapter -v
python3 -m pytest test_exfm.py -x
```
