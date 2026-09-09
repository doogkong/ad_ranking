# Rec-Distill

Reference implementation of the algorithmic components of **Rec-Distill: An Industrial Distillation Pipeline for Large-Scale Recommendation Models** (ByteDance AML, May 2026).

Paper: https://arxiv.org/abs/2605.29755

Scales teacher models up to **24B dense parameters** and **20K-length behavior sequences**, while distilling gains into lightweight students with transferability exceeding **60%** in the best setting. Deployed across Douyin/TikTok's advertising, recommendation, and live-streaming services, delivering measurable business gains in every deployed scenario (e.g. +1.0% ADVV, +1.1% ADSS in Ads; +1.27% Finish/U in Rec).

---

## Summary

Like GR2, LoopFM, ExFM, and AdvertiserPredictor, Rec-Distill is a **training pipeline**, not a single encoder architecture. Its problem: standard knowledge distillation (KD) either co-trains teacher and student together — which means the teacher's training cost lands on every student, and doesn't scale to a "1-to-N" setup with multiple students across a multi-stage pipeline — or uses a static, frozen teacher, which quickly goes stale against recommendation's constantly-shifting streaming data.

Rec-Distill's fix is a fully **decoupled** teacher-student paradigm: the teacher trains continuously on streaming data and simply *caches* its logits as it goes; students read those cached signals independently, at their own pace, with their own architecture, and without ever calling the teacher directly. The paper frames its whole optimization target as a decomposition:

```
DeltaGain_distill = DeltaGain_scale x eta
```

— the total distillation gain is the product of **how much better scaling up the teacher made it** (`DeltaGain_scale`) and **what fraction of that advantage actually reaches the student** (`eta`, the *distillation transferability*). Rec-Distill separately optimizes both halves: it scales the teacher (reusing `../RankMixer`/`../tokenmixer_large`-style architectures and `LONGER`-style long-sequence modeling, elsewhere in this repo), and it maximizes transferability through loss design, a debiasing mechanism, and a decoupled student architecture — the parts implemented here.

---

## Key Ideas

### Gain decomposition (Eq. 1, 4-5)

`teacher_gain` is the headroom scaling the teacher created; `distillation_transferability` is the fraction of it the student actually recovers; `distill_gain` is what the student gained in absolute terms — and by construction, `distill_gain == teacher_gain * distillation_transferability` (verified directly in `test_identity_holds`). This framing is what lets the paper reason about *where* to invest engineering effort: scaling the teacher without improving transferability is wasted, and vice versa.

### Decoupled "1-to-N" training (Sec 4.1)

The teacher trains continuously on streaming data and writes its logits to a cache exactly once per example; any number of student models — potentially at different pipeline stages, with different architectures — read that same cached signal without triggering additional teacher inference. `TeacherSignalCache` models this write-once/read-many amortization directly; `HybridTrainingScheduler` models the two-phase training schedule (an initial batch phase for fast convergence, then a streaming phase for continuous adaptation) and its key empirical finding: distillation supervision that's present from the very start of the streaming phase (`distill_from_step=0`) durably outperforms supervision added only mid-stream, even many months later.

### Black-box distillation loss (Eq. 2-3, 8-11)

Rec-Distill distills exclusively from the teacher's final output logits (no access to internal features — "black-box," the best tradeoff of efficacy vs. communication/storage overhead at industrial scale). `kd_loss_temperature` implements the general temperature-scaled KL-divergence formulation; at `tau=1` it has the *same gradient* as plain cross-entropy against the teacher's soft label (`kd_loss_ce` — verified via `test_tau_equals_one_gradient_matches_ce`, which checks gradient equivalence rather than value equality, since KL includes a student-independent entropy term CE doesn't). `mse_grad_factor` / `ce_grad_factor` make the paper's stated reason for preferring CE mechanically checkable: **MSE's gradient vanishes as the student's prediction saturates toward 0 or 1, regardless of how wrong it actually is; CE's gradient does not.**

### Decoupled-tower student (Sec 4.3.1, Eq. 6-7)

The student splits into a **Main Task Tower** (trained exclusively on ground truth — this is what actually serves traffic) and an **Auxiliary Tower** (trained on task loss *plus* the distillation loss). Both towers update a shared backbone, so its representations benefit from distilled knowledge either way — but because the towers themselves share no parameters, a stalled or corrupted distillation pipeline can only ever contaminate the Auxiliary Tower, never the Main Tower actually serving requests. `test_main_tower_isolated_from_distillation_gradient` verifies this mechanically: backpropagating only the auxiliary loss produces exactly zero gradient on the main tower's parameters.

### Sampling-aware student debias (Eq. 12-14)

Industrial training data is typically negative-sampled, and — critically — **the teacher and student are often sampled differently** (e.g. the teacher can afford a larger, more aggressively-sampled training set). Naively matching each side's own debias-corrected output would compare two probabilities living in *different* corrected spaces, introducing yet another bias from that mismatch. The fix: project the teacher's **raw** logit through the **student's** debias function (Eq. 13), so both sides of the distillation loss are evaluated in the same corrected space before comparing them (Eq. 14). `test_projection_through_wrong_params_differs` shows directly that projecting the same teacher logit through different sampling configurations lands in genuinely different places — which is exactly why the projection has to use the *student's* parameters, not the teacher's own.

---

## Key Components

| Component | Paper section |
|---|---|
| `teacher_gain`, `distillation_transferability`, `distill_gain` | Sec 3.2-3.3, Eq. 1, 4-5, 15 |
| `TeacherSignalCache`, `HybridTrainingScheduler` | Sec 4.1, 4.4 |
| `kd_loss_temperature`, `kd_loss_ce`, `student_objective`, `mse_grad_factor`, `ce_grad_factor` | Sec 4.3.2, Eq. 2-3, 8-11 |
| `DecoupledTowerStudent`, `main_tower_loss`, `auxiliary_tower_loss` | Sec 4.3.1, Eq. 6-7 |
| `DebiasParams`, `sampling_aware_debias`, `cross_debias_distill_loss` | Sec 4.3.3, Eq. 12-14 |

### What's not implemented

Teacher scaling itself (dense-parameter scaling via a TokenMixer-Large-style architecture, and long-sequence modeling via a LONGER-style architecture — see [`../RankMixer`](../RankMixer)/[`../tokenmixer_large`](../tokenmixer_large) for related implementations elsewhere in this repo) and the distributed batch/streaming infrastructure (Kafka-style intermediate storage, HDFS, snapshot coordination) are out of scope — this module is the model-agnostic scaffolding around them.

---

## Usage

```python
from rec_distill import (
    teacher_gain, distillation_transferability,
    TeacherSignalCache, HybridTrainingScheduler,
    kd_loss_ce, DecoupledTowerStudent, main_tower_loss, auxiliary_tower_loss,
    DebiasParams, cross_debias_distill_loss,
)

# Sec 3.2-3.3: measure how much of the teacher's advantage the student recovered
eta = distillation_transferability(p_teacher=0.8879, p_student_raw=0.8624, p_student_distill=0.8818)

# Sec 4.1: teacher writes once, students read independently
cache = TeacherSignalCache()
cache.write(example_id, teacher_logit)
teacher_signal = cache.read_batch(batch_example_ids)   # any number of students can call this

# Sec 4.3.1: decoupled-tower student
student = DecoupledTowerStudent(backbone, backbone_out_dim=256)
main_logit, aux_logit = student(features)
loss = main_tower_loss(main_logit, labels) + auxiliary_tower_loss(
    aux_logit, labels, kd_loss_ce(teacher_signal, aux_logit), alpha=0.1
)

# Sec 4.3.3: cross-debias distillation when teacher/student are sampled differently
student_debias = DebiasParams(r_s=0.1, p_x=0.5, r_plus=0.9, b_s=0.0)
distill_loss = cross_debias_distill_loss(teacher_raw_logit, student_raw_logit, student_debias)
```

---

## Files

```
Rec-Distill/
├── rec_distill.py       # full implementation + smoke test
├── test_rec_distill.py  # pytest test suite (33 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 rec_distill.py
```

Expected output (abridged):

```
--- Sec 3.2-3.3: Gain decomposition (Eq. 1, 4-5) ---
teacher_gain=0.0255, eta=0.7608, distill_gain=0.0194, identity holds: True

--- Sec 4.1: Decoupled '1-to-N' training ---
teacher writes: 5, total reads (2 students): 10

--- Sec 4.3.2: Black-box distillation loss (Eq. 2, 8-10) ---
near saturation (p_s=0.999, p_t=0): mse_grad_factor=0.000998 (vanishing), ce_grad_factor=0.999000 (stays informative)

--- Sec 4.3.1: Decoupled-tower student ---
L_aux backward: main tower gradient contamination = 0.000000 (should be 0.0)

--- Sec 4.3.3: Sampling-aware cross-debias (Eq. 12-14) ---
T2 (teacher-space) vs T2' (student-space) differ: True
```

### Test suite

```bash
python3 -m pytest test_rec_distill.py -v
```

Expected output:

```
collected 33 items
...
33 passed in <1s
```

Useful variants:

```bash
python3 -m pytest test_rec_distill.py::TestDecoupledTowerStudent -v
python3 -m pytest test_rec_distill.py -x
```
