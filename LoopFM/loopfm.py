"""LoopFM: Learning frOm HistOrical RePresentations of Foundation Model for Recommendation.

Reference PyTorch implementation of the LoopFM pipeline, from *"LoopFM:
Learning frOm HistOrical RePresentations of Foundation Model for
Recommendation"* (Meta AI, Jun 2026), arXiv:2605.29280.

Like GR2, LoopFM is not a single encoder architecture — it is a **modular
transfer pipeline** that opens a second, high-bandwidth channel between a
large foundation model (FM) and the compact vertical models (VMs) that
actually serve traffic, alongside the usual scalar knowledge-distillation
(KD) channel. Scalar KD compresses everything the FM learned about an
example into one soft-label number; LoopFM instead *materializes* the FM's
intermediate representations as structured input features (concretely: a
user-keyed temporal sequence of the FM's own historical embeddings) that the
VM can consume directly, without needing real-time FM inference at serving.

The pipeline has four stages (Figure 1 of the paper), each implemented here:

  Stage 1  Extraction   -> extract_embedding
  Stage 2  Compression  -> MatryoshkaAutoencoder, quantize_int4
  Stage 3  Structuring  -> build_user_sequence, group_by_key
  Stage 4  VM serving   -> MeanPoolSequenceEncoder, SumPoolSequenceEncoder,
                            AttentionPoolSequenceEncoder, loopfm_training_loss

Section 4's theoretical contribution (a gain decomposition of LoopFM's
transfer into temporal/cross-feature/compression-loss mutual-information
terms, and a transfer-ratio lower bound) is a population-level bound over
Bayes risk and mutual information under idealized assumptions (NTK regime,
benign overfitting) — it characterizes *why* the pipeline works, but isn't
itself an algorithm to implement. What IS directly implementable and is what
the paper's own experiments report is the empirical transfer ratio,
TR = delta_NE_VM / delta_NE_FM, provided here as `normalized_entropy` and
`transfer_ratio`.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Stage 1 — Extraction (Sec 3, Stage 1)
# ---------------------------------------------------------------------------

def extract_embedding(layer_activations: Sequence[Tensor]) -> Tensor:
    """Concatenates a chosen subset of an FM's layer activations into one raw
    embedding e^(i) in R^D for an example, per Stage 1: given layers
    l_1,...,l_M, select K of them and concatenate. D = sum of the selected
    layers' widths.

    Args:
        layer_activations: K tensors, each (B, d_lk).
    Returns:
        (B, D) concatenated raw embedding.
    """
    return torch.cat(list(layer_activations), dim=-1)


def stop_gradient(x: Tensor) -> Tensor:
    """No-op forward, blocks gradient flow — the autoencoder trains on the
    FM's activations, but its loss must never update the FM backbone.
    """
    return x.detach()


# ---------------------------------------------------------------------------
# Stage 2 — Compression: Matryoshka Autoencoder + INT4 quantization (Eq. 1)
# ---------------------------------------------------------------------------

class MatryoshkaAutoencoder(nn.Module):
    """Compresses a raw embedding e in R^D to z in R^d (d << D), Eq. 1:

        z = f_enc(e),  e_hat = f_dec(z),  L_AE = ||e - e_hat||^2

    The encoder's final layer is Tanh, bounding z to [-1, 1] (this is what
    makes INT4 quantization in `quantize_int4` lossless-ish). Trained with a
    Matryoshka-style loss (Kusupati et al., 2022): the decoder is also asked
    to reconstruct e from every zero-padded PREFIX z[:, :d'] for a
    configurable set of target dimensions D, so any prefix of z is itself a
    valid, directly usable representation — letting deployment trade off
    feature width against storage/latency without retraining.

    Args:
        raw_dim: D, the concatenated raw embedding width.
        latent_dim: d, the full compressed width.
        prefix_dims: target dimensions d' <= d to train as valid standalone
            prefixes (defaults to just [latent_dim], i.e. no Matryoshka
            nesting).
    """

    def __init__(self, raw_dim: int, latent_dim: int, prefix_dims: Optional[List[int]] = None) -> None:
        super().__init__()
        self.raw_dim = raw_dim
        self.latent_dim = latent_dim
        self.prefix_dims = sorted(prefix_dims or [latent_dim])
        if self.prefix_dims[-1] > latent_dim:
            raise ValueError("prefix_dims cannot exceed latent_dim")

        self.encoder = nn.Sequential(nn.Linear(raw_dim, latent_dim), nn.Tanh())
        self.decoder = nn.Linear(latent_dim, raw_dim)

    def encode(self, e: Tensor) -> Tensor:
        """e: (B, raw_dim) -> z: (B, latent_dim), bounded to [-1, 1]."""
        return self.encoder(e)

    def decode(self, z_prefix: Tensor) -> Tensor:
        """Reconstructs e from a (possibly truncated) prefix of z, zero-padded
        back to latent_dim before the shared decoder is applied.
        """
        B, d_prime = z_prefix.shape
        if d_prime < self.latent_dim:
            pad = torch.zeros(B, self.latent_dim - d_prime, device=z_prefix.device, dtype=z_prefix.dtype)
            z_prefix = torch.cat([z_prefix, pad], dim=-1)
        return self.decoder(z_prefix)

    def forward(self, e: Tensor) -> Tuple[Tensor, Dict[int, Tensor]]:
        """
        Returns:
            z: (B, latent_dim) full compressed embedding.
            reconstructions: {d': (B, raw_dim)} for each configured prefix dim.
        """
        z = self.encode(e)
        reconstructions = {d_prime: self.decode(z[:, :d_prime]) for d_prime in self.prefix_dims}
        return z, reconstructions

    def mrae_loss(self, e: Tensor, reconstructions: Dict[int, Tensor]) -> Tensor:
        """L_MRAE = sum over target dims d' of ||e - f_dec(z[:, :d'])||^2 (Matryoshka MAE loss)."""
        return sum(F.mse_loss(recon, e) for recon in reconstructions.values())


def quantize_int4(z: Tensor) -> Tensor:
    """INT4 quantization of a Tanh-bounded latent (Stage 2): z_quant =
    round(z * 8).clamp(-8, 7) / 8 — 16 evenly-spaced levels covering [-1, 1],
    a 4x storage reduction from FP32.
    """
    return torch.round(z * 8).clamp(-8, 7) / 8


# ---------------------------------------------------------------------------
# Stage 3 — Structuring: user-keyed temporal sequences (Eq. 2)
# ---------------------------------------------------------------------------

def build_user_sequence(history: List[Tuple[float, Tensor]], t_cur: float, max_len: int) -> Tensor:
    """Builds one key's LoopFM sequence S_k (Eq. 2):

        S_k = [z_{k,t_L}, ..., z_{k,t_2}, z_{k,t_1}],  t_L < ... < t_2 < t_1 < t_cur

    Only entries strictly before `t_cur` are used (the example currently
    being served/trained on is excluded — this is what lets the VM serve
    without any real-time FM inference), and the sequence is truncated to the
    `max_len` most recent qualifying entries, kept in chronological
    (oldest-first) order.

    Args:
        history: (timestamp, embedding) pairs for one key; embedding is (d,).
        t_cur: the current example's timestamp.
        max_len: L, maximum sequence length.
    Returns:
        (L', d) tensor, L' <= max_len; (0, d) if no qualifying history —
        d is inferred from `history` if non-empty, else 0.
    """
    past = sorted((t, z) for t, z in history if t < t_cur)
    past = past[-max_len:] if max_len > 0 else []
    if not past:
        d = history[0][1].shape[-1] if history else 0
        return torch.empty(0, d)
    return torch.stack([z for _, z in past], dim=0)


def group_by_key(records: List[dict], key_field: str = "key") -> Dict:
    """Groups flat (key, timestamp, embedding) records into per-key history
    lists, ready to pass to `build_user_sequence`.

    Args:
        records: dicts with at least {key_field, "timestamp", "embedding"}.
    Returns:
        {key: [(timestamp, embedding), ...]}
    """
    grouped: Dict = {}
    for r in records:
        grouped.setdefault(r[key_field], []).append((r["timestamp"], r["embedding"]))
    return grouped


# ---------------------------------------------------------------------------
# Stage 4 — VM-side sequence encoders (Sec 3, "VM-side integration"; Table 3)
# ---------------------------------------------------------------------------

class MeanPoolSequenceEncoder(nn.Module):
    """Mean-pools a LoopFM sequence over valid (non-padded) positions."""

    def forward(self, seq: Tensor, mask: Tensor) -> Tensor:
        """seq: (B, L, d), mask: (B, L) bool (True = valid). Returns (B, d)."""
        mask_f = mask.unsqueeze(-1).float()
        summed = (seq * mask_f).sum(dim=1)
        count = mask_f.sum(dim=1).clamp(min=1.0)
        return summed / count


class SumPoolSequenceEncoder(nn.Module):
    """Sum-pools a LoopFM sequence over valid (non-padded) positions.

    The paper finds this surprisingly strong — even a sum-pooled historical
    embedding as a single feature captures much of LoopFM's gain, at far
    lower storage/modeling cost than a full sequence architecture.
    """

    def forward(self, seq: Tensor, mask: Tensor) -> Tensor:
        mask_f = mask.unsqueeze(-1).float()
        return (seq * mask_f).sum(dim=1)


class AttentionPoolSequenceEncoder(nn.Module):
    """Target-aware attention pooling (a DIN/DMIN-style activation unit): the
    current query (e.g. candidate-item features) attends over the LoopFM
    sequence, weighting historical entries by relevance to the query rather
    than treating them uniformly.

    Args:
        d_model: dimension of both the sequence entries and the query.
        hidden_dim: hidden width of the attention-score MLP.
    """

    def __init__(self, d_model: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(4 * d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, seq: Tensor, mask: Tensor, query: Tensor) -> Tensor:
        """
        Args:
            seq: (B, L, d), mask: (B, L) bool, query: (B, d).
        Returns:
            (B, d) attention-weighted pooled representation.
        """
        B, L, d = seq.shape
        q = query.unsqueeze(1).expand(B, L, d)
        features = torch.cat([q, seq, q - seq, q * seq], dim=-1)  # classic DIN activation unit
        scores = self.score_mlp(features).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(mask.any(dim=-1, keepdim=True), weights, torch.zeros_like(weights))
        return (seq * weights.unsqueeze(-1)).sum(dim=1)


def loopfm_training_loss(
    vm_logits: Tensor,
    fm_logits: Tensor,
    labels: Tensor,
    kd_weight: float,
) -> Tensor:
    """The VM's combined training objective (Sec 3, "VM-side integration"):

        L = L_task(y_hat_V, y) + lambda * L_KD(y_hat_V, y_hat_F)

    Task loss is BCE against the true label; the KD term is BCE of the VM's
    prediction against the (detached) FM's soft-label prediction — LoopFM's
    structured-embedding channel and this scalar KD channel are shown to be
    largely complementary, so both are used together in production.
    """
    task_loss = F.binary_cross_entropy_with_logits(vm_logits, labels)
    kd_loss = F.binary_cross_entropy_with_logits(vm_logits, torch.sigmoid(fm_logits.detach()))
    return task_loss + kd_weight * kd_loss


# ---------------------------------------------------------------------------
# Evaluation metrics (Sec 5.1): Normalized Entropy & Transfer Ratio
# ---------------------------------------------------------------------------

def normalized_entropy(labels: Tensor, probs: Tensor, eps: float = 1e-7) -> float:
    """NE (Normalized Entropy), He et al. 2014: log-loss normalized by the
    entropy of the empirical label rate, so NE is (roughly) comparable across
    datasets/time windows with different base rates. Lower is better.
    """
    probs = probs.clamp(min=eps, max=1 - eps)
    log_loss = F.binary_cross_entropy(probs, labels.float()).item()
    p_bar = labels.float().mean().clamp(min=eps, max=1 - eps).item()
    background_entropy = -(p_bar * math.log(p_bar) + (1 - p_bar) * math.log(1 - p_bar))
    return log_loss / background_entropy


def transfer_ratio(ne_vm_before: float, ne_vm_after: float, ne_fm_before: float, ne_fm_after: float) -> float:
    """TR = delta_NE_VM / delta_NE_FM: the fraction of an FM's own improvement
    (from ne_fm_before to ne_fm_after) that shows up as VM improvement (from
    ne_vm_before to ne_vm_after) once transferred. NE is lower-is-better, so
    both deltas are computed as (before - after); a positive TR means the VM
    improved in the same direction as the FM.
    """
    delta_vm = ne_vm_before - ne_vm_after
    delta_fm = ne_fm_before - ne_fm_after
    if delta_fm == 0:
        raise ValueError("transfer_ratio is undefined when the FM shows no improvement (delta_NE_FM = 0).")
    return delta_vm / delta_fm


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)

    print("--- Stage 1: Extraction ---")
    layer_acts = [torch.randn(4, 32), torch.randn(4, 16), torch.randn(4, 8)]
    e = extract_embedding(layer_acts)
    print(f"raw embedding e: {e.shape}")

    print("\n--- Stage 2: Matryoshka Autoencoder + INT4 quantization ---")
    ae = MatryoshkaAutoencoder(raw_dim=56, latent_dim=32, prefix_dims=[8, 16, 32])
    z, recons = ae(stop_gradient(e))
    loss_ae = ae.mrae_loss(e.detach(), recons)
    loss_ae.backward()
    print(f"z: {z.shape}, prefixes reconstructed: {list(recons.keys())}, L_MRAE: {loss_ae.item():.4f}")
    z_q = quantize_int4(z.detach())
    print(f"quantized z range: [{z_q.min().item():.3f}, {z_q.max().item():.3f}], unique levels: {z_q.unique().numel()}")

    print("\n--- Stage 3: Structuring ---")
    history = [(float(t), torch.randn(32)) for t in range(20)]
    seq = build_user_sequence(history, t_cur=20.0, max_len=10)
    print(f"user sequence S_u: {seq.shape}  (most recent 10 of 20 historical entries)")

    records = [
        {"key": "u1", "timestamp": 1.0, "embedding": torch.randn(32)},
        {"key": "u2", "timestamp": 2.0, "embedding": torch.randn(32)},
        {"key": "u1", "timestamp": 3.0, "embedding": torch.randn(32)},
    ]
    grouped = group_by_key(records)
    print(f"grouped keys: {list(grouped.keys())}, u1 history length: {len(grouped['u1'])}")

    print("\n--- Stage 4: VM-side sequence encoders + combined loss ---")
    B, L, d = 4, 10, 32
    seq_batch = torch.randn(B, L, d)
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[0, 5:] = False  # simulate a shorter real history for one user

    mean_enc = MeanPoolSequenceEncoder()
    sum_enc = SumPoolSequenceEncoder()
    attn_enc = AttentionPoolSequenceEncoder(d_model=d)
    query = torch.randn(B, d)

    pooled_mean = mean_enc(seq_batch, mask)
    pooled_sum = sum_enc(seq_batch, mask)
    pooled_attn = attn_enc(seq_batch, mask, query)
    print(f"mean-pool: {pooled_mean.shape}, sum-pool: {pooled_sum.shape}, attn-pool: {pooled_attn.shape}")

    vm_head = nn.Linear(d, 1)
    vm_logits = vm_head(pooled_attn).squeeze(-1)
    fm_logits = torch.randn(B)
    labels = torch.randint(0, 2, (B,)).float()
    loss = loopfm_training_loss(vm_logits, fm_logits, labels, kd_weight=5.0)
    loss.backward()
    print(f"loopfm_training_loss: {loss.item():.4f}, backward OK")

    print("\n--- Evaluation metrics ---")
    labels_eval = torch.randint(0, 2, (1000,))
    probs_baseline = torch.rand(1000) * 0.5 + labels_eval.float() * 0.1
    probs_loopfm = torch.rand(1000) * 0.3 + labels_eval.float() * 0.3
    ne_before = normalized_entropy(labels_eval, probs_baseline)
    ne_after = normalized_entropy(labels_eval, probs_loopfm)
    print(f"NE before: {ne_before:.4f}, NE after: {ne_after:.4f}")
    tr = transfer_ratio(ne_vm_before=ne_before, ne_vm_after=ne_after, ne_fm_before=0.50, ne_fm_after=0.45)
    print(f"transfer_ratio: {tr:.4f}")


if __name__ == "__main__":
    _smoke_test()
