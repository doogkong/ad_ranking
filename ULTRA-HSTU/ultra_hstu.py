"""ULTRA-HSTU: Bending the Scaling Law Curve in Large-Scale Recommendation
Systems.

Reference PyTorch implementation.
Paper: "Bending the Scaling Law Curve in Large-Scale Recommendation Systems"
https://arxiv.org/abs/2602.16986 (Meta Recommendation Systems, Feb 2026).

Core idea: HSTU (arxiv.org/abs/2402.17152) showed that self-attention over a
unified user-history sequence scales well with compute, but its O(L^2) cost
in sequence length L becomes unaffordable once histories grow to 10k-100k
events. Prior work sidesteps this by *replacing* self-attention with cross-
attention against only the ranking candidates -- cheap, but the paper shows
(their Table 1/3/5) this loses real model quality at scale. ULTRA-HSTU
instead keeps self-attention and bends its cost curve via three
complementary, stackable model changes:

  1. Input sequence optimization (Sec 4.1): merge each item/action pair into
     one token (x_i = item_i + action_i) instead of interleaving them as two
     tokens, halving the effective sequence length before attention ever
     runs -- see `build_uih_sequence`. Action embeddings are additionally
     built from multiple heterogeneous signals rather than one action-type
     id -- see `HeterogeneousActionEncoder`.
  2. Semi-Local Attention, SLA (Sec 4.2.1, Eq. 6-7): a sparse causal
     attention mask combining a short local sliding window (K1, recency)
     with a small set of global anchor positions (K2, long-range summary
     signal), giving O((K1+K2)*L) attention cost instead of O(L^2) -- see
     `semi_local_attention_mask`. The paper finds *both* windows are
     necessary: local-only regresses 0.35% C-NE, global-only regresses
     0.03% C-NE, relative to using both (Sec 4.2.1).
  3. Attention Truncation, a dynamic topological design (Sec 4.3, Fig. 2d):
     rather than paying O(D*L) to stack D layers over the full sequence,
     run only N1 layers at full length L, then select the most recent L'
     positions and run N2 further layers on just that short segment. The
     paper finds truncating to the *latest* UIH segment works best among
     the segment-selection strategies they tried -- see `UltraHSTU`.

The layer itself keeps HSTU's pointwise-aggregated-attention block, but in
the pre-norm formulation given in the paper's Background section (Eq. 1-5):

    X = Norm(Z)                                                    [Eq. 1]
    U, Q, K, V = phi1(f1(X))                                       [Eq. 2]
    A = (phi2(QK^T) elementwise* M) V                               [Eq. 3]
    Y = f2(Norm(A) elementwise* U)                                 [Eq. 4]
    Z_out = Y + Z                                                  [Eq. 5]

where phi1 = phi2 = SiLU and M is the (semi-local or full-causal) attention
mask -- applied as an elementwise gate on the post-activation weights,
which is *how this paper defines M* (unlike vanilla HSTU/rab, there is no
additive relative-position bias term in this formulation).

Out of scope (system/hardware-level, not modeling contributions): the
FP8/INT4 mixed-precision GEMM kernels and custom FlashAttention-3-style SLA
CUDA/ROCm kernels (Sec 4.2.2, App. D), and the distributed Load-Balanced
Stochastic Length training scheduler (Sec 4.1, Algorithm 1 in the appendix),
which balances per-rank compute load across a multi-GPU world rather than
changing the model itself.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Input sequence optimization (Sec 4.1)
# ---------------------------------------------------------------------------

def build_uih_sequence(
    item_emb: Tensor,
    action_emb: Tensor,
    is_candidate: Optional[Tensor] = None,
) -> Tensor:
    """x_{i,j} = I_{i,j} + a_{i,j} (Sec 4.1): merges each item/action pair
    into a single token by addition instead of interleaving them as two
    separate sequence positions, halving the sequence length HSTU would see
    for the same raw UIH without hurting model quality (paper's ablation:
    -32.5% train FLOP, -63.5% inference FLOP, at a UIH length of 3072).

    Args:
        item_emb: (B, n, D) per-position item embeddings.
        action_emb: (B, n, D) per-position action embeddings.
        is_candidate: optional (B, n) bool, True at positions that are
            candidates to be ranked. Their action embedding is masked to
            zero to prevent leaking the label being predicted, per the
            paper's note that "we mask the action embeddings in all
            candidate positions ... if j is a candidate to be ranked."
    Returns:
        (B, n, D) merged sequence.
    """
    if is_candidate is not None:
        action_emb = action_emb.masked_fill(is_candidate.unsqueeze(-1), 0.0)
    return item_emb + action_emb


class HeterogeneousActionEncoder(nn.Module):
    """Builds one action embedding a_{i,j} from multiple heterogeneous
    action signals rather than a single action-type id (Sec 4.1: "we
    enhance this simplified design with heterogeneous action encodings,
    from both implicit and explicit signals and side information through
    user contextual features"). Signals are summed into one d_model-dim
    vector, mirroring how the merged item+action token itself is built by
    addition rather than concatenation.

    Args:
        d_model: embedding width D.
        num_action_types: explicit engagement types (like, comment, share, ...).
        num_intensity_buckets: bucketed implicit engagement intensity
            (e.g. a quantized watch-time-completion ratio).
        num_context_buckets: bucketed side/context information (e.g. time
            of day, surface/placement id).
    """

    def __init__(
        self,
        d_model: int,
        num_action_types: int,
        num_intensity_buckets: int,
        num_context_buckets: int,
    ) -> None:
        super().__init__()
        self.action_type_emb = nn.Embedding(num_action_types, d_model)
        self.intensity_emb = nn.Embedding(num_intensity_buckets, d_model)
        self.context_emb = nn.Embedding(num_context_buckets, d_model)

    def forward(self, action_type: Tensor, intensity_bucket: Tensor, context_bucket: Tensor) -> Tensor:
        """All args are integer id tensors of matching shape (B, n)."""
        return (
            self.action_type_emb(action_type)
            + self.intensity_emb(intensity_bucket)
            + self.context_emb(context_bucket)
        )


# ---------------------------------------------------------------------------
# Semi-Local Attention mask (Sec 4.2.1, Eq. 6-7)
# ---------------------------------------------------------------------------

def semi_local_attention_mask(
    seq_len: int,
    local_window: int,
    global_window: int,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Eq. 7's linear-complexity sparse attention mask: a causal local
    sliding window of size K1 (recency) unioned with a set of K2 causal
    global anchor positions (long-range summary signal), giving
    O((K1+K2)*L) nonzero entries instead of full causal attention's
    O(L^2/2). This reference implementation anchors the global window at
    the *earliest* K2 positions in the sequence (a fixed, always-visible
    "memory" prefix, in the spirit of Longformer/BigBird global tokens),
    which realizes the paper's stated O((K1+K2)*L) complexity; the paper's
    Eq. 7 index convention is not fully specified standalone, so the exact
    anchor placement should be treated as this repo's interpretation of the
    mechanism rather than a byte-exact reproduction.

    Args:
        seq_len: L.
        local_window: K1, number of most-recent keys (inclusive of self)
            each query may attend to.
        global_window: K2, number of earliest-position keys every causally
            valid query may attend to.
        device: target device for the returned mask.
    Returns:
        (L, L) bool tensor, True where query i may attend to key j.
    """
    q = torch.arange(seq_len, device=device).view(-1, 1)
    k = torch.arange(seq_len, device=device).view(1, -1)
    causal = k <= q
    local = (q - k) < local_window
    is_global_anchor = k < global_window
    return causal & (local | is_global_anchor)


def attention_complexity(seq_len: int, local_window: int, global_window: int) -> Tuple[int, int, float]:
    """Returns (sla_nnz, full_causal_nnz, sparsity_ratio) for a given
    configuration -- a lightweight stand-in for the paper's FLOP-scaling
    plots (Figure 1/5/8): SLA's nonzero count grows linearly in `seq_len`
    while full causal attention's grows quadratically.
    """
    sla_nnz = int(semi_local_attention_mask(seq_len, local_window, global_window).sum().item())
    full_nnz = seq_len * (seq_len + 1) // 2
    return sla_nnz, full_nnz, sla_nnz / full_nnz


# ---------------------------------------------------------------------------
# ULTRA-HSTU Layer (Eq. 1-5)
# ---------------------------------------------------------------------------

class UltraHSTULayer(nn.Module):
    """One ULTRA-HSTU layer per Eq. 1-5: pre-norm, pointwise-projected
    Q/K/V/U, mask-gated pointwise attention, gated pointwise output
    transform, residual connection.

    Args:
        d_model: model width D.
        num_heads: number of attention heads H.
        dqk: per-head query/key dimension.
        dv: per-head value dimension (also the per-head width of gate U).
    """

    def __init__(self, d_model: int, num_heads: int, dqk: int, dv: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dqk = dqk
        self.dv = dv
        self.d_inner = num_heads * dv
        self.norm_in = nn.LayerNorm(d_model)                            # Eq.1
        self.f1 = nn.Linear(d_model, num_heads * (2 * dqk + 2 * dv))    # Eq.2 input proj
        self.norm_attn = nn.LayerNorm(self.d_inner)                     # Eq.4, Norm(A)
        self.f2 = nn.Linear(self.d_inner, d_model)                      # Eq.4 output proj
        self.scale = dqk ** -0.5

    def forward(self, z: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            z: (B, n, D) layer input (the residual stream).
            mask: (n, n) bool, True where a query may attend to a key
                (broadcasts over batch and heads).
        Returns:
            (B, n, D) — Eq. 5 residual output.
        """
        B, n, D = z.shape
        H, dqk, dv = self.num_heads, self.dqk, self.dv

        x = self.norm_in(z)                                              # Eq.1
        proj = F.silu(self.f1(x))                                        # Eq.2, phi1
        U, Q, K, V = proj.split([H * dv, H * dqk, H * dqk, H * dv], dim=-1)
        Q = Q.view(B, n, H, dqk).transpose(1, 2)  # (B, H, n, dqk)
        K = K.view(B, n, H, dqk).transpose(1, 2)
        V = V.view(B, n, H, dv).transpose(1, 2)   # (B, H, n, dv)

        logits = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        A = F.silu(logits)                                                # Eq.3, phi2
        A = A * mask.view(1, 1, n, n)                                     # Eq.3, elementwise* M
        AV = torch.matmul(A, V)
        AV = AV.transpose(1, 2).reshape(B, n, self.d_inner)

        gated = self.norm_attn(AV) * U                                   # Eq.4
        Y = self.f2(gated)
        return Y + z                                                      # Eq.5


# ---------------------------------------------------------------------------
# ULTRA-HSTU Encoder: SLA + Attention Truncation (Sec 4.2-4.3)
# ---------------------------------------------------------------------------

class UltraHSTU(nn.Module):
    """Stack of ULTRA-HSTU layers combining Semi-Local Attention with the
    Attention Truncation dynamic topological design (Fig. 2d): N1 layers
    run over the full L-length sequence, then N2 further layers run only
    over the latest L' positions -- avoiding the O(D*L) cost of stacking
    all D=N1+N2 layers over the full sequence.

    Set `num_layers_truncated=0` (the default) to get a plain SLA-only
    encoder with no truncation stage.

    Args:
        d_model, num_heads, dqk, dv: as `UltraHSTULayer`.
        num_layers_full: N1, layers run over the full sequence.
        num_layers_truncated: N2, additional layers run over the truncated
            latest-L' segment only.
        local_window, global_window: SLA's K1, K2 (shared by both stages).
        truncate_len: L', the length of the latest segment N2 layers see.
            Required if `num_layers_truncated > 0`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dqk: int,
        dv: int,
        num_layers_full: int,
        num_layers_truncated: int = 0,
        local_window: int = 64,
        global_window: int = 16,
        truncate_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_layers_truncated > 0 and truncate_len is None:
            raise ValueError("truncate_len is required when num_layers_truncated > 0")
        self.local_window = local_window
        self.global_window = global_window
        self.truncate_len = truncate_len
        self.layers_full = nn.ModuleList(
            [UltraHSTULayer(d_model, num_heads, dqk, dv) for _ in range(num_layers_full)]
        )
        self.layers_truncated = nn.ModuleList(
            [UltraHSTULayer(d_model, num_heads, dqk, dv) for _ in range(num_layers_truncated)]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            x: (B, L, D) merged UIH sequence (e.g. from `build_uih_sequence`).
        Returns:
            h_full: (B, L, D) — output of the N1 full-sequence layers.
            h_trunc: (B, L', D) or None — output of the N2 truncated-segment
                layers over `h_full`'s latest L' positions, if any.
        """
        B, L, D = x.shape
        device = x.device
        mask_full = semi_local_attention_mask(L, self.local_window, self.global_window, device)

        h = x
        for layer in self.layers_full:
            h = layer(h, mask_full)

        if len(self.layers_truncated) == 0:
            return h, None

        Lp = min(self.truncate_len, L)
        h_trunc = h[:, -Lp:, :]
        mask_trunc = semi_local_attention_mask(Lp, self.local_window, self.global_window, device)
        for layer in self.layers_truncated:
            h_trunc = layer(h_trunc, mask_trunc)
        return h, h_trunc

    def final_representation(self, h_full: Tensor, h_trunc: Optional[Tensor]) -> Tensor:
        """The representation used for ranking read-off (last sequence
        position): the doubly-refined truncated-segment output if
        Attention Truncation is enabled, otherwise the full-sequence one.
        """
        return h_trunc[:, -1, :] if h_trunc is not None else h_full[:, -1, :]


# ---------------------------------------------------------------------------
# Ranking head (multi-task consumption/engagement predictions)
# ---------------------------------------------------------------------------

class RankingHead(nn.Module):
    """Small MLP mapping an ULTRA-HSTU output representation to multi-task
    logits (e.g. the paper's C-NE / E-NE consumption / engagement tasks)."""

    def __init__(self, d_model: int, hidden_dims: Optional[List[int]], num_tasks: int) -> None:
        super().__init__()
        hidden = hidden_dims or [d_model]
        dims = [d_model] + hidden + [num_tasks]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, h: Tensor) -> Tensor:
        return self.mlp(h)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B, L, D = 2, 256, 32
    H, dqk, dv = 4, 8, 8
    K1, K2 = 32, 8

    print("--- Input sequence optimization ---")
    item_emb = torch.randn(B, L, D)
    action_emb = torch.randn(B, L, D)
    is_candidate = torch.zeros(B, L, dtype=torch.bool)
    is_candidate[:, -4:] = True  # last 4 positions are ranking candidates
    x = build_uih_sequence(item_emb, action_emb, is_candidate)
    print(f"merged UIH sequence: {x.shape}  (vs. {2 * L} tokens if interleaved)")

    action_encoder = HeterogeneousActionEncoder(
        d_model=D, num_action_types=8, num_intensity_buckets=10, num_context_buckets=24
    )
    het_action = action_encoder(
        action_type=torch.randint(0, 8, (B, L)),
        intensity_bucket=torch.randint(0, 10, (B, L)),
        context_bucket=torch.randint(0, 24, (B, L)),
    )
    print(f"heterogeneous action embedding: {het_action.shape}")

    print("\n--- Semi-Local Attention complexity ---")
    for seq_len in (256, 512, 1024, 2048):
        sla_nnz, full_nnz, ratio = attention_complexity(seq_len, K1, K2)
        print(f"L={seq_len:5d}  SLA nnz={sla_nnz:7d}  full nnz={full_nnz:8d}  sparsity={ratio:.4f}")

    print("\n--- ULTRA-HSTU forward (SLA only) ---")
    model = UltraHSTU(
        d_model=D, num_heads=H, dqk=dqk, dv=dv,
        num_layers_full=4, local_window=K1, global_window=K2,
    )
    h_full, h_trunc = model(x)
    print(f"h_full: {h_full.shape}, h_trunc: {h_trunc}")

    print("\n--- ULTRA-HSTU forward (SLA + Attention Truncation) ---")
    model_at = UltraHSTU(
        d_model=D, num_heads=H, dqk=dqk, dv=dv,
        num_layers_full=3, num_layers_truncated=2,
        local_window=K1, global_window=K2, truncate_len=64,
    )
    h_full, h_trunc = model_at(x)
    print(f"h_full: {h_full.shape}, h_trunc: {h_trunc.shape}")

    head = RankingHead(D, hidden_dims=[D], num_tasks=2)
    logits = head(model_at.final_representation(h_full, h_trunc))
    loss = F.binary_cross_entropy_with_logits(logits, torch.zeros(B, 2))
    loss.backward()
    print(f"logits: {logits.shape}  loss: {loss.item():.4f}  backward: OK")

    total = sum(p.numel() for p in model_at.parameters())
    print(f"params: {total:,}")


if __name__ == "__main__":
    _smoke_test()
