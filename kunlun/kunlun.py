"""Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems
through Unified Architecture Design.

Reference PyTorch implementation.
Paper: https://arxiv.org/abs/2602.10016  (Meta Platforms, Feb 2026)

Architecture overview (two blocks per layer):

    Non-seq features X^(0) → EmbeddingLayer → X^(1) ∈ ℝ^(B, n_ns, d)
    Seq features     S^(0) → MLP + ROTE     → S^(1) ∈ ℝ^(B, T, d)

    for l = 0 .. L-1:
        ┌─ KunlunInteractionBlock ──────────────────────────────────────────┐
        │  X_sum = WeightGeneration(X)                [MLP compression]    │
        │  if even: H_summary = HSP(S)                [fresh pooling]      │
        │  if odd:  H_summary = cached                [CompSkip: reuse]    │
        │  X^(l+1) = GlobalInteraction(X_sum ∥ H_summary)  [Wukong MoE]  │
        └────────────────────────────────────────────────────────────────  ┘
        ┌─ KunlunTransformerBlock ──────────────────────────────────────────┐
        │  if even: S^(l+1) = GDPA(S, X_sum)         [personalized PFFN]  │
        │  if odd:  S^(l+1) = LN(SWA(S) + S)         [local refinement]   │
        └────────────────────────────────────────────────────────────────  ┘

    ŷ = σ(MLP(flatten(X^(L))))

CompSkip (§4.4.1): every-other-layer alternation between:
  - Global understanding: PFFN + fresh HSP (even layers)
  - Local refinement: SWA + reuse HSP cache (odd layers)
  Reduces FLOPs by 43.1%, improves QPS by 35%.

Key innovations vs. InterFormer:
  GDPA    — multi-head attention-style PFFN fusable into one kernel (6× MFU on PFFN)
  HSP     — hierarchical seed pooling via SumKronLinear (better init than PMA)
  SWA     — sliding window attention O(Tw) instead of O(T²), 31% QPS improvement
  CompSkip        — every-other-layer alternation, 43.1% FLOPs reduction
  Event Personalization — per-event-type d_model / n_heads / n_tokens / L / w
  MoE Wukong      — expert parallelism, 4% additional QPS

Scaling law: NE(C) = NE_0 − η·log(C/C_0), η ≈ 2× InterFormer.
Deployed at Meta Ads: +1.2% NE.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Rotary Temporal Embeddings (ROTE)   §4.2 + Appendix B
# ---------------------------------------------------------------------------

class ROTE(nn.Module):
    """Rotary Temporal Embeddings: extend RoPE with log-scaled temporal gaps.

    Standard RoPE treats token *position* as the signal. In recommendation
    sequences, temporal *gap* between events matters more — a click yesterday
    differs fundamentally from one a month ago even at adjacent positions.

    ROTE encodes τ_t = log(1 + Δt / t_scale) and combines positional and
    temporal frequencies θ_i, φ_i via the standard rotation matrix R_{τ,φ}.

    Args:
        d_model:  embedding dimension (must be even).
        t_scale:  temporal normalization constant in seconds (default: 1 day).
        base:     RoPE frequency base (default: 10000).
    """

    def __init__(self, d_model: int, t_scale: float = 86400.0, base: float = 10000.0) -> None:
        super().__init__()
        self.t_scale = t_scale
        half_d = d_model // 2
        freqs = 1.0 / (base ** (torch.arange(0, half_d, dtype=torch.float) / half_d))
        self.register_buffer("freqs", freqs)  # (d/2,)

    def forward(self, x: Tensor, timestamps: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:          (B, T, d)
            timestamps: (B, T) in seconds; None falls back to position indices.
        Returns:
            (B, T, d) with rotary temporal embeddings applied.
        """
        B, T, d = x.shape
        if timestamps is None:
            tau = torch.arange(T, device=x.device, dtype=torch.float).unsqueeze(0).expand(B, -1)
        else:
            gaps = torch.zeros_like(timestamps, dtype=torch.float)
            gaps[:, 1:] = (timestamps[:, 1:] - timestamps[:, :-1]).float().clamp(min=0)
            tau = torch.log1p(gaps / self.t_scale)                  # (B, T)

        angles = tau.unsqueeze(-1) * self.freqs                      # (B, T, d/2)
        cos_a, sin_a = torch.cos(angles), torch.sin(angles)          # (B, T, d/2)
        x1, x2 = x[..., ::2], x[..., 1::2]                          # (B, T, d/2)
        x_rot = torch.stack([x1 * cos_a - x2 * sin_a,
                              x1 * sin_a + x2 * cos_a], dim=-1)      # (B, T, d/2, 2)
        return x_rot.reshape(B, T, d)


# ---------------------------------------------------------------------------
# GDPA: Generalized Dot-Product Attention   §4.3.1
# ---------------------------------------------------------------------------

class GDPA(nn.Module):
    """Generalized Dot-Product Attention: multi-head cross-attention PFFN.

    Reformulates PFFN as a single attention operator to enable FlashAttention-
    style kernel fusion (up to 6× MFU improvement on the PFFN block):

        GDPA_h(Q, K, V) = Activation_h(Q K^T / τ) V

    where:
        Q = S^(l)             sequence tokens as queries    (B, T, d_h)
        K = w1_h(X_sum^(l))   head-h K projection of X_sum  (B, n_sum, d_h)
        V = w2_h(X_sum^(l))   head-h V projection of X_sum  (B, n_sum, d_h)
        τ = maxlen(seq)       temperature (better than 1/√d empirically)

    Prior PFFN: two sequential MLP steps — non-fusable and memory-bound.
    GDPA: fused cross-attention — compute-bound and kernelizable.

    Includes a residual connection for stable stacking across multiple layers.

    Args:
        n_sum:       non-seq summary token count n_sum.
        d_model:     embedding dimension d.
        n_heads:     number of attention heads H.
        max_seq_len: temperature τ (paper uses maxlen(seq)).
    """

    def __init__(self, n_sum: int, d_model: int, n_heads: int = 8,
                 max_seq_len: int = 1000) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.tau      = max_seq_len
        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj  = nn.Linear(d_model, 2 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, S: Tensor, X_sum: Tensor) -> Tensor:
        """
        Args:
            S:     (B, T, d)      sequence tokens (queries).
            X_sum: (B, n_sum, d)  non-seq summary (K, V source).
        Returns:
            (B, T, d)
        """
        B, T, d = S.shape
        n_sum = X_sum.size(1)
        H, d_h = self.n_heads, self.d_head

        Q  = self.q_proj(S).reshape(B, T, H, d_h).transpose(1, 2)       # (B, H, T, d_h)
        KV = self.kv_proj(X_sum)                                          # (B, n_sum, 2d)
        K, V = KV.chunk(2, dim=-1)
        K = K.reshape(B, n_sum, H, d_h).transpose(1, 2)                  # (B, H, n_sum, d_h)
        V = V.reshape(B, n_sum, H, d_h).transpose(1, 2)                  # (B, H, n_sum, d_h)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.tau           # (B, H, T, n_sum)
        attn = F.softmax(attn, dim=-1)
        out  = torch.matmul(attn, V)                                      # (B, H, T, d_h)
        out  = out.transpose(1, 2).reshape(B, T, d)                      # (B, T, d)
        return self.out_proj(out) + S                                     # residual


# ---------------------------------------------------------------------------
# Sliding Window Attention   §4.3.3
# ---------------------------------------------------------------------------

class SlidingWindowAttention(nn.Module):
    """Sliding window self-attention for linear-complexity sequence modeling.

    Full self-attention is O(T²). For T > 1000 this is prohibitive.
    SWA restricts position t to attend within [t-w, t+w], reducing to O(Tw).
    Consistent with temporal locality bias in recommendation sequences.

    Note: this reference implementation builds an explicit T×T attention mask,
    which retains O(T²) memory. A production implementation would use a
    custom Triton/CUDA kernel for true O(Tw) memory.

    Args:
        d_model: embedding dimension.
        n_heads: number of attention heads.
        window:  half-window size w (attends to 2w+1 positions).
    """

    def __init__(self, d_model: int, n_heads: int = 8, window: int = 100) -> None:
        super().__init__()
        self.window = window
        self.mha    = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, S: Tensor) -> Tensor:
        """
        Args:
            S: (B, T, d)
        Returns:
            (B, T, d)
        """
        B, T, d = S.shape
        if T <= 2 * self.window + 1:
            out, _ = self.mha(S, S, S)
            return out

        # Mask: -inf for positions outside the window
        mask = torch.full((T, T), float('-inf'), device=S.device)
        for t in range(T):
            lo, hi = max(0, t - self.window), min(T, t + self.window + 1)
            mask[t, lo:hi] = 0.0
        out, _ = self.mha(S, S, S, attn_mask=mask)
        return out


# ---------------------------------------------------------------------------
# SumKronLinear   §4.3.2
# ---------------------------------------------------------------------------

class SumKronLinear(nn.Module):
    """Kronecker product-based sequence compression (HSP Stage 3).

    Compresses S seed embeddings to T output tokens:

        Y_b = Σ_{i=1}^{k} Z_i^T X_b W_i

    where X_b ∈ ℝ^(S×D), Z_i ∈ ℝ^(S×T), W_i ∈ ℝ^(D×D), Y_b ∈ ℝ^(T×D).

    Advantages over full linear (O(S·D·T·D) params):
      - 14× parameter reduction: O(k·(S·T + D²)) for typical S=256, T=32, D=384, k=8.
      - Cross-dimensional expressiveness: captures joint S×D correlations
        (unlike separable rank-1 factorization Y = P_seq · X · P_emb).
      - Scaling: relative parameter savings grow as D increases.

    Args:
        S:  input seed count.
        T:  output token count (T < S).
        D:  embedding dimension.
        k:  Kronecker rank (higher k → more expressive, more parameters).
    """

    def __init__(self, S: int, T: int, D: int, k: int = 8) -> None:
        super().__init__()
        self.k = k
        self.Z = nn.Parameter(torch.empty(k, S, T))
        self.W = nn.Parameter(torch.empty(k, D, D))
        nn.init.trunc_normal_(self.Z, std=0.02)
        nn.init.trunc_normal_(self.W, std=0.02)

    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: (B, S, D)
        Returns:
            Y: (B, T, D)
        """
        ZX = torch.einsum('kst,bsd->bktd', self.Z, X)   # (B, k, T, D)
        return torch.einsum('bktd,kde->bte', ZX, self.W) # (B, T, D)


# ---------------------------------------------------------------------------
# HSP: Hierarchical Seed Pooling   §4.3.2
# ---------------------------------------------------------------------------

class HSP(nn.Module):
    """Hierarchical Seed Pooling: 3-stage sequence summarization.

    Improves upon PMA (Pooling by Multi-Head Attention) through hierarchical
    compression. The overcomplete seeds provide better initialization and more
    stable training than direct random queries used in PMA.

    Stage 1 — Seed Embedding Initialization:
        E_seed ∈ ℝ^(n_seeds × d), n_seeds > n_tokens (overcomplete, shared across batch).

    Stage 2 — Seed-level Attention:
        H_seed = MHA(Norm(E_seed), S, S) ∈ ℝ^(B × n_seeds × d)
        Seeds attend to the full input sequence, gaining context-aware representations.

    Stage 3 — Parameter-Efficient Seed Compression (SumKronLinear):
        H_summary = SumKronLinear(H_seed) ∈ ℝ^(B × n_tokens × d)
        Compresses n_seeds → n_tokens with 14× fewer params vs. full linear.

    Args:
        d_model:   embedding dimension.
        n_seeds:   overcomplete seed count (must be > n_tokens).
        n_tokens:  final summary token count passed to downstream interaction.
        n_heads:   MHA heads for Stage 2.
        kron_rank: Kronecker rank k for SumKronLinear.
    """

    def __init__(self, d_model: int, n_seeds: int, n_tokens: int,
                 n_heads: int = 8, kron_rank: int = 8) -> None:
        super().__init__()
        assert n_seeds > n_tokens, f"n_seeds ({n_seeds}) must be > n_tokens ({n_tokens})"
        self.E_seed  = nn.Parameter(torch.empty(1, n_seeds, d_model))
        self.norm    = nn.LayerNorm(d_model)
        self.mha     = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.compress = SumKronLinear(n_seeds, n_tokens, d_model, k=kron_rank)
        nn.init.trunc_normal_(self.E_seed, std=0.02)

    def forward(self, S: Tensor) -> Tensor:
        """
        Args:
            S: (B, T_seq, d)  input sequence.
        Returns:
            H_summary: (B, n_tokens, d)  compact representation.
        """
        B = S.size(0)
        E     = self.E_seed.expand(B, -1, -1)           # (B, n_seeds, d)
        H, _  = self.mha(self.norm(E), S, S)            # (B, n_seeds, d)  Stage 2
        return self.compress(H)                          # (B, n_tokens, d) Stage 3


# ---------------------------------------------------------------------------
# Weight Generation   §4.1 (Kunlun Interaction Block)
# ---------------------------------------------------------------------------

class WeightGeneration(nn.Module):
    """Weight Generation: derives non-seq summary X_sum for GDPA and GlobalInteraction.

    Compresses n_ns non-seq tokens to n_sum via token-axis projection, then
    applies a 2-layer MLP. The resulting X_sum is used as the personalized
    K, V source for GDPA in the Transformer Block, and as part of the input
    to the Global Interaction module.

    Args:
        n_ns:    input non-seq token count.
        n_sum:   output summary token count (n_sum ≤ n_ns).
        d_model: embedding dimension.
    """

    def __init__(self, n_ns: int, n_sum: int, d_model: int) -> None:
        super().__init__()
        self.compress = nn.Linear(n_ns, n_sum, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: (B, n_ns, d)
        Returns:
            X_sum: (B, n_sum, d)
        """
        X_sum = self.compress(X.transpose(1, 2)).transpose(1, 2)  # (B, n_sum, d)
        return self.mlp(X_sum)


# ---------------------------------------------------------------------------
# KunlunTransformerBlock   §4.1, §4.3
# ---------------------------------------------------------------------------

class KunlunTransformerBlock(nn.Module):
    """Kunlun Transformer Block: context-aware sequence modeling (§4.1).

    Applies one of two operations per layer via CompSkip:
      - Even layer: GDPA-enhanced PFFN — personalizes sequence tokens using
        non-seq context X_sum as K, V source (global understanding).
      - Odd layer: Multi-Head Self-Attention with Sliding Window — captures
        local dependencies within the sequence (local refinement).

    Args:
        n_sum:       non-seq summary token count (GDPA K, V input size).
        d_model:     embedding dimension.
        n_heads:     attention heads for both GDPA and SWA.
        window:      SWA half-window size w.
        max_seq_len: GDPA temperature τ.
    """

    def __init__(self, n_sum: int, d_model: int, n_heads: int = 8,
                 window: int = 100, max_seq_len: int = 1000) -> None:
        super().__init__()
        self.gdpa = GDPA(n_sum, d_model, n_heads, max_seq_len)
        self.swa  = SlidingWindowAttention(d_model, n_heads, window)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, S: Tensor, X_sum: Tensor, is_even_layer: bool) -> Tensor:
        """
        Args:
            S:            (B, T, d) sequence embeddings.
            X_sum:        (B, n_sum, d) non-seq summary (used only on even layers).
            is_even_layer: True → GDPA; False → SWA.
        Returns:
            (B, T, d)
        """
        if is_even_layer:
            return self.gdpa(S, X_sum)                  # PFFN via GDPA (residual inside)
        S_attn = self.swa(S)
        return self.norm(S_attn + S)                    # local SWA + residual


# ---------------------------------------------------------------------------
# Global Interaction: Mixture of Wukong Experts   §4.4.3
# ---------------------------------------------------------------------------

class WukongExpert(nn.Module):
    """Single Wukong expert: DOT product + deep interaction network.

    Each expert processes a designated feature partition and captures:
      - Linear interactions via DOT product (inner-product attention).
      - Hierarchical interactions via a deep MLP.

    Args:
        n_tokens: number of tokens assigned to this expert.
        d_model:  embedding dimension.
    """

    def __init__(self, n_tokens: int, d_model: int) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.scale    = d_model ** -0.5
        self.deep     = nn.Sequential(
            nn.Linear(n_tokens * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_tokens * d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: (B, n_tokens, d)
        Returns:
            (B, n_tokens, d)
        """
        B = X.size(0)
        scores = torch.bmm(X, X.transpose(1, 2)) * self.scale       # (B, n, n)
        X_dot  = torch.bmm(F.softmax(scores, dim=-1), X)             # (B, n, d)
        X_deep = self.deep(X.reshape(B, -1)).reshape(B, self.n_tokens, -1)
        return self.norm(X_dot + X_deep + X)


class GlobalInteraction(nn.Module):
    """Global Interaction: Mixture of Wukong Experts (§4.4.3).

    Processes the combined representation X_global = Concat(X_sum, H_summary)
    through M Wukong experts in parallel, each handling a feature partition.
    Enables horizontal scaling through expert parallelism (4% QPS improvement).

    Expert outputs are concatenated to produce the new non-seq representation
    X^(l+1), which becomes the input to the next layer.

    Args:
        n_global:  total input token count (n_sum + n_tokens).
        d_model:   embedding dimension.
        n_experts: M, number of Wukong experts.
    """

    def __init__(self, n_global: int, d_model: int, n_experts: int = 2) -> None:
        super().__init__()
        tpe = (n_global + n_experts - 1) // n_experts   # tokens per expert (ceiling)
        self.tpe      = tpe
        self.n_global = n_global
        self.experts  = nn.ModuleList([
            WukongExpert(min(tpe, n_global - i * tpe), d_model)
            for i in range(n_experts)
            if i * tpe < n_global
        ])

    def forward(self, X_global: Tensor) -> Tensor:
        """
        Args:
            X_global: (B, n_global, d)
        Returns:
            (B, n_global, d)
        """
        parts = []
        for i, expert in enumerate(self.experts):
            lo = i * self.tpe
            hi = min(lo + self.tpe, self.n_global)
            parts.append(expert(X_global[:, lo:hi, :]))
        return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# KunlunInteractionBlock   §4.1, §4.4
# ---------------------------------------------------------------------------

class KunlunInteractionBlock(nn.Module):
    """Kunlun Interaction Block: bidirectional cross-modal interaction (§4.1).

    Facilitates information exchange between sequential and non-sequential inputs:

    1. Weight Generation: X → X_sum (compressed non-seq summary for GDPA weights).
    2. HSP: S → H_summary  [fresh on even layers via CompSkip].
       Cache reuse:          [reuse previous H_summary on odd layers].
    3. Global Interaction:   Wukong MoE on Concat(X_sum, H_summary) → X^(l+1).

    CompSkip pattern (§4.4.1):
      Even layers: compute fresh HSP (global sequence understanding).
      Odd layers:  reuse cached HSP (pair with local SWA in Transformer Block).

    Args:
        n_ns:      input non-seq token count.
        n_sum:     WeightGeneration output token count.
        d_model:   embedding dimension.
        n_seeds:   HSP seed count (must be > n_tokens).
        n_tokens:  HSP output token count.
        n_heads:   MHA heads for HSP Stage 2.
        n_experts: Wukong expert count.
        kron_rank: SumKronLinear rank k.
    """

    def __init__(self, n_ns: int, n_sum: int, d_model: int,
                 n_seeds: int, n_tokens: int,
                 n_heads: int = 8, n_experts: int = 2, kron_rank: int = 8) -> None:
        super().__init__()
        self.weight_gen   = WeightGeneration(n_ns, n_sum, d_model)
        self.hsp          = HSP(d_model, n_seeds, n_tokens, n_heads, kron_rank)
        self.global_inter = GlobalInteraction(n_sum + n_tokens, d_model, n_experts)

    def forward(
        self,
        X: Tensor,
        S: Tensor,
        cached_summary: Optional[Tensor],
        is_even_layer: bool,
    ):
        """
        Args:
            X:               (B, n_ns, d)
            S:               (B, T_seq, d)
            cached_summary:  (B, n_tokens, d) from previous even layer; None on l=0.
            is_even_layer:   True → fresh HSP; False → reuse cache.
        Returns:
            X_new:     (B, n_sum+n_tokens, d)  new non-seq representation.
            X_sum:     (B, n_sum, d)           summary for GDPA.
            H_summary: (B, n_tokens, d)        sequence summary (cache for next odd layer).
        """
        X_sum = self.weight_gen(X)                                      # (B, n_sum, d)
        H_summary = self.hsp(S) if (is_even_layer or cached_summary is None) \
                    else cached_summary                                  # (B, n_tokens, d)
        X_global = torch.cat([X_sum, H_summary], dim=1)                 # (B, n_sum+n_tokens, d)
        X_new    = self.global_inter(X_global)                          # (B, n_sum+n_tokens, d)
        return X_new, X_sum, H_summary


# ---------------------------------------------------------------------------
# KunlunLayer   §4.5
# ---------------------------------------------------------------------------

class KunlunLayer(nn.Module):
    """Single Kunlun layer: InteractionBlock + TransformerBlock (§4.5).

    Maintains bidirectional information flow:
      Non-seq → Seq: X_sum from InteractionBlock guides GDPA in TransformerBlock.
      Seq → Non-seq: H_summary from HSP feeds GlobalInteraction.

    Layer behavior is governed by CompSkip (even vs. odd):
      Even: fresh HSP + GDPA (global seq understanding + personalized PFFN).
      Odd:  reuse HSP + SWA  (local seq refinement).
    """

    def __init__(self, n_ns: int, n_sum: int, d_model: int,
                 n_seeds: int, n_tokens: int,
                 n_heads: int = 8, window: int = 100, n_experts: int = 2,
                 kron_rank: int = 8, max_seq_len: int = 1000) -> None:
        super().__init__()
        self.interaction = KunlunInteractionBlock(
            n_ns, n_sum, d_model, n_seeds, n_tokens, n_heads, n_experts, kron_rank)
        self.transformer = KunlunTransformerBlock(
            n_sum, d_model, n_heads, window, max_seq_len)

    def forward(self, X: Tensor, S: Tensor,
                cached_summary: Optional[Tensor], layer_idx: int):
        """
        Args:
            X:               (B, n_ns, d)
            S:               (B, T, d)
            cached_summary:  (B, n_tokens, d) or None.
            layer_idx:       current layer index l.
        Returns:
            X_new:     (B, n_sum+n_tokens, d)
            S_new:     (B, T, d)
            X_sum:     (B, n_sum, d)
            H_summary: (B, n_tokens, d)
        """
        is_even = (layer_idx % 2 == 0)
        X_new, X_sum, H_summary = self.interaction(X, S, cached_summary, is_even)
        S_new = self.transformer(S, X_sum, is_even)
        return X_new, S_new, X_sum, H_summary


# ---------------------------------------------------------------------------
# Event-Level Personalization   §4.4.2
# ---------------------------------------------------------------------------

@dataclass
class EventConfig:
    """Per-event-type architecture configuration for Event-Level Personalization.

    High-value events (purchases, clicks) receive larger capacity;
    low-value events (impressions) use smaller configurations.
    Allocates compute proportional to each event type's importance.

    Attributes:
        d_model:   model dimension controlling capacity.
        n_heads:   attention heads.
        n_tokens:  HSP output tokens passed to non-seq interaction.
        n_layers:  number of Kunlun layers for this event type.
        window:    SWA half-window size (controls receptive field).
    """
    d_model:  int = 128
    n_heads:  int = 4
    n_tokens: int = 16
    n_layers: int = 2
    window:   int = 50


# Paper example configurations (§4.4.2)
CLICK_CONFIG      = EventConfig(d_model=256, n_heads=8, n_tokens=32, n_layers=3, window=100)
IMPRESSION_CONFIG = EventConfig(d_model=128, n_heads=4, n_tokens=16, n_layers=2, window=50)


# ---------------------------------------------------------------------------
# Full Kunlun Model
# ---------------------------------------------------------------------------

class Kunlun(nn.Module):
    """Kunlun: unified architecture for massive-scale CTR recommendation (§4).

    Builds on InterFormer (Zeng et al., 2024) with systematic model-efficiency
    co-design to achieve predictable scaling laws:

    Low-level (module optimization):
      GDPA  — fused multi-head PFFN (6× MFU on PFFN block)
      HSP   — hierarchical seed pooling (better quality than PMA)
      SWA   — O(Tw) sliding window attention (31% QPS improvement)

    High-level (computation reallocation):
      CompSkip     — 43.1% FLOPs reduction, 35% QPS improvement
      Event Personalization — per-event-type d_model, n_heads, n_tokens, L, w
      Wukong MoE   — 4% additional QPS from expert parallelism

    MFU: 17% → 37% on NVIDIA B200 GPUs.
    Achieves 2× scaling efficiency over state-of-the-art (InterFormer).
    Deployed at Meta Ads with +1.2% NE improvement.

    Args:
        dense_dim:     total concatenated dense feature dimension.
        sparse_dims:   vocabulary sizes for sparse (categorical) features.
        seq_input_dim: raw embedding dim per sequence item.
        d_model:       global embedding dimension d.
        num_layers:    L, number of Kunlun layers.
        n_sum:         WeightGeneration output token count.
        n_seeds:       HSP seed count (must be > n_tokens).
        n_tokens:      HSP output token count.
        n_heads:       attention heads for GDPA, SWA, HSP.
        window:        SWA half-window size w.
        n_experts:     Wukong expert count in GlobalInteraction.
        kron_rank:     SumKronLinear rank k.
        max_seq_len:   GDPA temperature τ (paper: maxlen(seq)).
        top_mlp_dims:  hidden dims for the final prediction MLP.
        num_tasks:     output logit count (1 for pCTR, 2 for pCTR+pCVR).
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: list[int],
        seq_input_dim: int,
        d_model: int = 128,
        num_layers: int = 3,
        n_sum: int = 8,
        n_seeds: int = 64,
        n_tokens: int = 16,
        n_heads: int = 8,
        window: int = 100,
        n_experts: int = 2,
        kron_rank: int = 8,
        max_seq_len: int = 1000,
        top_mlp_dims: Optional[list[int]] = None,
        num_tasks: int = 1,
    ) -> None:
        super().__init__()
        self.d_model    = d_model
        self.num_layers = num_layers
        self.n_sum      = n_sum
        self.n_tokens   = n_tokens

        # --- Feature preprocessing ---
        self.dense_proj  = nn.Linear(dense_dim, d_model)
        self.sparse_embs = nn.ModuleList([nn.Embedding(v, d_model) for v in sparse_dims])
        self.seq_proj    = nn.Sequential(
            nn.Linear(seq_input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.rote = ROTE(d_model)

        # n_ns for layer 0 = 1 dense + len(sparse_dims) sparse tokens
        n_ns_0   = 1 + len(sparse_dims)
        n_global = n_sum + n_tokens   # token count after layer 0 (stable thereafter)

        # --- Kunlun Layers ---
        self.layers = nn.ModuleList([
            KunlunLayer(
                n_ns       = n_ns_0 if l == 0 else n_global,
                n_sum      = n_sum,
                d_model    = d_model,
                n_seeds    = n_seeds,
                n_tokens   = n_tokens,
                n_heads    = n_heads,
                window     = window,
                n_experts  = n_experts,
                kron_rank  = kron_rank,
                max_seq_len= max_seq_len,
            )
            for l in range(num_layers)
        ])

        # --- Final prediction head: flatten X^(L) ∈ ℝ^(B, n_global, d) ---
        flat_dim = n_global * d_model
        hidden   = top_mlp_dims or [flat_dim // 2]
        dims     = [flat_dim] + hidden + [num_tasks]
        mlp: list[nn.Module] = []
        for i in range(len(dims) - 1):
            mlp.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mlp.append(nn.ReLU())
        self.classifier = nn.Sequential(*mlp)

    def _preprocess(
        self,
        dense_feat:   Tensor,
        sparse_feats: list[Tensor],
        seq_feat:     Tensor,
        timestamps:   Optional[Tensor] = None,
    ):
        """
        Args:
            dense_feat:   (B, dense_dim)
            sparse_feats: list of (B,) int tensors.
            seq_feat:     (B, T, seq_input_dim)
            timestamps:   (B, T) optional seconds.
        Returns:
            X: (B, n_ns, d),  S: (B, T, d)
        """
        x_dense  = self.dense_proj(dense_feat).unsqueeze(1)             # (B, 1, d)
        x_sparse = [emb(f).unsqueeze(1) for emb, f
                    in zip(self.sparse_embs, sparse_feats)]
        X = torch.cat([x_dense] + x_sparse, dim=1)                      # (B, n_ns, d)
        S = self.seq_proj(seq_feat)                                      # (B, T, d)
        S = self.rote(S, timestamps)
        return X, S

    def forward(
        self,
        dense_feat:   Tensor,
        sparse_feats: list[Tensor],
        seq_feat:     Tensor,
        timestamps:   Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            dense_feat:   (B, dense_dim)
            sparse_feats: list of (B,) int tensors.
            seq_feat:     (B, T, seq_input_dim)
            timestamps:   (B, T) optional timestamps in seconds.
        Returns:
            logits: (B, num_tasks)
        """
        X, S = self._preprocess(dense_feat, sparse_feats, seq_feat, timestamps)

        cached_summary: Optional[Tensor] = None
        for l, layer in enumerate(self.layers):
            X, S, _, H_summary = layer(X, S, cached_summary, layer_idx=l)
            if l % 2 == 0:
                cached_summary = H_summary      # cache even-layer HSP for next odd layer

        return self.classifier(X.reshape(X.size(0), -1))                # (B, num_tasks)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B, T = 4, 50

    dense_dim     = 32
    sparse_vocabs = [100, 200, 50]
    seq_input_dim = 16

    model = Kunlun(
        dense_dim     = dense_dim,
        sparse_dims   = sparse_vocabs,
        seq_input_dim = seq_input_dim,
        d_model       = 32,
        num_layers    = 4,
        n_sum         = 4,
        n_seeds       = 16,
        n_tokens      = 8,
        n_heads       = 4,
        window        = 10,
        n_experts     = 2,
        kron_rank     = 4,
        max_seq_len   = 100,
        top_mlp_dims  = [32],
        num_tasks     = 1,
    )

    dense_feat   = torch.randn(B, dense_dim)
    sparse_feats = [torch.randint(0, v, (B,)) for v in sparse_vocabs]
    seq_feat     = torch.randn(B, T, seq_input_dim)

    logits = model(dense_feat, sparse_feats, seq_feat)
    print(f"logits:   {logits.shape}  {logits.squeeze(-1).tolist()}")

    loss = F.binary_cross_entropy_with_logits(logits, torch.zeros(B, 1))
    loss.backward()
    print(f"loss:     {loss.item():.4f}")
    print("backward: OK")

    total = sum(p.numel() for p in model.parameters())
    print(f"params:   {total:,}")

    # Demonstrate CompSkip alternating behavior
    print("\nCompSkip layer pattern:")
    for l in range(model.num_layers):
        kind = "GDPA + fresh HSP" if l % 2 == 0 else "SWA  + cached HSP"
        print(f"  l={l}: {kind}")


if __name__ == "__main__":
    _smoke_test()
