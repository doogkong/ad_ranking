"""TokenMixer-Large: Scaling Up Large Ranking Models in Industrial Recommenders.

Reference PyTorch implementation.
Paper: https://arxiv.org/abs/2602.06563  (ByteDance AML, Feb 2026)

Architecture overview:
    Input features
        ↓
    SemanticGroupTokenizer  — embed per feature group + prepend global token
        ↓
    L × TokenMixerLargeBlock — (pre-norm, MixingReverting, pre-norm, S-P MoE)
        + inter-residual connections every `inter_residual_interval` blocks
        + auxiliary loss from intermediate layers
        ↓
    Mean pooling → task heads

Key design choices vs. original TokenMixer (RankMixer):
  - Mixing & Reverting: symmetric reshape restores Token Semantic Alignment in residuals.
  - Sparse-Pertoken MoE (S-P MoE): "First Enlarge, Then Sparse" with Gate Value
    Scaling, shared expert, and down-matrix small init.
  - Pre-Norm (RMSNorm) instead of Post-Norm for stability.
  - Pure model design: no legacy ops (LHUC, DCNv2) that hurt hardware utilisation.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation (no mean-centring, lighter than LayerNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class PertokenSwiGLU(nn.Module):
    """Per-token-position SwiGLU FFN.

    Each token position t has independent weight matrices W_up^t, W_gate^t,
    W_down^t — modelling heterogeneous feature semantics across positions.
    Implemented via batched einsum over the token axis.

    Args:
        num_tokens: T, number of token positions.
        dim:        D, feature dimension per token.
        expand:     hidden expansion factor n  (hidden = n * D).
        small_init: initialise W_down to 0.01 × xavier so the block starts
                    near identity, preventing output-value explosion in deep
                    models (Down-Matrix Small Init from the paper).
    """

    def __init__(self, num_tokens: int, dim: int, expand: int = 4,
                 small_init: bool = False) -> None:
        super().__init__()
        hidden = int(dim * expand)
        self.w_up   = nn.Parameter(torch.empty(num_tokens, dim, hidden))
        self.w_gate = nn.Parameter(torch.empty(num_tokens, dim, hidden))
        self.w_down = nn.Parameter(torch.empty(num_tokens, hidden, dim))
        self.b_up   = nn.Parameter(torch.zeros(num_tokens, hidden))
        self.b_gate = nn.Parameter(torch.zeros(num_tokens, hidden))
        self.b_down = nn.Parameter(torch.zeros(num_tokens, dim))
        self._init_weights(small_init)

    def _init_weights(self, small_init: bool) -> None:
        for w in (self.w_up, self.w_gate):
            nn.init.xavier_uniform_(w)
        nn.init.xavier_uniform_(self.w_down)
        if small_init:
            self.w_down.data.mul_(0.01)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D)
        up   = torch.einsum("btd,tdn->btn", x, self.w_up)   + self.b_up
        gate = torch.einsum("btd,tdn->btn", x, self.w_gate) + self.b_gate
        h    = F.silu(gate) * up
        return torch.einsum("btn,tnd->btd", h, self.w_down) + self.b_down


# ---------------------------------------------------------------------------
# Mixing & Reverting
# ---------------------------------------------------------------------------

class MixingReverting(nn.Module):
    """Symmetric two-layer Mixing–Reverting block.

    Layer 1 — Mixing:
      Reshape T×D tokens into H head-tokens of size T·(D/H) each.
      Apply per-head SwiGLU + residual + norm.
      This lets each head aggregate information across all T original tokens.

    Layer 2 — Reverting:
      Reverse the reshape back to T×D.
      Apply per-token SwiGLU with residual wired to the *original* input X
      (not the mixed representation), establishing Token Semantic Alignment:
      the residual positions always correspond to the same original token
      semantics throughout the entire network.

    This symmetry is what allows stable residual connections in deep stacks —
    the key fix over RankMixer's broken residual design.
    """

    def __init__(self, num_tokens: int, dim: int, num_heads: int,
                 moe_factory=None) -> None:
        """
        Args:
            num_tokens:  T.
            dim:         D (must be divisible by num_heads).
            num_heads:   H mixing heads.
            moe_factory: callable(num_tokens, dim) → nn.Module.
                         When provided, replaces PertokenSwiGLU with S-P MoE.
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.T = num_tokens
        self.H = num_heads
        head_mix_dim = num_tokens * (dim // num_heads)

        if moe_factory is not None:
            self.mix_ffn    = moe_factory(num_heads, head_mix_dim)
            self.revert_ffn = moe_factory(num_tokens, dim)
        else:
            self.mix_ffn    = PertokenSwiGLU(num_heads, head_mix_dim, small_init=True)
            self.revert_ffn = PertokenSwiGLU(num_tokens, dim, small_init=True)

        self.mix_norm    = RMSNorm(head_mix_dim)
        self.revert_norm = RMSNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        H        = self.H
        head_dim = D // H

        # --- Mixing ---
        # (B, T, D) → (B, T, H, D/H) → (B, H, T, D/H) → (B, H, T·D/H)
        h = x.view(B, T, H, head_dim).permute(0, 2, 1, 3).reshape(B, H, T * head_dim)
        h = self.mix_norm(self.mix_ffn(h) + h)          # residual on h

        # --- Reverting ---
        # (B, H, T·D/H) → (B, H, T, D/H) → (B, T, H, D/H) → (B, T, D)
        x_rev = h.view(B, H, T, head_dim).permute(0, 2, 1, 3).reshape(B, T, D)
        x_out = self.revert_norm(self.revert_ffn(x_rev) + x)  # residual on *original* x
        return x_out


# ---------------------------------------------------------------------------
# Sparse-Pertoken MoE (S-P MoE)
# ---------------------------------------------------------------------------

class SparsePerTokenMoE(nn.Module):
    """Sparse-Pertoken Mixture-of-Experts.

    "First Enlarge, Then Sparse": start from a dense PertokenSwiGLU,
    split its hidden dimension into E routing experts, then activate only
    top-k experts per token.  A shared expert is always active.

    Design choices:
      Gate Value Scaling (α): constant multiplier on the routed expert sum
        before adding the shared expert.  Compensates for softmax sum-to-1
        constraint and allows more gradient to flow through SwiGLU weights.
      Shared Expert: one always-active expert provides a stable learning
        signal and improves training stability (similar to DeepSeek-MoE).
      Down-Matrix Small Init: W_down of every expert initialised to 0.01×
        xavier, so the MoE block approximates identity at the start of
        training.

    Note: this reference implementation computes all experts eagerly (correct
    for training).  Production deployment uses custom MoEGroupedFFN + Token
    Parallel + FP8 quantisation for efficiency (see paper §3.5).
    """

    def __init__(self, num_tokens: int, dim: int, num_experts: int = 3,
                 top_k: int = 2, expand: int = 4,
                 gate_scale: float = 2.0) -> None:
        super().__init__()
        self.top_k = top_k
        self.alpha = gate_scale

        self.experts = nn.ModuleList([
            PertokenSwiGLU(num_tokens, dim, expand=expand, small_init=True)
            for _ in range(num_experts)
        ])
        self.shared_expert = PertokenSwiGLU(num_tokens, dim, expand=expand,
                                            small_init=True)
        self.router = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # Router logits per token: (B, T, E)
        logits = self.router(x)
        topk_w, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        gate_w = F.softmax(topk_w, dim=-1)  # (B, T, k); sums to 1 per token

        # Compute all expert outputs: (E, B, T, D)
        all_out = torch.stack([e(x) for e in self.experts], dim=0)

        # Gather top-k outputs per token: (B, T, k, D)
        idx      = topk_idx.unsqueeze(-1).expand(*topk_idx.shape, x.size(-1))
        selected = all_out.permute(1, 2, 0, 3).gather(2, idx)

        # Weighted sum of selected experts + shared expert
        routed = (gate_w.unsqueeze(-1) * selected).sum(dim=2)   # (B, T, D)
        return self.alpha * routed + self.shared_expert(x)


# ---------------------------------------------------------------------------
# TokenMixer-Large Block
# ---------------------------------------------------------------------------

class TokenMixerLargeBlock(nn.Module):
    """Single TokenMixer-Large block.

    Structure (Pre-Norm design):
        RMSNorm → MixingReverting  → residual add
        RMSNorm → Sparse-Pertoken MoE → residual add

    MixingReverting carries its own internal norms and residuals (eqs. 8
    and 16 in the paper), so the outer residuals here propagate the
    full-resolution token signal across blocks.
    """

    def __init__(self, num_tokens: int, dim: int, num_heads: int,
                 num_experts: int = 3, top_k: int = 2,
                 moe_expand: int = 4, gate_scale: float = 2.0) -> None:
        super().__init__()

        def _moe(n_tok: int, d: int) -> SparsePerTokenMoE:
            return SparsePerTokenMoE(n_tok, d, num_experts=num_experts,
                                     top_k=top_k, expand=moe_expand,
                                     gate_scale=gate_scale)

        self.norm1 = RMSNorm(dim)
        self.mix   = MixingReverting(num_tokens, dim, num_heads, moe_factory=_moe)
        self.norm2 = RMSNorm(dim)
        self.moe   = SparsePerTokenMoE(num_tokens, dim, num_experts=num_experts,
                                       top_k=top_k, expand=moe_expand,
                                       gate_scale=gate_scale)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mix(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

class SemanticGroupTokenizer(nn.Module):
    """Projects feature groups into a common token dimension D.

    Industrial recommendation systems have heterogeneous feature spaces:
    user features, item features, sequential features, cross features, etc.
    Features within each semantic group are concatenated and projected via
    a group-specific two-layer MLP to preserve heterogeneity.

    A learnable global token (analogous to [CLS] in BERT) is prepended to
    aggregate cross-group information.

    Args:
        feature_groups: list of input dims, one per semantic group.
        model_dim:      target token dimension D.
    """

    def __init__(self, feature_groups: list[int], model_dim: int) -> None:
        super().__init__()
        self.group_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(g_dim, model_dim),
                nn.SiLU(),
                nn.Linear(model_dim, model_dim),
            )
            for g_dim in feature_groups
        ])
        self.global_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.trunc_normal_(self.global_token, std=0.02)
        self.num_tokens = 1 + len(feature_groups)

    def forward(self, groups: list[Tensor]) -> Tensor:
        """
        Args:
            groups: list of (B, group_dim) tensors.
        Returns:
            tokens: (B, 1 + num_groups, D)
        """
        group_toks = torch.stack([mlp(g) for mlp, g in zip(self.group_mlps, groups)], dim=1)
        B = group_toks.size(0)
        return torch.cat([self.global_token.expand(B, -1, -1), group_toks], dim=1)


# ---------------------------------------------------------------------------
# Full TokenMixer-Large Model
# ---------------------------------------------------------------------------

class TokenMixerLarge(nn.Module):
    """TokenMixer-Large ranking backbone.

    Args:
        feature_groups:          input dims per semantic feature group.
        model_dim:               token dimension D.
        num_heads:               mixing heads H (D must be divisible by H).
        num_layers:              depth L.
        num_tasks:               number of prediction heads (e.g. 2 for pCTR+pVTR).
        num_experts:             routing experts per S-P MoE layer.
        top_k:                   experts activated per token (paper uses 2 for 1:2 sparsity).
        moe_expand:              SwiGLU hidden expansion factor n.
        gate_scale:              α for Gate Value Scaling (paper default: 2.0).
        inter_residual_interval: add inter-layer residual every N blocks (paper: 2–3).
                                 Set to None to disable.
        aux_loss_weight:         weight λ for auxiliary lower-layer loss (0 to disable).
    """

    def __init__(
        self,
        feature_groups: list[int],
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 6,
        num_tasks: int = 1,
        num_experts: int = 3,
        top_k: int = 2,
        moe_expand: int = 4,
        gate_scale: float = 2.0,
        inter_residual_interval: Optional[int] = 2,
        aux_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.tokenizer = SemanticGroupTokenizer(feature_groups, model_dim)
        num_tokens = self.tokenizer.num_tokens

        self.blocks = nn.ModuleList([
            TokenMixerLargeBlock(
                num_tokens, model_dim, num_heads,
                num_experts=num_experts, top_k=top_k,
                moe_expand=moe_expand, gate_scale=gate_scale,
            )
            for _ in range(num_layers)
        ])
        self.inter_residual_interval = inter_residual_interval
        self.aux_loss_weight = aux_loss_weight
        self.final_norm = RMSNorm(model_dim)

        # Auxiliary heads on intermediate layers for aux loss.
        # Placed at the end of each inter-residual interval window.
        aux_layer_indices = (
            [i for i in range(num_layers - 1)
             if (i + 1) % (inter_residual_interval or num_layers) == 0]
            if aux_loss_weight > 0 else []
        )
        self.aux_heads = nn.ModuleDict({
            str(i): nn.Linear(model_dim, num_tasks) for i in aux_layer_indices
        })
        self.heads = nn.ModuleList([nn.Linear(model_dim, 1) for _ in range(num_tasks)])

    def forward(self, groups: list[Tensor]) -> tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            groups: list of (B, group_dim) tensors, one per feature group.
        Returns:
            logits:   (B, num_tasks) — unnormalised prediction scores.
            aux_loss: scalar auxiliary loss from intermediate layers, or None.
        """
        x = self.tokenizer(groups)                    # (B, T, D)

        inter_buf: Optional[Tensor] = None
        aux_logits: list[Tensor]    = []

        for i, block in enumerate(self.blocks):
            # Apply buffered inter-residual at the start of each new interval.
            if (self.inter_residual_interval is not None
                    and i > 0 and i % self.inter_residual_interval == 0
                    and inter_buf is not None):
                x = x + inter_buf

            if (self.inter_residual_interval is not None
                    and i % self.inter_residual_interval == 0):
                inter_buf = x

            x = block(x)

            if str(i) in self.aux_heads:
                aux_logits.append(self.aux_heads[str(i)](x.mean(dim=1)))

        x      = self.final_norm(x)
        pooled = x.mean(dim=1)                        # (B, D)
        logits = torch.cat([h(pooled) for h in self.heads], dim=-1)  # (B, num_tasks)

        aux_loss = None
        if aux_logits and self.aux_loss_weight > 0:
            target   = logits.detach()
            aux_loss = self.aux_loss_weight * sum(
                F.mse_loss(al, target) for al in aux_logits
            )

        return logits, aux_loss


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B = 4

    # Three semantic groups: user (64-d), item (128-d), context (32-d)
    feature_groups = [64, 128, 32]
    groups = [torch.randn(B, d) for d in feature_groups]

    model = TokenMixerLarge(
        feature_groups=feature_groups,
        model_dim=64,
        num_heads=4,
        num_layers=4,
        num_tasks=2,           # e.g. pCTR + pVTR
        num_experts=3,
        top_k=2,
        moe_expand=2,
        gate_scale=2.0,
        inter_residual_interval=2,
        aux_loss_weight=0.1,
    )

    logits, aux_loss = model(groups)
    print(f"logits:   {logits.shape}   {logits}")
    print(f"aux_loss: {aux_loss.item():.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"params:   {total_params:,}")


if __name__ == "__main__":
    _smoke_test()
