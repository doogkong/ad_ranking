"""Wukong: Towards a Scaling Law for Large-Scale Recommendation.

Reference PyTorch implementation.
Paper: https://arxiv.org/abs/2403.02545  (Meta AI, ICML 2024)

Architecture overview:
    Dense + Sparse features
        ↓
    EmbeddingLayer  — per-group MLP → d-dimensional tokens X_0 ∈ ℝ^(n×d)
        ↓
    Interaction Stack  (l identical WukongLayers)
        each layer:
            FMBlock  — FM(X) = X·X^T·Y  →  flatten  →  LN  →  MLP  →  n_F tokens
            LCBBlock — W_L · X                                       →  n_L tokens
            concat(FMB, LCB) + residual  →  LN  →  X_{i+1}
        ↓
    Final MLP  →  logit(s)

Key design choices:
  - FMs capture pairwise (2nd-order) interactions; stacking l layers gives
    interactions up to order 2^l via binary-exponentiation analogy.
  - LCB preserves identity-order embeddings so each layer contains both
    low-order and high-order signals for the next FM to interact.
  - Optimized FM: X·X^T is rank-d (often d ≤ n), so project with learnable
    Y ∈ ℝ^(n×k) to reduce O(n²d) → O(nkd) with k << n.
    Y is made attentive by processing the linearly-compressed input through an MLP.
  - Residual across layers stabilises training; LN after each layer.

Scaling knobs (§3.8):
    l    — number of WukongLayers (highest impact on quality)
    n_F  — FMBlock output token count
    n_L  — LCB output token count  (n_L ≈ n_F in practice)
    k    — FM projection rank
    MLP  — depth and width inside FMBlock
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _make_mlp(dims: list[int], dropout: float = 0.0) -> nn.Sequential:
    """Build a ReLU MLP from a list of layer widths."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Embedding Layer (§3.2)
# ---------------------------------------------------------------------------

class FeatureGroupEmbedding(nn.Module):
    """Projects one feature group (dense or pooled sparse) to d_model via MLP.

    In production each categorical feature has its own embedding table and
    multi-hot lookups are sum-pooled. Here we accept a pre-computed dense
    vector per group (the output of embedding lookup + pooling) and project
    it to d_model.

    Args:
        input_dim:  dimensionality of the raw feature group vector.
        d_model:    target token dimension d.
    """

    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)  # (B, d_model)


class EmbeddingLayer(nn.Module):
    """Maps a list of feature groups to the token matrix X_0 ∈ ℝ^(B, n, d).

    Args:
        feature_dims:  list of raw input dims, one per feature group.
        d_model:       global embedding dimension d.
    """

    def __init__(self, feature_dims: list[int], d_model: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList([
            FeatureGroupEmbedding(dim, d_model) for dim in feature_dims
        ])
        self.n = len(feature_dims)  # number of tokens

    def forward(self, feature_groups: list[Tensor]) -> Tensor:
        """
        Args:
            feature_groups: list of (B, input_dim_i) tensors.
        Returns:
            X: (B, n, d_model)
        """
        tokens = [proj(feat) for proj, feat in zip(self.projections, feature_groups)]
        return torch.stack(tokens, dim=1)  # (B, n, d_model)


# ---------------------------------------------------------------------------
# Optimized Factorization Machine (§3.6)
# ---------------------------------------------------------------------------

class OptimizedFM(nn.Module):
    """Low-rank attentive FM.

    Standard FM computes XX^T ∈ ℝ^(n×n) — quadratic in the number of
    embeddings and often rank-limited (rank ≤ d when d ≤ n).

    Optimized FM projects XX^T down to n×k via a learnable matrix Y ∈ ℝ^(n×k):
        FM_opt(X) = X · (X^T · Y)     [associativity: O(nkd) instead of O(n²d)]

    Y is made attentive: it is derived from the (linearly compressed) input
    via a small MLP, allowing the projection to adapt to the current sample.

    Args:
        n_tokens:  n, number of input embeddings.
        d_model:   d, embedding dimension.
        k:         projection rank (k << n for efficiency).
    """

    def __init__(self, n_tokens: int, d_model: int, k: int = 24) -> None:
        super().__init__()
        self.k = k
        # Attentive Y: compress X to (n·d) → MLP → (n·k)
        self.attn_proj = nn.Linear(n_tokens * d_model, n_tokens * k, bias=True)

    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: (B, n, d)
        Returns:
            out: (B, n, k)  — FM interaction matrix
        """
        B, n, d = X.shape
        # Attentive projection matrix Y: (B, n, k)
        Y = self.attn_proj(X.reshape(B, n * d)).reshape(B, n, self.k)
        # FM_opt = X · (X^T · Y) = (B,n,d) · [(B,n,d)^T · (B,n,k)]
        #        = (B,n,d) · (B,d,k)
        XtY = X.transpose(1, 2) @ Y        # (B, d, k)
        return X @ XtY                      # (B, n, k)


# ---------------------------------------------------------------------------
# Factorization Machine Block (§3.4)
# ---------------------------------------------------------------------------

class FMBlock(nn.Module):
    """FMB: FM → flatten → LN → MLP → reshape to n_F tokens.

    FMB(X_i) = reshape(MLP(LN(flatten(FM(X_i)))))

    The output is n_F new embeddings of dimension d_model, which carry
    pairwise (and via stacking, higher-order) interaction information.

    Args:
        n_tokens:   n, number of input embeddings for this layer.
        d_model:    d.
        n_F:        number of output tokens.
        k:          FM projection rank.
        mlp_dims:   hidden layer widths for the internal MLP (excluding
                    input and output which are inferred).
        dropout:    dropout on MLP hidden layers.
    """

    def __init__(self, n_tokens: int, d_model: int, n_F: int,
                 k: int = 24, mlp_dims: Optional[list[int]] = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.n_F     = n_F
        self.d_model = d_model
        fm_out_dim   = n_tokens * k         # flattened FM output size
        out_dim      = n_F * d_model

        hidden = mlp_dims if mlp_dims is not None else [fm_out_dim]
        self.fm   = OptimizedFM(n_tokens, d_model, k)
        self.norm = nn.LayerNorm(fm_out_dim)
        self.mlp  = _make_mlp([fm_out_dim] + hidden + [out_dim], dropout)

    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: (B, n, d)
        Returns:
            (B, n_F, d)
        """
        B = X.size(0)
        fm_out  = self.fm(X)                            # (B, n, k)
        flat    = fm_out.reshape(B, -1)                 # (B, n*k)
        normed  = self.norm(flat)                       # (B, n*k)
        out     = self.mlp(normed)                      # (B, n_F*d)
        return out.reshape(B, self.n_F, self.d_model)   # (B, n_F, d)


# ---------------------------------------------------------------------------
# Linear Compression Block (§3.5)
# ---------------------------------------------------------------------------

class LCBBlock(nn.Module):
    """LCB: learnable linear recombination of input embeddings.

    LCB(X_i) = W_L · X_i

    where W_L ∈ ℝ^(n_L × n_i).  This preserves interaction-order invariance:
    the i-th layer still sees interactions from order 1 to 2^{i-1}, ensuring
    the next FMB can combine all orders up to 2^i.

    Args:
        n_in:   number of input tokens n_i.
        n_L:    number of output tokens.
        d_model: d (applied per embedding dimension — same W_L for all d).
    """

    def __init__(self, n_in: int, n_L: int) -> None:
        super().__init__()
        self.W = nn.Linear(n_in, n_L, bias=False)

    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: (B, n_in, d)
        Returns:
            (B, n_L, d)
        """
        # Apply W along the token dimension: (B, d, n_in) × W^T → (B, d, n_L)
        return self.W(X.transpose(1, 2)).transpose(1, 2)   # (B, n_L, d)


# ---------------------------------------------------------------------------
# Wukong Layer (§3.3)
# ---------------------------------------------------------------------------

class WukongLayer(nn.Module):
    """Single Wukong interaction layer.

    X_{i+1} = LN(concat(FMB(X_i), LCB(X_i)) + residual(X_i))

    When the residual has a different number of tokens than the concat output,
    a linear projection aligns the shape (paper §3.3: "residual can be
    linearly compressed to match the shape").

    Args:
        n_in:      number of input tokens n_i.
        d_model:   embedding dimension d.
        n_F:       FMBlock output tokens.
        n_L:       LCB output tokens.
        k:         FM projection rank.
        mlp_dims:  hidden dims for FMBlock's internal MLP.
        dropout:   dropout rate.
    """

    def __init__(self, n_in: int, d_model: int, n_F: int, n_L: int,
                 k: int = 24, mlp_dims: Optional[list[int]] = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        n_out = n_F + n_L
        self.fmb  = FMBlock(n_in, d_model, n_F, k, mlp_dims, dropout)
        self.lcb  = LCBBlock(n_in, n_L)
        self.norm = nn.LayerNorm(d_model)
        # Residual projection when n_in ≠ n_out
        self.res_proj = (
            nn.Linear(n_in, n_out, bias=False) if n_in != n_out else nn.Identity()
        )
        self.n_out = n_out

    def forward(self, X: Tensor) -> Tensor:
        """
        Args:
            X: (B, n_in, d)
        Returns:
            (B, n_F + n_L, d)
        """
        fmb_out  = self.fmb(X)                                     # (B, n_F, d)
        lcb_out  = self.lcb(X)                                     # (B, n_L, d)
        combined = torch.cat([fmb_out, lcb_out], dim=1)            # (B, n_F+n_L, d)
        # Residual: project token axis of X to match n_out
        residual = self.res_proj(X.transpose(1, 2)).transpose(1, 2)  # (B, n_out, d)
        return self.norm(combined + residual)                        # (B, n_out, d)


# ---------------------------------------------------------------------------
# Full Wukong Model
# ---------------------------------------------------------------------------

class Wukong(nn.Module):
    """Wukong ranking backbone.

    Args:
        feature_dims:   list of raw input dims per feature group.
        d_model:        global embedding dimension d.
        num_layers:     l, number of WukongLayers (primary scaling knob).
        n_F:            FMB output tokens per layer.
        n_L:            LCB output tokens per layer.
        k:              FM projection rank (optimized FM).
        mlp_dims:       hidden layer widths inside FMBlock MLP.
        top_mlp_dims:   dims of the final prediction MLP (excl. input/output).
        num_tasks:      number of output logits (e.g. 2 for CTR + CVR).
        dropout:        dropout inside MLPs.
    """

    def __init__(
        self,
        feature_dims: list[int],
        d_model: int = 128,
        num_layers: int = 4,
        n_F: int = 32,
        n_L: int = 32,
        k: int = 24,
        mlp_dims: Optional[list[int]] = None,
        top_mlp_dims: Optional[list[int]] = None,
        num_tasks: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = EmbeddingLayer(feature_dims, d_model)
        n_tokens = len(feature_dims)   # n_0

        # Build Interaction Stack
        layers: list[nn.Module] = []
        n_cur = n_tokens
        for _ in range(num_layers):
            layers.append(WukongLayer(n_cur, d_model, n_F, n_L, k, mlp_dims, dropout))
            n_cur = n_F + n_L
        self.interaction_stack = nn.ModuleList(layers)

        # Final MLP: input = flattened last-layer tokens
        flat_dim = n_cur * d_model
        hidden   = top_mlp_dims if top_mlp_dims is not None else [flat_dim // 2]
        self.top_mlp = _make_mlp([flat_dim] + hidden + [num_tasks], dropout)

    def forward(self, feature_groups: list[Tensor]) -> Tensor:
        """
        Args:
            feature_groups: list of (B, input_dim_i) tensors.
        Returns:
            logits: (B, num_tasks)
        """
        X = self.embedding(feature_groups)    # (B, n, d)
        for layer in self.interaction_stack:
            X = layer(X)                       # (B, n_F+n_L, d) after each layer
        B = X.size(0)
        flat = X.reshape(B, -1)               # (B, (n_F+n_L)*d)
        return self.top_mlp(flat)             # (B, num_tasks)


# ---------------------------------------------------------------------------
# Convenience: scaling configurations from paper (Table 5 / §3.8)
# ---------------------------------------------------------------------------

def wukong_small(feature_dims: list[int], **kwargs) -> Wukong:
    """~0.5 GFLOP/example base config (l=2, n_L=n_F=8, k=24)."""
    return Wukong(feature_dims, num_layers=2, n_F=8, n_L=8, k=24,
                  mlp_dims=[2048], **kwargs)


def wukong_medium(feature_dims: list[int], **kwargs) -> Wukong:
    """~2 GFLOP/example (l=8, n_L=n_F=32, k=24)."""
    return Wukong(feature_dims, num_layers=8, n_F=32, n_L=32, k=24,
                  mlp_dims=[2048], **kwargs)


def wukong_large(feature_dims: list[int], **kwargs) -> Wukong:
    """~22 GFLOP/example (l=8, n_L=n_F=96, k=96)."""
    return Wukong(feature_dims, num_layers=8, n_F=96, n_L=96, k=96,
                  mlp_dims=[8192], **kwargs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B = 4

    # 8 feature groups with varying raw dimensions (simulating user/item/ctx features)
    feature_dims = [64, 64, 32, 32, 128, 128, 16, 16]
    groups = [torch.randn(B, d) for d in feature_dims]

    # Small config for speed
    model = Wukong(
        feature_dims=feature_dims,
        d_model=32,
        num_layers=3,
        n_F=8,
        n_L=8,
        k=8,
        mlp_dims=[64],
        top_mlp_dims=[64],
        num_tasks=2,
        dropout=0.0,
    )

    logits = model(groups)
    print(f"logits:   {logits.shape}  {logits.tolist()}")

    loss = F.binary_cross_entropy_with_logits(logits, torch.zeros(B, 2))
    loss.backward()
    print(f"loss:     {loss.item():.4f}")
    print("backward: OK")

    total = sum(p.numel() for p in model.parameters())
    print(f"params:   {total:,}")

    # Demonstrate interaction order growth
    print("\nInteraction order per layer (2^l):")
    for l in range(1, 7):
        print(f"  l={l}: up to 2^{l} = {2**l}-order interactions")


if __name__ == "__main__":
    _smoke_test()
