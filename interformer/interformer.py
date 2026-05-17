"""InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction.

Reference PyTorch implementation.
Paper: https://arxiv.org/abs/2411.09852  (UIUC + Meta AI, Sep 2025)

Architecture overview (Algorithm 1):
    Non-seq features  X^(0)  →  EmbeddingLayer  →  X^(1) ∈ ℝ^(B, n, d)
    Seq features      S^(0)  →  MaskNet         →  S^(1) ∈ ℝ^(B, T, d)

    Compute X_sum^(1) = Gating(LCE(X^(1)))      [Cross Arch, Eq. 10]
    Prepend X_sum^(1) to S^(1) as CLS tokens     [once, at layer 1]

    for l = 1..L:
        Cross Arch:
            X_sum^(l) = Gating(LCE(X^(l)))                         [Eq. 10]
            S_sum^(l) = Gating([S_CLS^(l) || S_PMA^(l) || S_recent^(l)])  [Eq. 11]
        Interaction Arch:
            X^(l+1) = MLP(Interaction([X^(l) || S_sum^(l)]))       [Eq. 7]
        Sequence Arch:
            S^(l+1) = MHA(PFFN(X_sum^(l), S^(l)))                  [Eq. 9]

    ŷ = MLP([X_sum^(L) || S_sum^(L)])

Two key problems solved:
  1. Insufficient inter-mode interaction — bidirectional information flow:
       - Non-seq features receive sequence context via S_sum in Interaction Arch.
       - Sequence features receive non-seq context via X_sum in PFFN.
  2. Aggressive information aggregation — retain full dimensionality in both arches;
       summarize only at Cross Arch boundaries using selective self-gating.

Deployed at Meta Ads: +0.15% NE, +24% QPS vs. prior SOTA.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class SelfGating(nn.Module):
    """Gating(X) = σ(gate_mlp(X)) ⊙ X  (Eq. 10, §4.4).

    Sparse masking: learned gate retains high-signal dimensions and
    suppresses noise. Applied after LCE for non-seq summarization and
    after concatenation for sequence summarization.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.gate(x)) * x


class MaskNet(nn.Module):
    """Sequence preprocessing: self-masking to merge k sequences and filter noise.

    MaskNet(S) = MLP_lce(S ⊙ MLP_mask(S))   [Eq. 6]

    MLP_mask learns which items and dimensions are relevant.
    MLP_lce compresses the concatenated sequence embedding dim to d_model.

    Args:
        input_dim:  kd — concatenated embedding dim of k sequences.
        d_model:    target embedding dimension d.
    """

    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        self.mask_mlp = nn.Linear(input_dim, input_dim)
        self.lce_mlp  = nn.Linear(input_dim, d_model)

    def forward(self, S: Tensor) -> Tensor:
        # S: (B, T, input_dim)
        mask = torch.sigmoid(self.mask_mlp(S))
        return self.lce_mlp(S * mask)  # (B, T, d_model)


class LinearCompressedEmbedding(nn.Module):
    """LCE: compress n feature tokens to n_sum via linear transform on token axis.

    X ∈ ℝ^(n×d) → W^T X → ℝ^(n_sum×d),  W ∈ ℝ^(n×n_sum)
    """

    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.W = nn.Linear(n_in, n_out, bias=False)

    def forward(self, X: Tensor) -> Tensor:
        # X: (B, n_in, d) → (B, n_out, d)
        return self.W(X.transpose(1, 2)).transpose(1, 2)


# ---------------------------------------------------------------------------
# Cross Arch components
# ---------------------------------------------------------------------------

class PoolingByMHA(nn.Module):
    """PMA: sequence summarization with learnable query vectors (§3.3, Eq. 4).

    PMA(Q_PMA, S) = MHA(Q_PMA, K=S, V=S)

    Q_PMA ∈ ℝ^(n_pma × d) is a learned parameter that summarises S into
    n_pma fixed-size summary tokens capturing different aspects of the sequence.
    """

    def __init__(self, n_pma: int, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        self.Q   = nn.Parameter(torch.empty(1, n_pma, d_model))
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        nn.init.trunc_normal_(self.Q, std=0.02)

    def forward(self, S: Tensor) -> Tensor:
        # S: (B, T, d) → (B, n_pma, d)
        B = S.size(0)
        Q, _ = self.mha(self.Q.expand(B, -1, -1), S, S)
        return Q


class CrossArch(nn.Module):
    """Cross Arch: selective summarization bridging Interaction and Sequence Arches.

    Non-sequence summarization (Eq. 10):
        X_sum = Gating(LCE(X))             shape: (B, n_sum, d)

    Sequence summarization (Eq. 11):
        S_sum = Gating([S_CLS || S_PMA || S_recent])   shape: (B, s_sum_len, d)

    where:
        S_CLS    = S[:, :n_cls, :]          first n_cls tokens (CLS tokens from MHA)
        S_PMA    = PMA(Q_PMA, S)            learnable-query attention summarization
        S_recent = S[:, -k_recent:, :]      most recent k_recent behavior tokens

    Args:
        n_ns:      number of non-sequence tokens n.
        n_sum:     compressed non-sequence token count n_sum << n.
        d_model:   embedding dimension d.
        n_cls:     number of CLS tokens prepended to sequence (paper: 4).
        n_pma:     PMA learnable query count (paper: 2).
        k_recent:  recent token count (paper: 2).
        n_heads:   MHA heads for PMA.
    """

    def __init__(self, n_ns: int, n_sum: int, d_model: int,
                 n_cls: int = 4, n_pma: int = 2, k_recent: int = 2,
                 n_heads: int = 4) -> None:
        super().__init__()
        self.n_cls    = n_cls
        self.k_recent = k_recent
        self.s_sum_len = n_cls + n_pma + k_recent

        self.lce       = LinearCompressedEmbedding(n_ns, n_sum)
        self.ns_gate   = SelfGating(d_model)
        self.pma       = PoolingByMHA(n_pma, d_model, n_heads)
        self.seq_gate  = SelfGating(d_model)

    def ns_summarize(self, X: Tensor) -> Tensor:
        """X: (B, n_ns, d) → X_sum: (B, n_sum, d)"""
        return self.ns_gate(self.lce(X))

    def seq_summarize(self, S: Tensor) -> Tensor:
        """S: (B, T_S, d) → S_sum: (B, n_cls+n_pma+k_recent, d)"""
        S_cls    = S[:, :self.n_cls, :]
        S_pma    = self.pma(S)
        S_recent = S[:, -self.k_recent:, :]
        combined = torch.cat([S_cls, S_pma, S_recent], dim=1)
        return self.seq_gate(combined)

    def forward(self, X: Tensor, S: Tensor):
        return self.ns_summarize(X), self.seq_summarize(S)


# ---------------------------------------------------------------------------
# Interaction Arch
# ---------------------------------------------------------------------------

class InteractionArch(nn.Module):
    """Interaction Arch: behavior-aware non-sequence interaction learning (§4.2, Eq. 7).

    X^(l+1) = MLP(Interaction([X^(l) || S_sum^(l)]))

    Concatenates X and S_sum along the token axis so each non-sequence feature
    can directly interact with the sequence summary. The inner-product-based
    interaction computes pairwise attention scores across the combined token set
    and enriches each X token with sequence context. Only the n_ns X-token
    positions are retained for the next layer.

    Compatible with any interaction module (inner product, DCNv2, DHEN — set
    via `interaction` arg).
    """

    def __init__(self, n_ns: int, s_sum_len: int, d_model: int) -> None:
        super().__init__()
        self.n_ns  = n_ns
        self.scale = d_model ** -0.5
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, X: Tensor, S_sum: Tensor) -> Tensor:
        """
        Args:
            X:     (B, n_ns, d)
            S_sum: (B, s_sum_len, d)
        Returns:
            X_new: (B, n_ns, d)
        """
        tokens  = torch.cat([X, S_sum], dim=1)           # (B, n_total, d)
        # Inner-product interaction: each token attends to all others
        scores  = torch.bmm(tokens, tokens.transpose(1, 2)) * self.scale  # (B, n_total, n_total)
        interact = torch.bmm(F.softmax(scores, dim=-1), tokens)            # (B, n_total, d)
        # Retain X-token positions + residual
        X_inter = interact[:, :self.n_ns, :]             # (B, n_ns, d)
        return self.mlp(X_inter + X)                     # (B, n_ns, d)


# ---------------------------------------------------------------------------
# Sequence Arch
# ---------------------------------------------------------------------------

class PersonalizedFFN(nn.Module):
    """PFFN: non-sequence-conditioned sequence transformation (§4.3, Eq. 8).

    W_PFFN = X_sum · W ∈ ℝ^(B, d, d)      [sample-specific weight matrix]
    PFFN(X_sum, S) = S · W_PFFN            [transform each sequence token]

    W is a learnable parameter ∈ ℝ^(n_sum·d, d·d). For each sample, it
    derives a d×d projection from the compressed non-sequence summary,
    essentially making sequence token transformation context-aware.
    """

    def __init__(self, n_sum: int, d_model: int) -> None:
        super().__init__()
        self.W = nn.Linear(n_sum * d_model, d_model * d_model, bias=False)
        self.d = d_model

    def forward(self, X_sum: Tensor, S: Tensor) -> Tensor:
        """
        Args:
            X_sum: (B, n_sum, d)
            S:     (B, T_S, d)
        Returns:
            (B, T_S, d)
        """
        B, n_sum, d = X_sum.shape
        W_pffn = self.W(X_sum.reshape(B, n_sum * d)).reshape(B, d, d)  # (B, d, d)
        return torch.bmm(S, W_pffn)                                      # (B, T_S, d)


class SequenceArch(nn.Module):
    """Sequence Arch: context-aware sequence modeling (§4.3, Eq. 9).

    S^(l+1) = MHA^(l)(PFFN(X_sum^(l), S^(l)))

    PFFN integrates non-sequence context into sequence token transformations
    (personalised projection). MHA then captures long-range sequential
    dependencies, with rotary positional embeddings for positional awareness.
    """

    def __init__(self, n_sum: int, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        self.pffn = PersonalizedFFN(n_sum, d_model)
        self.mha  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, X_sum: Tensor, S: Tensor) -> Tensor:
        """
        Args:
            X_sum: (B, n_sum, d)
            S:     (B, T_S, d)
        Returns:
            (B, T_S, d)
        """
        S_pffn    = self.pffn(X_sum, S)              # (B, T_S, d)
        S_mha, _  = self.mha(S_pffn, S_pffn, S_pffn) # (B, T_S, d)
        return self.norm(S_mha + S)                   # residual + norm


# ---------------------------------------------------------------------------
# InterFormer Block (one layer l)
# ---------------------------------------------------------------------------

class InterFormerBlock(nn.Module):
    """Single InterFormer block (one iteration of the loop in Algorithm 1).

    Execution order per block:
        1. CrossArch   → X_sum^(l), S_sum^(l)
        2. InteractionArch → X^(l+1)   [X enriched with S context]
        3. SequenceArch    → S^(l+1)   [S enriched with X context]
    """

    def __init__(self, n_ns: int, n_sum: int, s_sum_len: int,
                 d_model: int, n_heads: int = 4,
                 n_cls: int = 4, n_pma: int = 2, k_recent: int = 2) -> None:
        super().__init__()
        self.cross       = CrossArch(n_ns, n_sum, d_model, n_cls, n_pma, k_recent, n_heads)
        self.interaction = InteractionArch(n_ns, s_sum_len, d_model)
        self.sequence    = SequenceArch(n_sum, d_model, n_heads)

    def forward(self, X: Tensor, S: Tensor):
        """
        Args:
            X: (B, n_ns, d)
            S: (B, T_S, d)   T_S includes prepended CLS tokens
        Returns:
            X_new:  (B, n_ns, d)
            S_new:  (B, T_S, d)
            X_sum:  (B, n_sum, d)   — saved for final prediction
            S_sum:  (B, s_sum_len, d) — saved for final prediction
        """
        X_sum, S_sum = self.cross(X, S)
        X_new        = self.interaction(X, S_sum)
        S_new        = self.sequence(X_sum, S)
        return X_new, S_new, X_sum, S_sum


# ---------------------------------------------------------------------------
# Full InterFormer Model
# ---------------------------------------------------------------------------

class InterFormer(nn.Module):
    """InterFormer: bidirectional heterogeneous interaction for CTR prediction.

    Args:
        dense_dim:     total dimension of concatenated dense features.
        sparse_dims:   list of vocabulary sizes for sparse (categorical) features.
        seq_input_dim: raw embedding dim per sequence item (after k-seq MaskNet merge).
        d_model:       global embedding dimension d.
        num_layers:    L, number of InterFormer blocks.
        n_ns:          number of non-sequence feature tokens after preprocessing.
                       Includes 1 dense token + len(sparse_dims) sparse tokens.
        n_sum:         compressed non-seq token count for Cross Arch (n_sum << n_ns).
        n_cls:         CLS token count prepended to sequence (paper: 4).
        n_pma:         PMA learnable query count (paper: 2).
        k_recent:      recent token count in sequence summary (paper: 2).
        n_heads:       MHA heads.
        top_mlp_dims:  hidden dims for final prediction MLP.
        num_tasks:     number of output logits.
    """

    def __init__(
        self,
        dense_dim: int,
        sparse_dims: list[int],
        seq_input_dim: int,
        d_model: int = 64,
        num_layers: int = 3,
        n_sum: int = 4,
        n_cls: int = 4,
        n_pma: int = 2,
        k_recent: int = 2,
        n_heads: int = 4,
        top_mlp_dims: Optional[list[int]] = None,
        num_tasks: int = 1,
    ) -> None:
        super().__init__()
        self.d_model  = d_model
        self.n_cls    = n_cls
        self.num_layers = num_layers

        # --- Feature preprocessing ---
        # Dense: concat all dense → Linear → d_model token
        self.dense_proj = nn.Linear(dense_dim, d_model)
        # Sparse: one embedding per categorical feature
        self.sparse_embs = nn.ModuleList([
            nn.Embedding(vocab, d_model) for vocab in sparse_dims
        ])
        # Sequence: MaskNet to unify multiple sequences into one (B, T, d)
        self.seq_masknet = MaskNet(seq_input_dim, d_model)

        # n_ns = 1 dense token + len(sparse_dims) sparse tokens
        n_ns = 1 + len(sparse_dims)
        self.n_ns = n_ns

        s_sum_len = n_cls + n_pma + k_recent

        # Initial X_sum for prepending CLS (computed from X^(1) before loop)
        self.init_cross = CrossArch(n_ns, n_sum, d_model, n_cls, n_pma, k_recent, n_heads)

        # InterFormer blocks
        self.blocks = nn.ModuleList([
            InterFormerBlock(n_ns, n_sum, s_sum_len, d_model, n_heads,
                             n_cls, n_pma, k_recent)
            for _ in range(num_layers)
        ])

        # Final prediction MLP on [X_sum || S_sum]
        in_dim   = (n_sum + s_sum_len) * d_model
        hidden   = top_mlp_dims or [in_dim // 2]
        mlp_dims = [in_dim] + hidden + [num_tasks]
        layers: list[nn.Module] = []
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:
                layers.append(nn.ReLU())
        self.classifier = nn.Sequential(*layers)

    def _preprocess(
        self,
        dense_feat: Tensor,
        sparse_feats: list[Tensor],
        seq_feat: Tensor,
    ):
        """
        Args:
            dense_feat:   (B, dense_dim)
            sparse_feats: list of (B,) integer tensors, one per sparse feature.
            seq_feat:     (B, T, seq_input_dim)  — pre-merged multi-sequence input.
        Returns:
            X: (B, n_ns, d),  S: (B, T, d)
        """
        x_dense  = self.dense_proj(dense_feat).unsqueeze(1)          # (B, 1, d)
        x_sparse = [emb(f).unsqueeze(1) for emb, f
                    in zip(self.sparse_embs, sparse_feats)]           # each (B, 1, d)
        X = torch.cat([x_dense] + x_sparse, dim=1)                   # (B, n_ns, d)
        S = self.seq_masknet(seq_feat)                                # (B, T, d)
        return X, S

    def forward(
        self,
        dense_feat: Tensor,
        sparse_feats: list[Tensor],
        seq_feat: Tensor,
    ) -> Tensor:
        """
        Args:
            dense_feat:   (B, dense_dim)
            sparse_feats: list of (B,) integer tensors.
            seq_feat:     (B, T, seq_input_dim)
        Returns:
            logits: (B, num_tasks)
        """
        X, S = self._preprocess(dense_feat, sparse_feats, seq_feat)

        # Compute initial X_sum^(1) and prepend as CLS tokens (Algorithm 1, step 2-3)
        X_sum_init = self.init_cross.ns_summarize(X)        # (B, n_sum, d)
        S = torch.cat([X_sum_init, S], dim=1)               # (B, n_cls + T, d)

        # L InterFormer blocks — interleaving Interaction and Sequence Arches
        X_sum_last, S_sum_last = None, None
        for block in self.blocks:
            X, S, X_sum_last, S_sum_last = block(X, S)

        # Final prediction from last layer's summaries
        B = X.size(0)
        x_sum_flat = X_sum_last.reshape(B, -1)               # (B, n_sum*d)
        s_sum_flat = S_sum_last.reshape(B, -1)               # (B, s_sum_len*d)
        return self.classifier(torch.cat([x_sum_flat, s_sum_flat], dim=-1))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B, T = 4, 20

    dense_dim     = 32
    sparse_vocabs = [100, 200, 50]   # 3 sparse features
    seq_input_dim = 16               # per-step sequence embedding dim

    model = InterFormer(
        dense_dim=dense_dim,
        sparse_dims=sparse_vocabs,
        seq_input_dim=seq_input_dim,
        d_model=32,
        num_layers=3,
        n_sum=4,
        n_cls=4,
        n_pma=2,
        k_recent=2,
        n_heads=4,
        top_mlp_dims=[32],
        num_tasks=1,
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


if __name__ == "__main__":
    _smoke_test()
