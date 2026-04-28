"""InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction.

Reference: Zeng et al., arXiv:2411.09852

Architecture overview:
  Each InterFormerBlock contains three parallel arches that interleave:
  - Interaction Arch  : behavior-aware non-sequence feature interaction
  - Sequence Arch     : context-aware sequence modelling
  - Cross Arch        : information selection & summarization between arches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Feature preprocessing
# ---------------------------------------------------------------------------


class MaskNet(nn.Module):
    """Unify k behavior sequences via self-masking (Section 4.1, eq. 6).

    MaskNet(S) = MLP_ice(S ⊙ MLP_mask(S))
    """

    def __init__(self, k: int, d: int) -> None:
        super().__init__()
        self.mlp_mask = nn.Linear(k * d, k * d)
        self.mlp_ice = nn.Linear(k * d, d)

    def forward(self, seqs: List[torch.Tensor]) -> torch.Tensor:
        """seqs: list of k tensors [B, T, d] → [B, T, d]"""
        S = torch.cat(seqs, dim=-1)           # [B, T, k*d]
        mask = torch.sigmoid(self.mlp_mask(S))
        return self.mlp_ice(S * mask)          # [B, T, d]


# ---------------------------------------------------------------------------
# Cross Arch helpers
# ---------------------------------------------------------------------------


class PoolingByMHA(nn.Module):
    """PMA: summarise sequence with learnable seed queries (Section 3.3, eq. 4)."""

    def __init__(self, d: int, n_heads: int, n_seeds: int) -> None:
        super().__init__()
        self.Q = nn.Parameter(torch.empty(n_seeds, d))
        nn.init.normal_(self.Q, std=0.02)
        self.mha = nn.MultiheadAttention(d, n_heads, batch_first=True)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """S: [B, T, d] → [B, n_seeds, d]"""
        B = S.size(0)
        Q = self.Q.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.mha(Q, S, S)
        return out


class CrossArch(nn.Module):
    """Selective summarization for both non-sequence and sequence modes (Section 4.4).

    Produces:
      X_sum = Gating(MLP(X))      — compact non-sequence context for Sequence Arch
      S_sum = Gating([S_CLS ‖ S_PMA ‖ S_recent])  — compact seq context for Interaction Arch

    where Gating(X) = σ(X ⊙ MLP(X))  (eq. 10)
    """

    def __init__(
        self,
        n_nonseq: int,
        d: int,
        n_sum_x: int,
        n_heads: int,
        n_pma_seeds: int,
        n_recent: int,
    ) -> None:
        super().__init__()
        self.n_sum_x = n_sum_x
        self.n_pma_seeds = n_pma_seeds
        self.n_recent = n_recent
        self.d = d

        # Non-sequence: compress n feature tokens → n_sum_x tokens
        self.x_proj = nn.Linear(n_nonseq * d, n_sum_x * d)
        self.x_gate = nn.Linear(n_nonseq * d, n_sum_x * d)

        # Sequence: PMA over sequence body
        self.pma = PoolingByMHA(d, n_heads, n_pma_seeds)
        # Gating for sequence summary tokens
        self.s_gate = nn.Linear(d, d)

    def get_x_sum(self, X: torch.Tensor) -> torch.Tensor:
        """X: [B, n_nonseq, d] → X_sum: [B, n_sum_x, d]"""
        B = X.size(0)
        x_flat = X.reshape(B, -1)                                      # [B, n*d]
        z = self.x_proj(x_flat).reshape(B, self.n_sum_x, self.d)       # [B, n_sum_x, d]
        gate = torch.sigmoid(self.x_gate(x_flat).reshape(B, self.n_sum_x, self.d))
        return gate * z

    def get_s_sum(self, S: torch.Tensor) -> torch.Tensor:
        """S: [B, T+1, d] with CLS at index 0 → S_sum: [B, n_sum_seq, d]"""
        s_cls = S[:, :1, :]                      # [B, 1, d]
        s_body = S[:, 1:, :]                     # [B, T, d]
        s_pma = self.pma(s_body)                 # [B, n_pma_seeds, d]
        s_recent = s_body[:, -self.n_recent:, :] # [B, n_recent, d]
        s_cat = torch.cat([s_cls, s_pma, s_recent], dim=1)  # [B, 1+n_pma+n_recent, d]
        gate = torch.sigmoid(self.s_gate(s_cat))
        return gate * s_cat


# ---------------------------------------------------------------------------
# Sequence Arch
# ---------------------------------------------------------------------------


class PFFN(nn.Module):
    """Personalised FFN: f(X_sum) * S  (Section 4.3, eq. 8).

    Modulates sequence token embeddings with non-sequence context, making
    attention context-aware of static user profile and item features.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())

    def forward(self, x_sum: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """x_sum: [B, n_sum_x, d], S: [B, T+1, d] → [B, T+1, d]"""
        ctx = x_sum.mean(dim=1, keepdim=True)  # [B, 1, d]
        scale = self.mlp(ctx)                  # [B, 1, d]
        return scale * S


class SequenceArch(nn.Module):
    """Context-aware sequence modelling (Section 4.3, eq. 9).

    S^(l+1) = MHA(PFFN(X_sum^(l), S^(l)))
    """

    def __init__(self, d: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.pffn = PFFN(d)
        self.mha = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_sum: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """x_sum: [B, n_sum_x, d], S: [B, T+1, d] → [B, T+1, d]"""
        S2 = self.pffn(x_sum, S)
        attn_out, _ = self.mha(S2, S2, S2)
        return self.norm(S + self.dropout(attn_out))


# ---------------------------------------------------------------------------
# Interaction Arch
# ---------------------------------------------------------------------------


class DCNv2Cross(nn.Module):
    """DCNv2 cross layer: x^(l+1) = x^(0) ⊙ (W x^(l) + b) + x^(l)  (Section 3.2)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.w = nn.Linear(d, d)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.w(xl) + xl


class InteractionArch(nn.Module):
    """Behavior-aware non-sequence feature interaction (Section 4.2, eq. 7).

    X^(l+1) = MLP(Interaction([X^(l) ‖ S_sum^(l)]))

    Uses DCNv2 as the interaction module by default; other backbones (inner
    product, DHEN) can be swapped in by subclassing.
    """

    def __init__(
        self,
        n_nonseq: int,
        d: int,
        n_sum_seq: int,
        n_cross_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_in = (n_nonseq + n_sum_seq) * d
        d_out = n_nonseq * d

        self.cross_layers = nn.ModuleList(
            [DCNv2Cross(d_in) for _ in range(n_cross_layers)]
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_in, d_out),
        )
        self.norm = nn.LayerNorm(d)
        self.n_nonseq = n_nonseq
        self.d = d

    def forward(self, X: torch.Tensor, s_sum: torch.Tensor) -> torch.Tensor:
        """X: [B, n_nonseq, d], s_sum: [B, n_sum_seq, d] → [B, n_nonseq, d]"""
        B = X.size(0)
        combined = torch.cat([X, s_sum], dim=1).reshape(B, -1)  # [B, (n+n_sum)*d]

        x0 = xl = combined
        for cross in self.cross_layers:
            xl = cross(x0, xl)

        out = self.mlp(xl).reshape(B, self.n_nonseq, self.d)
        return self.norm(out + X)


# ---------------------------------------------------------------------------
# InterFormer Block (one interleaving layer)
# ---------------------------------------------------------------------------


class InterFormerBlock(nn.Module):
    """Single InterFormer block: one step of interleaved non-seq / seq learning."""

    def __init__(
        self,
        n_nonseq: int,
        d: int,
        n_sum_x: int,
        n_heads: int,
        n_pma_seeds: int,
        n_recent: int,
        n_cross_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        n_sum_seq = 1 + n_pma_seeds + n_recent

        self.cross = CrossArch(
            n_nonseq=n_nonseq,
            d=d,
            n_sum_x=n_sum_x,
            n_heads=n_heads,
            n_pma_seeds=n_pma_seeds,
            n_recent=n_recent,
        )
        self.interaction = InteractionArch(
            n_nonseq=n_nonseq,
            d=d,
            n_sum_seq=n_sum_seq,
            n_cross_layers=n_cross_layers,
            dropout=dropout,
        )
        self.sequence = SequenceArch(d=d, n_heads=n_heads, dropout=dropout)

    def forward(
        self, X: torch.Tensor, S: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """X: [B, n_nonseq, d], S: [B, T+1, d] → updated (X, S)"""
        x_sum = self.cross.get_x_sum(X)  # [B, n_sum_x, d]
        s_sum = self.cross.get_s_sum(S)  # [B, n_sum_seq, d]
        X_new = self.interaction(X, s_sum)
        S_new = self.sequence(x_sum, S)
        return X_new, S_new


# ---------------------------------------------------------------------------
# Full InterFormer Model
# ---------------------------------------------------------------------------


class InterFormer(nn.Module):
    """InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction.

    Args:
        n_dense:        Number of continuous (dense) non-sequence features.
        vocab_sizes:    Vocabulary size for each sparse (categorical) feature.
        n_seq_items:    Item vocabulary size for sequence embeddings.
        d:              Embedding / hidden dimension.
        T:              Sequence length (number of historical interactions).
        k:              Number of behavior sequences (unified via MaskNet when k > 1).
        n_layers:       Number of stacked InterFormer blocks.
        n_sum_x:        Number of non-sequence summary tokens exchanged to Sequence Arch.
        n_heads:        Number of attention heads in MHA and PMA.
        n_pma_seeds:    Number of learnable PMA query tokens for sequence summarization.
        n_recent:       Number of most-recent tokens included in sequence summary.
        n_cross_layers: DCNv2 cross layers inside each Interaction Arch.
        dropout:        Dropout probability.

    Inputs:
        dense_feats:  [B, n_dense]        float  – age, price, ...
        sparse_feats: [B, n_sparse]       long   – user_id, item_id, category, ...
        sequences:    [B, T] or [B, k, T] long   – item-id interaction history (0 = pad)

    Output:
        logits: [B, 1]  (apply torch.sigmoid for click probability)
    """

    def __init__(
        self,
        n_dense: int,
        vocab_sizes: List[int],
        n_seq_items: int,
        d: int = 64,
        T: int = 50,
        k: int = 1,
        n_layers: int = 3,
        n_sum_x: int = 4,
        n_heads: int = 4,
        n_pma_seeds: int = 4,
        n_recent: int = 5,
        n_cross_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d = d
        self.T = T
        self.k = k

        # --- Non-sequence feature encoders ---
        self.dense_proj = nn.Linear(n_dense, d) if n_dense > 0 else None
        self.sparse_embeds = nn.ModuleList(
            [nn.Embedding(v, d) for v in vocab_sizes]
        )
        n_sparse = len(vocab_sizes)
        n_nonseq = (1 if n_dense > 0 else 0) + n_sparse

        # --- Sequence feature encoder ---
        self.seq_embed = nn.Embedding(n_seq_items + 1, d, padding_idx=0)
        self.pos_embed = nn.Embedding(T + 2, d)  # positions 1..T for seq, 0 for CLS
        if k > 1:
            self.masknet = MaskNet(k, d)

        # --- InterFormer blocks ---
        self.blocks = nn.ModuleList(
            [
                InterFormerBlock(
                    n_nonseq=n_nonseq,
                    d=d,
                    n_sum_x=n_sum_x,
                    n_heads=n_heads,
                    n_pma_seeds=n_pma_seeds,
                    n_recent=n_recent,
                    n_cross_layers=n_cross_layers,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # --- Classifier head ---
        n_sum_seq = 1 + n_pma_seeds + n_recent
        head_dim = n_nonseq * d + n_sum_seq * d
        self.head = nn.Sequential(
            nn.Linear(head_dim, head_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim // 2, 1),
        )

        self.n_nonseq = n_nonseq
        self.n_sum_seq = n_sum_seq
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def _encode_features(
        self,
        dense_feats: Optional[torch.Tensor],
        sparse_feats: Optional[torch.Tensor],
        sequences: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens: List[torch.Tensor] = []

        if dense_feats is not None and self.dense_proj is not None:
            tokens.append(self.dense_proj(dense_feats).unsqueeze(1))  # [B, 1, d]

        if sparse_feats is not None:
            for i, emb in enumerate(self.sparse_embeds):
                tokens.append(emb(sparse_feats[:, i]).unsqueeze(1))   # [B, 1, d]

        X = torch.cat(tokens, dim=1)  # [B, n_nonseq, d]

        if sequences.dim() == 2:
            sequences = sequences.unsqueeze(1)  # [B, 1, T]

        seq_list = [self.seq_embed(sequences[:, ki]) for ki in range(self.k)]
        S = self.masknet(seq_list) if self.k > 1 else seq_list[0]  # [B, T, d]

        pos_ids = torch.arange(1, self.T + 1, device=S.device).unsqueeze(0)
        S = S + self.pos_embed(pos_ids)

        return X, S

    def forward(
        self,
        dense_feats: Optional[torch.Tensor] = None,
        sparse_feats: Optional[torch.Tensor] = None,
        sequences: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert sequences is not None, "sequences is required"
        X, S = self._encode_features(dense_feats, sparse_feats, sequences)
        # X: [B, n_nonseq, d],  S: [B, T, d]

        # Prepend X_sum^(1) as CLS token at the first layer (Section 4.3)
        x_sum_init = self.blocks[0].cross.get_x_sum(X)     # [B, n_sum_x, d]
        cls = x_sum_init[:, :1, :]                          # [B, 1, d]
        cls = cls + self.pos_embed(
            torch.zeros(1, 1, dtype=torch.long, device=X.device)
        )
        S = torch.cat([cls, S], dim=1)  # [B, T+1, d]

        for block in self.blocks:
            X, S = block(X, S)

        # Aggregate for final prediction
        s_sum = self.blocks[-1].cross.get_s_sum(S)  # [B, n_sum_seq, d]
        B = X.size(0)
        h = torch.cat([X.reshape(B, -1), s_sum.reshape(B, -1)], dim=1)
        return self.head(h)  # [B, 1]


def binary_cross_entropy_loss(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Standard CTR training loss."""
    return F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
