"""Foundation-Expert Paradigm for Hyperscale Recommendation.

Reference PyTorch implementation.
Paper: https://arxiv.org/abs/2508.02929  (Meta AI, Aug 2025)
  "Realizing Scaling Laws in Recommender Systems:
   A Foundation-Expert Paradigm for Hyperscale Model Deployment"

Architecture overview:

  Stage 1 — Foundation Model (FM):                        (compute-heavy, cross-surface)

    ItemEmbedding (§3.1.1):
      Emb_x_i = f(Emb_p_i, Emb_c_i) + Emb_a_i     history items
      Emb_y_j = f(Emb_p_j, Emb_c_j)                candidate items (no action)
      Combining item + action via summation (not interleaving):
        → halves sequence length from 2(N+M) to (N+M)
        → 50% fewer linear projection FLOPs, 25% fewer attention FLOPs

    HSTU target-aware attention (§3.1.2):
      Input:  concat([hist_embs, cand_embs])  ∈ (B, N+M, d)
      Mask:   history → causal; candidates → attend to all history, not each other
      Output: target-aware embeddings (TAE) at candidate positions  ∈ (B, M, d)
      TAE captures user's contextual interest in each specific candidate item
      given their full interaction history and the item's own features.

    Loss (§3.1.3):
      L = Σ_s ω_s · L_main_s  +  Σ_t ω_t · L_aux_t
      L_main: generalizable cross-surface objectives (likes, shares, completions)
              applied directly to HSTU output via MultiTaskHead
      L_aux:  surface-specific alignment with auxiliary features
              L_aux_t = (1/Σδ_i^t) · Σ_i δ_i^t · loss_t(ŷ_i^t(θ_H, θ_aux_t), y_i^t)
              δ_i^t ∈ {0,1} indicates sample i is in task t's valid sample space

  Stage 2 — Expert Model (per surface):                   (lightweight, 20-40% FM compute)

    FMEmbeddingModule:  LayerNorm + Dropout → regularize and denoise TAE
    LightweightHSTU:    causal self-attention on short-term surface-specific history
    FMFusionModule:     concat(FM_emb, pooled_short_term) → MLP → fused
    ExpertFusionModule: concat(fused, surface_features)   → MLP → per-task logits

  Transfer Ratio (§4.3):
    TR = (NE(Expert_FM1) − NE(Expert_FM2)) / (NE(FM1) − NE(FM2))
    Measures fraction of FM scaling gain transferred to Expert.
    Achieved TR ∈ [0.64, 1.0] across surfaces/tasks at Meta scale.

Production (Meta): tens of billions daily requests; expert latency neutral vs. one-stage;
  data-to-trainer latency ~30 minutes via HyperCast streaming infrastructure.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Item Embedding   §3.1.1
# ---------------------------------------------------------------------------

class ItemEmbedding(nn.Module):
    """Combine product, contextual, and action features into a unified item token (§3.1.1).

    History items:   Emb_x = f(Emb_p, Emb_c) + Emb_a
    Candidate items: Emb_y = f(Emb_p, Emb_c)     (no action — target has no label yet)

    f is a two-layer MLP. Adding action via summation rather than appending to the sequence
    halves the effective sequence length, reducing projection FLOPs by 50% and attention
    FLOPs by 25% relative to interleaved item-action token designs.

    Args:
        prod_dim: product feature dimension (item ID, category, LLM repr, etc.).
        ctx_dim:  contextual feature dimension (timestamp, surface, query, etc.).
        act_dim:  action feature dimension (like, share, watch, etc.).
        d_model:  output embedding dimension.
    """

    def __init__(self, prod_dim: int, ctx_dim: int, act_dim: int,
                 d_model: int) -> None:
        super().__init__()
        self.item_proj = nn.Sequential(
            nn.Linear(prod_dim + ctx_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.act_proj = nn.Linear(act_dim, d_model, bias=False)

    def forward(self, prod: Tensor, ctx: Tensor,
                act: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            prod: (..., prod_dim)
            ctx:  (..., ctx_dim)
            act:  (..., act_dim) or None for candidate items.
        Returns:
            emb: (..., d_model)
        """
        emb = self.item_proj(torch.cat([prod, ctx], dim=-1))
        if act is not None:
            emb = emb + self.act_proj(act)
        return emb


# ---------------------------------------------------------------------------
# Target-Aware Attention Mask
# ---------------------------------------------------------------------------

def _build_tae_mask(n_hist: int, n_cand: int, device: torch.device) -> Tensor:
    """Build additive attention mask for target-aware HSTU (§3.1.2).

    - History → history: causal (lower-triangular). No future leakage within history.
    - Candidate → history: full attention. Each candidate sees the complete history,
      enabling target-aware embeddings that capture user interest in that specific item.
    - Candidate → candidate: diagonal only. Candidates are scored independently —
      a candidate's embedding must not depend on other candidates' ordering.
    - History → candidate: blocked. History cannot see future candidates.

    Returns:
        mask: (N+M, N+M) additive mask; 0.0 = attend, -inf = block.
    """
    L = n_hist + n_cand
    mask = torch.full((L, L), float('-inf'), device=device)
    # History: causal — 0 for j<=i (past/self), -inf for j>i (future)
    mask[:n_hist, :n_hist] = torch.triu(
        torch.full((n_hist, n_hist), float('-inf'), device=device), diagonal=1
    )
    # Candidates: full attention to all history
    mask[n_hist:, :n_hist] = 0.0
    # Candidates: self-only (diagonal)
    idx = torch.arange(n_cand, device=device)
    mask[n_hist + idx, n_hist + idx] = 0.0
    return mask


# ---------------------------------------------------------------------------
# HSTU Layer and Stack   §3.1.2
# ---------------------------------------------------------------------------

class HSTULayer(nn.Module):
    """Single HSTU transformer layer: pre-norm MHA + point-wise FFN (§3.1.2).

    Based on the HSTU architecture from Zhai et al. 2024 [37], simplified for
    this reference implementation using standard multi-head attention.
    RMSNorm pre-normalization for training stability in deep sequential models.

    Args:
        d_model:  embedding dimension.
        n_heads:  number of attention heads.
        ffn_dim:  FFN hidden dimension (default 4 × d_model).
        dropout:  dropout rate.
    """

    def __init__(self, d_model: int, n_heads: int,
                 ffn_dim: Optional[int] = None, dropout: float = 0.1) -> None:
        super().__init__()
        ffn_dim = ffn_dim or 4 * d_model
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim, bias=False),
            nn.SiLU(),
            nn.Linear(ffn_dim, d_model, bias=False),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:         (B, L, d_model)
            attn_mask: (L, L) additive mask; None = full bidirectional.
        Returns:
            (B, L, d_model)
        """
        n = self.norm1(x)
        attn_out, _ = self.attn(n, n, n, attn_mask=attn_mask)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class HSTU(nn.Module):
    """Hierarchical Sequential Transduction Unit stack (§3.1.2).

    Processes unified sequence of history + candidate item embeddings.
    Outputs target-aware embeddings (TAE) at candidate positions:
    TAE_j captures user's contextual interest in candidate j given full history.

    Args:
        d_model:  embedding dimension.
        n_layers: number of HSTU layers.
        n_heads:  attention heads per layer.
        ffn_dim:  FFN hidden dimension.
        dropout:  dropout rate.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int,
                 ffn_dim: Optional[int] = None, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            HSTULayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: Tensor, n_hist: int) -> Tensor:
        """
        Args:
            x:      (B, N+M, d_model)  history + candidate embeddings.
            n_hist: N, number of history positions.
        Returns:
            out: (B, N+M, d_model); TAE = out[:, n_hist:, :] for candidates.
        """
        n_cand = x.size(1) - n_hist
        mask = _build_tae_mask(n_hist, n_cand, x.device)
        for layer in self.layers:
            x = layer(x, attn_mask=mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Foundation Model components   §3.1.3
# ---------------------------------------------------------------------------

class MultiTaskHead(nn.Module):
    """Cross-surface main task prediction head for L_main (§3.1.3).

    Applied directly to HSTU output TAE. Generalizable objectives (likes,
    shares, video completions) shared across all surfaces.

    Args:
        d_model:    input embedding dimension.
        n_tasks:    number of cross-surface tasks.
        hidden_dim: MLP hidden dimension.
    """

    def __init__(self, d_model: int, n_tasks: int,
                 hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        h = hidden_dim or d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, h, bias=False),
            nn.ReLU(),
            nn.Linear(h, n_tasks, bias=False),
        )

    def forward(self, tae: Tensor) -> Tensor:
        """Args: tae (B, M, d). Returns: logits (B, M, n_tasks)."""
        return self.mlp(tae)


class AlignmentModule(nn.Module):
    """Surface-specific auxiliary alignment module for L_aux (§3.1.3).

    Aligns FM target-aware embeddings toward surface-specific objectives
    using auxiliary features (surface type, item metadata, etc.).
    Loss computed only over valid sample space per task via δ mask.

    L_aux_t = (1/Σδ_i^t) · Σ_i δ_i^t · loss_t(ŷ_i^t(θ_H, θ_aux), y_i^t)

    Args:
        d_model:   TAE embedding dimension.
        aux_dim:   auxiliary feature dimension.
        n_tasks:   number of surface-specific auxiliary tasks.
    """

    def __init__(self, d_model: int, aux_dim: int, n_tasks: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model + aux_dim, d_model, bias=False),
            nn.ReLU(),
            nn.Linear(d_model, n_tasks, bias=False),
        )

    def forward(self, tae: Tensor, aux_features: Tensor) -> Tensor:
        """
        Args:
            tae:          (B, M, d_model)
            aux_features: (B, M, aux_dim)
        Returns:
            logits: (B, M, n_tasks)
        """
        return self.mlp(torch.cat([tae, aux_features], dim=-1))

    @staticmethod
    def masked_loss(logits: Tensor, labels: Tensor,
                    delta: Optional[Tensor] = None) -> Tensor:
        """Compute L_aux_t over valid sample space defined by delta mask.

        Args:
            logits: (B, M, n_tasks)
            labels: (B, M, n_tasks)
            delta:  (B, M, n_tasks) bool or float; None = all valid.
        Returns:
            scalar loss averaged over valid samples.
        """
        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        if delta is not None:
            loss = loss * delta.float()
            return loss.sum() / (delta.float().sum().clamp(min=1.0))
        return loss.mean()


# ---------------------------------------------------------------------------
# Foundation Model   §3.1
# ---------------------------------------------------------------------------

class FoundationModel(nn.Module):
    """Foundation Model (FM): cross-surface HSTU generating target-aware embeddings (§3.1).

    Trained on lifelong, cross-surface user histories. Produces Target-Aware
    Embeddings (TAE) for each candidate item — representing the user's contextual
    interest in that specific item given their full interaction history.

    The FM is large and compute-heavy (HSTU-0.5B or HSTU-1B in production).
    Experts are lightweight (20-40% compute) and consume FM's TAE as features.

    Args:
        prod_dim:      product feature dimension (item ID, category, etc.).
        ctx_dim:       contextual feature dimension (time, surface, etc.).
        act_dim:       action feature dimension (like, share, watch, etc.).
        aux_dim:       auxiliary feature dimension for alignment module.
        d_model:       HSTU hidden dimension.
        n_layers:      HSTU depth.
        n_heads:       attention heads.
        n_main_tasks:  number of cross-surface main tasks.
        n_aux_tasks:   number of surface-specific auxiliary tasks.
        ffn_dim:       HSTU FFN hidden dimension (default 4×d_model).
        dropout:       dropout rate.
        main_weights:  per-main-task loss weights ω_s (default uniform).
        aux_weights:   per-aux-task loss weights ω_t (default uniform).
    """

    def __init__(
        self,
        prod_dim:     int,
        ctx_dim:      int,
        act_dim:      int,
        aux_dim:      int,
        d_model:      int,
        n_layers:     int,
        n_heads:      int,
        n_main_tasks: int,
        n_aux_tasks:  int,
        ffn_dim:      Optional[int] = None,
        dropout:      float = 0.1,
        main_weights: Optional[list[float]] = None,
        aux_weights:  Optional[list[float]] = None,
    ) -> None:
        super().__init__()
        self.n_main_tasks = n_main_tasks
        self.n_aux_tasks  = n_aux_tasks

        self.register_buffer(
            'main_w', torch.tensor(main_weights or [1.0] * n_main_tasks))
        self.register_buffer(
            'aux_w',  torch.tensor(aux_weights  or [1.0] * n_aux_tasks))

        self.item_emb  = ItemEmbedding(prod_dim, ctx_dim, act_dim, d_model)
        self.hstu      = HSTU(d_model, n_layers, n_heads, ffn_dim, dropout)
        self.main_head = MultiTaskHead(d_model, n_main_tasks)
        self.align     = AlignmentModule(d_model, aux_dim, n_aux_tasks)

    def forward(
        self,
        hist_prod:    Tensor,            # (B, N, prod_dim)
        hist_ctx:     Tensor,            # (B, N, ctx_dim)
        hist_act:     Tensor,            # (B, N, act_dim)
        cand_prod:    Tensor,            # (B, M, prod_dim)
        cand_ctx:     Tensor,            # (B, M, ctx_dim)
        aux_features: Tensor,            # (B, M, aux_dim)
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            hist_prod, hist_ctx, hist_act: history item features.
            cand_prod, cand_ctx:           candidate item features (no action).
            aux_features:                  surface-specific auxiliary features.
        Returns:
            tae:         (B, M, d_model)  target-aware embeddings for experts.
            main_logits: (B, M, n_main_tasks)
            aux_logits:  (B, M, n_aux_tasks)
        """
        B, N = hist_prod.shape[:2]
        M    = cand_prod.shape[1]

        # Build unified item sequence: history then candidates
        hist_embs = self.item_emb(hist_prod, hist_ctx, hist_act)  # (B, N, d)
        cand_embs = self.item_emb(cand_prod, cand_ctx)            # (B, M, d)
        seq       = torch.cat([hist_embs, cand_embs], dim=1)      # (B, N+M, d)

        # HSTU: target-aware attention
        out = self.hstu(seq, n_hist=N)                            # (B, N+M, d)
        tae = out[:, N:, :]                                       # (B, M, d)

        # Task heads
        main_logits = self.main_head(tae)                         # (B, M, n_main)
        aux_logits  = self.align(tae, aux_features)               # (B, M, n_aux)

        return tae, main_logits, aux_logits

    def compute_loss(
        self,
        main_logits: Tensor,              # (B, M, n_main_tasks)
        main_labels: Tensor,              # (B, M, n_main_tasks)
        aux_logits:  Tensor,              # (B, M, n_aux_tasks)
        aux_labels:  Tensor,              # (B, M, n_aux_tasks)
        aux_delta:   Optional[Tensor] = None,  # (B, M, n_aux_tasks) valid sample mask
    ) -> Tensor:
        """Compute L = Σ ω_s L_main_s + Σ ω_t L_aux_t (§3.1.3)."""
        # Main loss: per-task BCE, weighted sum
        main_loss = sum(
            self.main_w[s] * F.binary_cross_entropy_with_logits(
                main_logits[..., s], main_labels[..., s]
            )
            for s in range(self.n_main_tasks)
        )

        # Auxiliary loss: masked BCE per task
        aux_loss = sum(
            self.aux_w[t] * AlignmentModule.masked_loss(
                aux_logits[..., t:t+1],
                aux_labels[..., t:t+1],
                aux_delta[..., t:t+1] if aux_delta is not None else None,
            )
            for t in range(self.n_aux_tasks)
        )

        return main_loss + aux_loss


# ---------------------------------------------------------------------------
# Expert Model components   §3.2
# ---------------------------------------------------------------------------

class FMEmbeddingModule(nn.Module):
    """Preprocess FM target-aware embeddings: regularization and denoising (§3.2).

    The FM and expert operate in separate training loops with different data
    distributions. FMEmbeddingModule bridges this gap via LayerNorm (aligns
    scale/mean) and Dropout (regularization against FM distribution shift).

    Optionally projects FM dimension to a smaller expert dimension to reduce
    expert compute.

    Args:
        fm_dim:     FM target-aware embedding dimension.
        expert_dim: expert hidden dimension (= fm_dim if no projection needed).
        dropout:    dropout rate for regularization.
    """

    def __init__(self, fm_dim: int, expert_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(fm_dim)
        self.drop = nn.Dropout(dropout)
        self.proj = (nn.Linear(fm_dim, expert_dim, bias=False)
                     if fm_dim != expert_dim else nn.Identity())

    def forward(self, tae: Tensor) -> Tensor:
        """Args: tae (B, M, fm_dim). Returns: (B, M, expert_dim)."""
        return self.proj(self.drop(self.norm(tae)))


class FMFusionModule(nn.Module):
    """Fuse FM embeddings (long-term) with lightweight HSTU output (short-term) (§3.2).

    FM captures generalized, lifelong cross-surface knowledge.
    Lightweight HSTU captures recent, surface-specific behavioral patterns.
    Fusion via concatenation + MLP combines complementary signals.

        fused = MLP(concat(fm_emb, short_term_rep))

    Args:
        expert_dim: FM embedding projected dimension.
        short_dim:  lightweight HSTU output dimension.
        out_dim:    fused output dimension.
    """

    def __init__(self, expert_dim: int, short_dim: int,
                 out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(expert_dim + short_dim, out_dim, bias=False),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim, bias=False),
        )

    def forward(self, fm_emb: Tensor, short_rep: Tensor) -> Tensor:
        """
        Args:
            fm_emb:    (B, M, expert_dim)
            short_rep: (B, short_dim) mean-pooled short-term representation.
        Returns:
            fused: (B, M, out_dim)
        """
        M = fm_emb.size(1)
        short_exp = short_rep.unsqueeze(1).expand(-1, M, -1)  # (B, M, short_dim)
        return self.mlp(torch.cat([fm_emb, short_exp], dim=-1))


class ExpertFusionModule(nn.Module):
    """Surface-specific predictions from fused representation + surface features (§3.2).

    Flexible final module: simple MLP baseline (used in the paper for robustness),
    extensible to more advanced structures per surface needs.

    Args:
        fused_dim:  fused FM+HSTU representation dimension.
        surf_dim:   surface-specific feature dimension.
        n_tasks:    number of surface-specific prediction tasks.
        hidden_dim: MLP hidden dimension.
    """

    def __init__(self, fused_dim: int, surf_dim: int, n_tasks: int,
                 hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        h = hidden_dim or fused_dim
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim + surf_dim, h, bias=False),
            nn.ReLU(),
            nn.Linear(h, n_tasks, bias=False),
        )

    def forward(self, fused: Tensor, surf_feats: Tensor) -> Tensor:
        """
        Args:
            fused:      (B, M, fused_dim)
            surf_feats: (B, M, surf_dim)
        Returns:
            logits: (B, M, n_tasks)
        """
        return self.mlp(torch.cat([fused, surf_feats], dim=-1))


# ---------------------------------------------------------------------------
# Expert Model   §3.2
# ---------------------------------------------------------------------------

class ExpertModel(nn.Module):
    """Lightweight surface-specific expert model in the Foundation-Expert paradigm (§3.2).

    Replaces the traditional one-stage per-surface model. By offloading general
    knowledge acquisition to the FM, experts need only 20-40% of equivalent
    one-stage compute while achieving superior quality through FM transfer.

    Data flow:
      tae            → FMEmbeddingModule    → fm_processed  (B, M, expert_dim)
      short_history  → LightweightHSTU     → short_term     (B, T_short, d_short)
                                           → mean pool       (B, d_short)
      fm_processed + short_term → FMFusionModule → fused     (B, M, out_dim)
      fused + surface_features  → ExpertFusionModule → logits (B, M, n_tasks)

    Args:
        fm_dim:       FM target-aware embedding dimension.
        prod_dim:     product feature dim for short-term history.
        ctx_dim:      contextual feature dim for short-term history.
        act_dim:      action feature dim for short-term history.
        surf_dim:     surface-specific feature dimension.
        d_expert:     expert hidden dimension.
        n_layers:     lightweight HSTU layers (typically 1-2).
        n_heads:      attention heads.
        n_tasks:      number of surface-specific prediction tasks.
        dropout:      dropout rate.
    """

    def __init__(
        self,
        fm_dim:    int,
        prod_dim:  int,
        ctx_dim:   int,
        act_dim:   int,
        surf_dim:  int,
        d_expert:  int,
        n_layers:  int,
        n_heads:   int,
        n_tasks:   int,
        dropout:   float = 0.0,
    ) -> None:
        super().__init__()

        # FM embedding preprocessing
        self.fm_emb_module = FMEmbeddingModule(fm_dim, d_expert, dropout)

        # Short-term surface-specific history encoding
        self.item_emb   = ItemEmbedding(prod_dim, ctx_dim, act_dim, d_expert)
        self.light_hstu = HSTU(d_expert, n_layers, n_heads, dropout=dropout)

        # Fusion
        self.fm_fusion      = FMFusionModule(d_expert, d_expert, d_expert)
        self.expert_fusion  = ExpertFusionModule(d_expert, surf_dim, n_tasks)

    def forward(
        self,
        tae:            Tensor,   # (B, M, fm_dim)  from FoundationModel
        short_prod:     Tensor,   # (B, T, prod_dim) short-term history
        short_ctx:      Tensor,   # (B, T, ctx_dim)
        short_act:      Tensor,   # (B, T, act_dim)
        surface_feats:  Tensor,   # (B, M, surf_dim)
    ) -> Tensor:                  # (B, M, n_tasks)
        """
        Args:
            tae:           target-aware embeddings from the FM.
            short_{prod,ctx,act}: recent user history on this surface.
            surface_feats: surface-specific features per candidate.
        Returns:
            logits: (B, M, n_tasks)
        """
        # FM embedding preprocessing
        fm_processed = self.fm_emb_module(tae)              # (B, M, d_expert)

        # Short-term surface-specific encoding
        short_embs  = self.item_emb(short_prod, short_ctx, short_act)  # (B, T, d)
        T = short_embs.size(1)
        short_out   = self.light_hstu(short_embs, n_hist=T) # (B, T, d_expert) — full causal
        short_rep   = short_out.mean(dim=1)                 # (B, d_expert)

        # Fuse long-term (FM) + short-term (lightweight HSTU)
        fused  = self.fm_fusion(fm_processed, short_rep)    # (B, M, d_expert)

        # Surface-specific prediction
        return self.expert_fusion(fused, surface_feats)     # (B, M, n_tasks)


# ---------------------------------------------------------------------------
# Transfer Ratio utility   §4.3
# ---------------------------------------------------------------------------

def transfer_ratio(ne_expert_fm1: float, ne_expert_fm2: float,
                   ne_fm1: float, ne_fm2: float) -> float:
    """Transfer Ratio: fraction of FM scaling gain inherited by Expert (§4.3).

    TR = (NE(Expert_FM1) − NE(Expert_FM2)) / (NE(FM1) − NE(FM2))

    TR = 1.0: expert fully inherits FM improvement.
    TR > 1.0: possible when FM improves cross-surface features benefiting expert's
              local objectives through higher-order interactions (theoretically possible).
    TR in [0.64, 1.0] achieved across surfaces/tasks at Meta production scale.

    Lower NE is better (NE = Normalized Entropy, cross-entropy normalized by base rate).

    Args:
        ne_expert_fm1: NE of expert using stronger FM1.
        ne_expert_fm2: NE of expert using weaker FM2.
        ne_fm1:        NE of FM1 (used as teacher/embedding source).
        ne_fm2:        NE of FM2 (baseline FM).
    Returns:
        Transfer ratio TR.
    """
    fm_delta = ne_fm1 - ne_fm2
    if abs(fm_delta) < 1e-10:
        raise ValueError("FM NE delta is near zero; cannot compute transfer ratio.")
    return (ne_expert_fm1 - ne_expert_fm2) / fm_delta


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B, N, M, T_short = 4, 32, 8, 16

    # Feature dimensions
    prod_dim, ctx_dim, act_dim = 16, 8, 4
    aux_dim, surf_dim = 12, 10

    # --- Foundation Model ---
    fm = FoundationModel(
        prod_dim     = prod_dim,
        ctx_dim      = ctx_dim,
        act_dim      = act_dim,
        aux_dim      = aux_dim,
        d_model      = 32,
        n_layers     = 2,
        n_heads      = 4,
        n_main_tasks = 3,   # like, share, video_complete
        n_aux_tasks  = 2,   # surface-specific aux tasks
    )

    hist_prod = torch.randn(B, N, prod_dim)
    hist_ctx  = torch.randn(B, N, ctx_dim)
    hist_act  = torch.randn(B, N, act_dim)
    cand_prod = torch.randn(B, M, prod_dim)
    cand_ctx  = torch.randn(B, M, ctx_dim)
    aux_feats = torch.randn(B, M, aux_dim)

    tae, main_logits, aux_logits = fm(
        hist_prod, hist_ctx, hist_act, cand_prod, cand_ctx, aux_feats)
    print(f"FM TAE:         {tae.shape}")
    print(f"FM main logits: {main_logits.shape}")
    print(f"FM aux logits:  {aux_logits.shape}")

    main_labels = torch.zeros_like(main_logits)
    aux_labels  = torch.zeros_like(aux_logits)
    delta       = torch.randint(0, 2, aux_logits.shape).float()

    fm_loss = fm.compute_loss(main_logits, main_labels, aux_logits, aux_labels, delta)
    fm_loss.backward()
    print(f"FM loss:        {fm_loss.item():.4f}")
    print("FM backward:    OK")

    fm_params = sum(p.numel() for p in fm.parameters())
    print(f"FM params:      {fm_params:,}")

    # --- Expert Model ---
    expert = ExpertModel(
        fm_dim    = 32,   # must match FM d_model
        prod_dim  = prod_dim,
        ctx_dim   = ctx_dim,
        act_dim   = act_dim,
        surf_dim  = surf_dim,
        d_expert  = 16,   # experts are smaller
        n_layers  = 1,
        n_heads   = 4,
        n_tasks   = 2,
    )

    short_prod  = torch.randn(B, T_short, prod_dim)
    short_ctx   = torch.randn(B, T_short, ctx_dim)
    short_act   = torch.randn(B, T_short, act_dim)
    surf_feats  = torch.randn(B, M, surf_dim)

    with torch.no_grad():
        tae_detached = tae.detach()

    expert_logits = expert(tae_detached, short_prod, short_ctx, short_act, surf_feats)
    print(f"\nExpert logits:  {expert_logits.shape}")

    expert_loss = F.binary_cross_entropy_with_logits(
        expert_logits, torch.zeros_like(expert_logits))
    expert_loss.backward()
    print(f"Expert loss:    {expert_loss.item():.4f}")
    print("Expert backward: OK")

    expert_params = sum(p.numel() for p in expert.parameters())
    print(f"Expert params:  {expert_params:,}")
    ratio = expert_params / fm_params
    print(f"Expert/FM ratio: {ratio:.1%}  (paper: 20-40%)")

    # --- Transfer Ratio demo ---
    tr = transfer_ratio(
        ne_expert_fm1=-2.14, ne_expert_fm2=0.0,
        ne_fm1=-2.13, ne_fm2=0.0,
    )
    print(f"\nTransfer Ratio: {tr:.4f}  (paper range: [0.64, 1.0])")

    print("\nFoundation-Expert paradigm:")
    print("  FM:     generalizable, lifelong, cross-surface → TAE")
    print("  Expert: lightweight, surface-specific → surface predictions")
    print("  TAE transfers FM scaling gains to experts without joint training")


if __name__ == "__main__":
    _smoke_test()
