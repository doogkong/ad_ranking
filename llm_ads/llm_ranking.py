"""LLM-Enriched Ad Ranking with Predictability Regularization.

Extends the Foundation-Expert paradigm (arXiv 2508.02929) with LLM-generated
semantic features from "LLM Retrieval for Stable and Predictable Ad Recommendations"
(arXiv 2605.21969v1) to improve ranking quality and system predictability.

Key extensions over the baseline Foundation-Expert model:

  1. SemanticItemEmbedding  (§3 of 2605.21969v1 + §3.1.1 of 2508.02929):
       Emb_x = f(prod, ctx, llm_cat, llm_emb) + act
       LLM category logits and caption embeddings replace / augment raw ad IDs,
       eliminating cold-start sensitivity and improving semantic coverage.

  2. ConsistencyRegularizer  (§2, §5 of 2605.21969v1):
       L_consistency = (1/P) Σ_p ||TAE(ad_p) - TAE(ad_p')||²
       For LLM-identified semantic variant pairs (a, a'), penalise TAE divergence.
       This is the ranking-stage equivalent of the retrieval A/A' predictability metric.

  3. LLMEnrichedFoundationModel  (§3.1 of 2508.02929 + §4.1 of 2605.21969v1):
       FM inputs now include (prod, ctx, act, llm_cat_logits, llm_caption_emb).
       FM training loss = L_main + L_aux + λ * L_consistency

  4. LLMEnrichedExpertModel  (§3.2 of 2508.02929 + §4.4 of 2605.21969v1):
       Expert surface features augmented with LLM-derived ad similarity scores
       and category membership vectors — signals that improve cold-start ranking.

  5. PredictabilityAwareLoss:
       L_total = Σ_s ω_s L_main_s + Σ_t ω_t L_aux_t
                + λ L_consistency + μ L_semantic_alignment
       L_semantic_alignment: supervise category head with LLM-extracted labels
       so the ranking model's internal representation tracks LLM semantics.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Reuse Foundation-Expert components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'foundation_expert'))
from foundation_expert import (
    HSTU,
    AlignmentModule,
    FMEmbeddingModule,
    FMFusionModule,
    ExpertFusionModule,
    MultiTaskHead,
    _build_tae_mask,
)


# ---------------------------------------------------------------------------
# LLM Feature Encoder   §4.1 of 2605.21969v1
# ---------------------------------------------------------------------------

class LLMFeatureEncoder(nn.Module):
    """Project LLM-extracted semantic attributes into a dense ranking feature vector.

    Takes two LLM outputs:
      - category_logits: soft scores over the category vocabulary (from AdSemanticExtractor)
      - caption_emb:     L2-normalised dense LLM caption embedding

    Projects both to d_out and combines them for use as additional item features.

    Args:
        n_categories:   category vocabulary size.
        d_llm:          LLM caption embedding dimension.
        d_out:          output dimension (must match prod_dim or be concatenated).
    """

    def __init__(self, n_categories: int, d_llm: int, d_out: int) -> None:
        super().__init__()
        self.cat_proj     = nn.Linear(n_categories, d_out, bias=False)
        self.caption_proj = nn.Linear(d_llm, d_out, bias=False)
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_out, d_out, bias=False),
            nn.ReLU(),
        )

    def forward(self, category_logits: Tensor, caption_emb: Tensor) -> Tensor:
        """
        Args:
            category_logits: (..., n_categories)  raw logits or sigmoid scores.
            caption_emb:     (..., d_llm)          L2-normalised LLM embedding.
        Returns:
            llm_feat: (..., d_out)
        """
        cat_feat = self.cat_proj(torch.sigmoid(category_logits))
        cap_feat = self.caption_proj(caption_emb)
        return self.fusion(torch.cat([cat_feat, cap_feat], dim=-1))


# ---------------------------------------------------------------------------
# SemanticItemEmbedding   §3.1.1 of 2508.02929 + §4.1 of 2605.21969v1
# ---------------------------------------------------------------------------

class SemanticItemEmbedding(nn.Module):
    """Item embedding enriched with LLM semantic features (§3.1.1 + §4.1).

    Extends the Foundation-Expert ItemEmbedding:
      Emb_x = f(prod, ctx, llm_feat) + act     (history items)
      Emb_y = f(prod, ctx, llm_feat)            (candidate items, no action)

    LLM features (category logits + caption embeddings) replace or augment
    raw ad IDs, providing:
      - Semantic awareness: ads with similar content cluster in embedding space
      - Cold-start robustness: new ads with no history get meaningful embeddings
        from their LLM semantic attributes alone

    Args:
        prod_dim:     product feature dimension (may include raw ad ID embedding).
        ctx_dim:      contextual feature dimension.
        act_dim:      action feature dimension.
        n_categories: LLM category vocabulary size.
        d_llm:        LLM caption embedding dimension.
        d_model:      output embedding dimension.
    """

    def __init__(
        self,
        prod_dim:     int,
        ctx_dim:      int,
        act_dim:      int,
        n_categories: int,
        d_llm:        int,
        d_model:      int,
    ) -> None:
        super().__init__()
        self.llm_encoder = LLMFeatureEncoder(n_categories, d_llm, d_model)
        # Item projection takes: prod + ctx + llm_feat
        self.item_proj = nn.Sequential(
            nn.Linear(prod_dim + ctx_dim + d_model, d_model, bias=False),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.act_proj = nn.Linear(act_dim, d_model, bias=False)

    def forward(
        self,
        prod:             Tensor,            # (..., prod_dim)
        ctx:              Tensor,            # (..., ctx_dim)
        category_logits:  Tensor,            # (..., n_categories)
        caption_emb:      Tensor,            # (..., d_llm)
        act:              Optional[Tensor] = None,  # (..., act_dim)
    ) -> Tensor:                             # (..., d_model)
        """Fuse product, context, and LLM features; optionally add action."""
        llm_feat = self.llm_encoder(category_logits, caption_emb)
        emb = self.item_proj(torch.cat([prod, ctx, llm_feat], dim=-1))
        if act is not None:
            emb = emb + self.act_proj(act)
        return emb


# ---------------------------------------------------------------------------
# ConsistencyRegularizer   §2 of 2605.21969v1
# ---------------------------------------------------------------------------

class ConsistencyRegularizer(nn.Module):
    """Predictability regularization: penalise TAE divergence for semantic variant pairs (§2).

    For LLM-identified A/A' pairs (ad, semantic_variant_ad) that differ only
    in minor creative details (copy, image ID), their Target-Aware Embeddings
    should be similar. This is the ranking analogue of the retrieval A/A' metric.

        L_consistency = (1/P) Σ_p w_p ||TAE(ad_p) - TAE(ad_p')||²

    where w_p ∈ [0, 1] is the LLM-derived semantic similarity between the pair
    (high similarity → stronger consistency requirement).

    Args:
        reduction: 'mean' or 'sum' over the pair batch.
    """

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        tae_primary: Tensor,   # (P, M, d)  TAE for primary ads
        tae_shadow:  Tensor,   # (P, M, d)  TAE for shadow/variant ads
        weights:     Optional[Tensor] = None,  # (P,) per-pair semantic similarity
    ) -> Tensor:
        """
        Args:
            tae_primary: (P, M, d)  TAE from FM for P primary ads, M candidates each.
            tae_shadow:  (P, M, d)  TAE from FM for P shadow ads (same M candidates).
            weights:     (P,) LLM similarity scores; None = uniform weight.
        Returns:
            scalar consistency loss.
        """
        diff = tae_primary - tae_shadow.detach()        # shadow TAE is reference, no grad
        loss_per_pair = (diff ** 2).mean(dim=(-2, -1))  # (P,) — mean over M and d
        if weights is not None:
            loss_per_pair = loss_per_pair * weights
        if self.reduction == 'mean':
            return loss_per_pair.mean()
        return loss_per_pair.sum()


# ---------------------------------------------------------------------------
# LLMEnrichedFoundationModel   §3.1 + §4.1 + §2
# ---------------------------------------------------------------------------

class LLMEnrichedFoundationModel(nn.Module):
    """Foundation Model with LLM semantic features + predictability regularization.

    Extends Foundation-Expert FM (§3.1 of arXiv 2508.02929):
      - ItemEmbedding now includes LLM category logits + caption embeddings
      - Training adds consistency loss for A/A' semantic variant pairs
      - Category supervision auxiliary loss aligns internal representations with LLM

    Args:
        prod_dim:          product feature dimension.
        ctx_dim:           contextual feature dimension.
        act_dim:           action feature dimension.
        aux_dim:           auxiliary alignment feature dimension.
        n_categories:      LLM category vocabulary size.
        d_llm:             LLM caption embedding dimension.
        d_model:           HSTU hidden dimension.
        n_layers:          HSTU depth.
        n_heads:           attention heads.
        n_main_tasks:      number of cross-surface main tasks.
        n_aux_tasks:       number of surface-specific auxiliary tasks.
        consistency_weight: λ — weight for consistency regularization loss.
        semantic_weight:   μ — weight for LLM category alignment loss.
        ffn_dim:           HSTU FFN hidden dimension.
        dropout:           dropout rate.
        main_weights:      per-main-task loss weights.
        aux_weights:       per-aux-task loss weights.
    """

    def __init__(
        self,
        prod_dim:           int,
        ctx_dim:            int,
        act_dim:            int,
        aux_dim:            int,
        n_categories:       int,
        d_llm:              int,
        d_model:            int,
        n_layers:           int,
        n_heads:            int,
        n_main_tasks:       int,
        n_aux_tasks:        int,
        consistency_weight: float = 0.1,
        semantic_weight:    float = 0.05,
        ffn_dim:            Optional[int] = None,
        dropout:            float = 0.1,
        main_weights:       Optional[list[float]] = None,
        aux_weights:        Optional[list[float]] = None,
    ) -> None:
        super().__init__()
        self.n_main_tasks       = n_main_tasks
        self.n_aux_tasks        = n_aux_tasks
        self.consistency_weight = consistency_weight
        self.semantic_weight    = semantic_weight

        self.register_buffer('main_w', torch.tensor(main_weights or [1.0] * n_main_tasks))
        self.register_buffer('aux_w',  torch.tensor(aux_weights  or [1.0] * n_aux_tasks))

        # LLM-enriched item embedding (key difference from base FM)
        self.item_emb = SemanticItemEmbedding(
            prod_dim, ctx_dim, act_dim, n_categories, d_llm, d_model
        )
        self.hstu      = HSTU(d_model, n_layers, n_heads, ffn_dim, dropout)
        self.main_head = MultiTaskHead(d_model, n_main_tasks)
        self.align     = AlignmentModule(d_model, aux_dim, n_aux_tasks)

        # LLM semantic alignment head: TAE → predicted category distribution
        # Trained with soft supervision from LLM-extracted categories
        self.semantic_head = nn.Linear(d_model, n_categories, bias=False)

        self.consistency_reg = ConsistencyRegularizer()

    def forward(
        self,
        hist_prod:        Tensor,   # (B, N, prod_dim)
        hist_ctx:         Tensor,   # (B, N, ctx_dim)
        hist_act:         Tensor,   # (B, N, act_dim)
        hist_cat_logits:  Tensor,   # (B, N, n_categories) LLM categories for history
        hist_captions:    Tensor,   # (B, N, d_llm)
        cand_prod:        Tensor,   # (B, M, prod_dim)
        cand_ctx:         Tensor,   # (B, M, ctx_dim)
        cand_cat_logits:  Tensor,   # (B, M, n_categories) LLM categories for candidates
        cand_captions:    Tensor,   # (B, M, d_llm)
        aux_features:     Tensor,   # (B, M, aux_dim)
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns:
            tae:          (B, M, d_model)  target-aware embeddings.
            main_logits:  (B, M, n_main_tasks)
            aux_logits:   (B, M, n_aux_tasks)
            sem_logits:   (B, M, n_categories)  category prediction from TAE.
        """
        B, N = hist_prod.shape[:2]
        M    = cand_prod.shape[1]

        # Build unified sequence with LLM features
        hist_embs = self.item_emb(hist_prod, hist_ctx, hist_cat_logits, hist_captions, hist_act)
        cand_embs = self.item_emb(cand_prod, cand_ctx, cand_cat_logits, cand_captions)
        seq = torch.cat([hist_embs, cand_embs], dim=1)   # (B, N+M, d)

        out = self.hstu(seq, n_hist=N)
        tae = out[:, N:, :]                               # (B, M, d)

        main_logits = self.main_head(tae)
        aux_logits  = self.align(tae, aux_features)
        sem_logits  = self.semantic_head(tae)             # (B, M, n_categories)

        return tae, main_logits, aux_logits, sem_logits

    def compute_loss(
        self,
        main_logits:     Tensor,               # (B, M, n_main)
        main_labels:     Tensor,               # (B, M, n_main)
        aux_logits:      Tensor,               # (B, M, n_aux)
        aux_labels:      Tensor,               # (B, M, n_aux)
        sem_logits:      Tensor,               # (B, M, n_categories)
        cand_cat_labels: Tensor,               # (B, M, n_categories)  LLM soft labels
        aux_delta:       Optional[Tensor] = None,
        tae_primary:     Optional[Tensor] = None,   # (P, M, d) for consistency
        tae_shadow:      Optional[Tensor] = None,   # (P, M, d)
        pair_weights:    Optional[Tensor] = None,   # (P,)
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute L_total = L_main + L_aux + λ L_consistency + μ L_semantic."""
        # Main cross-surface task loss
        main_loss = sum(
            self.main_w[s] * F.binary_cross_entropy_with_logits(
                main_logits[..., s], main_labels[..., s]
            )
            for s in range(self.n_main_tasks)
        )

        # Auxiliary masked loss
        aux_loss = sum(
            self.aux_w[t] * AlignmentModule.masked_loss(
                aux_logits[..., t:t+1],
                aux_labels[..., t:t+1],
                aux_delta[..., t:t+1] if aux_delta is not None else None,
            )
            for t in range(self.n_aux_tasks)
        )

        # LLM semantic alignment: supervise TAE-predicted categories with LLM labels
        # Soft labels → BCE; teaches the model that LLM semantics ≈ good representations
        sem_loss = F.binary_cross_entropy_with_logits(
            sem_logits, torch.sigmoid(cand_cat_labels)
        )

        # Consistency regularization for A/A' pairs
        if tae_primary is not None and tae_shadow is not None:
            consist_loss = self.consistency_reg(tae_primary, tae_shadow, pair_weights)
        else:
            consist_loss = torch.tensor(0.0, device=main_logits.device)

        total = (main_loss + aux_loss
                 + self.semantic_weight    * sem_loss
                 + self.consistency_weight * consist_loss)

        breakdown = {
            'main':        main_loss.detach(),
            'aux':         aux_loss.detach(),
            'semantic':    sem_loss.detach(),
            'consistency': consist_loss.detach(),
        }
        return total, breakdown


# ---------------------------------------------------------------------------
# LLMEnrichedExpertModel   §3.2 of 2508.02929 + §4.4 of 2605.21969v1
# ---------------------------------------------------------------------------

class LLMEnrichedExpertModel(nn.Module):
    """Lightweight surface-specific expert enriched with LLM ad similarity features.

    Extends Foundation-Expert ExpertModel (§3.2) by adding LLM-derived
    ad-to-ad similarity scores and category membership as surface features.
    These signals are particularly valuable for:
      - Cold-start ads: similarity to known-good ads in the semantic graph
      - Predictability: stable category features reduce sensitivity to ID changes

    Args:
        fm_dim:       FM TAE dimension.
        prod_dim:     product feature dimension.
        ctx_dim:      contextual feature dimension.
        act_dim:      action feature dimension.
        surf_dim:     surface-specific feature dimension.
        n_categories: LLM category vocabulary (used as additional surface feature).
        d_llm:        LLM caption embedding dimension.
        d_expert:     expert hidden dimension.
        n_layers:     lightweight HSTU layers (1–2).
        n_heads:      attention heads.
        n_tasks:      number of surface-specific prediction tasks.
        dropout:      dropout rate.
    """

    def __init__(
        self,
        fm_dim:       int,
        prod_dim:     int,
        ctx_dim:      int,
        act_dim:      int,
        surf_dim:     int,
        n_categories: int,
        d_llm:        int,
        d_expert:     int,
        n_layers:     int,
        n_heads:      int,
        n_tasks:      int,
        dropout:      float = 0.0,
    ) -> None:
        super().__init__()
        # FM embedding preprocessing
        self.fm_emb_module = FMEmbeddingModule(fm_dim, d_expert, dropout)

        # Short-term surface history with LLM-enriched item embedding
        self.item_emb   = SemanticItemEmbedding(
            prod_dim, ctx_dim, act_dim, n_categories, d_llm, d_expert
        )
        self.light_hstu = HSTU(d_expert, n_layers, n_heads, dropout=dropout)

        # LLM ad similarity encoder: category logits for candidate → additional surface signal
        # Encodes "how semantically typical is this candidate for this surface"
        self.llm_surf_enc = nn.Sequential(
            nn.Linear(n_categories + d_llm, d_expert, bias=False),
            nn.ReLU(),
        )

        # Fusion layers
        # Surface features now include: original surf_dim + d_expert (LLM signal)
        self.fm_fusion     = FMFusionModule(d_expert, d_expert, d_expert)
        self.expert_fusion = ExpertFusionModule(d_expert, surf_dim + d_expert, n_tasks)

    def forward(
        self,
        tae:             Tensor,   # (B, M, fm_dim)   from FoundationModel, detached
        short_prod:      Tensor,   # (B, T, prod_dim)  short-term history
        short_ctx:       Tensor,   # (B, T, ctx_dim)
        short_act:       Tensor,   # (B, T, act_dim)
        short_cat:       Tensor,   # (B, T, n_categories)  LLM categories for history
        short_captions:  Tensor,   # (B, T, d_llm)
        cand_cat_logits: Tensor,   # (B, M, n_categories)  LLM categories for candidates
        cand_captions:   Tensor,   # (B, M, d_llm)
        surface_feats:   Tensor,   # (B, M, surf_dim)
    ) -> Tensor:                   # (B, M, n_tasks)
        """
        Args:
            tae:             TAE from FM (detached — no gradient to FM).
            short_*:         recent surface-specific interaction history with LLM features.
            cand_cat_logits, cand_captions: LLM semantic features for candidates.
            surface_feats:   surface context features per candidate.
        Returns:
            logits: (B, M, n_tasks)
        """
        # FM embedding preprocessing
        fm_processed = self.fm_emb_module(tae)          # (B, M, d_expert)

        # Short-term encoding with LLM-enriched embeddings
        short_embs = self.item_emb(
            short_prod, short_ctx, short_cat, short_captions, short_act
        )
        T = short_embs.size(1)
        short_out  = self.light_hstu(short_embs, n_hist=T)
        short_rep  = short_out.mean(dim=1)              # (B, d_expert)

        # Fuse FM (long-term) + short-term
        fused = self.fm_fusion(fm_processed, short_rep) # (B, M, d_expert)

        # LLM semantic signal for candidates: category + caption
        llm_cand = self.llm_surf_enc(
            torch.cat([cand_cat_logits, cand_captions], dim=-1)
        )  # (B, M, d_expert)

        # Augment surface features with LLM candidate signal
        surf_augmented = torch.cat([surface_feats, llm_cand], dim=-1)  # (B, M, surf_dim+d_expert)

        return self.expert_fusion(fused, surf_augmented)  # (B, M, n_tasks)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B, N, M, T_short = 4, 16, 6, 8
    P = 2   # A/A' pairs

    prod_dim, ctx_dim, act_dim = 12, 6, 4
    aux_dim, surf_dim = 8, 6
    n_categories, d_llm = 16, 12
    d_model, d_expert = 24, 12

    # --- LLMEnrichedFoundationModel ---
    fm = LLMEnrichedFoundationModel(
        prod_dim     = prod_dim,
        ctx_dim      = ctx_dim,
        act_dim      = act_dim,
        aux_dim      = aux_dim,
        n_categories = n_categories,
        d_llm        = d_llm,
        d_model      = d_model,
        n_layers     = 2,
        n_heads      = 4,
        n_main_tasks = 3,
        n_aux_tasks  = 2,
        consistency_weight = 0.1,
        semantic_weight    = 0.05,
    )

    hist_prod       = torch.randn(B, N, prod_dim)
    hist_ctx        = torch.randn(B, N, ctx_dim)
    hist_act        = torch.randn(B, N, act_dim)
    hist_cat        = torch.randn(B, N, n_categories)
    hist_captions   = F.normalize(torch.randn(B, N, d_llm), dim=-1)
    cand_prod       = torch.randn(B, M, prod_dim)
    cand_ctx        = torch.randn(B, M, ctx_dim)
    cand_cat        = torch.randn(B, M, n_categories)
    cand_captions   = F.normalize(torch.randn(B, M, d_llm), dim=-1)
    aux_feats       = torch.randn(B, M, aux_dim)

    tae, main_logits, aux_logits, sem_logits = fm(
        hist_prod, hist_ctx, hist_act, hist_cat, hist_captions,
        cand_prod, cand_ctx, cand_cat, cand_captions, aux_feats,
    )
    print(f"FM TAE:          {tae.shape}")
    print(f"FM main logits:  {main_logits.shape}")
    print(f"FM sem logits:   {sem_logits.shape}")

    # A/A' consistency: P pairs of (primary, shadow) FM runs
    tae_primary = tae[:P]                                     # (P, M, d)
    tae_shadow  = tae[:P] + 0.1 * torch.randn(P, M, d_model) # slight variant
    pair_weights = torch.tensor([0.9, 0.7])

    main_labels = torch.zeros_like(main_logits)
    aux_labels  = torch.zeros_like(aux_logits)
    cand_cat_labels = torch.randn(B, M, n_categories)
    delta = torch.randint(0, 2, aux_logits.shape).float()

    fm_loss, breakdown = fm.compute_loss(
        main_logits, main_labels,
        aux_logits, aux_labels,
        sem_logits, cand_cat_labels,
        aux_delta    = delta,
        tae_primary  = tae_primary,
        tae_shadow   = tae_shadow,
        pair_weights = pair_weights,
    )
    fm_loss.backward()
    print(f"FM loss:         {fm_loss.item():.4f}")
    print(f"  main={breakdown['main']:.4f}  aux={breakdown['aux']:.4f}"
          f"  sem={breakdown['semantic']:.4f}  consist={breakdown['consistency']:.4f}")
    print("FM backward:     OK")
    fm_params = sum(p.numel() for p in fm.parameters())
    print(f"FM params:       {fm_params:,}")

    # --- LLMEnrichedExpertModel ---
    expert = LLMEnrichedExpertModel(
        fm_dim       = d_model,
        prod_dim     = prod_dim,
        ctx_dim      = ctx_dim,
        act_dim      = act_dim,
        surf_dim     = surf_dim,
        n_categories = n_categories,
        d_llm        = d_llm,
        d_expert     = d_expert,
        n_layers     = 1,
        n_heads      = 4,
        n_tasks      = 2,
    )

    short_prod     = torch.randn(B, T_short, prod_dim)
    short_ctx      = torch.randn(B, T_short, ctx_dim)
    short_act      = torch.randn(B, T_short, act_dim)
    short_cat      = torch.randn(B, T_short, n_categories)
    short_captions = F.normalize(torch.randn(B, T_short, d_llm), dim=-1)
    surf_feats     = torch.randn(B, M, surf_dim)

    expert_logits = expert(
        tae.detach(), short_prod, short_ctx, short_act,
        short_cat, short_captions, cand_cat, cand_captions, surf_feats,
    )
    print(f"\nExpert logits:   {expert_logits.shape}")

    expert_loss = F.binary_cross_entropy_with_logits(
        expert_logits, torch.zeros_like(expert_logits)
    )
    expert_loss.backward()
    print(f"Expert loss:     {expert_loss.item():.4f}")
    print("Expert backward: OK")
    expert_params = sum(p.numel() for p in expert.parameters())
    print(f"Expert params:   {expert_params:,}")
    print(f"Expert/FM ratio: {expert_params/fm_params:.1%}  (target: 20-40%)")

    print("\nLLM-enriched ranking model:")
    print("  FM:     LLM category + caption → richer TAE; consistency loss for A/A' stability")
    print("  Expert: LLM similarity features → cold-start robust; surface-level predictability")


if __name__ == "__main__":
    _smoke_test()
