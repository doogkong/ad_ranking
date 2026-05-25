"""Scaling User Modeling (SUM) — Upstream Representation for Ads Ranking.

Reference: "Scaling User Modeling: Large-scale Online User Representations for
Ads Personalization in Meta" (Meta Platforms, arXiv 2311.09544v2).

SUM trains a single large-capacity user tower shared across hundreds of downstream
ads ranking tasks.  Downstream models receive a high-quality user embedding instead
of raw feature tables, saving capacity, cold-start, and training cost.

Architecture overview:

  User feature sequence (sparse IDs + dense behaviours)
            │
            ▼
  UserTower  (pyramid of N Interaction Modules)
    IM_1: X_1 = Concat(Interaction(X_0), X_dense) + Residual(X_0)   Eq. 1
    IM_2: X_2 = Concat(Interaction(X_1), X_dense) + Residual(X_1)
    …
    IM_N: X_N  (top of pyramid)
    → outputs K=2 pooled user embeddings of dimension D=96
            │
            ▼
  MixTower  (DHEN-style cross-feature interaction + per-task heads)
    → multi-task logits for click/conversion/etc.

  Training loss (Eq. 8):
    L = -1/N Σ_i Σ_t w_t [y_ti log(ŷ_ti) + (1-y_ti) log(1-ŷ_ti)]

  SOAP (SUM Online Asynchronous Platform):
    Write path:  request arrives → read cached embedding (k-1) → serve downstream
                 → async compute new embedding k → write back to feature store
    Rolling avg: Emb_served = mean(Emb_{k-1}, Emb_{k-2}, Emb_{k-3})
                 Mitigates distribution shift introduced by the async lag.

Interaction extractors (§3.1):
  MLP               — dense projection baseline
  DotCompressionWithAttention — Eq. 2-4: compress sequence, dot-product attention
  MLPMixerExtractor — Eq. 6-7: channel-mix + token-mix; parameter-efficient
  DeepCrossExtractor — Eq. 5: explicit multiplicative feature crosses (DCN-v2)
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Interaction Extractors (§3.1)
# ---------------------------------------------------------------------------

class MLPExtractor(nn.Module):
    """Two-layer MLP feature extractor (baseline)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, in_dim)
        return self.net(x)


class DotCompressionWithAttention(nn.Module):
    """Dot-compression with attention (§3.1, Eq. 2-4).

    Compresses a sequence of item embeddings to a fixed-length summary using
    learned query vectors and dot-product attention.

      Q  = W_q · X_dense                          (B, n_heads, d_k)   Eq. 2
      K  = W_k · X_seq                            (B, L, d_k)         Eq. 3
      A  = softmax(Q K^T / √d_k)                  (B, n_heads, L)
      Z  = A · X_seq  →  flatten  →  (B, out_dim)                    Eq. 4
    """

    def __init__(
        self,
        seq_dim: int,
        dense_dim: int,
        d_k: int,
        n_heads: int,
        out_dim: int,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.W_q = nn.Linear(dense_dim, n_heads * d_k, bias=False)
        self.W_k = nn.Linear(seq_dim, d_k, bias=False)
        self.out_proj = nn.Linear(n_heads * seq_dim, out_dim)

    def forward(self, x_seq: Tensor, x_dense: Tensor) -> Tensor:
        # x_seq:   (B, L, seq_dim)
        # x_dense: (B, dense_dim)
        B, L, _ = x_seq.shape
        Q = self.W_q(x_dense).view(B, self.n_heads, self.d_k)      # (B, H, d_k)
        K = self.W_k(x_seq)                                          # (B, L, d_k)
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.d_k ** 0.5)  # (B, H, L)
        A = F.softmax(scores, dim=-1)                                # (B, H, L)
        Z = torch.bmm(A, x_seq)                                      # (B, H, seq_dim)
        Z = Z.reshape(B, self.n_heads * x_seq.shape[-1])             # (B, H*seq_dim)
        return self.out_proj(Z)                                       # (B, out_dim)


class MLPMixerExtractor(nn.Module):
    """MLP-Mixer extractor (§3.1, Eq. 6-7).

    Parameter-efficient: shared weights across sequence positions (token-mix)
    followed by per-position channel-mix.

      Y_token   = X + W_2 · σ(W_1 · X^T)^T      (B, L, d)   Eq. 6 — token mix
      Y_channel = Y_token + W_4 · σ(W_3 · Y_token)            (B, L, d)   Eq. 7 — channel mix
    """

    def __init__(self, seq_len: int, in_dim: int, hidden_token: int, hidden_channel: int, out_dim: int) -> None:
        super().__init__()
        # token-mix: operates across the L dimension
        self.token_mix = nn.Sequential(
            nn.Linear(seq_len, hidden_token),
            nn.GELU(),
            nn.Linear(hidden_token, seq_len),
        )
        # channel-mix: operates across the d dimension
        self.channel_mix = nn.Sequential(
            nn.Linear(in_dim, hidden_channel),
            nn.GELU(),
            nn.Linear(hidden_channel, in_dim),
        )
        self.out_proj = nn.Linear(in_dim, out_dim)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, in_dim)
        # token mix: transpose so Linear sees L-dim
        y = self.norm1(x)
        y = y + self.token_mix(y.transpose(1, 2)).transpose(1, 2)
        # channel mix
        y = y + self.channel_mix(self.norm2(y))
        # pool over sequence
        pooled = y.mean(dim=1)                                       # (B, in_dim)
        return self.out_proj(pooled)                                  # (B, out_dim)


class DeepCrossExtractor(nn.Module):
    """Deep Cross Network v2 extractor (§3.1, Eq. 5).

      x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l   (explicit cross)   Eq. 5
    """

    def __init__(self, in_dim: int, n_cross_layers: int, out_dim: int) -> None:
        super().__init__()
        self.cross_layers = nn.ModuleList([
            nn.Linear(in_dim, in_dim) for _ in range(n_cross_layers)
        ])
        self.out_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, in_dim)
        x0, xl = x, x
        for layer in self.cross_layers:
            xl = x0 * layer(xl) + xl
        return self.out_proj(xl)                                      # (B, out_dim)


# ---------------------------------------------------------------------------
# Interaction Module — pyramid building block (§3.1, Eq. 1)
# ---------------------------------------------------------------------------

class InteractionModule(nn.Module):
    """One level of the SUM pyramid (§3.1, Eq. 1).

      X_n = Concat(Interaction(X_{n-1}), X_dense) + Residual(X_{n-1})

    Parallel extractors are summed before concatenation:
      Interaction(X) = MLP(X) + DotCompr(X) + Mixer(X) + DCN(X)
    """

    def __init__(
        self,
        in_dim: int,          # dimension of X_{n-1}
        dense_dim: int,       # dimension of X_dense (unchanged across levels)
        seq_len: int,         # history sequence length (for Mixer)
        extractor_out: int,   # output dim of each extractor
        out_dim: int,         # final output dim of this module
    ) -> None:
        super().__init__()
        self.mlp = MLPExtractor(in_dim, in_dim * 2, extractor_out)
        self.dot = DotCompressionWithAttention(
            seq_dim=in_dim,
            dense_dim=dense_dim,
            d_k=extractor_out // 4,
            n_heads=4,
            out_dim=extractor_out,
        )
        self.mixer = MLPMixerExtractor(
            seq_len=seq_len,
            in_dim=in_dim,
            hidden_token=seq_len * 2,
            hidden_channel=in_dim * 2,
            out_dim=extractor_out,
        )
        self.dcn = DeepCrossExtractor(in_dim, n_cross_layers=2, out_dim=extractor_out)

        # projection for element-wise extractors (need a flat input)
        self.seq_pool = nn.Linear(in_dim, in_dim)

        # combine: Concat(interaction, X_dense)  then  project to out_dim
        concat_dim = extractor_out + dense_dim
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, out_dim),
            nn.ReLU(),
        )

        # residual: align in_dim → out_dim if they differ
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x_seq: Tensor, x_dense: Tensor) -> Tensor:
        # x_seq:   (B, L, in_dim) — sequence representation at this pyramid level
        # x_dense: (B, dense_dim) — context / dense features (fixed)
        B, L, D = x_seq.shape
        x_flat = self.seq_pool(x_seq.mean(dim=1))                    # (B, in_dim)

        i_mlp   = self.mlp(x_flat)                                   # (B, extractor_out)
        i_dot   = self.dot(x_seq, x_dense)                           # (B, extractor_out)
        i_mix   = self.mixer(x_seq)                                   # (B, extractor_out)
        i_dcn   = self.dcn(x_flat)                                   # (B, extractor_out)
        interaction = i_mlp + i_dot + i_mix + i_dcn                  # (B, extractor_out)

        cat = torch.cat([interaction, x_dense], dim=-1)               # (B, extractor_out + dense_dim)
        out = self.fusion(cat)                                        # (B, out_dim)

        residual = self.residual(x_flat)                              # (B, out_dim)
        # Broadcast residual back to sequence shape for next level
        out_seq = out.unsqueeze(1).expand(B, L, -1)                  # (B, L, out_dim)
        res_seq = residual.unsqueeze(1).expand(B, L, -1)             # (B, L, out_dim)
        return self.norm(out_seq + res_seq)                           # (B, L, out_dim)


# ---------------------------------------------------------------------------
# UserTower — pyramid of N InteractionModules (§3)
# ---------------------------------------------------------------------------

class UserTower(nn.Module):
    """SUM UserTower: pyramid of Interaction Modules producing K user embeddings.

    Outputs K=2 pooled user embeddings of dimension D each.  In Meta production,
    these are cached in the SOAP feature store and served asynchronously.

    Architecture (N=3 levels shown):
      IM_1 → IM_2 → IM_3
        ↘       ↘
         pool    pool         K=2 embeddings tapped from intermediate + final level
    """

    def __init__(
        self,
        input_dim: int,      # initial item embedding dimension
        dense_dim: int,      # dense context feature dimension
        seq_len: int,        # history sequence length
        n_layers: int = 4,   # pyramid depth
        extractor_out: int = 128,
        hidden_dim: int = 256,
        user_emb_dim: int = 96,  # D in the paper
        n_outputs: int = 2,      # K in the paper
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_outputs = n_outputs

        # Input embedding
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Pyramid of Interaction Modules
        self.layers = nn.ModuleList([
            InteractionModule(
                in_dim=hidden_dim,
                dense_dim=dense_dim,
                seq_len=seq_len,
                extractor_out=extractor_out,
                out_dim=hidden_dim,
            )
            for _ in range(n_layers)
        ])

        # Tap K outputs from the top K layers
        assert n_outputs <= n_layers
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, user_emb_dim),
                nn.LayerNorm(user_emb_dim),
            )
            for _ in range(n_outputs)
        ])

    def forward(self, x_seq: Tensor, x_dense: Tensor) -> List[Tensor]:
        # x_seq:   (B, L, input_dim) — sequence of item embeddings
        # x_dense: (B, dense_dim)    — dense context features
        x = self.input_proj(x_seq)                                    # (B, L, hidden_dim)

        tapped = []
        for i, layer in enumerate(self.layers):
            x = layer(x, x_dense)                                    # (B, L, hidden_dim)
            # tap the last n_outputs layers
            tap_start = self.n_layers - self.n_outputs
            if i >= tap_start:
                tapped.append(x.mean(dim=1))                         # (B, hidden_dim)

        embeddings = [
            head(h) for head, h in zip(self.output_heads, tapped)
        ]                                                             # K × (B, user_emb_dim)
        return embeddings


# ---------------------------------------------------------------------------
# MixTower — DHEN-style cross + per-task heads (§3.2)
# ---------------------------------------------------------------------------

class MixTower(nn.Module):
    """SUM MixTower: fuses user embeddings with ad features, outputs per-task logits.

    Takes K user embeddings + ad features, applies cross interactions, and
    projects to per-task binary logits.
    """

    def __init__(
        self,
        user_emb_dim: int,    # D
        n_user_embs: int,     # K
        ad_feat_dim: int,
        n_tasks: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        fused_dim = user_emb_dim * n_user_embs + ad_feat_dim

        # Cross-feature interaction (DCN-style)
        self.cross = DeepCrossExtractor(fused_dim, n_cross_layers=3, out_dim=hidden_dim)

        # Deep branch
        self.deep = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Per-task output heads
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim * 2, 1) for _ in range(n_tasks)
        ])

    def forward(self, user_embs: List[Tensor], ad_feats: Tensor) -> Tensor:
        # user_embs: K × (B, user_emb_dim)
        # ad_feats:  (B, ad_feat_dim)
        fused = torch.cat(user_embs + [ad_feats], dim=-1)            # (B, fused_dim)
        cross_out = self.cross(fused)                                 # (B, hidden_dim)
        deep_out  = self.deep(fused)                                  # (B, hidden_dim)
        combined  = torch.cat([cross_out, deep_out], dim=-1)         # (B, 2*hidden_dim)
        logits = torch.cat([head(combined) for head in self.task_heads], dim=-1)  # (B, n_tasks)
        return logits


# ---------------------------------------------------------------------------
# Multi-task loss (§3, Eq. 8)
# ---------------------------------------------------------------------------

def multi_task_loss(
    logits: Tensor,
    labels: Tensor,
    task_weights: Optional[Tensor] = None,
) -> Tensor:
    """Weighted multi-task binary cross-entropy (Eq. 8).

      L = -1/N Σ_i Σ_t w_t [y_ti log(ŷ_ti) + (1-y_ti) log(1-ŷ_ti)]

    Args:
        logits:       (B, T) raw logits
        labels:       (B, T) binary labels ∈ {0, 1}
        task_weights: (T,)   per-task weights w_t; uniform if None
    """
    T = logits.shape[-1]
    if task_weights is None:
        task_weights = logits.new_ones(T)
    else:
        task_weights = task_weights.to(logits.device)

    bce = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')  # (B, T)
    weighted = (bce * task_weights.unsqueeze(0)).sum(dim=-1).mean()
    return weighted


# ---------------------------------------------------------------------------
# SOAP — SUM Online Asynchronous Platform (§4)
# ---------------------------------------------------------------------------

class SOAPFeatureStore:
    """In-memory SOAP feature store: rolling window of K=3 user embeddings.

    In production this is a low-latency key-value store (e.g. ZippyDB).
    The average of the K most recent embeddings is served to downstream models,
    mitigating distribution shift from the async compute lag.
    """

    def __init__(self, window_size: int = 3) -> None:
        self.window_size = window_size
        # user_id → deque of (B, D) tensors
        self._store: Dict[str, deque] = {}

    def write(self, user_id: str, embedding: Tensor) -> None:
        if user_id not in self._store:
            self._store[user_id] = deque(maxlen=self.window_size)
        self._store[user_id].append(embedding.detach().cpu())

    def read(self, user_id: str) -> Optional[Tensor]:
        if user_id not in self._store or len(self._store[user_id]) == 0:
            return None
        stack = torch.stack(list(self._store[user_id]), dim=0)      # (K, D)
        return stack.mean(dim=0)                                     # (D,) average pooling


class SOAPClient:
    """SOAP async serve pattern (§4):

    Read path (latency-critical):
      1. Downstream model requests user embedding.
      2. Return cached embedding immediately (from previous write cycle).
      3. Enqueue async task: compute new embedding from latest features.

    Write path (async, off critical path):
      4. Compute new embedding with UserTower.
      5. Write to feature store.
      6. Next request will use the updated (rolling-averaged) embedding.

    This decouples inference latency from embedding freshness.
    """

    def __init__(self, user_tower: UserTower, store: SOAPFeatureStore) -> None:
        self.user_tower = user_tower
        self.store = store

    @torch.no_grad()
    def serve(self, user_id: str, x_seq: Tensor, x_dense: Tensor) -> Tensor:
        """Return cached embedding (fast read) and schedule async update."""
        cached = self.store.read(user_id)

        # Async update (simulated synchronously here; production uses a queue)
        new_embs = self.user_tower(x_seq.unsqueeze(0), x_dense.unsqueeze(0))
        new_emb = torch.cat(new_embs, dim=-1).squeeze(0)             # concat K embeddings
        self.store.write(user_id, new_emb)

        if cached is None:
            return new_emb
        return cached                                                 # serve previous


# ---------------------------------------------------------------------------
# Full SUM Model
# ---------------------------------------------------------------------------

class SUMModel(nn.Module):
    """Full SUM model: UserTower + MixTower with multi-task training.

    Typical usage:
      - Train jointly on click/conversion/engagement tasks.
      - Extract UserTower weights → deploy as upstream shared model.
      - Downstream ranking models consume cached user embeddings from SOAP.
    """

    def __init__(
        self,
        input_dim: int,
        dense_dim: int,
        seq_len: int,
        ad_feat_dim: int,
        n_tasks: int,
        n_layers: int = 4,
        extractor_out: int = 128,
        hidden_dim: int = 256,
        user_emb_dim: int = 96,
        n_user_embs: int = 2,
    ) -> None:
        super().__init__()
        self.user_tower = UserTower(
            input_dim=input_dim,
            dense_dim=dense_dim,
            seq_len=seq_len,
            n_layers=n_layers,
            extractor_out=extractor_out,
            hidden_dim=hidden_dim,
            user_emb_dim=user_emb_dim,
            n_outputs=n_user_embs,
        )
        self.mix_tower = MixTower(
            user_emb_dim=user_emb_dim,
            n_user_embs=n_user_embs,
            ad_feat_dim=ad_feat_dim,
            n_tasks=n_tasks,
            hidden_dim=hidden_dim,
        )

    def forward(self, x_seq: Tensor, x_dense: Tensor, ad_feats: Tensor) -> Tuple[List[Tensor], Tensor]:
        user_embs = self.user_tower(x_seq, x_dense)                  # K × (B, user_emb_dim)
        logits    = self.mix_tower(user_embs, ad_feats)              # (B, n_tasks)
        return user_embs, logits

    def compute_loss(
        self,
        logits: Tensor,
        labels: Tensor,
        task_weights: Optional[Tensor] = None,
    ) -> Tensor:
        return multi_task_loss(logits, labels, task_weights)


# ---------------------------------------------------------------------------
# Solutions to sequence modeling limitations
# ---------------------------------------------------------------------------

class SIMRetriever(nn.Module):
    """Search-based Interest Model (SIM) — O(L) hard + soft two-stage retrieval.

    Addresses the O(L²) attention bottleneck for very long user histories.

    Stage 1 (hard search): exact category match reduces L=10000 → K=200.
    Stage 2 (soft attention): target-aware dot-product attention over K items.

    Reference: Pi et al. "Search-based User Interest Modeling", Alibaba, KDD 2020.
    """

    def __init__(self, item_dim: int, target_dim: int, top_k: int = 200) -> None:
        super().__init__()
        self.top_k = top_k
        self.attn_proj = nn.Linear(item_dim, target_dim, bias=False)
        self.out_proj   = nn.Linear(item_dim, item_dim)

    def forward(
        self,
        history: Tensor,          # (B, L, item_dim) — full long history
        history_cats: Tensor,     # (B, L)           — integer category IDs
        target_emb: Tensor,       # (B, target_dim)  — target item embedding
        target_cat: Tensor,       # (B,)             — target item category ID
    ) -> Tensor:
        B, L, D = history.shape

        # Stage 1: hard search — keep only items whose category matches target
        mask = (history_cats == target_cat.unsqueeze(1))              # (B, L) bool
        # If fewer than top_k match, pad with the first matching idx; else take top_k
        scores_stage1 = mask.float()                                  # (B, L) 0/1
        # Take top_k by stage-1 score (deterministic; production uses inverted index)
        k = min(self.top_k, L)
        _, idx = scores_stage1.topk(k, dim=1)                        # (B, k)
        retrieved = history.gather(1, idx.unsqueeze(-1).expand(B, k, D))  # (B, k, D)

        # Stage 2: soft target-aware attention over retrieved items
        q = self.attn_proj(target_emb).unsqueeze(2)                  # (B, target_dim, 1)
        attn = torch.bmm(retrieved, q).squeeze(-1) / (D ** 0.5)      # (B, k)
        attn = F.softmax(attn, dim=-1)                                # (B, k)
        out = (attn.unsqueeze(-1) * retrieved).sum(dim=1)            # (B, item_dim)
        return self.out_proj(out)                                     # (B, item_dim)


class RecencyWeightedPooling(nn.Module):
    """Exponential recency decay before sequence pooling.

    Addresses the session-boundary / recency-blindness limitation.
    Items are assumed to be ordered oldest-first (index 0 = oldest).

      w_l = exp(−λ · (L − l − 1))   for l = 0, …, L-1
      user_emb = Σ_l w_l · h_l  /  Σ_l w_l

    λ=0 → uniform average.  λ>0 → more recent items weighted higher.
    """

    def __init__(self, decay_lambda: float = 0.05) -> None:
        super().__init__()
        self.decay_lambda = decay_lambda

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, L, D) — ordered oldest-first
        B, L, D = x.shape
        positions = torch.arange(L, device=x.device, dtype=x.dtype)  # 0 … L-1
        weights = torch.exp(-self.decay_lambda * (L - positions - 1)) # (L,)
        weights = weights / weights.sum()                              # normalise
        return (x * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)    # (B, D)


class HybridUserTower(nn.Module):
    """Long-term SOAP embedding + real-time short-term GRU.

    Addresses the staleness limitation: cached long-term embedding is always
    served immediately; a cheap GRU encodes the last N events at request time
    to capture recent signals not yet reflected in the SOAP cache.

      user_emb = MLP(Concat(lt_emb, st_emb))
    """

    def __init__(
        self,
        lt_dim: int,     # long-term embedding dim (from SOAP)
        item_dim: int,   # item feature dim for short-term sequence
        st_hidden: int,  # GRU hidden size
        out_dim: int,    # fused output dim
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(item_dim, st_hidden, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(lt_dim + st_hidden, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, lt_emb: Tensor, recent_seq: Tensor) -> Tensor:
        # lt_emb:     (B, lt_dim)   — cached long-term embedding from SOAP
        # recent_seq: (B, N, item_dim) — last N events at serving time
        _, h_n = self.gru(recent_seq)                                 # h_n: (1, B, st_hidden)
        st_emb = h_n.squeeze(0)                                       # (B, st_hidden)
        return self.fusion(torch.cat([lt_emb, st_emb], dim=-1))      # (B, out_dim)


class GradientSurgery(torch.autograd.Function):
    """PCGrad: project conflicting task gradients before shared parameter update.

    Addresses negative transfer in multi-task training.

    For each pair of tasks (i, j): if ∇L_i · ∇L_j < 0 (conflicting),
    project ∇L_i onto the plane perpendicular to ∇L_j.

    Reference: Yu et al. "Gradient Surgery for Multi-Task Learning", NeurIPS 2020.

    Usage: call GradientSurgery.apply(task_gradients) during the backward pass.
    This reference implementation demonstrates the algorithm on explicit gradient
    tensors; in practice it is applied as an optimizer hook.
    """

    @staticmethod
    def pcgrad(grads: List[Tensor]) -> List[Tensor]:
        n_tasks = len(grads)
        proj_grads = [g.clone() for g in grads]
        for i in range(n_tasks):
            for j in range(n_tasks):
                if i == j:
                    continue
                dot = torch.dot(proj_grads[i].flatten(), grads[j].flatten())
                if dot < 0:
                    # project out the conflicting component
                    norm_sq = (grads[j] * grads[j]).sum() + 1e-12
                    proj_grads[i] = proj_grads[i] - (dot / norm_sq) * grads[j]
        return proj_grads


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    torch.manual_seed(0)

    B, L = 4, 20
    INPUT_DIM = 64
    DENSE_DIM = 32
    AD_DIM    = 48
    N_TASKS   = 3

    model = SUMModel(
        input_dim=INPUT_DIM,
        dense_dim=DENSE_DIM,
        seq_len=L,
        ad_feat_dim=AD_DIM,
        n_tasks=N_TASKS,
        n_layers=3,
        extractor_out=64,
        hidden_dim=128,
        user_emb_dim=96,
        n_user_embs=2,
    )

    x_seq   = torch.randn(B, L, INPUT_DIM)
    x_dense = torch.randn(B, DENSE_DIM)
    ad_feats = torch.randn(B, AD_DIM)
    labels  = torch.randint(0, 2, (B, N_TASKS)).float()

    user_embs, logits = model(x_seq, x_dense, ad_feats)

    print(f"UserTower embs: {len(user_embs)} × {user_embs[0].shape}")
    print(f"MixTower logits: {logits.shape}")

    loss = model.compute_loss(logits, labels)
    print(f"Loss: {loss.item():.4f}")

    loss.backward()
    print("Backward: OK")

    total = sum(p.numel() for p in model.parameters())
    user  = sum(p.numel() for p in model.user_tower.parameters())
    mix   = sum(p.numel() for p in model.mix_tower.parameters())
    print(f"Total params:      {total:,}")
    print(f"  UserTower:       {user:,}")
    print(f"  MixTower:        {mix:,}")

    # SOAP async serving demo
    store  = SOAPFeatureStore(window_size=3)
    client = SOAPClient(model.user_tower, store)
    emb1 = client.serve('user_001', x_seq[0], x_dense[0])
    emb2 = client.serve('user_001', x_seq[0], x_dense[0])
    print(f"\nSOAP serve  (cold):  {emb1.shape}")
    print(f"SOAP serve  (warm):  {emb2.shape}  (cached previous embedding)")

    print("\n--- Limitation solutions ---")

    # SIM retrieval
    sim = SIMRetriever(item_dim=INPUT_DIM, target_dim=INPUT_DIM, top_k=10)
    long_history = torch.randn(B, 100, INPUT_DIM)
    history_cats = torch.randint(0, 8, (B, 100))
    target_emb   = torch.randn(B, INPUT_DIM)
    target_cat   = torch.randint(0, 8, (B,))
    sim_out = sim(long_history, history_cats, target_emb, target_cat)
    print(f"SIM (L=100→K=10):    {sim_out.shape}")

    # Recency-weighted pooling
    rw = RecencyWeightedPooling(decay_lambda=0.1)
    rw_out = rw(x_seq)
    print(f"RecencyWeightedPool: {rw_out.shape}")

    # Hybrid long-term + short-term
    hybrid = HybridUserTower(lt_dim=192, item_dim=INPUT_DIM, st_hidden=32, out_dim=128)
    lt_emb     = torch.randn(B, 192)
    recent_seq = torch.randn(B, 5, INPUT_DIM)
    hybrid_out = hybrid(lt_emb, recent_seq)
    print(f"HybridUserTower:     {hybrid_out.shape}")

    # PCGrad gradient surgery
    g1 = torch.randn(64)
    g2 = torch.randn(64)
    proj = GradientSurgery.pcgrad([g1, g2])
    print(f"PCGrad projected:    {len(proj)} tensors of shape {proj[0].shape}")
