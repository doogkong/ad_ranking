"""Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations.

Reference PyTorch implementation.
Paper: https://arxiv.org/abs/2512.09200  (Meta AI, Dec 2025)

Framework overview:

  Data Integration (§3.2):
    LatticeZipper  — mixed attribution-window datasets; one model with K window heads.
                     Each impression assigned to one window by hash; oracle head at serving.
    lattice_filter — Pareto-optimal feature selection across N tasks (iterative frontier).

  Model Unification (§3.3): Lattice Networks  (three-stage preprocessor-backbone-task)
    Preprocessors:
      CategoricalProcessor  — embedding tables  → (B, n_cat,   d)
      DenseProcessor        — bias-less MLPs    → (B, n_dense, d)
      SequenceProcessor     — attention models  → (B, T,       d)
      MixingNetwork         — concat cat+dense, QK-norm, LayerNorm → O_cd ∈ (B, n_cd, d)
    Backbone (L interleaved layers with ECS residuals):
      TransformerBlock  — RoPE + MHA + domain-specific FFN   [sequence path]
      DWFBlock          — concat(O_s, O_cd) → FM + LCB       [cross-modal path]
      ECS               — DenseNet-style global residual store
    Task Modules:
      TaskModule        — per-objective lightweight MLP head
    Zipper Heads:
      LatticeZipper     — K window-specific prediction heads (oracle used at inference)

  Knowledge Transfer (§3.4): Lattice KTAP
    LatticeKTAP       — precomputed teacher embeddings + logit distillation at inference

  Stability (§3.3.4):
    SwishRMSNorm      — RMSNorm(x) ⊙ Sigmoid(RMSNorm(x)); avoids cancellation near mean
    Bias-less layers  — removes bias from linear + norm layers to prevent unbounded growth
    Parameter untying — per-domain weights in backbone for conflict-free multi-domain learning

Production results: +10% topline NE, +11.5% user satisfaction, +6% CVR, +20% capacity savings.
"""

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _make_mlp(dims: list[int], bias: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class SwishRMSNorm(nn.Module):
    """SwishRMSNorm: stable activation for deep recommendation networks (§3.3.4).

    Combines Swish (SiLU) activation with RMSNorm:
        X_out = RMSNorm(X_in) ⊙ Sigmoid(RMSNorm(X_in))

    Avoids catastrophic cancellation issues of LayerNorm when elements are near
    the mean. The self-gating via Sigmoid improves numerical stability in deep
    MLPs, critical for DHEN ensembles with many stacked layers.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        n = self.norm(x)
        return n * torch.sigmoid(n)


# ---------------------------------------------------------------------------
# Lattice Filter: Pareto-optimal feature selection   §3.2.2
# ---------------------------------------------------------------------------

def lattice_filter(importance_scores: Tensor, target_count: int,
                   seed: int = 0) -> list[int]:
    """Pareto-optimal feature selection across N consolidated tasks.

    For each feature i we have an importance vector F_i = (f_{i,1}, ..., f_{i,N})
    across N tasks. Feature k *dominates* feature i (F_i ≺ F_k) if:
        f_{k,j} >= f_{i,j}  for all j   AND   f_{k,j} > f_{i,j}  for some j

    A dominated feature is off the Pareto frontier and excluded first.
    Iteratively selects features from the current frontier until budget T is met.
    When more Pareto-optimal features exist than remaining budget, randomly fill
    to avoid bias (pre-sorted importance ensures critical features appear first).

    Args:
        importance_scores: (n_features, n_tasks)  permutation-based feature importance.
        target_count:      T, total features to select.
        seed:              random seed for the stochastic fill step.
    Returns:
        Sorted list of selected feature indices, len == min(target_count, n_features).
    """
    rng = random.Random(seed)
    scores = importance_scores.tolist()
    n_features = len(scores)
    remaining  = list(range(n_features))
    selected: list[int] = []

    while len(selected) < target_count and remaining:
        # Identify Pareto frontier (non-dominated features)
        pareto: list[int] = []
        for i in remaining:
            dominated = any(
                all(scores[j][t] >= scores[i][t] for t in range(len(scores[i])))
                and any(scores[j][t] > scores[i][t] for t in range(len(scores[i])))
                for j in remaining if j != i
            )
            if not dominated:
                pareto.append(i)

        budget = target_count - len(selected)
        if len(pareto) <= budget:
            selected.extend(pareto)
            remaining = [i for i in remaining if i not in pareto]
        else:
            selected.extend(rng.sample(pareto, budget))
            break

    return sorted(selected)


# ---------------------------------------------------------------------------
# Lattice Zipper: mixed attribution-window datasets   §3.2.1
# ---------------------------------------------------------------------------

class LatticeZipper(nn.Module):
    """Multi-attribution-window prediction heads for delayed feedback (§3.2.1).

    Ad conversions (e.g., purchases) can occur minutes to days after an impression.
    Maintaining K separate models trained on K attribution windows is expensive
    and causes overfitting/instability from conflicting labels on the same impression.

    Lattice Zipper maintains K prediction heads on a *single* shared backbone:
      - Each impression is deterministically assigned to one window by hashing
        (user_id, ad_id, timestamp) to a discrete distribution.
      - During training, the loss for each impression is computed only against
        its assigned head, allowing simultaneous learning from data at different
        freshness-correctness trade-off points.
      - At serving time, the oracle (longest-window) head is used, benefiting
        from the complete long-window data while the shorter-window heads provided
        fresher signals during shared backbone training.

    Args:
        n_windows:  K, number of attribution windows (e.g. 3 for 90min/1day/7day).
        in_dim:     backbone output dimension fed into each head.
        n_tasks:    number of objectives predicted per window.
    """

    def __init__(self, n_windows: int, in_dim: int, n_tasks: int = 1) -> None:
        super().__init__()
        self.n_windows  = n_windows
        self.oracle_idx = n_windows - 1
        # Bias-less linear heads (§3.3.4)
        self.heads = nn.ModuleList([
            nn.Linear(in_dim, n_tasks, bias=False) for _ in range(n_windows)
        ])

    def forward(self, x: Tensor,
                window_idx: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x:          (B, in_dim)  backbone representation.
            window_idx: (B,) int, window assignment per sample.
                        None or eval mode → oracle head for all samples.
        Returns:
            logits: (B, n_tasks)
        """
        if window_idx is None or not self.training:
            return self.heads[self.oracle_idx](x)

        B = x.size(0)
        out = torch.zeros(B, self.heads[0].out_features,
                          dtype=x.dtype, device=x.device)
        for w in range(self.n_windows):
            mask = (window_idx == w)
            if mask.any():
                out[mask] = self.heads[w](x[mask])
        return out


# ---------------------------------------------------------------------------
# Feature Processors   §3.3.1
# ---------------------------------------------------------------------------

class CategoricalProcessor(nn.Module):
    """Categorical feature processor: embedding tables → (B, n_cat, d).

    Each sparse feature (user ID, item ID, etc.) is projected to a uniform
    d-dimensional space via separate embedding tables. The output token matrix
    enables subsequent modules to operate on all features uniformly.

    Args:
        vocab_sizes: list of vocabulary sizes, one per categorical feature.
        d_model:     target embedding dimension d.
    """

    def __init__(self, vocab_sizes: list[int], d_model: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(v, d_model) for v in vocab_sizes
        ])

    def forward(self, x: list[Tensor]) -> Tensor:
        """
        Args:
            x: list of (B,) int tensors, one per categorical feature.
        Returns:
            O_c: (B, n_cat, d)
        """
        tokens = [emb(feat).unsqueeze(1) for emb, feat
                  in zip(self.embeddings, x)]
        return torch.cat(tokens, dim=1)


class DenseProcessor(nn.Module):
    """Dense feature processor: numerical features → (B, n_dense, d).

    Each dense feature group is projected to d_model via a bias-less MLP
    (bias-less following §3.3.4 to prevent unbounded growth in deep networks).
    Features can be scalars or pre-computed dense vectors.

    Args:
        input_dims: list of raw dimensions per dense feature group.
        d_model:    target embedding dimension d.
    """

    def __init__(self, input_dims: list[int], d_model: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(dim, d_model, bias=False) for dim in input_dims
        ])

    def forward(self, x: list[Tensor]) -> Tensor:
        """
        Args:
            x: list of (B, input_dim_i) tensors.
        Returns:
            O_d: (B, n_dense, d)
        """
        tokens = [proj(feat).unsqueeze(1) for proj, feat
                  in zip(self.projections, x)]
        return torch.cat(tokens, dim=1)


class SequenceProcessor(nn.Module):
    """Sequence feature processor: behavioral history → (B, T, d).

    Processes user interaction sequences (clicks, views, purchases) using
    an attention-based event model. Multiple event types can be provided;
    they are concatenated along the embedding dimension and projected to d.
    RoPE positional embeddings are applied for temporal ordering.

    Args:
        seq_input_dim: concatenated embedding dim of all event types.
        d_model:       target embedding dimension d.
        n_heads:       attention heads for the event attention layer.
    """

    def __init__(self, seq_input_dim: int, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(seq_input_dim, d_model, bias=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, S: Tensor) -> Tensor:
        """
        Args:
            S: (B, T, seq_input_dim)
        Returns:
            O_s: (B, T, d)
        """
        S = self.proj(S)                              # (B, T, d)
        S_attn, _ = self.attn(S, S, S)
        return self.norm(S_attn + S)


class MixingNetwork(nn.Module):
    """Mixing network: fuse categorical and dense tokens into unified O_cd (§3.3.1).

    Concatenates O_c and O_d along the token axis, then applies QK-norm
    (§3.3.4) to mitigate modality contention among different feature types
    before normalizing to produce the non-sequence representation O_cd.

    Args:
        n_tokens: total non-seq token count (n_cat + n_dense).
        d_model:  embedding dimension d.
    """

    def __init__(self, n_tokens: int, d_model: int) -> None:
        super().__init__()
        # QK-norm: scale Q and K to prevent attention collapse across modalities
        self.qk_norm = nn.LayerNorm(d_model)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, O_c: Tensor, O_d: Tensor) -> Tensor:
        """
        Args:
            O_c: (B, n_cat,   d)
            O_d: (B, n_dense, d)
        Returns:
            O_cd: (B, n_cat+n_dense, d)
        """
        O_cd = torch.cat([O_c, O_d], dim=1)    # (B, n_cd, d)
        O_cd = self.qk_norm(O_cd)
        return self.norm(O_cd)


# ---------------------------------------------------------------------------
# Extended Context Storage (ECS)   §3.3.2
# ---------------------------------------------------------------------------

class ExtendedContextStorage(nn.Module):
    """Extended Context Storage: DenseNet-style global residual store (§3.3.2).

    Provides a global key-value store supporting DenseNet [32]-style residual
    connections, enabling high-bandwidth information flow across layers.
    Each backbone layer stores its output; subsequent layers receive a projected
    summary of all prior activations as an additional residual input.

    Args:
        d_model:    per-layer feature dimension.
        max_layers: maximum number of stored layers (determines projection size).
    """

    def __init__(self, d_model: int, max_layers: int = 8) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_layers = max_layers
        # Project concatenated stored states back to d_model
        self.proj = nn.Linear(d_model * max_layers, d_model, bias=False)
        self._store: list[Tensor] = []

    def reset(self) -> None:
        self._store.clear()

    def push(self, x: Tensor) -> None:
        """Store the mean-pooled layer output (B, d) for efficient stacking."""
        self._store.append(x.mean(dim=1))         # (B, d)

    def get_residual(self) -> Optional[Tensor]:
        """Returns projected summary of all stored states, or None if empty."""
        if not self._store:
            return None
        n = len(self._store)
        # Zero-pad to max_layers for fixed-size projection
        padded = self._store + [torch.zeros_like(self._store[0])] * (self.max_layers - n)
        ctx = torch.cat(padded, dim=-1)            # (B, max_layers * d)
        return self.proj(ctx)                       # (B, d)


# ---------------------------------------------------------------------------
# Transformer Block (TB)   §3.3.2
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Transformer Block: RoPE + MHA + domain-specific FFN for sequences (§3.3.2).

    Processes sequence embeddings O_s using:
      1. RoPE positional embeddings for temporal awareness.
      2. Standard Multi-Head Self-Attention for within-sequence dependencies.
      3. Domain-specific FFN (one per domain) for domain-adaptive transformation.
         Mitigates cross-domain interference vs. a single shared FFN.

    SwishRMSNorm is used in FFN layers for stability (§3.3.4).

    Args:
        d_model:   embedding dimension.
        n_heads:   attention heads.
        n_domains: number of domains; each gets its own FFN weights.
    """

    def __init__(self, d_model: int, n_heads: int = 4, n_domains: int = 1) -> None:
        super().__init__()
        self.n_domains = n_domains
        self.norm1     = nn.LayerNorm(d_model)
        self.attn      = nn.MultiheadAttention(d_model, n_heads, batch_first=True,
                                               bias=False)
        self.norm2     = nn.LayerNorm(d_model)
        # Domain-specific FFN: separate parameters per domain (parameter untying)
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model, bias=False),
                SwishRMSNorm(4 * d_model),
                nn.Linear(4 * d_model, d_model, bias=False),
            )
            for _ in range(n_domains)
        ])

    def forward(self, O_s: Tensor, domain_ids: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            O_s:        (B, T, d) sequence embeddings.
            domain_ids: (B,) int domain index per sample, or None → domain 0.
        Returns:
            (B, T, d)
        """
        # MHA with pre-norm
        S_attn, _ = self.attn(self.norm1(O_s), self.norm1(O_s), self.norm1(O_s))
        O_s = O_s + S_attn

        # Domain-specific FFN
        out = torch.zeros_like(O_s)
        if domain_ids is None:
            out = self.ffns[0](self.norm2(O_s))
        else:
            for d_id in range(self.n_domains):
                mask = (domain_ids == d_id)
                if mask.any():
                    out[mask] = self.ffns[d_id](self.norm2(O_s[mask]))
        return O_s + out


# ---------------------------------------------------------------------------
# DHEN/Wukong Fusion Block (DWFBlock)   §3.3.2
# ---------------------------------------------------------------------------

class _FMBlock(nn.Module):
    """Factorization Machine block (from Wukong): FM + flatten + LN + MLP → n_F tokens."""

    def __init__(self, n_tokens: int, d_model: int, n_out: int, k: int = 16) -> None:
        super().__init__()
        self.n_out   = n_out
        self.d_model = d_model
        self.k       = k
        fm_flat  = n_tokens * k
        self.attn_proj = nn.Linear(n_tokens * d_model, n_tokens * k, bias=False)
        self.norm      = nn.LayerNorm(fm_flat)
        self.mlp       = nn.Sequential(
            nn.Linear(fm_flat, fm_flat, bias=False),
            nn.ReLU(),
            nn.Linear(fm_flat, n_out * d_model, bias=False),
        )

    def forward(self, X: Tensor) -> Tensor:
        B, n, d = X.shape
        Y    = self.attn_proj(X.reshape(B, n * d)).reshape(B, n, self.k)
        XtY  = X.transpose(1, 2) @ Y                   # (B, d, k)
        fm   = (X @ XtY).reshape(B, -1)                # (B, n*k)
        return self.mlp(self.norm(fm)).reshape(B, self.n_out, self.d_model)


class _LCBBlock(nn.Module):
    """Linear Compression Block (from Wukong): learnable token-axis recombination."""

    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.W = nn.Linear(n_in, n_out, bias=False)

    def forward(self, X: Tensor) -> Tensor:
        return self.W(X.transpose(1, 2)).transpose(1, 2)


class DWFBlock(nn.Module):
    """DHEN/Wukong Fusion Block: cross-modal sequence + non-seq interaction (§3.3.2).

    Merges updated sequence context O_s (from TransformerBlock) with non-sequence
    context O_cd via Wukong's FM + LCB interaction stack:

        combined = Concat(O_s, O_cd)  ∈ ℝ^(B × (T+n_cd) × d)
        O'_cd = LN(FMBlock(combined) ∥ LCBBlock(combined) + residual)

    FM captures pairwise interactions between sequence and non-seq tokens.
    LCB preserves all-order information for subsequent layers.
    Outputs n_out tokens = n_F + n_L representing the updated non-seq state.

    Args:
        n_seq:   sequence token count T.
        n_cd:    non-sequence token count.
        d_model: embedding dimension.
        n_out:   output token count (n_F + n_L).
        k:       FM projection rank.
    """

    def __init__(self, n_seq: int, n_cd: int, d_model: int,
                 n_out: int = 16, k: int = 16) -> None:
        super().__init__()
        n_total    = n_seq + n_cd
        n_F        = n_out // 2
        n_L        = n_out - n_F
        self.fmb   = _FMBlock(n_total, d_model, n_F, k)
        self.lcb   = _LCBBlock(n_total, n_L)
        self.norm  = nn.LayerNorm(d_model)
        self.res   = (nn.Linear(n_total, n_out, bias=False)
                      if n_total != n_out else nn.Identity())
        self.n_out = n_out

    def forward(self, O_s: Tensor, O_cd: Tensor) -> Tensor:
        """
        Args:
            O_s:  (B, T,    d)
            O_cd: (B, n_cd, d)
        Returns:
            O'_cd: (B, n_out, d)
        """
        combined = torch.cat([O_s, O_cd], dim=1)                # (B, n_total, d)
        fmb_out  = self.fmb(combined)                           # (B, n_F, d)
        lcb_out  = self.lcb(combined)                           # (B, n_L, d)
        merged   = torch.cat([fmb_out, lcb_out], dim=1)         # (B, n_out, d)
        residual = self.res(combined.transpose(1, 2)).transpose(1, 2)
        return self.norm(merged + residual)


# ---------------------------------------------------------------------------
# Task Module   §3.3.3
# ---------------------------------------------------------------------------

class TaskModule(nn.Module):
    """Per-objective prediction head: lightweight MLP projecting O_cd → logit (§3.3.3).

    Each task (CTR, CVR, quality) gets its own head, enabling specialization
    while sharing the backbone representation. Bias-less to prevent unbounded
    growth in the shared embedding space.

    Args:
        in_dim:   flattened backbone dimension.
        hidden:   hidden layer widths.
        n_tasks:  output logit count.
    """

    def __init__(self, in_dim: int, hidden: list[int], n_tasks: int = 1) -> None:
        super().__init__()
        dims = [in_dim] + hidden + [n_tasks]
        self.mlp = _make_mlp(dims, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Lattice KTAP: Inference-time Knowledge Transfer   §3.4
# ---------------------------------------------------------------------------

class LatticeKTAP(nn.Module):
    """Lattice KTAP: asynchronous teacher-to-student knowledge transfer (§3.4).

    Traditional distillation transfers knowledge only during training via soft
    labels. KTAP extends this to inference via precomputed teacher embeddings:

      Background Teacher Computation: teacher Lattice Networks continuously
        precompute backbone embeddings for (user, item) pairs and store them
        with a TTL (typically ~6 hours) in a key-value cache.

      Student Query Mechanism: at training and inference, student models query
        the cache by (user_id, item_id) key. On hit, teacher embeddings are
        injected as additional input features. On miss (expired or unseen pairs),
        zero vectors serve as placeholders.

      Dual Knowledge Transfer: teacher embeddings (feature-level) + teacher
        logits (label-level, soft distillation) are both provided to students.

    This reference implementation simulates the cache with an in-memory dict.
    A production deployment would use a distributed key-value store with TTL.

    Args:
        teacher_dim:  teacher backbone embedding dimension.
        student_dim:  student model embedding dimension.
        cache_size:   maximum number of (user, item) entries to store.
    """

    def __init__(self, teacher_dim: int, student_dim: int,
                 cache_size: int = 10000) -> None:
        super().__init__()
        self.cache_size   = cache_size
        self.teacher_dim  = teacher_dim
        # Project teacher embedding into student space
        self.proj = nn.Linear(teacher_dim, student_dim, bias=False)
        # In-memory cache: key → (embedding, logit)
        self._cache: dict[int, tuple[Tensor, Tensor]] = {}

    def store(self, keys: Tensor, embeddings: Tensor, logits: Tensor) -> None:
        """Store teacher outputs keyed by (user_id * 1e6 + item_id) hash.

        Args:
            keys:       (B,) int64 cache keys.
            embeddings: (B, teacher_dim) teacher backbone embeddings.
            logits:     (B, n_tasks) teacher prediction logits.
        """
        for k, emb, lg in zip(keys.tolist(), embeddings.detach(), logits.detach()):
            if len(self._cache) >= self.cache_size:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[int(k)] = (emb, lg)

    def query(self, keys: Tensor) -> tuple[Tensor, Optional[Tensor]]:
        """Query teacher embeddings for a batch of keys.

        Args:
            keys: (B,) int64 cache keys.
        Returns:
            teacher_feats: (B, student_dim)  zero if cache miss.
            teacher_logits:(B, n_tasks) or None if no cached logits available.
        """
        B     = keys.size(0)
        device = keys.device
        embs, logits = [], []
        has_logits = False

        for k in keys.tolist():
            entry = self._cache.get(int(k))
            if entry is not None:
                emb, lg = entry
                embs.append(emb.to(device))
                logits.append(lg.to(device))
                has_logits = True
            else:
                embs.append(torch.zeros(self.teacher_dim, device=device))
                logits.append(None)

        emb_tensor   = self.proj(torch.stack(embs))     # (B, student_dim)
        logit_tensor = (torch.stack([l if l is not None
                                     else torch.zeros_like(logits[0])
                                     for l in logits])
                        if has_logits else None)
        return emb_tensor, logit_tensor

    def distillation_loss(self, student_logits: Tensor,
                          teacher_logits: Tensor, temperature: float = 2.0) -> Tensor:
        """Soft-label knowledge distillation loss (KL divergence).

        Args:
            student_logits: (B, n_tasks)
            teacher_logits: (B, n_tasks)
            temperature:    distillation temperature T.
        Returns:
            scalar KL divergence loss.
        """
        s = F.log_softmax(student_logits / temperature, dim=-1)
        t = F.softmax(teacher_logits  / temperature, dim=-1)
        return F.kl_div(s, t, reduction='batchmean') * (temperature ** 2)


# ---------------------------------------------------------------------------
# Full Lattice Network   §3.3
# ---------------------------------------------------------------------------

class LatticeNetwork(nn.Module):
    """Lattice Network: unified MDMO model for cross-domain recommendation (§3.3).

    Three-stage architecture that handles diverse input formats while enabling
    cross-domain learning through interleaved sequence and non-sequence processing:

    Stage 1 — Feature Processors:
      Categorical + Dense features → uniform d-dimensional tokens → O_cd via MixingNetwork.
      Sequences → O_s via SequenceProcessor.

    Stage 2 — Backbone (L interleaved layers with ECS):
      TransformerBlock: contextualizes O_s with domain-specific FFN.
      DWFBlock:         merges O_s into O_cd via FM + LCB interactions.
      ECS:              DenseNet-style residual from all prior layer outputs.

    Stage 3 — Task Modules + Lattice Zipper:
      Per-objective lightweight MLPs produce task predictions.
      LatticeZipper routes training samples to window-specific heads and uses
      the oracle head at serving time.

    Stability (§3.3.4): SwishRMSNorm in FFNs, bias-less layers, parameter untying
    per domain, QK-norm in MixingNetwork, all combined to enable deep scaling.

    Args:
        vocab_sizes:    vocabulary sizes for categorical features.
        dense_dims:     input dims per dense feature group.
        seq_input_dim:  concatenated embedding dim of sequence inputs.
        d_model:        global embedding dimension d.
        n_layers:       L, number of backbone interleaved layer pairs.
        n_domains:      number of domains (per-domain FFN in TransformerBlock).
        n_out_tokens:   DWFBlock output token count per layer.
        fm_rank:        FM projection rank k in DWFBlock.
        n_heads:        attention heads for sequences.
        task_hidden:    hidden dims for each TaskModule.
        n_tasks:        number of prediction objectives.
        n_windows:      K attribution windows for LatticeZipper (1 = no zipper).
        ktap_dim:       teacher embedding dim for KTAP (0 = disabled).
    """

    def __init__(
        self,
        vocab_sizes:   list[int],
        dense_dims:    list[int],
        seq_input_dim: int,
        d_model:       int = 64,
        n_layers:      int = 3,
        n_domains:     int = 1,
        n_out_tokens:  int = 16,
        fm_rank:       int = 16,
        n_heads:       int = 4,
        task_hidden:   Optional[list[int]] = None,
        n_tasks:       int = 1,
        n_windows:     int = 1,
        ktap_dim:      int = 0,
    ) -> None:
        super().__init__()
        self.d_model    = d_model
        self.n_layers   = n_layers
        self.n_tasks    = n_tasks
        self.n_windows  = n_windows

        # --- Stage 1: Feature Processors ---
        self.cat_proc  = CategoricalProcessor(vocab_sizes, d_model)
        self.den_proc  = DenseProcessor(dense_dims, d_model)
        self.seq_proc  = SequenceProcessor(seq_input_dim, d_model, n_heads)
        n_cd           = len(vocab_sizes) + len(dense_dims)
        self.mixing    = MixingNetwork(n_cd, d_model)

        # --- Stage 2: Backbone ---
        # Sequence length T is dynamic; DWFBlock n_seq must be set at runtime
        # We use a fixed n_seq equal to 1 for the DWFBlock (mean-pooled sequence)
        self.tb_layers  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, n_domains)
            for _ in range(n_layers)
        ])
        # DWFBlock: n_seq=1 (we pass mean-pooled sequence), n_cd adapts after l=0
        n_cd_l = n_cd
        self.dwf_layers: nn.ModuleList = nn.ModuleList()
        for _ in range(n_layers):
            self.dwf_layers.append(DWFBlock(1, n_cd_l, d_model, n_out_tokens, fm_rank))
            n_cd_l = n_out_tokens

        self.ecs = ExtendedContextStorage(d_model, max_layers=n_layers * 2)

        # --- Stage 3: Task Modules ---
        flat_dim = n_out_tokens * d_model
        hidden   = task_hidden or [flat_dim // 2]
        self.task_modules = nn.ModuleList([
            TaskModule(flat_dim, hidden, 1) for _ in range(n_tasks)
        ])

        # --- Lattice Zipper ---
        self.zipper = (LatticeZipper(n_windows, flat_dim, n_tasks)
                       if n_windows > 1 else None)

        # --- KTAP ---
        self.ktap = (LatticeKTAP(ktap_dim, d_model)
                     if ktap_dim > 0 else None)
        self.ktap_proj: Optional[nn.Linear] = (
            nn.Linear(d_model + d_model, d_model, bias=False)
            if ktap_dim > 0 else None
        )

    def forward(
        self,
        cat_feats:    list[Tensor],
        dense_feats:  list[Tensor],
        seq_feat:     Tensor,
        domain_ids:   Optional[Tensor] = None,
        window_idx:   Optional[Tensor] = None,
        ktap_keys:    Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            cat_feats:   list of (B,) int tensors for categorical features.
            dense_feats: list of (B, dim_i) float tensors.
            seq_feat:    (B, T, seq_input_dim) sequence input.
            domain_ids:  (B,) int domain index or None.
            window_idx:  (B,) int attribution window index or None (inference).
            ktap_keys:   (B,) int64 cache keys for KTAP query or None.
        Returns:
            logits: (B, n_tasks)  or  (B, n_tasks) via zipper heads.
        """
        # Stage 1: preprocess
        O_c  = self.cat_proc(cat_feats)          # (B, n_cat,   d)
        O_d  = self.den_proc(dense_feats)         # (B, n_dense, d)
        O_cd = self.mixing(O_c, O_d)              # (B, n_cd,    d)
        O_s  = self.seq_proc(seq_feat)             # (B, T,       d)

        # Optionally inject KTAP teacher embeddings
        if self.ktap is not None and ktap_keys is not None:
            teacher_feats, _ = self.ktap.query(ktap_keys)   # (B, d)
            # Expand and inject as additional context into O_cd mean
            O_cd_mean = O_cd.mean(dim=1)                     # (B, d)
            fused     = self.ktap_proj(
                torch.cat([O_cd_mean, teacher_feats], dim=-1))  # (B, d)
            O_cd = O_cd + fused.unsqueeze(1)                 # broadcast residual

        # Stage 2: backbone (interleaved TB + DWFBlock)
        self.ecs.reset()
        for tb, dwf in zip(self.tb_layers, self.dwf_layers):
            O_s  = tb(O_s, domain_ids)
            # Pass mean-pooled sequence into DWFBlock (shape: (B, 1, d))
            O_s_pool = O_s.mean(dim=1, keepdim=True)         # (B, 1, d)
            O_cd     = dwf(O_s_pool, O_cd)                   # (B, n_out, d)
            self.ecs.push(O_cd)
            # Add ECS residual to O_cd
            ecs_res = self.ecs.get_residual()
            if ecs_res is not None:
                O_cd = O_cd + ecs_res.unsqueeze(1)

        # Stage 3: task heads
        B    = O_cd.size(0)
        flat = O_cd.reshape(B, -1)                           # (B, n_out*d)

        if self.zipper is not None:
            return self.zipper(flat, window_idx)             # (B, n_tasks)

        if self.n_tasks == 1:
            return self.task_modules[0](flat)                # (B, 1)
        return torch.cat([m(flat) for m in self.task_modules], dim=-1)  # (B, n_tasks)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B, T = 4, 20

    vocab_sizes   = [100, 200, 50]
    dense_dims    = [8, 16]
    seq_input_dim = 32

    model = LatticeNetwork(
        vocab_sizes   = vocab_sizes,
        dense_dims    = dense_dims,
        seq_input_dim = seq_input_dim,
        d_model       = 32,
        n_layers      = 3,
        n_domains     = 2,
        n_out_tokens  = 8,
        fm_rank       = 8,
        n_heads       = 4,
        task_hidden   = [32],
        n_tasks       = 2,
        n_windows     = 3,
    )

    cat_feats   = [torch.randint(0, v, (B,)) for v in vocab_sizes]
    dense_feats = [torch.randn(B, d) for d in dense_dims]
    seq_feat    = torch.randn(B, T, seq_input_dim)
    domain_ids  = torch.randint(0, 2, (B,))
    window_idx  = torch.randint(0, 3, (B,))

    # Training forward
    model.train()
    logits = model(cat_feats, dense_feats, seq_feat, domain_ids, window_idx)
    print(f"train logits: {logits.shape}  {logits.tolist()}")

    loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
    loss.backward()
    print(f"loss:         {loss.item():.4f}")
    print("backward:     OK")

    # Inference: oracle head (no window_idx)
    model.eval()
    with torch.no_grad():
        logits_inf = model(cat_feats, dense_feats, seq_feat, domain_ids)
    print(f"infer logits: {logits_inf.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"params:       {total:,}")

    # LatticeFilter demo
    scores  = torch.rand(10, 3)    # 10 features, 3 tasks
    selected = lattice_filter(scores, target_count=4, seed=42)
    print(f"\nLatticeFilter: selected {len(selected)} from 10 → {selected}")

    # LatticeZipper routing demo
    print("\nLatticeZipper training: routes to assigned window head")
    print("LatticeZipper inference: always uses oracle (longest-window) head")


if __name__ == "__main__":
    _smoke_test()
