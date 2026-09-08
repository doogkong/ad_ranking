"""HSTU: Hierarchical Sequential Transduction Units for Generative Recommenders.

Reference PyTorch implementation.
Paper: "Actions Speak Louder than Words: Trillion-Parameter Sequential
Transducers for Generative Recommendations" https://arxiv.org/abs/2402.17152
(Meta AI, Feb 2024).

Core idea: reformulate ranking/retrieval as sequential transduction over a
single unified time series of interleaved content and action tokens (a
"Generative Recommender", GR), and replace vanilla Transformer self-attention
with HSTU — an encoder tailored to recommendation data:

    U(X), V(X), Q(X), K(X) = Split(phi1(f1(X)))                     [Eq. 1]
    A(X)V(X) = phi2(Q(X)K(X)^T + rab^(p,t)) V(X)                    [Eq. 2]
    Y(X) = f2(Norm(A(X)V(X)) (elementwise*) U(X))                   [Eq. 3]

where f1, f2 are single linear layers, phi1 = phi2 = SiLU, Norm is LayerNorm,
and rab^(p,t) is a learned bias over relative position and (optionally)
relative time between two sequence entries. Each HSTU layer is wrapped in a
residual connection: X_out = X + Y(X).

Two design choices this replaces relative to a standard Transformer block:
  1. Softmax attention is replaced by a *pointwise* nonlinearity (SiLU)
     applied directly to the (biased, causally-masked) attention logits, with
     no row-wise normalization. The paper motivates this because softmax
     normalizes away exactly the information recommendation needs most: how
     *many* strong signals exist in a user's history, not just their relative
     order. It's also more forgiving of the non-stationary, unbounded
     vocabularies recommendation systems must serve online.
  2. The attention output is fused with a SwiGLU-style gate U(X) and a single
     output projection f2, replacing the usual attention-block + separate
     feed-forward block. This drops the linear-layer count outside attention
     from six (Q,K,V,O + 2 FFN layers) to two (f1, f2) per layer, which is
     what lets HSTU scale to far deeper stacks at the same memory budget.

This module also includes reference implementations of two secondary
techniques from the paper:
  - Stochastic Length (Sec 3.2, Eq. 4): a training-time trick that randomly
    subsamples long user histories to a shorter length most of the time,
    algorithmically increasing sparsity without materially hurting quality.
  - M-FALCON (Sec 3.4): cost-amortized inference. Because HSTU attention is
    causal, a user's history representation at each layer can be computed
    once and cached; scoring m candidate actions against that same history
    then costs O(m*n*d) instead of O(m*n^2*d). `HSTU.score_candidates`
    implements this caching + microbatching idea in simplified form.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Relative Attention Bias — rab^(p,t)
# ---------------------------------------------------------------------------

def _relative_position_bucket(relative_position: Tensor, num_buckets: int, max_distance: int) -> Tensor:
    """Log-scale bucketing of a (non-negative) relative distance into
    `num_buckets` bins: exact for small distances, logarithmically coarser
    for large ones (the same scheme popularized by T5's relative position
    bias, cited by the HSTU paper as the basis for rab^(p,t)).
    """
    relative_position = relative_position.clamp(min=0).round().long()
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact
    large = max_exact + (
        torch.log(relative_position.float().clamp(min=1) / max_exact)
        / math.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).long()
    large = large.clamp(max=num_buckets - 1)
    return torch.where(is_small, relative_position, large)


class RelativeAttentionBias(nn.Module):
    """rab^(p,t): a learned bias over relative position (p) and, optionally,
    relative time (t) between two sequence entries, added to raw attention
    logits before the pointwise nonlinearity (Eq. 2).

    Args:
        num_heads: number of attention heads H (bias is per-head).
        num_pos_buckets / max_pos_distance: bucketing for relative position.
        use_time_bias: also learn a bias from relative wall-clock time gaps
            (recommendation-specific: how *recently* something happened
            matters in a way it doesn't for token position in language).
        num_time_buckets / max_time_distance: bucketing for relative time.
    """

    def __init__(
        self,
        num_heads: int,
        num_pos_buckets: int = 32,
        max_pos_distance: int = 128,
        use_time_bias: bool = False,
        num_time_buckets: int = 32,
        max_time_distance: int = 86_400,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_pos_buckets = num_pos_buckets
        self.max_pos_distance = max_pos_distance
        self.pos_bias = nn.Embedding(num_pos_buckets, num_heads)

        self.use_time_bias = use_time_bias
        if use_time_bias:
            self.num_time_buckets = num_time_buckets
            self.max_time_distance = max_time_distance
            self.time_bias = nn.Embedding(num_time_buckets, num_heads)

    def _pos_bucket(self, rel: Tensor) -> Tensor:
        return _relative_position_bucket(rel, self.num_pos_buckets, self.max_pos_distance)

    def _time_bucket(self, rel: Tensor) -> Tensor:
        return _relative_position_bucket(rel.abs(), self.num_time_buckets, self.max_time_distance)

    def forward(self, seq_len: int, timestamps: Optional[Tensor] = None, device=None) -> Tensor:
        """Returns the full (n, n) relative bias matrix.

        Shape is (H, n, n) if `timestamps` is None (shared across the batch),
        or (B, H, n, n) if per-sample timestamps (B, n) are given.
        """
        device = device or self.pos_bias.weight.device
        idx = torch.arange(seq_len, device=device)
        rel = idx.view(-1, 1) - idx.view(1, -1)  # (n, n): query_pos - key_pos
        bias = self.pos_bias(self._pos_bucket(rel)).permute(2, 0, 1)  # (H, n, n)

        if self.use_time_bias and timestamps is not None:
            rel_t = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)  # (B, n, n)
            tbias = self.time_bias(self._time_bucket(rel_t)).permute(0, 3, 1, 2)  # (B, H, n, n)
            return bias.unsqueeze(0) + tbias
        return bias

    def offsets_bias(self, offsets: Tensor) -> Tensor:
        """Position-only bias for arbitrary relative offsets, shape (H, *offsets.shape).

        Used by `HSTU.score_candidates` to evaluate rab^(p,t) only at the
        handful of relative distances a new candidate token actually needs,
        instead of materializing a full (n, n) matrix.
        """
        bias = self.pos_bias(self._pos_bucket(offsets))  # (*offsets.shape, H)
        return bias.movedim(-1, 0).contiguous()  # (H, *offsets.shape)

    def self_bias(self, device=None) -> Tensor:
        """Bias at relative distance 0 (a token attending to itself), shape (H,)."""
        device = device or self.pos_bias.weight.device
        zero = torch.zeros(1, dtype=torch.long, device=device)
        return self.pos_bias(self._pos_bucket(zero)).squeeze(0)


# ---------------------------------------------------------------------------
# HSTU Layer (Eq. 1-3)
# ---------------------------------------------------------------------------

class HSTULayer(nn.Module):
    """One HSTU layer: pointwise projection, pointwise-aggregated attention,
    and a gated pointwise transformation, wrapped in a residual connection.

    Args:
        d_model: model width D.
        num_heads: number of attention heads H.
        dqk: per-head query/key dimension.
        dv: per-head value dimension (also the per-head width of the gate U).
    """

    def __init__(self, d_model: int, num_heads: int, dqk: int, dv: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dqk = dqk
        self.dv = dv
        self.d_inner = num_heads * dv
        self.f1 = nn.Linear(d_model, num_heads * (2 * dqk + 2 * dv))  # Eq.1 input proj
        self.f2 = nn.Linear(self.d_inner, d_model)                     # Eq.3 output proj
        self.norm = nn.LayerNorm(self.d_inner)
        self.scale = dqk ** -0.5

    def _project(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Eq. 1: U(X), V(X), Q(X), K(X) = Split(phi1(f1(X)))."""
        B, n, _ = x.shape
        H, dqk, dv = self.num_heads, self.dqk, self.dv
        proj = F.silu(self.f1(x))
        U, V, Q, K = proj.split([H * dv, H * dv, H * dqk, H * dqk], dim=-1)
        Q = Q.view(B, n, H, dqk).transpose(1, 2)  # (B, H, n, dqk)
        K = K.view(B, n, H, dqk).transpose(1, 2)
        V = V.view(B, n, H, dv).transpose(1, 2)   # (B, H, n, dv)
        return U, Q, K, V                          # U stays (B, n, H*dv)

    def forward(self, x: Tensor, rab: Tensor, causal_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            x: (B, n, D)
            rab: (1 or B, H, n, n) relative attention bias, added to logits.
            causal_mask: (1, 1, n, n) bool, True where the key position is in
                the future relative to the query position (masked out).
        Returns:
            x_out: (B, n, D) — residual output, Eq. 3 + skip connection.
            K, V: (B, H, n, dqk)/(B, H, n, dv) — this layer's keys/values,
                exposed so a caller can cache them (see HSTU.score_candidates).
        """
        B, n, D = x.shape
        U, Q, K, V = self._project(x)
        logits = torch.matmul(Q, K.transpose(-1, -2)) * self.scale  # Eq.2, QK^T
        logits = logits + rab
        A = F.silu(logits)  # Eq.2, phi2 — pointwise, NOT row-normalized like softmax
        # Masking logits with -inf before SiLU (unlike softmax) yields -inf*sigmoid(-inf) = NaN,
        # so causal masking is applied to the post-activation weights instead.
        A = A.masked_fill(causal_mask, 0.0)
        AV = torch.matmul(A, V)
        AV = AV.transpose(1, 2).reshape(B, n, self.d_inner)
        gated = self.norm(AV) * U  # Eq.3, SwiGLU-style gate
        Y = self.f2(gated)
        return x + Y, K, V

    def forward_incremental(
        self,
        cand_x: Tensor,
        K_hist: Tensor,
        V_hist: Tensor,
        rab_hist: Tensor,
        rab_self: Tensor,
    ) -> Tensor:
        """Scores a batch of candidate tokens that all occupy the same new
        sequence position, against a cached history's keys/values.

        Args:
            cand_x: (B, m, D) — m candidates' hidden state entering this layer.
            K_hist, V_hist: (B, H, n, dqk)/(B, H, n, dv) — cached from a prior
                full forward pass over the history (see HSTU.forward).
            rab_hist: (H, n) — bias from the (shared) candidate position to
                each history position.
            rab_self: (H,) — bias at relative distance 0 (self-attention term).
        Returns:
            (B, m, D) — updated candidate hidden state after this layer.
        """
        B, m, D = cand_x.shape
        n = K_hist.shape[2]
        U, Q, K, V = self._project(cand_x)  # Q,K: (B,H,m,dqk); V: (B,H,m,dv)

        logits_hist = torch.einsum("bhmd,bhnd->bhmn", Q, K_hist) * self.scale
        logits_hist = logits_hist + rab_hist.view(1, -1, 1, n)
        logits_self = (Q * K).sum(-1, keepdim=True) * self.scale  # diagonal self-term
        logits_self = logits_self + rab_self.view(1, -1, 1, 1)

        A_hist = F.silu(logits_hist)
        A_self = F.silu(logits_self)
        AV = torch.einsum("bhmn,bhnd->bhmd", A_hist, V_hist) + A_self * V
        AV = AV.transpose(1, 2).reshape(B, m, self.d_inner)
        gated = self.norm(AV) * U
        Y = self.f2(gated)
        return cand_x + Y


# ---------------------------------------------------------------------------
# HSTU Encoder (stack of L layers)
# ---------------------------------------------------------------------------

class HSTU(nn.Module):
    """Stack of L HSTULayers sharing one RelativeAttentionBias module.

    Consumes an already-unified sequence of d_model-dim tokens (the paper's
    "sequentialized, model-ready unified features" — interleaved content and
    action embeddings; see Fig. 2 of the paper). Tokenization / embedding of
    raw heterogeneous features is a DLRM-embedding-table concern orthogonal
    to the HSTU architecture itself and is out of scope here.

    Args:
        d_model: D.
        num_layers: L.
        num_heads: H.
        dqk / dv: per-head query-key / value dimensions.
        use_time_bias: enable the relative-time component of rab^(p,t).
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dqk: int,
        dv: int,
        use_time_bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            HSTULayer(d_model, num_heads, dqk, dv) for _ in range(num_layers)
        ])
        self.rab = RelativeAttentionBias(num_heads, use_time_bias=use_time_bias)

    def forward(
        self,
        x: Tensor,
        timestamps: Optional[Tensor] = None,
        return_cache: bool = False,
    ):
        """
        Args:
            x: (B, n, D) unified sequence of content/action embeddings.
            timestamps: optional (B, n) event times, for the time-bias term.
            return_cache: also return per-layer (K, V) for `score_candidates`.
        Returns:
            (B, n, D) encoded representations, and — if return_cache — a list
            of L (K, V) tensor pairs.
        """
        B, n, D = x.shape
        device = x.device
        rab = self.rab(n, timestamps, device=device)
        if rab.dim() == 3:
            rab = rab.unsqueeze(0)  # (1, H, n, n), broadcasts over batch
        causal_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)
        causal_mask = causal_mask.view(1, 1, n, n)

        cache: List[Tuple[Tensor, Tensor]] = []
        h = x
        for layer in self.layers:
            h, K, V = layer(h, rab, causal_mask)
            cache.append((K, V))
        if return_cache:
            return h, cache
        return h

    def score_candidates(
        self,
        cache: List[Tuple[Tensor, Tensor]],
        candidate_x: Tensor,
        microbatch_size: Optional[int] = None,
    ) -> Tensor:
        """M-FALCON-style amortized scoring (Sec. 3.4): scores `m` candidate
        tokens against ONE already-encoded history, reusing that history's
        cached per-layer keys/values instead of re-running the full O(n^2)
        encoder once per candidate. All m candidates are treated as
        alternatives for the same next sequence position, so they share an
        identical relative-position bias to the history and to themselves —
        exactly the simplification the paper uses to make their attention
        operations "exactly the same" and batchable.

        Args:
            cache: per-layer (K_hist, V_hist), from `forward(..., return_cache=True)`
                on a batch of size 1 (one user's history).
            candidate_x: (m, D) — m candidate token embeddings.
            microbatch_size: process candidates in chunks of this size
                (b_m in the paper); defaults to scoring all m at once.
        Returns:
            (m, D) — each candidate's encoded representation.
        """
        m = candidate_x.shape[0]
        history_len = cache[0][0].shape[2]
        device = candidate_x.device

        offsets = history_len - torch.arange(history_len, device=device)  # candidate_pos - j
        rab_hist = self.rab.offsets_bias(offsets)  # (H, n)
        rab_self = self.rab.self_bias(device=device)  # (H,)

        microbatch_size = microbatch_size or m
        outputs = []
        for start in range(0, m, microbatch_size):
            h = candidate_x[start:start + microbatch_size].unsqueeze(0)  # (1, b_m, D)
            for layer, (K_hist, V_hist) in zip(self.layers, cache):
                h = layer.forward_incremental(h, K_hist, V_hist, rab_hist, rab_self)
            outputs.append(h.squeeze(0))
        return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Task heads (Table 1: ranking & retrieval as sequential transduction)
# ---------------------------------------------------------------------------

class RankingHead(nn.Module):
    """Small MLP turning an HSTU output representation into multi-task logits,
    matching the "apply a small neural network ... into multi-task
    predictions" step used for the target-aware ranking formulation (Sec 2.2).
    """

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


def retrieval_scores(user_repr: Tensor, candidate_embeddings: Tensor) -> Tensor:
    """Dot-product retrieval score, p(Phi | u) surrogate (Sec 2.2):
    argmax_Phi p(Phi | u_i) is approximated by ranking candidates on
    <u_i, Phi> similarity.

    Args:
        user_repr: (..., D)
        candidate_embeddings: (C, D)
    Returns:
        (..., C) similarity scores.
    """
    return user_repr @ candidate_embeddings.transpose(-1, -2)


# ---------------------------------------------------------------------------
# Stochastic Length (Sec 3.2, Eq. 4)
# ---------------------------------------------------------------------------

def stochastic_length_target(
    seq_len: int,
    alpha: float,
    max_corpus_len: int,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Reference approximation of Stochastic Length's target-length rule (Eq. 4).

    Sequences no longer than max_corpus_len^(alpha/2) are always kept whole.
    Longer sequences are usually truncated to that same short length, but are
    kept at full length with probability max_corpus_len^alpha / seq_len^2 — so
    the encoder still occasionally trains on the full tail of long histories,
    which the paper shows preserves quality while cutting average compute.
    """
    short_len = max(1, int(round(max_corpus_len ** (alpha / 2))))
    if seq_len <= short_len:
        return seq_len
    keep_full_prob = min(1.0, (max_corpus_len ** alpha) / (seq_len ** 2))
    r = torch.rand((), generator=generator).item()
    return seq_len if r < keep_full_prob else short_len


def apply_stochastic_length(
    x: Tensor,
    alpha: float,
    max_corpus_len: int,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Subsamples a batch of sequences (B, n, D) down to a Stochastic-Length
    target, keeping the relative temporal order of the kept positions (a
    random ordered subset, rather than the paper's exact subsequence
    construction in Appendix F.1, which this is a simplified stand-in for).
    """
    B, n, D = x.shape
    target = stochastic_length_target(n, alpha, max_corpus_len, generator)
    if target >= n:
        return x
    idx = torch.randperm(n, generator=generator)[:target]
    idx = idx.sort().values
    return x[:, idx, :]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B, n, D = 2, 20, 32
    H, dqk, dv, L = 4, 8, 8, 3

    model = HSTU(d_model=D, num_layers=L, num_heads=H, dqk=dqk, dv=dv, use_time_bias=True)
    head = RankingHead(D, hidden_dims=[D], num_tasks=1)

    x = torch.randn(B, n, D)
    timestamps = torch.arange(n, dtype=torch.float32).unsqueeze(0).expand(B, n).contiguous()

    out = model(x, timestamps=timestamps)
    print(f"encoded:    {out.shape}")

    logits = head(out[:, -1, :])  # target-aware ranking read-off at the last position
    loss = F.binary_cross_entropy_with_logits(logits, torch.zeros(B, 1))
    loss.backward()
    print(f"loss:       {loss.item():.4f}")
    print("backward:   OK")

    total = sum(p.numel() for p in model.parameters())
    print(f"params:     {total:,}")

    print("\n--- M-FALCON-style candidate scoring ---")
    model.zero_grad()
    history = x[:1]  # score candidates against a single user's history
    _, cache = model(history, return_cache=True)
    candidates = torch.randn(6, D)
    scores = model.score_candidates(cache, candidates, microbatch_size=4)
    print(f"candidate reprs: {scores.shape}")
    candidate_logits = head(scores)
    candidate_logits.sum().backward()
    print("backward:        OK")

    print("\n--- Stochastic Length ---")
    long_x = torch.randn(1, 4096, D)
    short_x = apply_stochastic_length(long_x, alpha=1.6, max_corpus_len=4096)
    print(f"4096 -> {short_x.shape[1]} tokens (alpha=1.6)")


if __name__ == "__main__":
    _smoke_test()
