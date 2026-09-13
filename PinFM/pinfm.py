"""PinFM: Foundation Model for User Activity Sequences at a Billion-scale
Visual Discovery Platform.

Reference PyTorch implementation.
Paper: https://arxiv.org/abs/2507.12704 (Pinterest, Jul 2025).

Core idea: a single foundation model (FM), pretrained once on years of raw
cross-application user activity sequences, is fine-tuned into many
downstream ranking models (Home Feed, Related Items, ...) as a reusable
sequence-encoding module, rather than training a bespoke large sequence
model per application.

Two stages:

  1. Pretraining (Sec 3.1): a GPT2-style, pre-LN, causal decoder-only
     transformer backbone `M` consumes a sequence of (item id, action type,
     surface type) embeddings summed together, projected/L2-normalized by a
     small MLP phi_in, and produces a user representation sequence H via
     another MLP + L2-norm phi_out:

         H = phi_out(M(phi_in(E_item + E_action + E_surface)))            [Eq. 1]

     It is trained with three complementary InfoNCE-based (Eq. 2)
     next-item-prediction losses (`next_token_loss`, `multi_token_loss`,
     `future_token_loss`) rather than a single next-token loss, because
     users have multiple concurrent interests that a single-step objective
     under-captures.

  2. Fine-tuning (Sec 3.2): the pretrained backbone is spliced into a
     downstream ranking model (DLRM/DCN-style feature-crossing model) as the
     user-sequence encoder. The candidate item can be fused either late
     (encode the user sequence alone, reuse across all candidates, but not
     target-aware — `late_fusion_representation`) or early (append the
     candidate to the sequence so attention can directly model the
     user-candidate interaction — this is what the paper finds works best,
     see Table 1). Two techniques address candidate-id cold start when using
     early fusion (Sec 3.2): Candidate Item Randomization
     (`candidate_id_randomization`) and Item-age Dependent Dropout
     (`item_age_dependent_dropout`).

The paper's headline systems contribution is the Deduplicated
Cross-Attention Transformer (DCAT, Sec 4.1, Eq. 3-4), implemented here as
`PinFMDCAT`: because the same user's raw activity sequence is repeated for
every one of the ~1000s of candidates scored in a request, the (expensive)
context self-attention pass is run once per *unique* sequence in the batch,
its per-layer K/V cached, then broadcast back and cheaply cross-attended
against each candidate — a mechanism structurally similar to HSTU's
M-FALCON candidate caching (see `../HSTU/hstu.py`), but deduplicating on
identical raw input sequences within a batch rather than caching a single
user's history across a scoring pass.

Out of scope (infrastructure, not modeling contributions): the custom
Triton kernels implementing DCAT's deduplication/broadcast and the
CPU-embedding-table serving stack (Sec 4.2-4.3). `quantize_dequantize_embedding`
below is a numerical *simulation* of the paper's int4/int8 embedding
post-training quantization for measuring its accuracy impact, not the
actual bit-packed storage format.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Input embedding (Eq. 1, phi_in) and target-item encoder (psi)
# ---------------------------------------------------------------------------

class UserActivityEmbedding(nn.Module):
    """Builds the per-position transformer input: item id + action type +
    surface type embeddings, summed (not concatenated/interleaved), then
    projected and L2-normalized by phi_in (Eq. 1). Candidate items have no
    action embedding (they haven't been interacted with yet), so
    `action_ids`/`surface_ids` are optional.
    """

    def __init__(
        self,
        vocab_size: int,
        num_actions: int,
        num_surfaces: int,
        d_model: int,
        d_hidden: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.item_emb = nn.Embedding(vocab_size, d_model)
        self.action_emb = nn.Embedding(num_actions, d_model)
        self.surface_emb = nn.Embedding(num_surfaces, d_model)
        hidden = d_hidden or d_model
        self.phi_in = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )

    def forward(
        self,
        item_ids: Tensor,
        action_ids: Optional[Tensor] = None,
        surface_ids: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.item_emb(item_ids)
        if action_ids is not None:
            x = x + self.action_emb(action_ids)
        if surface_ids is not None:
            x = x + self.surface_emb(surface_ids)
        return F.normalize(self.phi_in(x), p=2, dim=-1)


class TargetItemEncoder(nn.Module):
    """psi: projects a raw item embedding into the contrastive target space
    used by the InfoNCE losses below (z_{i+1} = psi(emb(id_{i+1})), Eq. 2).
    Kept separate from `UserActivityEmbedding.phi_in` since the paper treats
    the "as a future target" representation and "as sequence input"
    representation as two different learned views of the same raw id
    embedding.
    """

    def __init__(self, d_model: int, d_hidden: Optional[int] = None) -> None:
        super().__init__()
        hidden = d_hidden or d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )

    def forward(self, item_emb: Tensor) -> Tensor:
        return F.normalize(self.mlp(item_emb), p=2, dim=-1)


class OutputProjection(nn.Module):
    """phi_out: projects the backbone's final hidden states into the
    L2-normalized user representation space H (Eq. 1)."""

    def __init__(self, d_model: int, d_hidden: Optional[int] = None) -> None:
        super().__init__()
        hidden = d_hidden or d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model)
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.mlp(x), p=2, dim=-1)


# ---------------------------------------------------------------------------
# Backbone: GPT2-style pre-LN causal transformer (Sec 3.1)
# ---------------------------------------------------------------------------

class CausalSelfAttentionLayer(nn.Module):
    """One pre-LN transformer block. Exposes `forward` (standard causal
    self-attention, also returning this layer's K/V for caching) and
    `forward_cross` (attend a small set of query tokens against an
    *externally supplied* K/V cache instead of self-attention) — the latter
    is what `PinFMDCAT`'s crossing component (Eq. 4) uses.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: Optional[int] = None) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        d_ff = d_ff or 4 * d_model
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def _qkv(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B, n, _ = x.shape
        qkv = self.qkv(x).view(B, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]  # each (B, H, n, head_dim)

    @staticmethod
    def _attend(Q: Tensor, K: Tensor, V: Tensor, causal: bool) -> Tensor:
        B, H, nq, hd = Q.shape
        nk = K.shape[2]
        logits = torch.matmul(Q, K.transpose(-1, -2)) * (hd ** -0.5)
        if causal:
            mask = torch.triu(torch.ones(nq, nk, dtype=torch.bool, device=Q.device), diagonal=1)
            logits = logits.masked_fill(mask, float("-inf"))
        weights = F.softmax(logits, dim=-1)
        out = torch.matmul(weights, V)
        return out.transpose(1, 2).reshape(B, nq, H * hd)

    def forward(self, x: Tensor, causal: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
        """Standard self-attention block. Returns (output, K, V) so K/V can
        be cached (this is the paper's context component, Eq. 3)."""
        h = self.ln1(x)
        Q, K, V = self._qkv(h)
        attn = self._attend(Q, K, V, causal=causal)
        x = x + self.out_proj(attn)
        x = x + self.ffn(self.ln2(x))
        return x, K, V

    def forward_cross(self, x_query: Tensor, K_ctx: Tensor, V_ctx: Tensor) -> Tensor:
        """Crossing component (Eq. 4): `x_query` (the candidate token(s))
        attends to `K_ctx`/`V_ctx` (broadcast context cache) concatenated
        with its own K/V — i.e. the candidate sees the full context plus
        itself, with no causal mask needed since it is always the most
        recent token."""
        h = self.ln1(x_query)
        Qc, Kc, Vc = self._qkv(h)
        K = torch.cat([K_ctx, Kc], dim=2)
        V = torch.cat([V_ctx, Vc], dim=2)
        attn = self._attend(Qc, K, V, causal=False)
        x = x_query + self.out_proj(attn)
        x = x + self.ffn(self.ln2(x))
        return x


class CausalTransformer(nn.Module):
    """Stack of `CausalSelfAttentionLayer`s: the paper's backbone `M`."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, d_ff: Optional[int] = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [CausalSelfAttentionLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, return_kv_cache: bool = False):
        cache: List[Tuple[Tensor, Tensor]] = []
        h = x
        for layer in self.layers:
            h, K, V = layer(h, causal=True)
            cache.append((K, V))
        h = self.final_ln(h)
        if return_kv_cache:
            return h, cache
        return h

    def forward_cross(self, x_query: Tensor, kv_cache: List[Tuple[Tensor, Tensor]]) -> Tensor:
        h = x_query
        for layer, (K_ctx, V_ctx) in zip(self.layers, kv_cache):
            h = layer.forward_cross(h, K_ctx, V_ctx)
        return self.final_ln(h)


# ---------------------------------------------------------------------------
# Pretraining losses (Sec 3.1, Eq. 2): InfoNCE next/multi/future-token loss
# ---------------------------------------------------------------------------

def info_nce_loss(anchor: Tensor, positive: Tensor, negatives: Tensor, temperature: Tensor) -> Tensor:
    """Eq. 2: InfoNCE contrastive loss with one positive and K in-batch
    negatives per anchor.

    Args:
        anchor: (K, d)
        positive: (K, d)
        negatives: (K, num_neg, d)
        temperature: scalar tensor (the paper makes this learnable).
    """
    pos_sim = (anchor * positive).sum(-1) / temperature
    neg_sim = torch.einsum("kd,knd->kn", anchor, negatives) / temperature
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def _sample_in_batch_negatives(pool: Tensor, num_samples: int, num_negatives: int, generator=None) -> Tensor:
    idx = torch.randint(0, pool.shape[0], (num_samples, num_negatives), generator=generator, device=pool.device)
    return pool[idx]


def next_token_loss(
    H: Tensor,
    Z: Tensor,
    is_positive: Tensor,
    num_negatives: int = 8,
    temperature: float = 0.1,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """L_ntl: predicts the immediate next positively-engaged item from each
    position's user representation.

    Args:
        H: (B, m, d) user representations (`OutputProjection` output).
        Z: (B, m, d) target-item embeddings (`TargetItemEncoder` output),
            aligned so Z[:, i] is the item interacted with at position i.
        is_positive: (B, m) bool, True where the action at position i is a
            positive engagement type (Sec 3.1, A_pos).
        num_negatives: in-batch negatives per anchor. Note: sampled from the
            whole batch's targets without excluding same-user positives, a
            simplification of the paper's exact negative-sampling scheme.
        temperature: InfoNCE temperature (paper makes this learnable).
    """
    B, m, d = H.shape
    anchors = H[:, :-1, :].reshape(-1, d)
    targets = Z[:, 1:, :].reshape(-1, d)
    valid = is_positive[:, 1:].reshape(-1)
    anchors, targets = anchors[valid], targets[valid]
    if anchors.shape[0] == 0:
        return H.new_tensor(0.0)
    pool = Z.reshape(-1, d)
    negatives = _sample_in_batch_negatives(pool, anchors.shape[0], num_negatives, generator)
    return info_nce_loss(anchors, targets, negatives, temperature)


def multi_token_loss(
    H: Tensor,
    Z: Tensor,
    is_positive: Tensor,
    window: int,
    num_negatives: int = 8,
    temperature: float = 0.1,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """L_mtl: in addition to the immediate next item, also predicts every
    positively-engaged item within a future window of length `window`,
    since user interests tend to stay consistent over short spans."""
    B, m, d = H.shape
    losses = []
    for offset in range(1, window + 1):
        if offset >= m:
            break
        anchors = H[:, :-offset, :].reshape(-1, d)
        targets = Z[:, offset:, :].reshape(-1, d)
        valid = is_positive[:, offset:].reshape(-1)
        a, t = anchors[valid], targets[valid]
        if a.shape[0] == 0:
            continue
        pool = Z.reshape(-1, d)
        negs = _sample_in_batch_negatives(pool, a.shape[0], num_negatives, generator)
        losses.append(info_nce_loss(a, t, negs, temperature))
    if not losses:
        return H.new_tensor(0.0)
    return torch.stack(losses).mean()


def future_token_loss(
    H: Tensor,
    Z: Tensor,
    is_positive: Tensor,
    l_d: int,
    window: int,
    num_negatives: int = 8,
    temperature: float = 0.1,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """L_ftl: predicts a future window of positive items using ONLY the
    representation at the fixed position `l_d` (the downstream ranking
    model's real-time sequence length) — sharpening the model specifically
    at the sequence length it will actually see at fine-tuning/serving time,
    analogous to instruction fine-tuning's fixed prompt-then-generate setup.
    """
    B, m, d = H.shape
    if l_d >= m:
        return H.new_tensor(0.0)
    anchor = H[:, l_d, :]
    losses = []
    for offset in range(1, window + 1):
        j = l_d + offset
        if j >= m:
            break
        targets = Z[:, j, :]
        valid = is_positive[:, j]
        a, t = anchor[valid], targets[valid]
        if a.shape[0] == 0:
            continue
        pool = Z.reshape(-1, d)
        negs = _sample_in_batch_negatives(pool, a.shape[0], num_negatives, generator)
        losses.append(info_nce_loss(a, t, negs, temperature))
    if not losses:
        return H.new_tensor(0.0)
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Deduplicated Cross-Attention Transformer, DCAT (Sec 4.1, Eq. 3-4)
# ---------------------------------------------------------------------------

def deduplicate_sequences(seq_ids: Tensor) -> Tuple[Tensor, Tensor]:
    """Psi (Eq. 3): deduplicates identical raw id sequences along the batch
    dimension.

    Args:
        seq_ids: (B, ...) integer tensor (e.g. a user's item/action/surface
            ids stacked along a trailing dim), with possibly many duplicate
            rows — the paper observes a 1:1000 (serving) / 1:10 (training)
            ratio of unique sequences to total scored candidates, since the
            same user's history is resent once per candidate.
    Returns:
        unique_seqs: (B_u, ...) deduplicated rows.
        inverse: (B,) LongTensor with unique_seqs[inverse] == seq_ids; index-
            selecting a per-unique-row tensor by `inverse` realizes Psi^-1.
    """
    flat = seq_ids.reshape(seq_ids.shape[0], -1)
    unique_flat, inverse = torch.unique(flat, dim=0, return_inverse=True)
    unique_seqs = unique_flat.reshape(unique_flat.shape[0], *seq_ids.shape[1:])
    return unique_seqs, inverse


def compute_savings(batch_size: int, num_unique: int) -> float:
    """Fraction of context-component compute avoided by deduplication. The
    paper reports ~1:1000 unique:total at serving time and ~1:10 at
    training time, i.e. compute_savings close to 0.999 / 0.9 respectively.
    """
    return 1.0 - num_unique / batch_size


class PinFMDCAT(nn.Module):
    """Deduplicated Cross-Attention Transformer (Fig. 1 early-fusion path +
    Sec 4.1): runs the (expensive) context self-attention pass only once per
    UNIQUE user sequence in the batch, then cross-attends every candidate
    item — one per batch row, including repeats — against its user's cached
    K/V, broadcast back via the dedup inverse indices.
    """

    def __init__(
        self,
        vocab_size: int,
        num_actions: int,
        num_surfaces: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
    ) -> None:
        super().__init__()
        self.user_emb = UserActivityEmbedding(vocab_size, num_actions, num_surfaces, d_model)
        self.backbone = CausalTransformer(d_model, n_layers, n_heads)
        self.out_proj = OutputProjection(d_model)

    def forward(
        self,
        user_item_ids: Tensor,
        user_action_ids: Tensor,
        user_surface_ids: Tensor,
        candidate_item_ids: Tensor,
    ) -> Tuple[Tensor, int]:
        """
        Args:
            user_item_ids, user_action_ids, user_surface_ids: (B, L) — one
                row per (user, candidate) request; rows repeat whenever the
                same user's context is scored against multiple candidates.
            candidate_item_ids: (B,) one candidate id per row.
        Returns:
            candidate_repr: (B, d) — early-fusion, target-aware candidate
                representation, ready for downstream feature crossing.
            num_unique: number of distinct context sequences the context
                component actually ran on (<=B), for reporting compute
                savings via `compute_savings`.
        """
        B, L = user_item_ids.shape
        stacked = torch.stack([user_item_ids, user_action_ids, user_surface_ids], dim=-1)  # (B, L, 3)
        unique_seqs, inverse = deduplicate_sequences(stacked)
        num_unique = unique_seqs.shape[0]
        u_item, u_action, u_surface = unique_seqs.unbind(-1)

        # Context component (Eq. 3): run the transformer once per unique sequence.
        ctx_emb = self.user_emb(u_item, u_action, u_surface)          # (B_u, L, d)
        _, kv_cache = self.backbone(ctx_emb, return_kv_cache=True)

        # Broadcast K/V back to the full batch (Psi^-1).
        kv_cache_full = [(K.index_select(0, inverse), V.index_select(0, inverse)) for K, V in kv_cache]

        # Crossing component (Eq. 4): cross-attend each candidate against its user's context.
        cand_emb = self.user_emb(candidate_item_ids.unsqueeze(1))     # (B, 1, d), no action/surface
        h = self.backbone.forward_cross(cand_emb, kv_cache_full)      # (B, 1, d)
        candidate_repr = self.out_proj(h.squeeze(1))
        return candidate_repr, num_unique


# ---------------------------------------------------------------------------
# Fusion strategies (Sec 3.2, Table 1)
# ---------------------------------------------------------------------------

def late_fusion_representation(H: Tensor, mode: str = "mean") -> Tensor:
    """Late fusion (Sec 3.2, "PinFM-lite-*" in Table 1): the pretrained
    module encodes the user sequence WITHOUT the candidate item appended,
    producing one user-level vector reusable across every candidate in the
    request. Cheaper and trivially cacheable, but not target-aware — the
    paper measures this as the weakest (though cheapest) input-sequence
    design (Table 1: +1.87-1.93% Save HIT@3 vs. +2.91-3.76% for early
    fusion / `PinFMDCAT`).

    Args:
        H: (B, m, d) user representation sequence (context only, no candidate).
        mode: "mean" (PinFM-lite-mean) or "last" (PinFM-lite-last).
    """
    if mode == "mean":
        return H.mean(dim=1)
    if mode == "last":
        return H[:, -1, :]
    raise ValueError(f"unknown mode: {mode!r}, expected 'mean' or 'last'")


# ---------------------------------------------------------------------------
# Cold-start handling (Sec 3.2)
# ---------------------------------------------------------------------------

def candidate_id_randomization(
    candidate_ids: Tensor,
    vocab_size: int,
    p: float = 0.1,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Candidate Item Randomization (CIR): replaces a `p` fraction of
    candidate item ids with a random id during fine-tuning, simulating
    cold-start (unseen-id) conditions so the downstream model does not
    over-rely on candidate id embeddings for items with no interaction
    history yet.
    """
    mask = torch.rand(candidate_ids.shape, generator=generator, device=candidate_ids.device) < p
    random_ids = torch.randint(
        0, vocab_size, candidate_ids.shape, generator=generator, device=candidate_ids.device
    )
    return torch.where(mask, random_ids, candidate_ids)


def item_age_dependent_dropout(
    embeddings: Tensor,
    item_age_days: Tensor,
    training: bool = True,
) -> Tensor:
    """Item-age Dependent Dropout (IDD): applies heavier dropout to
    PinFM-derived candidate embeddings for fresher items, so the downstream
    ranking model relies less on (unreliable, id-embedding-derived) signal
    for candidates with little interaction history:

        p = 0.7  if age <  7 days
        p = 0.5  if 7 <= age < 28 days
        p = 0.0  otherwise (item is "warm" enough for its id embedding to
                  be trustworthy)

    Args:
        embeddings: (..., d) PinFM candidate representations.
        item_age_days: (...) matching leading shape, item age in days.
        training: dropout is a no-op at eval time.
    """
    if not training:
        return embeddings
    p = torch.zeros_like(item_age_days, dtype=embeddings.dtype)
    p = torch.where(item_age_days < 7, torch.full_like(p, 0.7), p)
    p = torch.where((item_age_days >= 7) & (item_age_days < 28), torch.full_like(p, 0.5), p)
    keep_prob = (1.0 - p).clamp(min=1e-6)
    mask = torch.bernoulli(keep_prob)
    return embeddings * (mask / keep_prob).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Embedding quantization simulation (Sec 4.2)
# ---------------------------------------------------------------------------

def quantize_dequantize_embedding(x: Tensor, num_bits: int = 4) -> Tensor:
    """Simulates FBGEMM-style per-row min-max post-training quantization
    (Sec 4.2: 32-dim fp16 sub-embeddings -> int8/int4 + fp16 scale + fp16
    bias) by fake-quantizing then immediately dequantizing, so downstream
    code can measure accuracy impact without implementing the actual
    bit-packed storage format.
    """
    qmax = 2 ** num_bits - 1
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    scale = (x_max - x_min).clamp(min=1e-8) / qmax
    q = torch.round((x - x_min) / scale).clamp(0, qmax)
    return q * scale + x_min


def quantization_error(x: Tensor, num_bits: int = 4) -> float:
    """Relative L2 quantization error ||x - dequant(x)|| / ||x|| — the
    paper reports 0.45% at int8 and 7.8% at int4 on their production
    embedding tables (Sec 4.2)."""
    xq = quantize_dequantize_embedding(x, num_bits)
    denom = torch.linalg.vector_norm(x).clamp(min=1e-8)
    return (torch.linalg.vector_norm(x - xq) / denom).item()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    vocab_size, num_actions, num_surfaces = 500, 6, 3
    d_model, n_layers, n_heads = 32, 2, 4
    B, m = 4, 24

    print("--- Pretraining forward + losses ---")
    user_emb = UserActivityEmbedding(vocab_size, num_actions, num_surfaces, d_model)
    backbone = CausalTransformer(d_model, n_layers, n_heads)
    out_proj = OutputProjection(d_model)
    target_encoder = TargetItemEncoder(d_model)
    temperature = torch.tensor(0.1)

    item_ids = torch.randint(0, vocab_size, (B, m))
    action_ids = torch.randint(0, num_actions, (B, m))
    surface_ids = torch.randint(0, num_surfaces, (B, m))
    is_positive = torch.rand(B, m) > 0.4

    x = user_emb(item_ids, action_ids, surface_ids)
    h = backbone(x)
    H = out_proj(h)
    Z = target_encoder(user_emb.item_emb(item_ids))
    print(f"H: {H.shape}, Z: {Z.shape}")

    l_ntl = next_token_loss(H, Z, is_positive, temperature=temperature)
    l_mtl = multi_token_loss(H, Z, is_positive, window=4, temperature=temperature)
    l_ftl = future_token_loss(H, Z, is_positive, l_d=16, window=4, temperature=temperature)
    total = l_ntl + l_mtl + l_ftl
    total.backward()
    print(f"L_ntl={l_ntl.item():.4f}  L_mtl={l_mtl.item():.4f}  L_ftl={l_ftl.item():.4f}  backward: OK")

    print("\n--- Deduplicated Cross-Attention Transformer (DCAT) ---")
    dcat = PinFMDCAT(vocab_size, num_actions, num_surfaces, d_model, n_layers, n_heads)
    n_users, n_candidates_per_user, L = 2, 6, 16
    base_item = torch.randint(0, vocab_size, (n_users, L))
    base_action = torch.randint(0, num_actions, (n_users, L))
    base_surface = torch.randint(0, num_surfaces, (n_users, L))
    user_item_ids = base_item.repeat_interleave(n_candidates_per_user, dim=0)
    user_action_ids = base_action.repeat_interleave(n_candidates_per_user, dim=0)
    user_surface_ids = base_surface.repeat_interleave(n_candidates_per_user, dim=0)
    candidate_ids = torch.randint(0, vocab_size, (n_users * n_candidates_per_user,))

    candidate_repr, num_unique = dcat(user_item_ids, user_action_ids, user_surface_ids, candidate_ids)
    savings = compute_savings(user_item_ids.shape[0], num_unique)
    print(f"candidate_repr: {candidate_repr.shape}, unique contexts: {num_unique}/{user_item_ids.shape[0]}"
          f" (compute savings: {savings:.1%})")
    candidate_repr.sum().backward()
    print("backward: OK")

    print("\n--- Fusion strategies ---")
    late_mean = late_fusion_representation(H, mode="mean")
    late_last = late_fusion_representation(H, mode="last")
    print(f"late fusion (mean): {late_mean.shape}, (last): {late_last.shape}")

    print("\n--- Cold-start handling ---")
    many_candidate_ids = torch.randint(0, vocab_size, (2000,))
    randomized = candidate_id_randomization(many_candidate_ids, vocab_size, p=0.1)
    changed = (randomized != many_candidate_ids).float().mean().item()
    print(f"CIR changed {changed:.1%} of candidate ids (target ~10%)")

    item_age_days = torch.tensor([2.0, 10.0, 40.0, 3.0])
    dropped = item_age_dependent_dropout(candidate_repr[:4].detach(), item_age_days)
    print(f"IDD output: {dropped.shape}")

    print("\n--- Embedding quantization simulation ---")
    raw = torch.randn(1000, 32)
    err8 = quantization_error(raw, num_bits=8)
    err4 = quantization_error(raw, num_bits=4)
    print(f"relative L2 error: int8={err8:.4f}, int4={err4:.4f}  (paper: 0.0045, 0.078)")


if __name__ == "__main__":
    _smoke_test()
