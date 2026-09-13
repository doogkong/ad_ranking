"""UniPinRec: Unifying Generative Retrieval and Ranking at Pinterest Scale.

Reference PyTorch implementation.
Paper: https://arxiv.org/abs/2606.00422 (Pinterest, 2026).

Core idea: retrieval and ranking are conventionally trained as two separate
transformers over the *same* user action history, duplicating parameters,
training compute, and (crucially) the O(n^2) cost of self-attention over
that history at serving time. UniPinRec, which builds directly on
`../PinRec`, unifies both stages into one model with one input format, one
training stage, and one shared serving computation, via three ideas:

  1. Masked Action Modeling, MAM (Sec 3.1.1): rather than interleaving item
     and action tokens (doubling sequence length, as HSTU/PinRec-style
     ranking extensions do), the action is concatenated with the item
     embedding along the *feature* dimension at the same sequence position,
     and randomly masked out (replaced by a dedicated [MASK] embedding)
     during training. This adds ranking supervision -- "what action did/
     will the user take here?" -- to the exact same non-interleaved
     sequence retrieval already uses, without inflating context length.
     Future/candidate positions have their action ALWAYS masked (Eq. 2),
     since at inference the action on an unscored candidate is unknown by
     definition -- this is what makes ranking a well-posed prediction task
     on the same architecture retrieval uses.

  2. A single attention pattern serving both stages (Fig. 2, "M-FALCON
     pattern", following HSTU/`../HSTU`'s M-FALCON candidate-scoring
     design): the model processes one concatenated sequence of past
     (history) tokens followed by future (candidate) tokens. Past tokens
     keep standard causal attention; each candidate attends to ALL past
     tokens (target-aware, like `../foundation_expert`'s TAE mask) but is
     blocked from attending to other candidates, so a candidate's score
     never depends on which other candidates happen to be in the same
     batch (`build_unipinrec_attention_mask`).

  3. A joint loss (Eq. 3-5): the same next-item sampled-softmax retrieval
     objective PinRec uses (`item_retrieval_loss`), applied at every past
     position (item embeddings are never masked, only actions are), PLUS a
     per-action-type binary cross-entropy ranking objective
     (`action_prediction_loss`) evaluated at every position whose action was
     masked -- both masked-out past positions and all candidate positions.

Because retrieval and ranking now share one backbone and one non-inflated
sequence, the (expensive) O(n^2) self-attention pass over a user's history
computed during retrieval can be cached and reused as-is by ranking, which
then only pays the O(nk) cost of cross-attending k candidates against that
cache (`encode_history` / `score_candidates` below) -- avoiding a second,
duplicate O(n^2) encoding pass. This is the same K/V-caching idea as HSTU's
M-FALCON and `../PinFM`'s DCAT, but applied *across* two different serving
stages (retrieval "prefill" -> ranking "decode") rather than within one.

Out of scope (infrastructure, not modeling): the flex-attention / fused CUDA
kernels that materialize the block-sparse mask efficiently, FP8 mixed
precision, and the cross-process GPU KV-cache pool used to physically share
memory between the retrieval and ranking Triton services (Sec 3.3, 4.2).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Masked Action Modeling input encoder (Sec 3.1.1)
# ---------------------------------------------------------------------------

def sample_action_mask(
    batch_size: int, n_positions: int, p_mask: float, generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Sec 3.1.1 masking schedule for PAST positions: each position's action
    is independently masked (input replaced with [MASK]) with probability
    `p_mask`. Returns a bool (B, n) tensor, True where the action must be
    masked. Future/candidate positions are always masked and don't need
    sampling -- just use `torch.ones(..., dtype=torch.bool)`.
    """
    return torch.rand(batch_size, n_positions, generator=generator, device=device) < p_mask


class MaskedActionInputEncoder(nn.Module):
    """Builds one sequence token per (item, action) pair by concatenating
    the action embedding onto the item embedding along the FEATURE
    dimension (not interleaving two separate sequence positions), then
    projecting to d_model. When a position's action is masked, its action
    embedding is replaced with a dedicated learned [MASK] embedding, an
    extra class appended to the action vocabulary -- Eq. 2's guarantee that
    the hidden state at a masked position depends only on item content and
    on *other, unmasked* positions' actions, never its own.
    """

    def __init__(self, d_item: int, num_action_types: int, d_model: int) -> None:
        super().__init__()
        self.mask_action_id = num_action_types  # dedicated [MASK] class
        self.action_emb = nn.Embedding(num_action_types + 1, d_item)
        self.proj = nn.Linear(2 * d_item, d_model)

    def forward(self, item_emb: Tensor, action_ids: Tensor, is_masked: Tensor) -> Tensor:
        """
        Args:
            item_emb: (B, n, d_item).
            action_ids: (B, n) long, ground-truth action type ids (ignored
                wherever `is_masked` is True).
            is_masked: (B, n) bool, True where the action input must be
                replaced with [MASK].
        Returns:
            (B, n, d_model) input sequence.
        """
        masked_action_ids = torch.where(
            is_masked, torch.full_like(action_ids, self.mask_action_id), action_ids
        )
        action_repr = self.action_emb(masked_action_ids)
        return self.proj(torch.cat([item_emb, action_repr], dim=-1))


# ---------------------------------------------------------------------------
# Unified attention pattern (Sec 3.1.1(3), Fig. 2): past causal + M-FALCON future
# ---------------------------------------------------------------------------

def build_unipinrec_attention_mask(n_past: int, n_future: int, device: Optional[torch.device] = None) -> Tensor:
    """The single attention pattern serving both retrieval and ranking in
    one forward pass over the concatenated [past; future] sequence:

        past   -> past:    standard causal (lower-triangular)
        future -> past:    full (every candidate sees the whole history --
                            target-aware, like `../foundation_expert`'s TAE mask)
        future -> future:  diagonal only (a candidate sees only itself, so
                            its score never depends on the other candidates
                            in the same batch -- the "M-FALCON pattern")
        past   -> future:  blocked (no leakage from candidates into history)

    Returns:
        (n_past+n_future, n_past+n_future) bool tensor, True = may attend.
    """
    n = n_past + n_future
    mask = torch.zeros(n, n, dtype=torch.bool, device=device)
    past_causal = ~torch.triu(torch.ones(n_past, n_past, dtype=torch.bool, device=device), diagonal=1)
    mask[:n_past, :n_past] = past_causal
    mask[n_past:, :n_past] = True
    mask[n_past:, n_past:] = torch.eye(n_future, dtype=torch.bool, device=device)
    return mask


# ---------------------------------------------------------------------------
# Transformer backbone: dense (training) + KV-cached (cross-stage serving) paths
# ---------------------------------------------------------------------------

class UniPinRecLayer(nn.Module):
    """One pre-LN self-attention + FFN block. `forward_masked` takes an
    arbitrary boolean attend-mask (used for the dense training pass and for
    the retrieval "prefill" causal pass); `forward_cross` implements the
    ranking "decode" step, cross-attending future/candidate tokens against a
    cached past K/V while enforcing the same future-blocks-future rule.
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
        return qkv[0], qkv[1], qkv[2]

    def forward_masked(self, x: Tensor, attn_mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """attn_mask: (n, n) bool, True = attend. Returns (out, K, V)."""
        h = self.ln1(x)
        Q, K, V = self._qkv(h)
        logits = torch.matmul(Q, K.transpose(-1, -2)) * (self.head_dim ** -0.5)
        logits = logits.masked_fill(~attn_mask, float("-inf"))
        attn = torch.matmul(F.softmax(logits, dim=-1), V)
        B, n, D = x.shape
        attn = attn.transpose(1, 2).reshape(B, n, D)
        x = x + self.out_proj(attn)
        return x + self.ffn(self.ln2(x)), K, V

    def forward_cross(self, x_future: Tensor, K_past: Tensor, V_past: Tensor) -> Tensor:
        """Ranking decode step: `x_future` (k candidates) attends to all of
        cached `K_past`/`V_past` plus itself only (future-future blocked to
        the diagonal), without recomputing self-attention over the past.
        """
        B, k, D = x_future.shape
        h = self.ln1(x_future)
        Qf, Kf, Vf = self._qkv(h)
        K = torch.cat([K_past, Kf], dim=2)
        V = torch.cat([V_past, Vf], dim=2)
        logits = torch.matmul(Qf, K.transpose(-1, -2)) * (self.head_dim ** -0.5)
        n_past = K_past.shape[2]
        future_logits = logits[..., n_past:]
        block_others = ~torch.eye(k, dtype=torch.bool, device=x_future.device)
        future_logits = future_logits.masked_fill(block_others, float("-inf"))
        logits = torch.cat([logits[..., :n_past], future_logits], dim=-1)
        attn = torch.matmul(F.softmax(logits, dim=-1), V)
        attn = attn.transpose(1, 2).reshape(B, k, D)
        x = x_future + self.out_proj(attn)
        return x + self.ffn(self.ln2(x))


class UniPinRecTransformer(nn.Module):
    """Stack of `UniPinRecLayer`s with two equivalent forward paths:
    `forward_combined` (one dense pass over [past; future], used in
    training) and `forward_self` + `forward_cross` (the KV-cached path used
    at serving time, where retrieval's prefill is reused by ranking's
    decode). `test_unipinrec.py` verifies both paths produce the same
    candidate hidden states.
    """

    def __init__(self, d_model: int, n_layers: int, n_heads: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([UniPinRecLayer(d_model, n_heads) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)

    def forward_combined(self, x_past: Tensor, x_future: Tensor) -> Tuple[Tensor, Tensor]:
        n_past, n_future = x_past.shape[1], x_future.shape[1]
        mask = build_unipinrec_attention_mask(n_past, n_future, device=x_past.device)
        x = torch.cat([x_past, x_future], dim=1)
        for layer in self.layers:
            x, _, _ = layer.forward_masked(x, mask)
        x = self.final_ln(x)
        return x[:, :n_past, :], x[:, n_past:, :]

    def forward_self(self, x_past: Tensor, return_kv_cache: bool = False):
        n = x_past.shape[1]
        causal = ~torch.triu(torch.ones(n, n, dtype=torch.bool, device=x_past.device), diagonal=1)
        cache: List[Tuple[Tensor, Tensor]] = []
        h = x_past
        for layer in self.layers:
            h, K, V = layer.forward_masked(h, causal)
            cache.append((K, V))
        h = self.final_ln(h)
        if return_kv_cache:
            return h, cache
        return h

    def forward_cross(self, x_future: Tensor, kv_cache: List[Tuple[Tensor, Tensor]]) -> Tensor:
        h = x_future
        for layer, (K_past, V_past) in zip(self.layers, kv_cache):
            h = layer.forward_cross(h, K_past, V_past)
        return self.final_ln(h)


# ---------------------------------------------------------------------------
# Output heads
# ---------------------------------------------------------------------------

class ItemOutputHead(nn.Module):
    """Retrieval head: projects a hidden state to an L2-normalized item
    representation for the next-item sampled-softmax objective."""

    def __init__(self, d_model: int, d_hidden: Optional[int] = None) -> None:
        super().__init__()
        hidden = d_hidden or d_model
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.mlp(x), p=2, dim=-1)


class TargetItemEncoder(nn.Module):
    """psi: projects a raw (d_item-dim) item embedding into the same
    d_model space as `ItemOutputHead`'s output, producing the contrastive
    targets Z used by `item_retrieval_loss` (mirrors `../PinRec`'s
    `TargetItemEncoder`)."""

    def __init__(self, d_item: int, d_model: int, d_hidden: Optional[int] = None) -> None:
        super().__init__()
        hidden = d_hidden or d_model
        self.mlp = nn.Sequential(nn.Linear(d_item, hidden), nn.GELU(), nn.Linear(hidden, d_model))

    def forward(self, item_emb: Tensor) -> Tensor:
        return F.normalize(self.mlp(item_emb), p=2, dim=-1)


class ActionPredictionHeads(nn.Module):
    """Ranking heads: one dedicated scalar-logit MLP head h_{psi_c} per
    action type c in {1..C} (Sec 3.1.1(3)), each independently predicting
    whether that action occurred/will occur at a given position."""

    def __init__(self, d_model: int, num_action_types: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(num_action_types)])

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat([head(x) for head in self.heads], dim=-1)  # (..., C)


# ---------------------------------------------------------------------------
# Count-Min Sketch + sampled softmax retrieval loss (Sec 3.0.1, shared with PinRec)
# ---------------------------------------------------------------------------

class CountMinSketch:
    """Frequency estimator for the retrieval loss's popularity bias
    correction Q(i_c) (Sec 3.0.1, Eq. 1) -- see `../PinRec/pinrec.py` for the
    identical mechanism; reimplemented here so this folder is self-contained.
    """

    def __init__(self, width: int = 2048, depth: int = 4, seed: int = 0) -> None:
        self.width = width
        self.depth = depth
        g = torch.Generator().manual_seed(seed)
        self.seeds = torch.randint(1, 2 ** 31 - 1, (depth,), generator=g)
        self.table = torch.zeros(depth, width)

    def _hash(self, ids: Tensor) -> Tensor:
        flat = ids.reshape(-1).long()
        return ((flat.unsqueeze(0) * self.seeds.unsqueeze(1)) % self.width).abs() % self.width

    def update(self, ids: Tensor, count: float = 1.0) -> None:
        idx = self._hash(ids)
        for d in range(self.depth):
            self.table[d].scatter_add_(0, idx[d], torch.full((idx.shape[1],), float(count)))

    def estimate(self, ids: Tensor) -> Tensor:
        idx = self._hash(ids)
        counts = torch.stack([self.table[d, idx[d]] for d in range(self.depth)], dim=0)
        return counts.min(dim=0).values.reshape(ids.shape)

    def log_bias(self, ids: Tensor, eps: float = 1.0) -> Tensor:
        return torch.log(self.estimate(ids) + eps)


def item_retrieval_loss(
    item_repr_past: Tensor, target_embeds: Tensor, item_ids: Tensor, cms: CountMinSketch, lam: float = 1.0
) -> Tensor:
    """Eq. 1/3: next-item sampled-softmax loss, applied at EVERY past
    position -- item content is never masked by MAM, only actions are, so
    the retrieval objective is unaffected by the ranking-side masking
    schedule.

    Args:
        item_repr_past: (B, n, d_model), `ItemOutputHead` output.
        target_embeds: (B, n, d_model), `TargetItemEncoder` output (NOT the
            raw d_item-dim item embeddings -- project them first).
        item_ids: (B, n) ids, for the CMS popularity-bias lookup.
    """
    B, n, d = item_repr_past.shape
    anchors = item_repr_past[:, :-1, :].reshape(-1, d)
    positives = target_embeds[:, 1:, :].reshape(-1, d)
    positive_ids = item_ids[:, 1:].reshape(-1)
    pool = target_embeds.reshape(-1, d)
    pool_ids = item_ids.reshape(-1)

    pos_score = lam * (anchors * positives).sum(-1) - cms.log_bias(positive_ids)
    neg_score = lam * (anchors @ pool.T) - cms.log_bias(pool_ids).unsqueeze(0)
    logits = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)
    labels = torch.zeros(anchors.shape[0], dtype=torch.long, device=anchors.device)
    return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Action prediction (ranking) loss (Eq. 4)
# ---------------------------------------------------------------------------

def action_prediction_loss(
    action_logits_past: Tensor,
    action_labels_past: Tensor,
    past_masked: Tensor,
    action_logits_future: Tensor,
    action_labels_future: Tensor,
    action_weights: Tensor,
) -> Tensor:
    """Eq. 4: for each action type c, a per-head BCE loss averaged over the
    masked past positions M_past, PLUS a BCE loss averaged over all k
    candidate positions (always masked) -- summed across action types with
    hyperparameter weights `action_weights` (w_c).

    Args:
        action_logits_past, action_labels_past: (B, n_past, C).
        past_masked: (B, n_past) bool, True at masked past positions (M_past).
        action_logits_future, action_labels_future: (B, k, C).
        action_weights: (C,).
    """
    C = action_logits_past.shape[-1]
    total = action_logits_past.new_tensor(0.0)
    for c in range(C):
        past_logits_c = action_logits_past[..., c][past_masked]
        past_labels_c = action_labels_past[..., c][past_masked]
        past_term = (
            F.binary_cross_entropy_with_logits(past_logits_c, past_labels_c)
            if past_logits_c.numel() > 0
            else action_logits_past.new_tensor(0.0)
        )
        future_term = F.binary_cross_entropy_with_logits(
            action_logits_future[..., c].reshape(-1), action_labels_future[..., c].reshape(-1)
        )
        total = total + action_weights[c] * (past_term + future_term)
    return total


# ---------------------------------------------------------------------------
# Full model: joint training + cross-stage KV-cache serving
# ---------------------------------------------------------------------------

class UniPinRecModel(nn.Module):
    """Wraps `MaskedActionInputEncoder` + `UniPinRecTransformer` +
    `ItemOutputHead` + `ActionPredictionHeads` into the paper's full
    training and serving flow.
    """

    def __init__(self, d_item: int, num_action_types: int, d_model: int, n_layers: int, n_heads: int) -> None:
        super().__init__()
        self.input_encoder = MaskedActionInputEncoder(d_item, num_action_types, d_model)
        self.backbone = UniPinRecTransformer(d_model, n_layers, n_heads)
        self.item_head = ItemOutputHead(d_model)
        self.action_heads = ActionPredictionHeads(d_model, num_action_types)

    def forward_train(
        self,
        item_emb_past: Tensor,
        action_ids_past: Tensor,
        item_emb_future: Tensor,
        p_mask: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Single joint forward pass (Sec 3.1.1(4)): builds the combined
        past+future sequence with the MAM masking schedule applied, and runs
        ONE transformer pass that produces both the retrieval signal
        (item_repr_past) and the ranking signal (both action logit tensors).
        """
        B, n_past, _ = item_emb_past.shape
        k = item_emb_future.shape[1]
        device = item_emb_past.device

        past_masked = sample_action_mask(B, n_past, p_mask, generator, device)
        future_masked = torch.ones(B, k, dtype=torch.bool, device=device)

        x_past = self.input_encoder(item_emb_past, action_ids_past, past_masked)
        placeholder_action = torch.zeros(B, k, dtype=torch.long, device=device)
        x_future = self.input_encoder(item_emb_future, placeholder_action, future_masked)

        z_past, z_future = self.backbone.forward_combined(x_past, x_future)

        item_repr_past = self.item_head(z_past)
        action_logits_past = self.action_heads(z_past)
        action_logits_future = self.action_heads(z_future)
        return item_repr_past, action_logits_past, action_logits_future, past_masked

    def encode_history(
        self, item_emb_past: Tensor, action_ids_past: Tensor, past_masked: Tensor
    ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """Retrieval "prefill": encode the user's history once, returning
        both the retrieval item representations and a K/V cache for reuse."""
        x_past = self.input_encoder(item_emb_past, action_ids_past, past_masked)
        z_past, kv_cache = self.backbone.forward_self(x_past, return_kv_cache=True)
        return self.item_head(z_past), kv_cache

    def score_candidates(self, kv_cache: List[Tuple[Tensor, Tensor]], item_emb_future: Tensor) -> Tensor:
        """Ranking "decode": reuse retrieval's cached K/V, paying only
        O(n*k) instead of re-running O(n^2) self-attention over history."""
        B, k, _ = item_emb_future.shape
        device = item_emb_future.device
        placeholder_action = torch.zeros(B, k, dtype=torch.long, device=device)
        future_masked = torch.ones(B, k, dtype=torch.bool, device=device)
        x_future = self.input_encoder(item_emb_future, placeholder_action, future_masked)
        z_future = self.backbone.forward_cross(x_future, kv_cache)
        return self.action_heads(z_future)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    d_item, d_model, n_layers, n_heads = 24, 32, 2, 4
    num_action_types = 4
    B, n_past, k = 3, 16, 5

    model = UniPinRecModel(d_item, num_action_types, d_model, n_layers, n_heads)
    target_encoder = TargetItemEncoder(d_item, d_model)

    item_emb_past = torch.randn(B, n_past, d_item)
    action_ids_past = torch.randint(0, num_action_types, (B, n_past))
    item_emb_future = torch.randn(B, k, d_item)

    print("--- Joint training forward pass ---")
    item_repr_past, action_logits_past, action_logits_future, past_masked = model.forward_train(
        item_emb_past, action_ids_past, item_emb_future, p_mask=0.2
    )
    print(f"item_repr_past: {item_repr_past.shape}, action_logits_past: {action_logits_past.shape}, "
          f"action_logits_future: {action_logits_future.shape}")
    print(f"fraction of past positions masked: {past_masked.float().mean().item():.2f} (target ~0.2)")

    item_ids = torch.randint(0, 1000, (B, n_past))
    cms = CountMinSketch()
    cms.update(item_ids)
    Z = target_encoder(item_emb_past)
    l_item = item_retrieval_loss(item_repr_past, Z, item_ids, cms)

    action_labels_past = torch.randint(0, 2, action_logits_past.shape).float()
    action_labels_future = torch.randint(0, 2, action_logits_future.shape).float()
    action_weights = torch.ones(num_action_types)
    l_action = action_prediction_loss(
        action_logits_past, action_labels_past, past_masked,
        action_logits_future, action_labels_future, action_weights,
    )
    total_loss = l_item + l_action
    total_loss.backward()
    print(f"L_item={l_item.item():.4f}  L_action={l_action.item():.4f}  L_total={total_loss.item():.4f}  "
          f"backward: OK")

    print("\n--- Cross-stage KV-cache reuse (retrieval prefill -> ranking decode) ---")
    model.zero_grad()
    with torch.no_grad():
        item_repr_cached, kv_cache = model.encode_history(item_emb_past, action_ids_past, past_masked)
        action_logits_cached = model.score_candidates(kv_cache, item_emb_future)

        # naive: recompute the SAME masked sequence in one dense joint pass
        # and compare the future (candidate) block -- both paths must agree.
        future_masked = torch.ones(B, k, dtype=torch.bool)
        x_past = model.input_encoder(item_emb_past, action_ids_past, past_masked)
        placeholder_action = torch.zeros(B, k, dtype=torch.long)
        x_future = model.input_encoder(item_emb_future, placeholder_action, future_masked)
        _, z_future_dense = model.backbone.forward_combined(x_past, x_future)
        action_logits_dense = model.action_heads(z_future_dense)
    print(f"cached candidate logits:  {action_logits_cached.shape}")
    match = torch.allclose(action_logits_cached, action_logits_dense, atol=1e-4)
    print(f"matches dense joint forward pass (same masking): {match}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nparams: {total_params:,}")


if __name__ == "__main__":
    _smoke_test()
