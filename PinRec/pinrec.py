"""PinRec: Unified Generative Retrieval for Pinterest Recommender Systems.

Reference PyTorch implementation.
Paper: https://arxiv.org/abs/2504.10507 (Pinterest, KDD 2026).

Core idea: a single generative-retrieval transformer serves ALL of
Pinterest's recommendation surfaces (Home Feed, Search, Related Pins)
instead of one bespoke model per surface, via the same pretrain-then-
fine-tune paradigm as `../PinFM`. PinRec's own contributions are what makes
a single *retrieval* model usable when each surface defines success
differently (saves vs. outbound clicks vs. session depth):

  1. Outcome-Conditioned Generation (Sec 4.1, Eq. 1): the output head is
     conditioned on a desired outcome (e.g. "generate a Pin the user will
     Save" vs. "...will click through to a product") via learned condition
     embeddings summed into the hidden state before projection:

         i_hat_{u,t} = O(h_{u,t}, c_1, c_2, ..., c_k)

     This lets one model be steered, at generation time, toward whichever
     business metric a surface cares about, instead of training toward one
     generic engagement target and hoping it transfers.

  2. Cross-surface pretrain/fine-tune with impression negatives (Eq. 3-4):
     pretrained once on lifelong, cross-surface action sequences, then
     fine-tuned per-surface on that surface's impression logs, where
     impression-only (shown-but-not-acted-on) items become extra in-batch
     *hard* negatives for the sampled-softmax loss.

  3. Multi-step, multi-embedding autoregressive generation (Sec 4.4): unlike
     standard next-item generation, PinRec generates a whole sequence of
     candidate embeddings, one per outcome per step, feeding a sampled
     choice back into the sequence so generation adapts across outcomes.
     Because this produces far more embeddings than a two-tower model,
     PinRec needs `allocate_budget` (split retrieval budget N across
     outcomes per their target proportions) and `compress_embeddings`
     (merge near-duplicate generated embeddings before ANN search, which
     would otherwise retrieve overlapping candidate sets and hurt
     diversity).

The training objective (Eq. 2) is a sampled-softmax next-item loss with an
explicit popularity bias-correction term Q(i_c), estimated via a
Count-Min Sketch frequency counter (`CountMinSketch`) over item ids seen as
in-batch negatives -- the same style of correction used in PinnerFormer,
made explicit here.

Out of scope (infrastructure, not modeling contributions): the NVIDIA
Triton serving pipeline, KV-cache/CUDA-graph decode optimizations, and the
Faiss IVF-HNSW ANN index (Sec 4.5). Generation below re-runs the backbone
each step for clarity rather than using an incremental KV cache.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Item / temporal input encoding (Sec 4.2)
# ---------------------------------------------------------------------------

class ItemEmbedder(nn.Module):
    """f_tau: projects an entity-type-specific pretrained feature embedding
    (e.g. OmniSage for Pins, OmniSearchSage for search queries) into the
    model's shared d_model space via a small MLP + L2 norm. Because this
    operates on externally-computed content/engagement features rather than
    a learned item-id embedding table, it generalizes to items never seen
    during PinRec training -- the basis of PinRec's zero-shot retrieval
    (Sec 6.3).
    """

    def __init__(self, raw_dim: int, d_model: int, hidden_layers: int = 2) -> None:
        super().__init__()
        dims = [raw_dim] + [d_model] * hidden_layers
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.LayerNorm(dims[i + 1])]
        self.mlp = nn.Sequential(*layers)

    def forward(self, raw_feature: Tensor) -> Tensor:
        return F.normalize(self.mlp(raw_feature), p=2, dim=-1)


class TemporalEncoder(nn.Module):
    """Sinusoidal absolute-time embeddings at several predefined periods
    (Sec 4.2: daily/weekly/seasonal patterns) plus a log-scale relative-time
    embedding (time since the previous interaction) with a learnable phase,
    capturing both global temporal context and local session dynamics. This
    is a compact reference realization of the paper's description rather
    than a byte-exact reproduction (the paper does not give the encoder's
    exact functional form).
    """

    def __init__(self, d_model: int, periods: Tuple[float, ...] = (1.0, 7.0, 30.0)) -> None:
        super().__init__()
        self.periods = periods
        self.abs_proj = nn.Linear(2 * len(periods), d_model)
        self.rel_phase = nn.Parameter(torch.zeros(len(periods)))
        self.rel_proj = nn.Linear(2 * len(periods), d_model)

    def _sinusoid(self, t: Tensor, phase: Optional[Tensor] = None) -> Tensor:
        feats = []
        for i, p in enumerate(self.periods):
            angle = 2 * math.pi * t / p
            if phase is not None:
                angle = angle + phase[i]
            feats.append(torch.sin(angle))
            feats.append(torch.cos(angle))
        return torch.stack(feats, dim=-1)

    def forward(self, abs_time: Tensor, prev_time: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            abs_time: (...,) absolute timestamp (e.g. days since epoch).
            prev_time: (...,) timestamp of the previous interaction; if
                given, adds a log-scale relative-time component.
        """
        out = self.abs_proj(self._sinusoid(abs_time))
        if prev_time is not None:
            rel = torch.log1p((abs_time - prev_time).clamp(min=0))
            out = out + self.rel_proj(self._sinusoid(rel, phase=self.rel_phase))
        return out


class PinRecInputEncoder(nn.Module):
    """Builds the per-position transformer input x_{u,j} by summing an
    already-projected item/query representation with learned action and
    surface embeddings and the temporal encoding (Sec 4.2, "Sequence
    Construction")."""

    def __init__(
        self,
        d_model: int,
        num_actions: int,
        num_surfaces: int,
        periods: Tuple[float, ...] = (1.0, 7.0, 30.0),
    ) -> None:
        super().__init__()
        self.action_emb = nn.Embedding(num_actions, d_model)
        self.surface_emb = nn.Embedding(num_surfaces, d_model)
        self.temporal = TemporalEncoder(d_model, periods)

    def forward(
        self,
        item_repr: Tensor,
        action_ids: Tensor,
        surface_ids: Tensor,
        abs_time: Tensor,
        prev_time: Optional[Tensor] = None,
    ) -> Tensor:
        x = item_repr + self.action_emb(action_ids) + self.surface_emb(surface_ids)
        return x + self.temporal(abs_time, prev_time)


# ---------------------------------------------------------------------------
# Causal transformer backbone (Sec 4.1): h_{u,t} = TransformerStack(x_{u,1:t})
# ---------------------------------------------------------------------------

class CausalSelfAttentionLayer(nn.Module):
    """One pre-LN causal self-attention + FFN block."""

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

    def forward(self, x: Tensor) -> Tensor:
        B, n, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        logits = torch.matmul(Q, K.transpose(-1, -2)) * (self.head_dim ** -0.5)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
        logits = logits.masked_fill(mask, float("-inf"))
        attn = torch.matmul(F.softmax(logits, dim=-1), V)
        attn = attn.transpose(1, 2).reshape(B, n, D)
        x = x + self.out_proj(attn)
        return x + self.ffn(self.ln2(x))


class CausalTransformer(nn.Module):
    """Stack of causal self-attention layers, position embeddings included
    at the caller's discretion (paper adds learned position embeddings
    alongside the featurized interaction sequence, Sec 4.1)."""

    def __init__(self, d_model: int, n_layers: int, n_heads: int, max_len: int = 4096) -> None:
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([CausalSelfAttentionLayer(d_model, n_heads) for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        B, n, D = x.shape
        positions = torch.arange(n, device=x.device)
        h = x + self.pos_emb(positions).unsqueeze(0)
        for layer in self.layers:
            h = layer(h)
        return self.final_ln(h)


# ---------------------------------------------------------------------------
# Outcome-Conditioned Generation (Sec 4.1, Eq. 1)
# ---------------------------------------------------------------------------

class OutcomeEmbedding(nn.Module):
    """Learnable embeddings for outcome-conditioning signals (e.g. desired
    action type, surface constraint) -- the c_1, ..., c_k terms in Eq. 1."""

    def __init__(self, num_outcomes: int, d_model: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_outcomes, d_model)

    def forward(self, outcome_ids: Tensor) -> Tensor:
        return self.emb(outcome_ids)


class OutcomeConditionedHead(nn.Module):
    """Output head O (Eq. 1): maps a hidden state h_{u,t}, optionally summed
    with one or more outcome-condition embeddings c_1..c_k, to a predicted
    item representation. With no conditions this is unconditioned generation
    (PinRec-UC); with conditions, outcome-conditioned generation (PinRec-OC).
    """

    def __init__(self, d_model: int, d_hidden: Optional[int] = None) -> None:
        super().__init__()
        hidden = d_hidden or d_model
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))

    def forward(self, h: Tensor, conditions: Optional[List[Tensor]] = None) -> Tensor:
        x = h
        if conditions:
            for c in conditions:
                x = x + c
        return F.normalize(self.mlp(x), p=2, dim=-1)


# ---------------------------------------------------------------------------
# Count-Min Sketch popularity estimator (Eq. 2's bias term Q)
# ---------------------------------------------------------------------------

class CountMinSketch:
    """A small Count-Min Sketch frequency estimator (Cormode & Muthukrishnan,
    cited in the paper) used to estimate how often an item is sampled as an
    in-batch negative, so the sampled-softmax loss can subtract a popularity
    bias Q(i_c) (Eq. 2) -- without it, popular items would be systematically
    over-penalized as negatives relative to their true prevalence.
    """

    def __init__(self, width: int = 2048, depth: int = 4, seed: int = 0) -> None:
        self.width = width
        self.depth = depth
        g = torch.Generator().manual_seed(seed)
        self.seeds = torch.randint(1, 2 ** 31 - 1, (depth,), generator=g)
        self.table = torch.zeros(depth, width)

    def _hash(self, ids: Tensor) -> Tensor:
        flat = ids.reshape(-1).long()
        return ((flat.unsqueeze(0) * self.seeds.unsqueeze(1)) % self.width).abs() % self.width  # (depth, N)

    def update(self, ids: Tensor, count: float = 1.0) -> None:
        idx = self._hash(ids)
        for d in range(self.depth):
            self.table[d].scatter_add_(0, idx[d], torch.full((idx.shape[1],), float(count)))

    def estimate(self, ids: Tensor) -> Tensor:
        idx = self._hash(ids)
        counts = torch.stack([self.table[d, idx[d]] for d in range(self.depth)], dim=0)
        return counts.min(dim=0).values.reshape(ids.shape)

    def log_bias(self, ids: Tensor, eps: float = 1.0) -> Tensor:
        """Q(i_c) = log(estimated_frequency + eps)."""
        return torch.log(self.estimate(ids) + eps)


# ---------------------------------------------------------------------------
# Sampled softmax loss with popularity bias correction (Eq. 2-4)
# ---------------------------------------------------------------------------

def sampled_softmax_loss(
    anchor: Tensor,
    positive: Tensor,
    positive_ids: Tensor,
    negative_pool: Tensor,
    negative_pool_ids: Tensor,
    cms: CountMinSketch,
    lam: float = 1.0,
) -> Tensor:
    """Eq. 2: s(i_hat, i_c) = lambda * i_hat^T i_c - Q(i_c), softmax cross
    entropy over {positive} union {in-batch negatives}.

    Args:
        anchor: (B, d) predicted item representations i_hat.
        positive: (B, d) ground-truth next-item embeddings.
        positive_ids: (B,) ids of the positives, for the CMS bias lookup.
        negative_pool: (P, d) shared pool of candidate negative embeddings
            (e.g. all positives observed in the current batch).
        negative_pool_ids: (P,) ids of the negative pool.
        cms: a `CountMinSketch` already updated with item-frequency counts.
        lam: temperature/scale on the raw similarity (paper's lambda).
    """
    pos_score = lam * (anchor * positive).sum(-1) - cms.log_bias(positive_ids)
    neg_score = lam * (anchor @ negative_pool.T) - cms.log_bias(negative_pool_ids).unsqueeze(0)
    logits = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)
    labels = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)


def pretraining_next_item_loss(
    H: Tensor, Z: Tensor, item_ids: Tensor, cms: CountMinSketch, lam: float = 1.0
) -> Tensor:
    """Eq. 3: next-item prediction loss averaged over every position of a
    lifelong, action-only (impression-free) user sequence."""
    B, m, d = H.shape
    anchors = H[:, :-1, :].reshape(-1, d)
    positives = Z[:, 1:, :].reshape(-1, d)
    positive_ids = item_ids[:, 1:].reshape(-1)
    pool = Z.reshape(-1, d)
    pool_ids = item_ids.reshape(-1)
    return sampled_softmax_loss(anchors, positives, positive_ids, pool, pool_ids, cms, lam)


def finetuning_next_item_loss(
    H: Tensor,
    Z: Tensor,
    item_ids: Tensor,
    impression_embeds: Tensor,
    impression_ids: Tensor,
    cms: CountMinSketch,
    lam: float = 1.0,
) -> Tensor:
    """Eq. 4: like `pretraining_next_item_loss`, but also folds in
    impression-only (shown-but-not-acted-on) items from the surface's feed
    view as additional in-batch hard negatives (Fig. 2, "IN")."""
    B, m, d = H.shape
    anchors = H[:, :-1, :].reshape(-1, d)
    positives = Z[:, 1:, :].reshape(-1, d)
    positive_ids = item_ids[:, 1:].reshape(-1)
    pool = torch.cat([Z.reshape(-1, d), impression_embeds.reshape(-1, d)], dim=0)
    pool_ids = torch.cat([item_ids.reshape(-1), impression_ids.reshape(-1)], dim=0)
    return sampled_softmax_loss(anchors, positives, positive_ids, pool, pool_ids, cms, lam)


# ---------------------------------------------------------------------------
# Autoregressive generation (Sec 4.4, Fig. 3)
# ---------------------------------------------------------------------------

def generate_unconditional(
    backbone: CausalTransformer, head: OutcomeConditionedHead, context_emb: Tensor, num_steps: int
) -> Tensor:
    """PinRec-UC generation: repeatedly append the predicted embedding to
    the sequence and re-run the backbone. Reference implementation re-runs
    the full backbone each step for clarity; production serving instead
    reuses a KV cache (Sec 4.5, out of scope here).
    """
    seq = context_emb
    outputs = []
    for _ in range(num_steps):
        h = backbone(seq)[:, -1, :]
        pred = head(h)
        outputs.append(pred)
        seq = torch.cat([seq, pred.unsqueeze(1)], dim=1)
    return torch.stack(outputs, dim=1)  # (B, num_steps, d)


def generate_outcome_conditioned(
    backbone: CausalTransformer,
    head: OutcomeConditionedHead,
    context_emb: Tensor,
    outcome_condition_sets: List[List[Tensor]],
    num_steps: int,
    generator: Optional[torch.Generator] = None,
) -> Dict[int, Tensor]:
    """PinRec-OC generation (Fig. 3): at each step, a single forward pass
    produces one candidate embedding per outcome (multi-task-head style,
    via `outcome_condition_sets`); one outcome's embedding is then sampled
    and fed back into the sequence for the next step, so generation adapts
    jointly across outcomes, while candidate embeddings for EVERY outcome at
    every step are retained as separate retrieval candidates.

    Args:
        outcome_condition_sets: list of length K (num outcomes); each
            element is the list of condition embeddings (Eq. 1's c_1..c_k)
            for that outcome, each shaped (B, d).
    Returns:
        {outcome_index: (B, num_steps, d)} generated embeddings per outcome.
    """
    K = len(outcome_condition_sets)
    seq = context_emb
    outputs: Dict[int, List[Tensor]] = {k: [] for k in range(K)}
    for _ in range(num_steps):
        h = backbone(seq)[:, -1, :]
        step_preds = [head(h, outcome_condition_sets[k]) for k in range(K)]
        for k, pred in enumerate(step_preds):
            outputs[k].append(pred)
        chosen = torch.randint(0, K, (1,), generator=generator).item()
        seq = torch.cat([seq, step_preds[chosen].unsqueeze(1)], dim=1)
    return {k: torch.stack(v, dim=1) for k, v in outputs.items()}


# ---------------------------------------------------------------------------
# Budget Allocation & Embedding Compression (Sec 4.4)
# ---------------------------------------------------------------------------

def allocate_budget(total_budget: int, outcome_fractions: Dict[int, float]) -> Dict[int, int]:
    """Sec 4.4 Budget Allocation: splits the total retrieval budget N across
    outcome-conditioned generation branches according to specified
    fractions B (must sum to ~1), so each outcome receives its designated
    proportion of the total candidates retrieved.
    """
    total_frac = sum(outcome_fractions.values())
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(f"outcome_fractions must sum to 1.0, got {total_frac}")
    allocation = {k: int(round(total_budget * frac)) for k, frac in outcome_fractions.items()}
    drift = total_budget - sum(allocation.values())
    if drift != 0:
        k0 = max(outcome_fractions, key=outcome_fractions.get)
        allocation[k0] += drift
    return allocation


def compress_embeddings(
    embeddings: Tensor, budgets: Tensor, threshold: float = 0.9
) -> Tuple[Tensor, Tensor]:
    """Sec 4.4 Embedding Compression: merges each generated embedding (in
    generation order) into a previously *uncompressed* embedding if their
    cosine similarity exceeds `threshold`, summing their retrieval budgets.
    This reduces the overlapping candidate sets that near-duplicate
    autoregressive steps would otherwise retrieve, improving diversity.

    Args:
        embeddings: (n, d), assumed L2-normalized, in generation order.
        budgets: (n,) retrieval budget assigned to each embedding.
        threshold: cosine-similarity merge threshold (paper tunes ~0.9).
    Returns:
        compressed_embeddings: (m, d), m <= n.
        compressed_budgets: (m,)
    """
    kept_embeds: List[Tensor] = []
    kept_budgets: List[Tensor] = []
    for i in range(embeddings.shape[0]):
        e, b = embeddings[i], budgets[i]
        merged = False
        for j in range(len(kept_embeds)):
            sim = torch.dot(kept_embeds[j], e)
            if sim.item() >= threshold:
                kept_budgets[j] = kept_budgets[j] + b
                merged = True
                break
        if not merged:
            kept_embeds.append(e)
            kept_budgets.append(b)
    return torch.stack(kept_embeds), torch.stack(kept_budgets)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    d_model, n_layers, n_heads = 32, 2, 4
    num_actions, num_surfaces, num_outcomes = 6, 3, 4
    B, m, raw_dim = 4, 20, 64

    print("--- Input encoding ---")
    pin_embedder = ItemEmbedder(raw_dim, d_model)
    input_encoder = PinRecInputEncoder(d_model, num_actions, num_surfaces)
    raw_feats = torch.randn(B, m, raw_dim)
    item_repr = pin_embedder(raw_feats)
    action_ids = torch.randint(0, num_actions, (B, m))
    surface_ids = torch.randint(0, num_surfaces, (B, m))
    abs_time = torch.rand(B, m) * 365
    prev_time = torch.cat([abs_time[:, :1], abs_time[:, :-1]], dim=1)
    x = input_encoder(item_repr, action_ids, surface_ids, abs_time, prev_time)
    print(f"input sequence x: {x.shape}")

    print("\n--- Backbone + Outcome-Conditioned Head ---")
    backbone = CausalTransformer(d_model, n_layers, n_heads)
    head = OutcomeConditionedHead(d_model)
    outcome_emb = OutcomeEmbedding(num_outcomes, d_model)
    h = backbone(x)
    H_uc_seq = head(h)  # unconditioned, PinRec-UC, all positions
    H_uc = H_uc_seq[:, -1, :]
    cond = [outcome_emb(torch.zeros(B, dtype=torch.long))]
    H_oc = head(h[:, -1, :], cond)  # outcome-conditioned, PinRec-OC (last position)
    print(f"H (UC): {H_uc.shape}, H (OC): {H_oc.shape}")

    print("\n--- Sampled softmax loss with CMS bias correction ---")
    item_ids = torch.randint(0, 1000, (B, m))
    cms = CountMinSketch()
    cms.update(item_ids)
    l_pre = pretraining_next_item_loss(H_uc_seq, item_repr, item_ids, cms)
    impression_embeds = torch.randn(B, 5, d_model)
    impression_ids = torch.randint(0, 1000, (B, 5))
    cms.update(impression_ids)
    l_ft = finetuning_next_item_loss(H_uc_seq, item_repr, item_ids, impression_embeds, impression_ids, cms)
    (l_pre + l_ft).backward()
    print(f"L_pretrain={l_pre.item():.4f}  L_finetune={l_ft.item():.4f}  backward: OK")

    print("\n--- Autoregressive generation ---")
    context = x.detach()
    gen_uc = generate_unconditional(backbone, head, context, num_steps=3)
    print(f"unconditional generation: {gen_uc.shape}")

    K = 2
    condition_sets = [[outcome_emb(torch.full((B,), k, dtype=torch.long))] for k in range(K)]
    gen_oc = generate_outcome_conditioned(backbone, head, context, condition_sets, num_steps=3)
    for k, v in gen_oc.items():
        print(f"outcome {k} generation: {v.shape}")

    print("\n--- Budget allocation & embedding compression ---")
    allocation = allocate_budget(total_budget=100, outcome_fractions={0: 0.7, 1: 0.3})
    print(f"budget allocation: {allocation}")

    all_embeds = torch.cat([gen_oc[0][0], gen_oc[1][0]], dim=0)  # (2*num_steps, d), one user
    all_embeds = F.normalize(all_embeds, p=2, dim=-1)
    all_budgets = torch.ones(all_embeds.shape[0])
    compressed, comp_budgets = compress_embeddings(all_embeds.detach(), all_budgets, threshold=0.9)
    print(f"compressed {all_embeds.shape[0]} -> {compressed.shape[0]} embeddings, "
          f"budgets sum to {comp_budgets.sum().item():.0f} (should equal {all_embeds.shape[0]})")


if __name__ == "__main__":
    _smoke_test()
