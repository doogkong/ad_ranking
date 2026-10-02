"""LIGE-GR: A Smooth Leap from Ranking to Generative Recommendation in the LLM Era.

Reference PyTorch implementation of LIGE-GR (Srinivas, He, Woodmansee, Lian,
Hu, Jiang, ... Liu; Meta), arXiv:2609.18148.

Industrial recommenders are itemwise: a ranker scores every candidate
independently, an item value model (VM) collapses the predicted engagement
signals into a scalar, and a greedy decoder takes the top-T (with a rule-based
"control layer" patching in diversity / business constraints). LIGE-GR upgrades
this into a listwise *generative* system without replacing the stack, by
upgrading exactly three components, each of which can be switched back to the
incumbent behavior:

  Eq. 3        Item value model                  -> ItemValueModel
  Eq. 5        Control layer CL(v_t | V_{t-1})   -> ControlLayer
  Sec 3.1      Context-free (CF) ranker          -> ContextFreeModel
               Context-aware (CA) causal refiner -> ContextAwareModel
  Eq. 8        ListVM_vanilla                    -> list_value(golden=False)
  Eq. 9        ListVM_golden (continuation-wtd)  -> list_value(golden=True)
  Eq. 10       Q(V+c) = ListVM_golden(V+c) + F   -> palette_decode
  Alg. 1       Palette beam decoder              -> palette_decode
  Eq. 11-12    Step-based future value           -> future_value_step
  Eq. 13-14    Duration-aware future value       -> future_value_duration
  Alg. 2       Serving path (pool, fallback)     -> LigeGR.generate
  Sec 5.1      Normalized entropy (NE)           -> normalized_entropy

Strict-generalization property (tested): Palette with CF in place of CA,
b=1, p_continue=1, F=0 is exactly the incumbent itemwise greedy decoder.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# score_fn(prefixes [N, t]) -> (task_probs [N, C, K], p_continue [N, C])
#   For each of N retained prefixes and every candidate c in C, the predicted
#   engagement probabilities and continuation probability of appending c.
ScoreFn = Callable[[Tensor], Tuple[Tensor, Tensor]]
# control_fn(prefixes [N, t]) -> CL [N, C]   (finite adjustment or -inf mask)
ControlFn = Callable[[Tensor], Tensor]


# ---------------------------------------------------------------------------
# Eq. 3 — item value model
# ---------------------------------------------------------------------------

class ItemValueModel(nn.Module):
    """itemVM(p) = w_1*p_like + w_2*p_follow + ... (Eq. 3): a fixed linear
    combination of predicted engagement signals. Weights are a buffer because
    in production they are a product-tuned value function, not learned."""

    def __init__(self, weights: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer("weights", torch.tensor(list(weights), dtype=torch.float32))

    @property
    def num_tasks(self) -> int:
        return self.weights.numel()

    def forward(self, probs: Tensor) -> Tensor:
        """probs [..., K] -> scalar value [...]."""
        return (probs * self.weights).sum(dim=-1)


# ---------------------------------------------------------------------------
# Section 2.2 — control layer CL(v_t | V_{t-1})
# ---------------------------------------------------------------------------

class ControlLayer:
    """Additive control-layer adjustment for every candidate given a prefix.

    Finite values shift the candidate's value-model score; -inf masks it.
    Three examples from Sec. 2.2 are supported (any subset can be enabled):

      * hard business rule: categories (a, b) in `forbidden_pairs` may not
        co-occur in a list                            -> CL = -inf
      * gap demotion: nearest same-category item in the prefix at position t'
        (1-indexed), candidate at t:  CL = -exp(t' - t + 1) * gap_coef
      * DPP diversity: CL = logdet(Phi_t^T Phi_t) - logdet(Phi_{t-1}^T Phi_{t-1})
        with unit-normalized item embeddings, scaled by `dpp_weight`

    Already-selected candidates are always masked (-inf): lists have distinct
    items.
    """

    def __init__(
        self,
        num_candidates: int,
        categories: Optional[Tensor] = None,
        embeddings: Optional[Tensor] = None,
        forbidden_pairs: Optional[Set[Tuple[int, int]]] = None,
        gap_coef: float = 0.0,
        dpp_weight: float = 0.0,
        dpp_eps: float = 1e-4,
    ) -> None:
        self.C = num_candidates
        self.categories = categories
        self.gap_coef = gap_coef
        self.dpp_weight = dpp_weight
        self.dpp_eps = dpp_eps
        self.forbidden = torch.zeros(0, 0, dtype=torch.bool)
        if forbidden_pairs:
            assert categories is not None, "forbidden_pairs needs categories"
            K = int(categories.max().item()) + 1
            self.forbidden = torch.zeros(K, K, dtype=torch.bool)
            for a, b in forbidden_pairs:
                self.forbidden[a, b] = self.forbidden[b, a] = True
        if gap_coef != 0.0:
            assert categories is not None, "gap demotion needs categories"
        self.embeddings = None
        if dpp_weight != 0.0:
            assert embeddings is not None, "DPP needs item embeddings"
            self.embeddings = F.normalize(embeddings, dim=-1)

    def __call__(self, prefixes: Tensor) -> Tensor:
        N, t = prefixes.shape
        C = self.C
        out = torch.zeros(N, C)
        # distinct items
        if t > 0:
            out.scatter_(1, prefixes, float("-inf"))
        if t == 0:
            return out
        if self.categories is not None:
            cat_prefix = self.categories[prefixes]                       # [N, t]
            same = cat_prefix.unsqueeze(1) == self.categories.view(1, C, 1)  # [N, C, t]
            if self.forbidden.numel() > 0:
                bad = self.forbidden[self.categories.view(1, C, 1), cat_prefix.unsqueeze(1)]
                out = out.masked_fill(bad.any(dim=-1), float("-inf"))
            if self.gap_coef != 0.0:
                pos = torch.arange(1, t + 1).view(1, 1, t)
                nearest = torch.where(same, pos, torch.zeros_like(pos)).amax(dim=-1)  # [N, C]; 0 = none
                gap = -torch.exp((nearest - (t + 1) + 1).float()) * self.gap_coef
                out = out + torch.where(nearest > 0, gap, torch.zeros_like(gap))
        if self.embeddings is not None:
            out = out + self.dpp_weight * self._dpp_increment(prefixes)
        return out

    def _dpp_increment(self, prefixes: Tensor) -> Tensor:
        N, t = prefixes.shape
        phi_prefix = self.embeddings[prefixes]                           # [N, t, De]
        phi = torch.cat([phi_prefix.unsqueeze(1).expand(N, self.C, t, -1),
                         self.embeddings.view(1, self.C, 1, -1).expand(N, self.C, 1, -1)], dim=2)
        gram = phi @ phi.transpose(-1, -2)                               # [N, C, t+1, t+1]
        eye = torch.eye(t + 1).expand_as(gram)
        logdet_new = torch.logdet(gram + self.dpp_eps * eye)
        gram_old = gram[:, 0, :t, :t]                                    # prefix is shared across c
        logdet_old = torch.logdet(gram_old + self.dpp_eps * torch.eye(t))
        return logdet_new - logdet_old.unsqueeze(1)


# ---------------------------------------------------------------------------
# Section 3.1 — listwise model: context-free + context-aware predictors
# ---------------------------------------------------------------------------

class ContextFreeModel(nn.Module):
    """Stand-in for the incumbent itemwise ranker (Fig. 3, left).

    Each candidate is scored with the user features by an interaction network
    producing the intermediate representation v'_t, then per-task heads (shared
    across items) output engagement logits. The last head is the continuation
    probability used by ListVM_golden.

    Args:
        user_dim / item_dim: dense input feature sizes.
        d_model: size of v'.
        num_tasks: K engagement tasks (the continue head is added on top).
    """

    def __init__(self, user_dim: int, item_dim: int, d_model: int, num_tasks: int, hidden: int = 64) -> None:
        super().__init__()
        self.interaction = nn.Sequential(
            nn.Linear(user_dim + item_dim + item_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, d_model),
        )
        self.item_proj = nn.Linear(item_dim, item_dim)
        self.task_nn = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, num_tasks + 1))

    def forward(self, user: Tensor, items: Tensor) -> Tuple[Tensor, Tensor]:
        """user [B, Du], items [B, C, Di] -> (v' [B, C, D], logits [B, C, K+1])."""
        B, C, _ = items.shape
        u = user.unsqueeze(1).expand(B, C, -1)
        v = self.interaction(torch.cat([u, items, self.item_proj(items) * items], dim=-1))
        return v, self.task_nn(v)


class ContextAwareModel(nn.Module):
    """Lightweight GPT-style causal Transformer refining the CF model (Fig. 3,
    right). Input is the sequence of CF intermediate representations
    (v'_1..v'_t); the output at position t is the context-aware prediction
    y_t | v_1..v_{t-1}. The paper uses 4 layers and 4 heads.

    v' is treated like LLM token embeddings; a learned positional embedding
    carries list position. Task heads share weights across positions.
    """

    def __init__(self, d_model: int, num_tasks: int, num_layers: int = 4, num_heads: int = 4,
                 max_len: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.task_nn = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_tasks + 1))

    def forward(self, v_seq: Tensor) -> Tensor:
        """v_seq [B, L, D] -> logits [B, L, K+1]; position t sees only <= t."""
        L = v_seq.shape[1]
        x = v_seq + self.pos(torch.arange(L, device=v_seq.device))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=v_seq.device), diagonal=1)
        return self.task_nn(self.norm(self.blocks(x, mask=causal)))


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def context_aware_loss(logits: Tensor, labels: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Multi-task BCE on logged lists. logits/labels [B, L, K+1]; mask [B, L]
    marks real (non-padded) positions. The CF model is preserved untouched, so
    train CA on detached v' (see `LigeGR.ca_training_step`)."""
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none").mean(dim=-1)
    if mask is None:
        return loss.mean()
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def normalized_entropy(probs: Tensor, labels: Tensor, eps: float = 1e-7) -> Tensor:
    """NE (Sec. 5.1): mean log-loss divided by the entropy of the background
    CTR. Lower is better."""
    p = probs.clamp(eps, 1 - eps)
    ll = -(labels * p.log() + (1 - labels) * (1 - p).log()).mean()
    base = labels.mean().clamp(eps, 1 - eps)
    h = -(base * base.log() + (1 - base) * (1 - base).log())
    return ll / h


def relative_ne_improvement(ne_cf: float, ne_ca: float) -> float:
    """Table 1: relative NE improvement of CA over CF; positive = CA better."""
    return (ne_cf - ne_ca) / ne_cf


def list_composition_metrics(topics: Sequence[int]) -> Dict[str, float]:
    """A few Table-4 list diagnostics on one list's topic ids: topic entropy,
    number of distinct topics, longest same-topic streak."""
    topics = list(topics)
    counts: Dict[int, int] = {}
    for x in topics:
        counts[x] = counts.get(x, 0) + 1
    n = len(topics)
    entropy = -sum((c / n) * math.log(c / n) for c in counts.values())
    streak = best = 1
    for a, b in zip(topics, topics[1:]):
        streak = streak + 1 if a == b else 1
        best = max(best, streak)
    return {"topic_entropy": entropy, "distinct_topics": float(len(counts)), "longest_streak": float(best)}


# ---------------------------------------------------------------------------
# Eq. 8-9 — listwise value model
# ---------------------------------------------------------------------------

def list_value(
    score_fn: ScoreFn,
    control_fn: ControlFn,
    item_vm: ItemValueModel,
    order: Sequence[int],
    golden: bool = True,
) -> float:
    """ListVM of an ordered list `order` of candidate indices.

    vanilla (Eq. 8): sum_t [itemVM(CA(u, v_t | V_{t-1})) + CL(v_t | V_{t-1})]
    golden  (Eq. 9): same, each term weighted by p_continue(V_{t-1}), with
                     p_continue(V_t) = p_continue(V_{t-1}) * CA_continue(v_t | V_{t-1}).
    """
    total, p_cont = 0.0, 1.0
    for t, c in enumerate(order):
        prefix = torch.tensor([list(order[:t])], dtype=torch.long).view(1, t)
        probs, cont = score_fn(prefix)
        s = item_vm(probs)[0, c].item() + control_fn(prefix)[0, c].item()
        total += (p_cont if golden else 1.0) * s
        if golden:
            p_cont *= cont[0, c].item()
    return total


# ---------------------------------------------------------------------------
# Eq. 11-14 — future-value estimators
# ---------------------------------------------------------------------------

def _geometric_sum(q: Tensor, remaining: int) -> Tensor:
    """sum_{j=1}^{remaining} q^j, elementwise over q; zero if remaining == 0."""
    if remaining <= 0:
        return torch.zeros_like(q)
    j = torch.arange(1, remaining + 1, dtype=q.dtype)
    return (q.unsqueeze(-1) ** j).sum(dim=-1)


def future_value_step(s_bar: Tensor, p_last: Tensor, remaining: int) -> Tensor:
    """Eq. 12: F_step(V_t) = s_bar(V_t) * sum_{j=1}^{T-t} p_last^j, where
    s_bar is the mean per-position VM+CL score of the prefix (Eq. 11) and
    p_last = CA_continue(u, v_t | V_{t-1}). Zero when nothing remains."""
    return s_bar * _geometric_sum(p_last, remaining)


def future_value_duration(s_bar: Tensor, p_last: Tensor, d_last: Tensor, d_bar: Tensor, remaining: int) -> Tensor:
    """Eq. 13-14: like `future_value_step`, but the last item's whole-item
    continuation probability is rescaled from its own duration d_t to the
    prefix-average duration d_bar via p ** (d_bar / d_t). Removes the
    step-based estimator's bias against prefixes ending in long items."""
    q = p_last ** (d_bar / d_last)
    return s_bar * _geometric_sum(q, remaining)


# ---------------------------------------------------------------------------
# Algorithm 1 — Palette decoding
# ---------------------------------------------------------------------------

@dataclass
class DecodeResult:
    order: List[int]                       # best list (candidate indices)
    list_value: float                      # its ListVM (golden or vanilla)
    beam: List[Tuple[List[int], float]] = field(default_factory=list)  # final (list, ListVM) per retained beam


def palette_decode(
    score_fn: ScoreFn,
    control_fn: ControlFn,
    item_vm: ItemValueModel,
    num_candidates: int,
    list_len: int,
    beam_width: int = 1,
    golden: bool = True,
    future: str = "none",
    durations: Optional[Tensor] = None,
) -> DecodeResult:
    """Palette: RL-style sequence decoder (Algorithm 1).

    State = selected prefix V, action = next candidate c, return so far =
    ListVM_golden(V+c), value-to-go = F_hat(V+c). Each step expands every
    retained prefix with every admissible candidate (CL > -inf), scores
    Q(V+c) = ListVM_golden(V+c) + F_hat(V+c) (Eq. 10), and keeps the top-b.
    The best full-length list by ListVM is returned.

    Args:
        golden: use continuation weighting (Eq. 9). False gives Eq. 8
            (p_continue == 1).
        future: "none" (F=0), "step" (Eq. 12), or "duration" (Eq. 14).
            Requires golden continuation probabilities; `durations` [C] is
            required for "duration".
    Setting score_fn to the CF scorer, beam_width=1, golden=False and
    future="none" recovers the incumbent itemwise greedy decoder.
    """
    assert future in ("none", "step", "duration")
    if future == "duration":
        assert durations is not None, "duration-aware estimator needs per-candidate durations"
    T = min(list_len, num_candidates)
    C = num_candidates

    prefixes = torch.zeros(1, 0, dtype=torch.long)
    lvm = torch.zeros(1)
    p_cont = torch.ones(1)
    score_sum = torch.zeros(1)
    dur_sum = torch.zeros(1)

    for t in range(T):
        probs, cont = score_fn(prefixes)                              # [N, C, K], [N, C]
        cl = control_fn(prefixes)                                     # [N, C]
        s = item_vm(probs) + cl                                       # [N, C]  (-inf where masked)
        new_lvm = lvm.unsqueeze(1) + (p_cont.unsqueeze(1) if golden else 1.0) * s
        new_pc = p_cont.unsqueeze(1) * cont if golden else torch.ones_like(s)
        new_sum = score_sum.unsqueeze(1) + s
        s_bar = new_sum / (t + 1)

        remaining = T - (t + 1)
        if future == "none" or remaining == 0:
            fut = torch.zeros_like(s)
        elif future == "step":
            fut = future_value_step(s_bar, cont, remaining)
        else:
            d_last = durations.view(1, C).expand_as(s)
            d_bar = (dur_sum.unsqueeze(1) + d_last) / (t + 1)
            fut = future_value_duration(s_bar, cont, d_last, d_bar, remaining)

        q = new_lvm + fut
        q = torch.where(torch.isfinite(s), q, torch.full_like(q, float("-inf")))
        k = min(beam_width, int(torch.isfinite(q).sum().item()))
        assert k > 0, "no admissible extension (control layer masked every candidate)"
        top = q.flatten().topk(k).indices
        n_idx, c_idx = top // C, top % C

        prefixes = torch.cat([prefixes[n_idx], c_idx.unsqueeze(1)], dim=1)
        lvm = new_lvm[n_idx, c_idx]
        p_cont = new_pc[n_idx, c_idx]
        score_sum = new_sum[n_idx, c_idx]
        if durations is not None:
            dur_sum = dur_sum[n_idx] + durations[c_idx]

    best = int(lvm.argmax().item())
    beam = [(prefixes[i].tolist(), lvm[i].item()) for i in range(prefixes.shape[0])]
    return DecodeResult(prefixes[best].tolist(), lvm[best].item(), beam)


# ---------------------------------------------------------------------------
# Scorers + end-to-end system (Alg. 2)
# ---------------------------------------------------------------------------

def make_cf_scorer(cf_probs: Tensor) -> ScoreFn:
    """Incumbent scorer: prefix-independent probs cf_probs [C, K+1] (K engagement
    + continue). Continuation is reported but unused by the incumbent decoder."""
    def score(prefixes: Tensor) -> Tuple[Tensor, Tensor]:
        N = prefixes.shape[0]
        return cf_probs[:, :-1].unsqueeze(0).expand(N, -1, -1), cf_probs[:, -1].unsqueeze(0).expand(N, -1)
    return score


def make_ca_scorer(ca: ContextAwareModel, v_cf: Tensor) -> ScoreFn:
    """Context-aware scorer over cached CF representations v_cf [C, D].

    For N prefixes and every candidate c, runs the lightweight causal
    Transformer on (v'_{V_1..V_t}, v'_c) and reads the last position: one
    batched forward over N*C short sequences, no re-encoding of the heavy CF
    model (Sec. 4.1)."""
    C, D = v_cf.shape

    @torch.no_grad()
    def score(prefixes: Tensor) -> Tuple[Tensor, Tensor]:
        N, t = prefixes.shape
        prefix_v = v_cf[prefixes].unsqueeze(1).expand(N, C, t, D)
        seq = torch.cat([prefix_v, v_cf.view(1, C, 1, D).expand(N, C, 1, D)], dim=2).reshape(N * C, t + 1, D)
        p = torch.sigmoid(ca(seq)[:, -1]).view(N, C, -1)
        return p[..., :-1], p[..., -1]
    return score


class LigeGR(nn.Module):
    """CF ranker + CA refiner + item VM, with the two-phase serving path."""

    def __init__(self, cf: ContextFreeModel, ca: ContextAwareModel, item_vm: ItemValueModel) -> None:
        super().__init__()
        self.cf, self.ca, self.item_vm = cf, ca, item_vm

    def ca_training_step(self, user: Tensor, list_items: Tensor, labels: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Loss for the CA module on logged lists; CF is run under no_grad so
        the incumbent model is preserved and CA can be updated independently
        (Sec. 4.2). user [B, Du], list_items [B, T, Di] in display order,
        labels [B, T, K+1]."""
        with torch.no_grad():
            v, _ = self.cf(user, list_items)
        return context_aware_loss(self.ca(v), labels, mask)

    @torch.no_grad()
    def generate(
        self,
        user: Tensor,
        items: Tensor,
        list_len: int,
        control_fn: ControlFn,
        beam_width: int = 1,
        golden: bool = True,
        future: str = "none",
        durations: Optional[Tensor] = None,
        pool_frac: float = 1.0,
        latency_budget_ms: Optional[float] = None,
        use_context_aware: bool = True,
    ) -> DecodeResult:
        """Algorithm 2 for one request (user [Du], items [C, Di]).

        Phase 1: CF forward once; cache v' and CF scores.
        Phase 2: Palette over the top `pool_frac` of candidates by CF item
        value (Sec. 4.3: paper re-scores roughly the top third). If phase 2
        exceeds `latency_budget_ms` or fails, fall back to the incumbent
        itemwise greedy decoder on cached CF scores. `use_context_aware=False`
        is the global configuration-level revert.
        """
        C = items.shape[0]
        v_cf, logits = self.cf(user.unsqueeze(0), items.unsqueeze(0))
        v_cf, cf_probs = v_cf[0], torch.sigmoid(logits[0])
        full_control = control_fn

        def incumbent() -> DecodeResult:
            return palette_decode(make_cf_scorer(cf_probs), full_control, self.item_vm, C, list_len,
                                  beam_width=1, golden=False, future="none")

        if not use_context_aware:
            return incumbent()

        start = time.perf_counter()
        try:
            pool_size = max(min(C, list_len), int(math.ceil(pool_frac * C)))
            pool = self.item_vm(cf_probs[:, :-1]).topk(pool_size).indices.sort().values
            # Control layer sees the full candidate space; restrict its columns to the pool.
            def pool_control(prefixes: Tensor) -> Tensor:
                return full_control(pool[prefixes] if prefixes.numel() else prefixes)[:, pool]
            res = palette_decode(
                make_ca_scorer(self.ca, v_cf[pool]),
                pool_control, self.item_vm, pool_size, list_len, beam_width, golden, future,
                None if durations is None else durations[pool],
            )
            if latency_budget_ms is not None and (time.perf_counter() - start) * 1e3 > latency_budget_ms:
                return incumbent()
            res.order = pool[res.order].tolist()
            res.beam = [(pool[o].tolist(), v) for o, v in res.beam]
            return res
        except RuntimeError:
            return incumbent()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    Du, Di, D, K, C, T = 12, 10, 32, 3, 30, 8

    cf = ContextFreeModel(Du, Di, D, K)
    ca = ContextAwareModel(D, K, num_layers=4, num_heads=4)
    item_vm = ItemValueModel([1.0, 2.0, 1.5])
    model = LigeGR(cf, ca, item_vm)

    # --- CA training on synthetic logged lists (CF frozen) ---
    user = torch.randn(16, Du)
    lst = torch.randn(16, T, Di)
    labels = (torch.rand(16, T, K + 1) < 0.3).float()
    opt = torch.optim.Adam(ca.parameters(), lr=1e-3)
    for step in range(3):
        loss = model.ca_training_step(user, lst, labels)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"CA training loss after 3 steps: {loss.item():.4f}")

    # --- decoding ---
    u, items = torch.randn(Du), torch.randn(C, Di)
    cats = torch.randint(0, 5, (C,))
    durs = torch.rand(C) * 50 + 5
    control = ControlLayer(C, categories=cats, gap_coef=0.5, embeddings=items, dpp_weight=0.2)

    base = model.generate(u, items, T, control, use_context_aware=False)
    print("\nincumbent greedy      :", base.order)
    for name, kw in [
        ("CA  b=1 vanilla      ", dict(beam_width=1, golden=False)),
        ("CA  b=6 vanilla      ", dict(beam_width=6, golden=False)),
        ("CA  b=6 golden+F_dur ", dict(beam_width=6, golden=True, future="duration", durations=durs)),
    ]:
        r = model.generate(u, items, T, control, pool_frac=1 / 3, **kw)
        print(f"{name}: {r.order}  ListVM={r.list_value:.3f}")
    print("topic metrics of last list:", list_composition_metrics(cats[r.order].tolist()))
