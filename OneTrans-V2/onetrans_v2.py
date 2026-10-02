"""OneTrans-V2: Unifying Retrieval, Pre-rank, and Fine-rank with One Transformer.

Reference PyTorch implementation of OneTrans-V2 (ByteDance Global E-Commerce
Recommendation Foundation Team), arXiv:2609.28589.

A recommendation cascade (retrieval -> pre-rank -> fine-rank) is usually three
separately trained/served models that each re-encode the same user behavior
sequence. OneTrans-V2 is ONE causal Transformer for all three stages:

  * the candidate-independent behavior sequence S is encoded once into a shared
    user context (a per-layer K/V cache);
  * each stage appends its own stage-specific tokens (retrieval: CTX + decision
    + SID tokens; pre-rank: 1 token; fine-rank: several tokens) that use
    token-specific Q/K/V/FFN parameters (mixed parameterization) and read the
    shared context through a *stage visibility mask*;
  * all three tasks are trained jointly, with in-model distillation from
    fine-rank (teacher) to pre-rank (student).

Paper section -> code:

  4.1      Stage visibility mask, shared context  -> build_stage_mask, OneTransV2Backbone
  4.2.1    DCGR (decisions -> SID codes), Eq. 1-4  -> OneTransV2.forward / .retrieve
           Business steering beta*phi(z), Eq. 4    -> OneTransV2.retrieve(beta, phi)
           Classifier-free guidance, Eq. 5         -> OneTransV2.retrieve(cfg_weight)
  4.2.2    Pre-/fine-rank BCE, Eq. 6-7             -> OneTransV2.loss
           Mean-centered KD, Eq. 8                 -> distillation_loss
  4.2.3    Overall objective, Eq. 9                -> OneTransV2.loss
  4.3      TransBlock (Alg. 1): pre-RMSNorm, fused QKVG, QKNorm, GQA, gated
           attention, residual multiplier 1/sqrt(2N), sparse MoE (shared expert
           + sigmoid router) -> TransBlock, SparseMoE
           muP / Depth-muP (Table 1), AdamW        -> mup_param_groups, build_optimizers
  4.4      Sequence-Native Training (one encoding, many exposures with causal
           anchors) and request-relative time rotary encoding, Eq. 10-11
                                                   -> compute_anchors, rotate_time
  5.2      Single-pass decision enumeration + two-stage top-k beam search
                                                   -> OneTransV2.retrieve, two_stage_topk
  6.1      HitRate@M                               -> hit_rate_at_m

Not reproduced: kernel fusion, FP8/FP16 serving, sparse embedding tables, and
the proprietary feature pipeline / RQ-KMeans SID training (SIDs are inputs).
"""

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

STAGES = ("R", "P", "F")        # retrieval, pre-rank, fine-rank
N_RETR_TOKENS = 5               # CTX, DT^p, DT^d, SID_0, SID_1  (SID_2 is only an output)


@dataclass
class Config:
    # vocabularies / feature sizes
    item_vocab: int = 1000
    n_actions: int = 4
    mm_dim: int = 0                   # multimodal side-information dim per behavior (0 = off)
    ctx_dim: int = 16                 # retrieval features NS_r
    pre_dim: int = 8                  # pre-rank features NS_p (small subset)
    fine_dim: int = 24                # fine-rank features NS_f (rich)
    n_fine_tokens: int = 3
    n_pre_targets: int = 2            # CTR, CVR
    n_fine_targets: int = 2
    # decision space: purchase level, discovery, supply type (parallel); spending level (dependent)
    n_oc: int = 3
    n_disc: int = 3
    n_ad: int = 2
    n_aov: int = 3                    # 0 is "no purchase"
    sid_vocab: int = 32               # per-level codebook size (paper: 8192)
    sid_levels: int = 3
    # backbone
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2               # GQA: heads per KV group = n_heads // n_kv_heads
    head_dim: int = 16
    time_periods: Tuple[float, ...] = (3600.0, 86400.0)   # temporal RoPE periods (seconds)
    # sparse MoE FFN
    n_routed: int = 4
    top_k: int = 2
    d_expert: int = 32
    routed_scale: float = 1.0
    # loss weights (Sec 6.1) and distillation
    lam_r: float = 0.1
    lam_p: float = 1.0
    lam_f: float = 1.0
    lam_kd: float = 1.0
    tau: float = 0.5
    kd_momentum: float = 0.9999

    @property
    def stage_sizes(self) -> Dict[str, int]:
        return {"R": N_RETR_TOKENS, "P": 1, "F": self.n_fine_tokens}

    @property
    def per_exposure(self) -> int:
        return sum(self.stage_sizes.values())

    @property
    def n_slots(self) -> int:
        return self.per_exposure

    @property
    def stage_offset(self) -> Dict[str, int]:
        off, out = 0, {}
        for s in STAGES:
            out[s] = off
            off += self.stage_sizes[s]
        return out


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MixedLinear(nn.Module):
    """Mixed parameterization (Sec 3/4.1): behavior tokens share one weight;
    each stage-specific token *slot* has its own. Fan-in init (variance 1/d_in),
    the muP-compatible choice for hidden weights."""

    def __init__(self, d_in: int, d_out: int, n_slots: int) -> None:
        super().__init__()
        std = d_in ** -0.5
        self.shared = nn.Parameter(torch.randn(d_in, d_out) * std)
        self.slot = nn.Parameter(torch.randn(n_slots, d_in, d_out) * std)

    def forward_shared(self, x: Tensor) -> Tensor:
        return x @ self.shared

    def forward_slots(self, x: Tensor, slot: Tensor) -> Tensor:
        """x [B, M, d_in], slot [M] -> [B, M, d_out]."""
        return torch.einsum("bmi,mio->bmo", x, self.slot[slot])


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        self.w_in = nn.Linear(d, 2 * hidden, bias=False)
        self.w_out = nn.Linear(hidden, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(F.silu(gate) * up)


class SparseMoE(nn.Module):
    """DeepSeekMoE-style FFN: one always-on shared expert + fine-grained routed
    SwiGLU experts. A *sigmoid* router scores experts independently (not
    winner-take-all like softmax); the top-k scores are renormalized to sum to
    one and multiplied by a fixed routed scaling factor."""

    def __init__(self, d: int, n_routed: int, top_k: int, d_expert: int, routed_scale: float = 1.0) -> None:
        super().__init__()
        self.top_k, self.routed_scale = top_k, routed_scale
        self.router = nn.Linear(d, n_routed, bias=False)
        self.shared = SwiGLU(d, d_expert)
        self.experts = nn.ModuleList([SwiGLU(d, d_expert) for _ in range(n_routed)])

    def route(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        scores = torch.sigmoid(self.router(x))
        top_v, top_i = scores.topk(self.top_k, dim=-1)
        return top_v / top_v.sum(-1, keepdim=True), top_i

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        w, idx = self.route(flat)
        out = self.shared(flat)
        routed = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            sel = (idx == e)
            rows = sel.any(dim=-1).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            weight = (w * sel)[rows].sum(-1, keepdim=True)
            routed.index_add_(0, rows, expert(flat[rows]) * weight)
        return (out + self.routed_scale * routed).reshape(shape)


# ---------------------------------------------------------------------------
# Eq. 11 — request-relative time rotary encoding
# ---------------------------------------------------------------------------

def rotate_time(x: Tensor, t: Tensor, periods: Sequence[float]) -> Tensor:
    """Rotate temporal channels. x [..., 2P] (P (cos,sin) pairs), t broadcastable
    to x[..., 0]. Pair p is rotated by angle 2*pi*t/period_p, so
    <R(t_e) q, R(t_i) k> = q^T R(t_i - t_e) k depends only on the time
    difference (Eq. 11)."""
    P = len(periods)
    omega = 2 * math.pi / torch.tensor(periods, dtype=x.dtype, device=x.device)      # [P]
    ang = t.unsqueeze(-1) * omega                                                     # [..., P]
    cos, sin = ang.cos(), ang.sin()
    xp = x.reshape(*x.shape[:-1], P, 2)
    x1, x2 = xp[..., 0], xp[..., 1]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).reshape(x.shape)


def compute_anchors(behavior_times: Tensor, request_times: Tensor) -> Tensor:
    """Anchor a_e = index of the last behavior that had arrived by request time
    t_e (-1 if none). behavior_times [B, L] ascending, request_times [B, E]."""
    return (behavior_times.unsqueeze(1) <= request_times.unsqueeze(2)).sum(-1) - 1


# ---------------------------------------------------------------------------
# Stage tokens and the stage visibility mask (Sec 4.1, Fig 2c)
# ---------------------------------------------------------------------------

@dataclass
class StageBatch:
    x: Tensor          # [B, M, d]   stage-specific token embeddings
    slot: Tensor       # [M]         token-slot id (selects token-specific parameters)
    group: Tensor      # [M]         (exposure, stage) id; attention among stage tokens is within-group
    anchor: Tensor     # [B, M]      visible behavior prefix end (inclusive) per token
    t_req: Tensor      # [B, M]      request timestamp per token


def build_stage_mask(stage: StageBatch, L: int) -> Tensor:
    """Bool mask [B, M, L+M], True = may attend.

    * every stage token sees the behavior prefix S[:anchor+1] of its exposure;
    * it also sees itself and *preceding* tokens of the same (exposure, stage)
      (causal in the predefined token order);
    * tokens of other stages or other exposures are masked.
    Behavior tokens never see stage tokens (handled by the behavior stream)."""
    B, M = stage.anchor.shape
    dev = stage.anchor.device
    to_s = torch.arange(L, device=dev).view(1, 1, L) <= stage.anchor.unsqueeze(-1)
    order = torch.arange(M, device=dev)
    to_t = (stage.group.view(M, 1) == stage.group.view(1, M)) & (order.view(1, M) <= order.view(M, 1))
    return torch.cat([to_s, to_t.unsqueeze(0).expand(B, M, M)], dim=-1)


# ---------------------------------------------------------------------------
# Algorithm 1 — TransBlock
# ---------------------------------------------------------------------------

class TransBlock(nn.Module):
    """One pre-norm Transformer block with mixed parameterization.

    X~ = RMSNorm(X); Q,K,V,G = split(X~ W^QKVG); Q,K = RMSNorm_head(Q,K);
    X += gamma * (Attn(Q,K,V; mask) * sigmoid(G)) W^O; X += gamma * MoE(RMSNorm(X)),
    with residual multiplier gamma = 1/sqrt(2N).

    Behavior tokens use shared parameters and causal attention; stage tokens
    use slot-specific QKVG/O/MoE parameters and attend to the behavior K/V
    (cached once per request) plus their own stage group.
    """

    def __init__(self, cfg: Config, num_blocks: int) -> None:
        super().__init__()
        self.cfg = cfg
        d, hd, Hq, Hk = cfg.d_model, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads
        assert Hq % Hk == 0 and 2 * len(cfg.time_periods) <= hd
        self.gamma = 1.0 / math.sqrt(2 * num_blocks)
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.qkvg = MixedLinear(d, (2 * Hq + 2 * Hk) * hd, cfg.n_slots)
        self.out = MixedLinear(Hq * hd, d, cfg.n_slots)
        self.q_norm, self.k_norm = RMSNorm(hd), RMSNorm(hd)
        mk = lambda: SparseMoE(d, cfg.n_routed, cfg.top_k, cfg.d_expert, cfg.routed_scale)
        self.moe_shared = mk()
        self.moe_slots = nn.ModuleList([mk() for _ in range(cfg.n_slots)])
        self.time_dims = 2 * len(cfg.time_periods)

    def _split(self, z: Tensor):
        B, T, _ = z.shape
        c = self.cfg
        hd = c.head_dim
        q, k, v, g = z.split([c.n_heads * hd, c.n_kv_heads * hd, c.n_kv_heads * hd, c.n_heads * hd], dim=-1)
        q = q.view(B, T, c.n_heads, hd).transpose(1, 2)
        k = k.view(B, T, c.n_kv_heads, hd).transpose(1, 2)
        v = v.view(B, T, c.n_kv_heads, hd).transpose(1, 2)
        return self.q_norm(q), self.k_norm(k), v, g

    def _attend(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        rep = self.cfg.n_heads // self.cfg.n_kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.unsqueeze(1))
        B, H, T, hd = a.shape
        return a.transpose(1, 2).reshape(B, T, H * hd)

    def forward_behavior(self, x: Tensor, t: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Behavior stream. x [B, L, d], t [B, L]. Returns (x', (K, V)) where K/V
        (post-QKNorm, keys time-rotated) are the shared user-context cache."""
        B, L, _ = x.shape
        q, k, v, g = self._split(self.qkvg.forward_shared(self.norm1(x)))
        td = self.time_dims
        k = torch.cat([k[..., :-td], rotate_time(k[..., -td:], t.unsqueeze(1), self.cfg.time_periods)], dim=-1)
        # Behavior self-attention is left unchanged by the time encoding: no temporal query term.
        q = torch.cat([q[..., :-td], torch.zeros_like(q[..., -td:])], dim=-1)
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device)).expand(B, L, L)
        a = self._attend(q, k, v, causal) * torch.sigmoid(g)
        x = x + self.gamma * self.out.forward_shared(a)
        x = x + self.gamma * self.moe_shared(self.norm2(x))
        return x, (k, v)

    def forward_stage(self, x: Tensor, stage: StageBatch, mask: Tensor, cache: Tuple[Tensor, Tensor]) -> Tensor:
        """Stage stream. x [B, M, d]; cache = behavior (K, V) [B, Hk, L, hd]."""
        q, k, v, g = self._split(self.qkvg.forward_slots(self.norm1(x), stage.slot))
        td, P = self.time_dims, self.cfg.time_periods
        t = stage.t_req.unsqueeze(1)
        # query and (own-exposure) stage keys rotate by the shared request time t_e
        q = torch.cat([q[..., :-td], rotate_time(q[..., -td:], t, P)], dim=-1)
        k = torch.cat([k[..., :-td], rotate_time(k[..., -td:], t, P)], dim=-1)
        K = torch.cat([cache[0].expand(x.shape[0], -1, -1, -1), k], dim=2)
        V = torch.cat([cache[1].expand(x.shape[0], -1, -1, -1), v], dim=2)
        a = self._attend(q, K, V, mask) * torch.sigmoid(g)
        x = x + self.gamma * self.out.forward_slots(a, stage.slot)
        h = self.norm2(x)
        f = torch.zeros_like(x)
        for s in stage.slot.unique().tolist():
            cols = (stage.slot == s).nonzero(as_tuple=True)[0]
            f[:, cols] = self.moe_slots[s](h[:, cols])
        return x + self.gamma * f


class OneTransV2Backbone(nn.Module):
    """N TransBlocks. `encode_context` runs the shared behavior stream once and
    returns the per-layer K/V cache C_u reused by every stage; `run_stage`
    executes stage-specific tokens against it (Sec 5.1)."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([TransBlock(cfg, cfg.n_layers) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)

    def encode_context(self, x: Tensor, t: Tensor):
        caches = []
        for blk in self.blocks:
            x, kv = blk.forward_behavior(x, t)
            caches.append(kv)
        return self.norm(x), caches

    def run_stage(self, stage: StageBatch, caches) -> Tensor:
        L = caches[0][0].shape[2]
        mask = build_stage_mask(stage, L)
        x = stage.x
        for blk, kv in zip(self.blocks, caches):
            x = blk.forward_stage(x, stage, mask, kv)
        return self.norm(x)

    def forward(self, beh_x: Tensor, beh_t: Tensor, stage: StageBatch):
        _, caches = self.encode_context(beh_x, beh_t)
        return self.run_stage(stage, caches)


# ---------------------------------------------------------------------------
# Eq. 8 — mean-centered knowledge distillation
# ---------------------------------------------------------------------------

def distillation_loss(teacher_logit: Tensor, student_logit: Tensor, mu_t: Tensor, mu_s: Tensor,
                      tau: float = 0.5) -> Tensor:
    """L_kd = -sum_t [p_T log p_S + (1-p_T) log(1-p_S)], with p = sigmoid((o - mu)/tau).

    Logit standardization with the std replaced by a fixed temperature tau; mu are
    (EMA) means per target. The teacher is detached so distillation moves only the
    student. Logits [..., T]; mu [T]. Sum over targets, mean over the rest."""
    p_t = torch.sigmoid((teacher_logit.detach() - mu_t) / tau)
    z_s = (student_logit - mu_s) / tau
    bce = -(p_t * F.logsigmoid(z_s) + (1 - p_t) * F.logsigmoid(-z_s))
    return bce.sum(-1).mean()


def two_stage_topk(scores: Tensor, k: int) -> Tuple[Tensor, Tensor, Tensor]:
    """Sec 5.2: keep the k best extensions of each beam, then the global top-k
    among them. Identical to the global top-k over all (beam, token) pairs but
    the intermediate pool is k per beam instead of beams*vocab.
    scores [n_beams, V] -> (values, beam index, token index)."""
    kk = min(k, scores.shape[1])
    v1, i1 = scores.topk(kk, dim=1)
    v2, j = v1.flatten().topk(min(k, v1.numel()))
    beam = j // kk
    return v2, beam, i1[beam, j % kk]


def hit_rate_at_m(topm_sids: Sequence[Tuple[int, ...]], target_sid: Tuple[int, ...], m: int) -> float:
    """HR@M for one request: 1 if the ground-truth SID is among the top-M returned."""
    return float(tuple(target_sid) in {tuple(s) for s in topm_sids[:m]})


# ---------------------------------------------------------------------------
# muP / Depth-muP and optimizers (Sec 4.3, 6.1)
# ---------------------------------------------------------------------------

def mup_hidden_lr(base_lr: float, width: int, depth: int, base_width: int, base_depth: int) -> float:
    """Table 1: hidden-weight Adam lr = eta_0 / (m_d * sqrt(m_N)), m_d = d/d0, m_N = N/N0."""
    return base_lr / ((width / base_width) * math.sqrt(depth / base_depth))


def mup_param_groups(model: "OneTransV2", base_lr: float, base_width: int, base_depth: int,
                     weight_decay: float = 0.01) -> List[dict]:
    """AdamW param groups: hidden (block) matrices get the muP lr; the rest
    (norms, heads, tokenizers) keep base_lr. Hidden init variance 1/fan_in and
    gamma = 1/sqrt(2N) are already built into TransBlock."""
    cfg = model.cfg
    hidden_lr = mup_hidden_lr(base_lr, cfg.d_model, cfg.n_layers, base_width, base_depth)
    hidden, other = [], []
    for name, p in model.named_parameters():
        if name.startswith("embed."):
            continue
        (hidden if name.startswith("backbone.blocks") and p.ndim >= 2 else other).append(p)
    return [{"params": hidden, "lr": hidden_lr, "weight_decay": weight_decay},
            {"params": other, "lr": base_lr, "weight_decay": weight_decay}]


def build_optimizers(model: "OneTransV2", base_lr: float = 1e-3, base_width: int = 64, base_depth: int = 2):
    """Dual optimizer (Sec 6.1): AdaGrad (lr 0.1, init accumulator 1.0) for
    embedding tables; AdamW (beta1=0, beta2=0.99999, eps=1e-5, wd 0.01) for dense."""
    emb = list(model.embed.parameters())
    sparse = torch.optim.Adagrad(emb, lr=0.1, initial_accumulator_value=1.0)
    dense = torch.optim.AdamW(mup_param_groups(model, base_lr, base_width, base_depth),
                              betas=(0.0, 0.99999), eps=1e-5)
    return sparse, dense


# ---------------------------------------------------------------------------
# The full model
# ---------------------------------------------------------------------------

@dataclass
class Retrieved:
    sid: Tuple[int, ...]       # generated semantic ID
    score: float               # steered decision score + sum of SID log-probs
    decision: Tuple[int, ...]  # (z_oc, z_disc, z_ad, z_aov) of the prefix that produced it


class OneTransV2(nn.Module):
    """OneTrans-V2: shared behavior context + retrieval (DCGR) / pre-rank / fine-rank."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        # embedding tables (sparse-optimizer group)
        self.embed = nn.ModuleDict({
            "item": nn.Embedding(cfg.item_vocab, d), "action": nn.Embedding(cfg.n_actions, d),
            "oc": nn.Embedding(cfg.n_oc, d), "disc": nn.Embedding(cfg.n_disc, d),
            "ad": nn.Embedding(cfg.n_ad, d), "aov": nn.Embedding(cfg.n_aov, d),
            "sid": nn.Embedding(cfg.sid_levels * cfg.sid_vocab, d),
        })
        self.mm_proj = nn.Linear(cfg.mm_dim, d, bias=False) if cfg.mm_dim else None
        # tokenizers (stage-specific NS features)
        self.ctx_tok = nn.Linear(cfg.ctx_dim, d)
        self.pre_tok = nn.Linear(cfg.pre_dim, d)
        self.fine_tok = nn.Linear(cfg.fine_dim, cfg.n_fine_tokens * d)
        self.null_dp = nn.Parameter(torch.zeros(d))     # CFG null tokens (Eq. 5)
        self.null_dd = nn.Parameter(torch.zeros(d))
        self.backbone = OneTransV2Backbone(cfg)
        # heads
        self.oc_head, self.disc_head = nn.Linear(d, cfg.n_oc), nn.Linear(d, cfg.n_disc)
        self.ad_head, self.aov_head = nn.Linear(d, cfg.n_ad), nn.Linear(d, cfg.n_aov)
        self.sid_heads = nn.ModuleList([nn.Linear(d, cfg.sid_vocab) for _ in range(cfg.sid_levels)])
        self.pre_head = nn.Linear(d, cfg.n_pre_targets)
        self.fine_head = nn.Linear(cfg.n_fine_tokens * d, cfg.n_fine_targets)
        self.register_buffer("kd_mu_t", torch.zeros(cfg.n_pre_targets))
        self.register_buffer("kd_mu_s", torch.zeros(cfg.n_pre_targets))

    # ----- tokenization -----------------------------------------------------
    def embed_behaviors(self, items: Tensor, actions: Tensor, mm: Optional[Tensor] = None) -> Tensor:
        x = self.embed["item"](items) + self.embed["action"](actions)
        if self.mm_proj is not None and mm is not None:
            x = x + self.mm_proj(mm)          # multimodal embedding fed directly as side info
        return x

    def retrieval_tokens(self, ctx: Tensor, n: int, zp: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
                         z_aov: Optional[Tensor] = None, sid_prefix: Sequence[Tensor] = (),
                         null: Optional[Tensor] = None) -> Tensor:
        """First n of [CTX, DT^p, DT^d, SID_0, SID_1] (Eq. 1). DT^p = SUM of the parallel
        decision embeddings (no chained order); DT^d = embedding of the dependent
        decision. `null` [B] bool swaps both DT tokens for learned null tokens."""
        toks = [self.ctx_tok(ctx)]
        if n >= 2:
            dp = self.embed["oc"](zp[0]) + self.embed["disc"](zp[1]) + self.embed["ad"](zp[2])
            toks.append(dp)
        if n >= 3:
            toks.append(self.embed["aov"](z_aov))
        for lvl, s in enumerate(sid_prefix):
            if len(toks) >= n:
                break
            toks.append(self.embed["sid"](s + lvl * self.cfg.sid_vocab))
        x = torch.stack(toks, dim=1)
        if null is not None and n >= 2:
            x = x.clone()
            m = null.view(-1, 1)
            x[:, 1] = torch.where(m, self.null_dp.expand_as(x[:, 1]), x[:, 1])
            if n >= 3:
                x[:, 2] = torch.where(m, self.null_dd.expand_as(x[:, 2]), x[:, 2])
        return x

    # ----- training ---------------------------------------------------------
    def forward(self, batch: Dict[str, Tensor], null_prob: float = 0.0) -> Dict[str, Tensor]:
        """Joint forward of all three tasks for E exposures sharing ONE encoding of
        the user's behavior sequence (Sequence-Native Training).

        batch keys: seq_items/seq_actions/seq_times [B, L] (seq_mm [B, L, mm_dim]);
        t_req [B, E]; ctx [B, E, ctx_dim]; z_oc/z_disc/z_ad/z_aov [B, E]; sid [B, E, 3];
        pre_feat [B, E, pre_dim]; fine_feat [B, E, fine_dim].
        """
        c = self.cfg
        B, E = batch["t_req"].shape
        beh = self.embed_behaviors(batch["seq_items"], batch["seq_actions"], batch.get("seq_mm"))
        anchors = compute_anchors(batch["seq_times"], batch["t_req"])            # [B, E]

        null = (torch.rand(B, E, device=beh.device) < null_prob) if (self.training and null_prob > 0) \
            else torch.zeros(B, E, dtype=torch.bool, device=beh.device)
        per = c.per_exposure
        toks = []
        for e in range(E):
            r = self.retrieval_tokens(
                batch["ctx"][:, e], N_RETR_TOKENS,
                zp=(batch["z_oc"][:, e], batch["z_disc"][:, e], batch["z_ad"][:, e]),
                z_aov=batch["z_aov"][:, e],
                sid_prefix=[batch["sid"][:, e, 0], batch["sid"][:, e, 1]], null=null[:, e])
            p = self.pre_tok(batch["pre_feat"][:, e]).unsqueeze(1)
            f = self.fine_tok(batch["fine_feat"][:, e]).view(B, c.n_fine_tokens, c.d_model)
            toks.append(torch.cat([r, p, f], dim=1))
        x = torch.cat(toks, dim=1)                                                # [B, E*per, d]
        dev = x.device
        slot = torch.arange(per, device=dev).repeat(E)
        stage_id = torch.cat([torch.full((n,), i) for i, n in enumerate(c.stage_sizes.values())]).to(dev)
        group = (torch.arange(E, device=dev).repeat_interleave(per) * len(STAGES)) + stage_id.repeat(E)
        stage = StageBatch(x, slot, group,
                           anchors.repeat_interleave(per, dim=1), batch["t_req"].repeat_interleave(per, dim=1))
        h = self.backbone(beh, batch["seq_times"], stage).view(B, E, per, c.d_model)

        off = c.stage_offset
        hr, hp, hf = h[:, :, off["R"]:off["R"] + N_RETR_TOKENS], h[:, :, off["P"]], \
            h[:, :, off["F"]:off["F"] + c.n_fine_tokens]
        return {
            "oc": self.oc_head(hr[:, :, 0]), "disc": self.disc_head(hr[:, :, 0]), "ad": self.ad_head(hr[:, :, 0]),
            "aov": self.aov_head(hr[:, :, 1]),
            "sid": torch.stack([self.sid_heads[l](hr[:, :, 2 + l]) for l in range(c.sid_levels)], dim=2),  # [B,E,3,V]
            "pre": self.pre_head(hp),
            "fine": self.fine_head(hf.reshape(B, E, -1)),
            "null": null,
        }

    def loss(self, batch: Dict[str, Tensor], out: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Eq. 9: L = lam_r L_r + lam_p L_p + lam_f L_f + lam_kd L_kd."""
        c = self.cfg
        ce = lambda lg, y: F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none")
        keep = (~out["null"]).float().reshape(-1)          # aov conditioned on a nulled DT^p is unsupervised
        l_dec = ce(out["oc"], batch["z_oc"]).mean() + ce(out["disc"], batch["z_disc"]).mean() \
            + ce(out["ad"], batch["z_ad"]).mean() \
            + (ce(out["aov"], batch["z_aov"]) * keep).sum() / keep.sum().clamp(min=1)
        l_sid = sum(ce(out["sid"][:, :, l], batch["sid"][:, :, l]).mean() for l in range(c.sid_levels))
        l_r = l_dec + l_sid
        l_p = F.binary_cross_entropy_with_logits(out["pre"], batch["y_pre"], reduction="none").sum(-1).mean()
        l_f = F.binary_cross_entropy_with_logits(out["fine"], batch["y_fine"], reduction="none").sum(-1).mean()
        teacher = out["fine"][..., :c.n_pre_targets]       # fine-rank is the teacher for the pre-rank targets
        if self.training:
            with torch.no_grad():
                m = c.kd_momentum
                self.kd_mu_t.mul_(m).add_((1 - m) * teacher.detach().reshape(-1, c.n_pre_targets).mean(0))
                self.kd_mu_s.mul_(m).add_((1 - m) * out["pre"].detach().reshape(-1, c.n_pre_targets).mean(0))
        l_kd = distillation_loss(teacher, out["pre"], self.kd_mu_t, self.kd_mu_s, c.tau)
        total = c.lam_r * l_r + c.lam_p * l_p + c.lam_f * l_f + c.lam_kd * l_kd
        return {"loss": total, "retrieval": l_r, "pre": l_p, "fine": l_f, "kd": l_kd}

    # ----- serving: shared user context ------------------------------------
    @torch.no_grad()
    def encode_user(self, items: Tensor, actions: Tensor, times: Tensor, mm: Optional[Tensor] = None):
        """Once per request: encode the behavior sequence into the cache C_u."""
        return self.backbone.encode_context(self.embed_behaviors(items, actions, mm), times)[1]

    def _single_stage(self, x: Tensor, first_slot: int, anchor: int, t_req: float) -> StageBatch:
        B, n, _ = x.shape
        dev = x.device
        return StageBatch(x, torch.arange(first_slot, first_slot + n, device=dev),
                          torch.zeros(n, dtype=torch.long, device=dev),
                          torch.full((B, n), anchor, dtype=torch.long, device=dev),
                          torch.full((B, n), t_req, dtype=x.dtype, device=dev))

    @torch.no_grad()
    def prerank(self, caches, pre_feat: Tensor, anchor: int, t_req: float) -> Tensor:
        """Pre-rank logits [N, n_pre_targets] for N candidates against the shared cache."""
        x = self.pre_tok(pre_feat).unsqueeze(1)
        h = self.backbone.run_stage(self._single_stage(x, self.cfg.stage_offset["P"], anchor, t_req), caches)
        return self.pre_head(h[:, 0])

    @torch.no_grad()
    def finerank(self, caches, fine_feat: Tensor, anchor: int, t_req: float) -> Tensor:
        """Fine-rank logits [N, n_fine_targets] for the surviving candidates."""
        c = self.cfg
        x = self.fine_tok(fine_feat).view(fine_feat.shape[0], c.n_fine_tokens, c.d_model)
        h = self.backbone.run_stage(self._single_stage(x, c.stage_offset["F"], anchor, t_req), caches)
        return self.fine_head(h.reshape(h.shape[0], -1))

    # ----- serving: DCGR decoding ------------------------------------------
    @torch.no_grad()
    def retrieve(self, caches, ctx: Tensor, anchor: int, t_req: float, k: int,
                 beta: float = 0.0, phi: Optional[Dict[str, Tensor]] = None,
                 cfg_weight: Optional[float] = None) -> List[Retrieved]:
        """DCGR decoding for one request (batch size 1) against the shared cache.

        1. Enumerate every valid decision prefix z = (oc, disc, ad, aov) and score
           log P(z|H) in two batched passes over the cached context (parallel
           decisions from the CTX state; aov from each DT^p state), instead of
           sequentially (Sec 5.2).
        2. Add the business offset beta * phi(z) (Eq. 4); phi = {"oc","disc","ad","aov"}
           -> per-value offsets. beta=0 leaves the model's own ranking unchanged.
        3. Beam search over the SID codes, each beam carrying its steered decision
           score; two-stage top-k per level. If cfg_weight is set, SID codes are
           scored with classifier-free guidance (Eq. 5):
           log P(. | null) + w (log P(. | z) - log P(. | null)).
        """
        c = self.cfg
        dev = ctx.device
        grid = torch.tensor(list(itertools.product(range(c.n_oc), range(c.n_disc), range(c.n_ad), range(c.n_aov))),
                            device=dev)                                        # [Z, 4]
        valid = (grid[:, 3] == 0) == (grid[:, 0] == 0)       # a spending level exists iff the user purchases
        grid = grid[valid]

        def run(x: Tensor, cache_batch: int) -> Tensor:
            stage = self._single_stage(x, 0, anchor, t_req)
            cs = [(kk.expand(cache_batch, -1, -1, -1), vv.expand(cache_batch, -1, -1, -1)) for kk, vv in caches]
            return self.backbone.run_stage(stage, cs)

        # pass 1: parallel decisions from the CTX state
        h0 = run(self.retrieval_tokens(ctx, 1), 1)[:, 0]
        lp_oc = F.log_softmax(self.oc_head(h0), -1)[0]
        lp_disc = F.log_softmax(self.disc_head(h0), -1)[0]
        lp_ad = F.log_softmax(self.ad_head(h0), -1)[0]
        # pass 2: aov given each parallel prefix z^p
        zp = torch.tensor(list(itertools.product(range(c.n_oc), range(c.n_disc), range(c.n_ad))), device=dev)
        n_p = zp.shape[0]
        h1 = run(self.retrieval_tokens(ctx.expand(n_p, -1), 2, zp=(zp[:, 0], zp[:, 1], zp[:, 2])), n_p)[:, 1]
        lp_aov = F.log_softmax(self.aov_head(h1), -1)                           # [n_p, n_aov]
        zp_index = (grid[:, 0] * c.n_disc + grid[:, 1]) * c.n_ad + grid[:, 2]
        score = lp_oc[grid[:, 0]] + lp_disc[grid[:, 1]] + lp_ad[grid[:, 2]] + lp_aov[zp_index, grid[:, 3]]
        if beta != 0.0 and phi is not None:
            offset = sum(phi[name][grid[:, i]] for i, name in enumerate(("oc", "disc", "ad", "aov")) if name in phi)
            score = score + beta * offset

        beams_z, beam_score = grid, score
        sids = torch.zeros(grid.shape[0], 0, dtype=torch.long, device=dev)
        for lvl in range(c.sid_levels):
            nb = beams_z.shape[0]
            args = dict(zp=(beams_z[:, 0], beams_z[:, 1], beams_z[:, 2]), z_aov=beams_z[:, 3],
                        sid_prefix=[sids[:, j] for j in range(lvl)])
            n_tok = 3 + lvl
            h = run(self.retrieval_tokens(ctx.expand(nb, -1), n_tok, **args), nb)[:, -1]
            lp = F.log_softmax(self.sid_heads[lvl](h), -1)
            if cfg_weight is not None:
                null = torch.ones(nb, dtype=torch.bool, device=dev)
                h_n = run(self.retrieval_tokens(ctx.expand(nb, -1), n_tok, null=null, **args), nb)[:, -1]
                lp_null = F.log_softmax(self.sid_heads[lvl](h_n), -1)
                lp = lp_null + cfg_weight * (lp - lp_null)
            top, beam, tok = two_stage_topk(beam_score.unsqueeze(1) + lp, k)
            beams_z, beam_score = beams_z[beam], top
            sids = torch.cat([sids[beam], tok.unsqueeze(1)], dim=1)
        return [Retrieved(tuple(s.tolist()), sc.item(), tuple(z.tolist()))
                for s, sc, z in zip(sids, beam_score, beams_z)]


def sid_to_items(retrieved: Sequence[Retrieved], index: Dict[Tuple[int, ...], List[int]]) -> List[int]:
    """Items may share codes; every item mapped to a retrieved SID is returned."""
    return [item for r in retrieved for item in index.get(r.sid, [])]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def make_batch(cfg: Config, B: int = 2, L: int = 12, E: int = 3, seed: int = 0) -> Dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    ri = lambda hi, *s: torch.randint(0, hi, s, generator=g)
    times = torch.sort(torch.rand(B, L, generator=g) * 1000, dim=1).values
    t_req = torch.sort(times[:, -E:] + torch.rand(B, E, generator=g) * 10, dim=1).values
    oc = ri(cfg.n_oc, B, E)
    return dict(
        seq_items=ri(cfg.item_vocab, B, L), seq_actions=ri(cfg.n_actions, B, L), seq_times=times, t_req=t_req,
        ctx=torch.randn(B, E, cfg.ctx_dim, generator=g), pre_feat=torch.randn(B, E, cfg.pre_dim, generator=g),
        fine_feat=torch.randn(B, E, cfg.fine_dim, generator=g),
        z_oc=oc, z_disc=ri(cfg.n_disc, B, E), z_ad=ri(cfg.n_ad, B, E),
        z_aov=torch.where(oc == 0, torch.zeros_like(oc), ri(cfg.n_aov - 1, B, E) + 1),
        sid=ri(cfg.sid_vocab, B, E, cfg.sid_levels),
        y_pre=(torch.rand(B, E, cfg.n_pre_targets, generator=g) < 0.3).float(),
        y_fine=(torch.rand(B, E, cfg.n_fine_targets, generator=g) < 0.3).float(),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = Config()
    model = OneTransV2(cfg)
    batch = make_batch(cfg)
    print("params:", sum(p.numel() for p in model.parameters()))

    sparse, dense = build_optimizers(model)
    for step in range(5):
        out = model(batch, null_prob=0.1)
        losses = model.loss(batch, out)
        sparse.zero_grad(); dense.zero_grad()
        losses["loss"].backward()
        sparse.step(); dense.step()
    print({k: round(v.item(), 4) for k, v in losses.items()})

    model.eval()
    b1 = {k: v[:1] for k, v in batch.items()}
    anchor = int(compute_anchors(b1["seq_times"], b1["t_req"])[0, -1])
    t_req = float(b1["t_req"][0, -1])
    caches = model.encode_user(b1["seq_items"], b1["seq_actions"], b1["seq_times"])      # encoded ONCE
    ret = model.retrieve(caches, b1["ctx"][:, -1], anchor, t_req, k=5)
    print("\nretrieval (beta=0):")
    for r in ret:
        print("  ", r)
    phi = {"ad": torch.tensor([0.0, 1.0])}
    ret_ad = model.retrieve(caches, b1["ctx"][:, -1], anchor, t_req, k=5, beta=3.0, phi=phi)
    print("retrieval steered to sponsored (beta=3):", [r.decision[2] for r in ret_ad], "vs", [r.decision[2] for r in ret])
    cand_pre = torch.randn(20, cfg.pre_dim)
    pre = model.prerank(caches, cand_pre, anchor, t_req)
    keep = pre[:, 0].topk(5).indices
    fine = model.finerank(caches, torch.randn(5, cfg.fine_dim), anchor, t_req)
    print("\npre-rank logits", tuple(pre.shape), "-> kept", keep.tolist(), "-> fine-rank logits", tuple(fine.shape))
