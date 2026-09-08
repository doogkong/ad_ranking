"""RankMixer: Scaling Up Ranking Models in Industrial Recommenders.

Reference PyTorch implementation.
Paper: https://arxiv.org/abs/2507.15551  (ByteDance, Jul 2025)

Architecture overview (Eq. 1):
    Raw features  -->  FeatureTokenizer  -->  X^(0) in R^(B, T, D)

    for n = 1..L:
        S^(n-1) = LN(TokenMixing(X^(n-1)) + X^(n-1))     [Eq. 3-5]
        X^(n)   = LN(PFFN(S^(n-1)) + S^(n-1))            [Eq. 6-9]

    o_output = mean_pool(X^(L))
    y_hat    = task_heads(o_output)

Two problems solved relative to Transformer/DHEN-style ranking models:
  1. Quadratic self-attention is a poor fit for heterogeneous, high-cardinality
     ID feature spaces (inner-product similarity across incomparable semantic
     spaces) and is memory-bound on GPUs. Multi-head Token Mixing replaces it
     with a parameter-free reshape/transpose "shuffle" that still lets every
     feature subspace interact with every other one, at a fraction of the FLOPs.
  2. Sharing one FFN across all tokens lets high-frequency feature groups
     dominate low-frequency ones. The Per-token FFN (PFFN) gives every token
     its own private MLP, trading a large but structured parameter increase
     (independent of sequence-length-style compute blowup) for much higher
     model capacity at the same FLOPs.

Deployed at Douyin (feed ranking + ads): 0.3% active-days, 1.08% duration,
0.73% ads AUC lift, scaling dense parameters 70x at roughly flat latency.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Input Layer: Feature Tokenization (Sec 3.2, Eq. 2)
# ---------------------------------------------------------------------------

class FeatureTokenizer(nn.Module):
    """Projects N semantically-grouped raw feature vectors into T aligned tokens.

    x_i = Proj(e_input[d*(i-1) : d*i]),  i = 1..T      [Eq. 2]

    Each group of raw feature embeddings (already concatenated upstream, e.g.
    user profile, candidate, sequence-summary, cross features) is projected by
    its own linear layer into the shared model dimension D. This keeps
    per-token compute balanced regardless of how many raw features feed into
    that token's group, avoiding both "one token per feature" (thousands of
    tiny fragments) and "one giant token" (degenerates to a plain DNN).

    Args:
        group_dims: raw (pre-projection) dimension of each of the T feature
            groups, e.g. [128, 64, 256, ...].
        d_model: shared token dimension D after projection.
    """

    def __init__(self, group_dims: list[int], d_model: int) -> None:
        super().__init__()
        self.num_tokens = len(group_dims)
        self.proj = nn.ModuleList([nn.Linear(g, d_model) for g in group_dims])

    def forward(self, groups: list[Tensor]) -> Tensor:
        """
        Args:
            groups: list of T tensors, group i has shape (B, group_dims[i]).
        Returns:
            X0: (B, T, D)
        """
        tokens = [proj(g) for proj, g in zip(self.proj, groups)]
        return torch.stack(tokens, dim=1)


# ---------------------------------------------------------------------------
# Multi-head Token Mixing (Sec 3.3.1, Eq. 3-5)
# ---------------------------------------------------------------------------

class TokenMixing(nn.Module):
    """Parameter-free cross-token mixing via a head-wise transpose.

    Each token x_t in R^D is split into H heads x_t^(1),...,x_t^(H), each of
    size D/H [Eq. 3]. The h-th output token is the concatenation, across all
    T input tokens, of their h-th head:

        s^h = Concat(x_1^h, x_2^h, ..., x_T^h)             [Eq. 4]

    This is implemented as a single reshape + transpose (no learnable
    parameters): view (T, D) as (T, H, D/H), swap the T and H axes, and
    flatten back to (H, T*D/H). With H = T (the paper's setting, used to keep
    the residual connection dimension-preserving), this is exactly an
    axis-swap "shuffle" of feature-subspace slices across token positions —
    every output token now contains a slice from every input token, so
    information mixes globally without any inner-product attention.

    The paper finds this outperforms self-attention here specifically because
    ranking features live in heterogeneous, non-comparable ID subspaces:
    inner-product similarity across such spaces is unreliable and expensive,
    while this shuffle mixes subspaces without assuming any shared geometry.
    """

    def __init__(self, num_tokens: int, d_model: int, num_heads: Optional[int] = None) -> None:
        super().__init__()
        self.num_heads = num_heads or num_tokens
        if self.num_heads != num_tokens:
            raise ValueError(
                "RankMixer requires num_heads == num_tokens to keep the "
                "residual connection dimension-preserving (Sec 3.3.1)."
            )
        if d_model % self.num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={self.num_heads}")
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.head_dim = d_model // self.num_heads

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, D) -> (B, T, H, D/H) -> (B, H, T, D/H) -> (B, H, T*D/H)
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        x = x.transpose(1, 2).contiguous()
        return x.reshape(B, self.num_heads, T * self.head_dim)


# ---------------------------------------------------------------------------
# Per-token FFN (Sec 3.3.2, Eq. 6-9)
# ---------------------------------------------------------------------------

class PerTokenFFN(nn.Module):
    """Dense Per-token FFN: every token index gets its own private 2-layer MLP.

    v_t = W2_t . Gelu(W1_t . s_t + b1_t) + b2_t          [Eq. 6-7]

    Unlike a Transformer FFN (one shared MLP for every token) or an MMoE
    expert (every expert sees the same input), each per-token FFN here both
    sees a distinct token and owns distinct parameters — isolating capacity
    per feature subspace so high-frequency fields cannot drown out long-tail
    ones, while keeping FLOPs identical to a shared FFN of the same width.

    Args:
        num_tokens: T.
        d_model: D.
        expand_ratio: k, hidden-dim multiplier for the FFN (hidden = k*D).
    """

    def __init__(self, num_tokens: int, d_model: int, expand_ratio: int = 4) -> None:
        super().__init__()
        hidden = expand_ratio * d_model
        self.W1 = nn.Parameter(torch.empty(num_tokens, d_model, hidden))
        self.b1 = nn.Parameter(torch.zeros(num_tokens, hidden))
        self.W2 = nn.Parameter(torch.empty(num_tokens, hidden, d_model))
        self.b2 = nn.Parameter(torch.zeros(num_tokens, d_model))
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)

    def forward(self, s: Tensor) -> Tensor:
        # s: (B, T, D)
        h = torch.einsum("btd,tdk->btk", s, self.W1) + self.b1
        h = F.gelu(h)
        v = torch.einsum("btk,tkd->btd", h, self.W2) + self.b2
        return v


# ---------------------------------------------------------------------------
# Sparse-MoE Per-token FFN (Sec 3.4, Eq. 10-11)
# ---------------------------------------------------------------------------

class SparseMoEPerTokenFFN(nn.Module):
    """ReLU-routed Sparse-MoE variant of the Per-token FFN.

    Each token owns its own bank of `num_experts` small FFNs plus its own
    router. Instead of Top-k + softmax, the gate is a plain ReLU:

        G_{t,j} = ReLU(h_t(s_t))_j                            [Eq. 10]
        v_t     = sum_j G_{t,j} * expert_{t,j}(s_t)

    ReLU routing lets each token activate a *variable* number of experts
    (rather than a fixed top-k), so high-information tokens can recruit more
    experts and low-information tokens fall back to few or zero. Sparsity is
    steered — not hard-selected — by an L1 penalty on the gate values:

        L = L_task + lambda * L_reg,   L_reg = sum_t sum_j G_{t,j}   [Eq. 11]

    `reg_loss()` returns L_reg for the caller to add (scaled by lambda) to the
    task loss.

    Note on Dense-Training/Sparse-Inference (DTSI, Sec 3.4): the paper trains
    two routers — a dense `h_train` (no L1 penalty, so gradients reach all
    experts and none starve) and a sparse `h_infer` (L1-penalized, used only
    at serving time). This module approximates that recipe: during training
    the *train* router's (denser) gates are used for the forward value so all
    experts receive signal, while the *infer* router is trained in parallel
    with an L1 penalty plus a distillation term pulling it toward the train
    router's routing decisions; at eval/inference only the infer router's
    (sparser) gates are used. This is a reference approximation of DTSI, not
    a reproduction of the production training pipeline.
    """

    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        num_experts: int = 4,
        expand_ratio: int = 4,
        dtsi: bool = False,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.dtsi = dtsi
        hidden = expand_ratio * d_model

        self.expert_W1 = nn.Parameter(torch.empty(num_tokens, num_experts, d_model, hidden))
        self.expert_b1 = nn.Parameter(torch.zeros(num_tokens, num_experts, hidden))
        self.expert_W2 = nn.Parameter(torch.empty(num_tokens, num_experts, hidden, d_model))
        self.expert_b2 = nn.Parameter(torch.zeros(num_tokens, num_experts, d_model))
        nn.init.xavier_uniform_(self.expert_W1)
        nn.init.xavier_uniform_(self.expert_W2)

        self.router_infer = nn.Parameter(torch.empty(num_tokens, d_model, num_experts))
        nn.init.xavier_uniform_(self.router_infer)
        if dtsi:
            self.router_train = nn.Parameter(torch.empty(num_tokens, d_model, num_experts))
            nn.init.xavier_uniform_(self.router_train)

        self._last_gates_infer: Optional[Tensor] = None
        self._last_gates_train: Optional[Tensor] = None

    def _experts(self, s: Tensor) -> Tensor:
        # s: (B, T, D) -> expert outputs (B, T, E, D)
        h = torch.einsum("btd,tedk->btek", s, self.expert_W1) + self.expert_b1
        h = F.gelu(h)
        return torch.einsum("btek,tekd->bted", h, self.expert_W2) + self.expert_b2

    def forward(self, s: Tensor) -> Tensor:
        # s: (B, T, D)
        expert_out = self._experts(s)  # (B, T, E, D)

        gates_infer = F.relu(torch.einsum("btd,tde->bte", s, self.router_infer))
        self._last_gates_infer = gates_infer

        if self.dtsi and self.training:
            gates_train = F.relu(torch.einsum("btd,tde->bte", s, self.router_train))
            self._last_gates_train = gates_train
            gates = gates_train
        else:
            gates = gates_infer

        return torch.einsum("bte,bted->btd", gates, expert_out)

    def reg_loss(self) -> Tensor:
        """L_reg = sum over tokens/experts of the (sparse, infer-router) gate values [Eq. 11]."""
        if self._last_gates_infer is None:
            raise RuntimeError("Call forward() before reg_loss().")
        return self._last_gates_infer.sum(dim=(-2, -1)).mean()

    def distill_loss(self) -> Tensor:
        """Auxiliary term pulling the sparse infer router toward the dense train router."""
        if not self.dtsi or self._last_gates_train is None:
            return torch.zeros((), device=self._last_gates_infer.device)
        return F.mse_loss(self._last_gates_infer, self._last_gates_train.detach())

    def active_expert_ratio(self) -> Tensor:
        """Fraction of (token, expert) gates that are strictly positive — a measure of sparsity."""
        if self._last_gates_infer is None:
            raise RuntimeError("Call forward() before active_expert_ratio().")
        return (self._last_gates_infer > 0).float().mean()


# ---------------------------------------------------------------------------
# RankMixer Block (Eq. 1)
# ---------------------------------------------------------------------------

class RankMixerBlock(nn.Module):
    """One RankMixer layer: TokenMixing followed by a (Sparse-)PFFN, each with
    a residual connection and LayerNorm (Eq. 1):

        S^(n-1) = LN(TokenMixing(X^(n-1)) + X^(n-1))
        X^(n)   = LN(PFFN(S^(n-1)) + S^(n-1))
    """

    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        expand_ratio: int = 4,
        moe: bool = False,
        num_experts: int = 4,
        dtsi: bool = False,
    ) -> None:
        super().__init__()
        self.token_mixing = TokenMixing(num_tokens, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        if moe:
            self.pffn = SparseMoEPerTokenFFN(num_tokens, d_model, num_experts, expand_ratio, dtsi)
        else:
            self.pffn = PerTokenFFN(num_tokens, d_model, expand_ratio)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        s = self.norm1(self.token_mixing(x) + x)
        return self.norm2(self.pffn(s) + s)


# ---------------------------------------------------------------------------
# Full RankMixer Model
# ---------------------------------------------------------------------------

class RankMixer(nn.Module):
    """RankMixer: hardware-aware, highly-parallel feature-interaction backbone.

    Args:
        group_dims: raw dimension of each of the T semantic feature groups
            fed into the FeatureTokenizer.
        d_model: model width D.
        num_layers: L, number of stacked RankMixerBlocks.
        expand_ratio: k, PFFN hidden-dim multiplier (hidden = k*D).
        moe: use Sparse-MoE Per-token FFN instead of the dense variant.
        num_experts: experts per token when moe=True.
        dtsi: use the Dense-Training/Sparse-Inference dual-router
            approximation (Sec 3.4) when moe=True.
        top_mlp_dims: hidden dims of the final task head MLP.
        num_tasks: number of output logits.
    """

    def __init__(
        self,
        group_dims: list[int],
        d_model: int = 64,
        num_layers: int = 2,
        expand_ratio: int = 4,
        moe: bool = False,
        num_experts: int = 4,
        dtsi: bool = False,
        top_mlp_dims: Optional[list[int]] = None,
        num_tasks: int = 1,
    ) -> None:
        super().__init__()
        num_tokens = len(group_dims)
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.num_layers = num_layers
        self.moe = moe

        self.tokenizer = FeatureTokenizer(group_dims, d_model)
        self.blocks = nn.ModuleList([
            RankMixerBlock(num_tokens, d_model, expand_ratio, moe, num_experts, dtsi)
            for _ in range(num_layers)
        ])

        hidden = top_mlp_dims or [d_model]
        mlp_dims = [d_model] + hidden + [num_tasks]
        layers: list[nn.Module] = []
        for i in range(len(mlp_dims) - 1):
            layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            if i < len(mlp_dims) - 2:
                layers.append(nn.ReLU())
        self.classifier = nn.Sequential(*layers)

    def forward(self, groups: list[Tensor]) -> Tensor:
        """
        Args:
            groups: list of T raw feature-group tensors, group i shaped
                (B, group_dims[i]).
        Returns:
            logits: (B, num_tasks)
        """
        x = self.tokenizer(groups)
        for block in self.blocks:
            x = block(x)
        pooled = x.mean(dim=1)  # mean pooling over T tokens (Sec 3.1)
        return self.classifier(pooled)

    def moe_reg_loss(self) -> Tensor:
        """Sum of L_reg (Eq. 11) across all Sparse-MoE blocks; 0.0 if moe=False."""
        if not self.moe:
            return torch.zeros((), device=next(self.parameters()).device)
        return sum(block.pffn.reg_loss() for block in self.blocks)

    @staticmethod
    def estimate_params_and_flops(num_layers: int, num_tokens: int, d_model: int, expand_ratio: int) -> tuple[int, int]:
        """Dense-model scaling-law estimate from Sec 3.5, Eq. 12:

            #Param ~= 2 * k * L * T * D^2
            FLOPs  ~= 4 * k * L * T * D^2  (per sample)
        """
        params = 2 * expand_ratio * num_layers * num_tokens * d_model ** 2
        flops = 4 * expand_ratio * num_layers * num_tokens * d_model ** 2
        return params, flops


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B = 4
    group_dims = [32, 48, 16, 64, 24, 40, 20, 56]  # T = 8 semantic feature groups

    model = RankMixer(
        group_dims=group_dims,
        d_model=64,
        num_layers=3,
        expand_ratio=4,
        moe=False,
        top_mlp_dims=[64],
        num_tasks=1,
    )

    groups = [torch.randn(B, g) for g in group_dims]
    logits = model(groups)
    print(f"logits:   {logits.shape}  {logits.squeeze(-1).tolist()}")

    loss = F.binary_cross_entropy_with_logits(logits, torch.zeros(B, 1))
    loss.backward()
    print(f"loss:     {loss.item():.4f}")
    print("backward: OK")

    total = sum(p.numel() for p in model.parameters())
    print(f"params:   {total:,}")

    est_params, est_flops = RankMixer.estimate_params_and_flops(
        num_layers=3, num_tokens=len(group_dims), d_model=64, expand_ratio=4
    )
    print(f"Eq.12 estimate (PFFN-only, no tokenizer/head): params~={est_params:,} flops~={est_flops:,}")

    print("\n--- Sparse-MoE variant ---")
    moe_model = RankMixer(
        group_dims=group_dims,
        d_model=64,
        num_layers=2,
        expand_ratio=4,
        moe=True,
        num_experts=4,
        dtsi=True,
        top_mlp_dims=[64],
        num_tasks=1,
    )
    moe_model.train()
    logits = moe_model(groups)
    reg = moe_model.moe_reg_loss()
    task_loss = F.binary_cross_entropy_with_logits(logits, torch.zeros(B, 1))
    (task_loss + 1e-3 * reg).backward()
    print(f"logits:      {logits.shape}")
    print(f"reg_loss:    {reg.item():.4f}")
    print("backward:    OK")


if __name__ == "__main__":
    _smoke_test()
