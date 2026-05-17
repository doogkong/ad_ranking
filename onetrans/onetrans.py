"""OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer.

Reference PyTorch implementation.
Paper: https://arxiv.org/abs/2510.26104  (ByteDance / NTU, WWW 2026)

Architecture overview:
    Sequential features  (behavior histories)  →  S-tokens  ]
                                                              ├─ concat ─→ OneTrans Pyramid Stack ─→ Task heads
    Non-sequential features (user/item/ctx)    →  NS-tokens ]

Key design:
  - Unified tokenizer converts both feature types into one token sequence.
  - Mixed parameterization: S-tokens share one set of Q/K/V+FFN weights;
    each NS-token has its own token-specific Q/K/V+FFN (preserves heterogeneity).
  - Causal mask: S-tokens attend only to preceding S-tokens (autoregressive
    over behaviors); NS-tokens attend to the entire S-history + preceding NS-tokens.
  - Pyramid stack: at each layer, only the tail L' S-token queries are computed
    (K/V span the full sequence), shrinking L' linearly from L_S → L_NS across
    layers and progressively concentrating information into NS-tokens.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------

class AutoSplitNSTokenizer(nn.Module):
    """Non-sequential tokenizer: concat all NS features → MLP → split into L_NS tokens.

    Preferred over group-wise tokenizer (ablation Table 3 in paper): auto-split
    lets the model learn feature groupings rather than relying on manual semantics,
    and uses a single dense kernel reducing launch overhead.
    """

    def __init__(self, ns_input_dim: int, d_model: int, L_NS: int) -> None:
        super().__init__()
        self.L_NS = L_NS
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(ns_input_dim, d_model * L_NS),
            nn.SiLU(),
            nn.Linear(d_model * L_NS, d_model * L_NS),
        )

    def forward(self, ns_features: Tensor) -> Tensor:
        # ns_features: (B, ns_input_dim)
        B = ns_features.size(0)
        return self.proj(ns_features).view(B, self.L_NS, self.d_model)  # (B, L_NS, d)


class SequentialTokenizer(nn.Module):
    """Projects each behavior sequence to d_model and merges with [SEP] delimiters.

    Supports two merge strategies (paper §3.2.2):
      timestamp_aware=True  — interleave events by time with sequence-type indicators.
      timestamp_aware=False — concatenate sequences by impact with [SEP] tokens.

    This implementation uses the timestamp-agnostic strategy (concatenation + SEP),
    which the paper shows performs slightly worse than timestamp-aware when timestamps
    are available, but is simpler and still benefits from SEP tokens.
    """

    def __init__(self, seq_dims: list[int], d_model: int) -> None:
        super().__init__()
        self.seq_projs = nn.ModuleList([nn.Linear(d, d_model) for d in seq_dims])
        self.sep_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_emb  = nn.Embedding(len(seq_dims), d_model)
        nn.init.trunc_normal_(self.sep_token, std=0.02)

    def forward(self, sequences: list[Tensor],
                type_ids: Optional[list[int]] = None) -> Tensor:
        """
        Args:
            sequences: list of (B, L_i, raw_dim_i) tensors, one per behavior type.
            type_ids:  optional integer type label per sequence.
        Returns:
            S-tokens: (B, L_S, d_model)  where L_S = Σ L_i + (n_seqs − 1) SEP tokens.
        """
        B = sequences[0].size(0)
        parts: list[Tensor] = []
        for i, (seq, proj) in enumerate(zip(sequences, self.seq_projs)):
            tok = proj(seq)  # (B, L_i, d_model)
            if type_ids is not None:
                tok = tok + self.type_emb(
                    torch.tensor(type_ids[i], device=seq.device)
                )
            parts.append(tok)
            if i < len(sequences) - 1:
                parts.append(self.sep_token.expand(B, -1, -1))
        return torch.cat(parts, dim=1)  # (B, L_S, d_model)


# ---------------------------------------------------------------------------
# Mixed Causal Attention
# ---------------------------------------------------------------------------

class MixedCausalAttention(nn.Module):
    """Causal MHA with mixed shared/token-specific parameterization.

    S-tokens share one W_S^{Q,K,V} projection.
    Each NS-token i has its own W_{NS,i}^{Q,K,V} projection.

    Causal mask enforces:
      - S-tokens attend only to preceding S-tokens (autoregressive).
      - NS-tokens attend to the full S-history + preceding NS-tokens.

    Pyramid: at each layer, only the tail L_q = L_S - query_start S-tokens
    produce queries; keys/values span the full current sequence.
    """

    def __init__(self, d_model: int, n_heads: int, L_NS: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.d_model = d_model
        self.L_NS    = L_NS
        self.scale   = self.d_head ** -0.5

        # Shared QKV for all S-tokens
        self.W_S_qkv = nn.Linear(d_model, 3 * d_model, bias=False)

        # Token-specific QKV for each NS-token: (L_NS, d_model, 3*d_model)
        self.W_NS_qkv = nn.Parameter(torch.empty(L_NS, d_model, 3 * d_model))
        nn.init.xavier_uniform_(self.W_NS_qkv.reshape(L_NS * d_model, 3 * d_model))

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor, L_S: int, query_start: int = 0) -> Tensor:
        """
        Args:
            x:            (B, L_S + L_NS, d_model)
            L_S:          number of S-tokens in x.
            query_start:  index of first S-token used as query (pyramid truncation).
                          Queries = x[:, query_start:L_S] (S) + x[:, L_S:] (NS).
        Returns:
            (B, L_q + L_NS, d_model)  where L_q = L_S − query_start.
        """
        B, L_total, D = x.shape
        H, Dh = self.n_heads, self.d_head
        L_NS  = self.L_NS
        L_q   = L_S - query_start

        x_S  = x[:, :L_S]           # (B, L_S, D)
        x_NS = x[:, L_S:L_S + L_NS] # (B, L_NS, D)

        # QKV for both token types (compute S once for reuse)
        qkv_S  = self.W_S_qkv(x_S)                                      # (B, L_S, 3D)
        qkv_NS = torch.einsum("bnd,ndo->bno", x_NS, self.W_NS_qkv)      # (B, L_NS, 3D)

        Q_S,  K_S,  V_S  = qkv_S.chunk(3,  dim=-1)   # each (B, L_S, D)
        Q_NS, K_NS, V_NS = qkv_NS.chunk(3, dim=-1)   # each (B, L_NS, D)

        # Pyramid: tail queries only for S-tokens
        Q_S = Q_S[:, query_start:]   # (B, L_q, D)
        Q   = torch.cat([Q_S, Q_NS], dim=1)          # (B, L_q + L_NS, D)
        K   = torch.cat([K_S, K_NS], dim=1)          # (B, L_total, D)
        V   = torch.cat([V_S, V_NS], dim=1)          # (B, L_total, D)

        # Causal mask: query at position p can attend to key at position j only if j ≤ p.
        # Query positions: [query_start .. L_S-1, L_S .. L_S+L_NS-1]
        q_pos = torch.arange(query_start, L_S + L_NS, device=x.device)  # (L_q+L_NS,)
        k_pos = torch.arange(L_total, device=x.device)                   # (L_total,)
        causal_mask = q_pos[:, None] < k_pos[None, :]  # (L_q+L_NS, L_total); True=masked

        # Multi-head attention
        def to_heads(t: Tensor) -> Tensor:
            s = t.size(1)
            return t.view(B, s, H, Dh).transpose(1, 2)  # (B, H, s, Dh)

        Q, K, V = to_heads(Q), to_heads(K), to_heads(V)
        scores  = (Q @ K.transpose(-2, -1)) * self.scale          # (B, H, L_q+L_NS, L_total)
        scores  = scores.masked_fill(causal_mask[None, None], float("-inf"))
        attn    = F.softmax(scores, dim=-1)
        attn    = torch.nan_to_num(attn)   # rows all-masked (empty S history) → zeros

        out = (attn @ V).transpose(1, 2).reshape(B, L_q + L_NS, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Mixed FFN
# ---------------------------------------------------------------------------

class MixedFFN(nn.Module):
    """FFN with mixed shared/token-specific parameterization.

    S-tokens: one shared (W1, W2).
    NS-tokens: token-specific (W1_i, W2_i) per position.
    Activation: SiLU.
    """

    def __init__(self, d_model: int, L_NS: int, expand: int = 4) -> None:
        super().__init__()
        hidden = d_model * expand
        self.L_NS = L_NS

        # Shared FFN for S-tokens
        self.W_S1 = nn.Linear(d_model, hidden,  bias=False)
        self.W_S2 = nn.Linear(hidden,  d_model, bias=False)

        # Token-specific FFN for NS-tokens
        self.W_NS1 = nn.Parameter(torch.empty(L_NS, d_model, hidden))
        self.W_NS2 = nn.Parameter(torch.empty(L_NS, hidden,  d_model))
        self.b_NS1 = nn.Parameter(torch.zeros(L_NS, hidden))
        self.b_NS2 = nn.Parameter(torch.zeros(L_NS, d_model))
        nn.init.xavier_uniform_(self.W_NS1.reshape(L_NS * d_model, hidden))
        nn.init.xavier_uniform_(self.W_NS2.reshape(L_NS * hidden,  d_model))

    def forward(self, x: Tensor, L_S_q: int) -> Tensor:
        """
        Args:
            x:     (B, L_S_q + L_NS, d_model)  — query-position tokens only.
            L_S_q: number of S-token queries in x (first L_S_q rows).
        """
        x_S  = x[:, :L_S_q]
        x_NS = x[:, L_S_q:]

        out_S  = self.W_S2(F.silu(self.W_S1(x_S)))
        h_NS   = F.silu(torch.einsum("bnd,ndo->bno", x_NS, self.W_NS1) + self.b_NS1)
        out_NS = torch.einsum("bno,nod->bnd", h_NS, self.W_NS2) + self.b_NS2

        return torch.cat([out_S, out_NS], dim=1)


# ---------------------------------------------------------------------------
# OneTrans Block
# ---------------------------------------------------------------------------

class OneTransBlock(nn.Module):
    """Single OneTrans block: Pre-Norm → MixedMHA + residual → Pre-Norm → MixedFFN + residual."""

    def __init__(self, d_model: int, n_heads: int, L_NS: int,
                 ffn_expand: int = 4) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = MixedCausalAttention(d_model, n_heads, L_NS)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = MixedFFN(d_model, L_NS, expand=ffn_expand)

    def forward(self, x: Tensor, L_S: int, query_start: int = 0) -> Tensor:
        """
        Args:
            x:            (B, L_S + L_NS, d_model)
            L_S:          current number of S-tokens in x.
            query_start:  first S-token index to use as query (pyramid truncation).
        Returns:
            (B, L_q + L_NS, d_model)  where L_q = L_S − query_start.
        """
        L_q = L_S - query_start

        # Attention sub-layer (pre-norm, residual to query slice of x)
        attn_out = self.attn(self.norm1(x), L_S, query_start)  # (B, L_q+L_NS, D)
        z = x[:, query_start:] + attn_out                       # residual

        # FFN sub-layer
        return z + self.ffn(self.norm2(z), L_q)


# ---------------------------------------------------------------------------
# Pyramid schedule
# ---------------------------------------------------------------------------

def _pyramid_query_counts(
    L_S: int, L_NS: int, n_layers: int, multiple_of: int = 32
) -> list[int]:
    """S-token query counts per layer, linearly decreasing from L_S to L_NS.

    At layer 0 all L_S tokens are queries; at the final layer only L_NS remain
    (matching the NS-token count), so the pyramid terminates with equal depth
    for both token types.
    """
    counts = []
    for l in range(n_layers):
        frac  = l / max(n_layers - 1, 1)
        count = round(L_S + frac * (L_NS - L_S))
        count = max(L_NS, (count // multiple_of) * multiple_of or L_NS)
        if l == n_layers - 1:
            count = L_NS
        counts.append(count)
    return counts


# ---------------------------------------------------------------------------
# Full OneTrans model
# ---------------------------------------------------------------------------

class OneTrans(nn.Module):
    """OneTrans: unified sequence modeling + feature interaction in one Transformer.

    Args:
        seq_dims:      raw embedding dims per behavior sequence type.
        ns_input_dim:  total dim of all non-sequential features concatenated.
        d_model:       hidden dimension d.
        n_heads:       attention heads H.
        n_layers:      stack depth L.
        L_NS:          number of non-sequential tokens.
        max_seq_len:   maximum total S-token length (used for positional embeddings).
        num_tasks:     number of prediction heads (e.g. 2 for CTR + CVR).
        ffn_expand:    FFN hidden expansion factor.
        pyramid_mult:  round pyramid query counts to multiples of this value.
    """

    def __init__(
        self,
        seq_dims: list[int],
        ns_input_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        L_NS: int = 32,
        max_seq_len: int = 512,
        num_tasks: int = 2,
        ffn_expand: int = 4,
        pyramid_mult: int = 32,
    ) -> None:
        super().__init__()
        self.d_model      = d_model
        self.L_NS         = L_NS
        self.n_layers     = n_layers
        self.pyramid_mult = pyramid_mult

        self.s_tokenizer  = SequentialTokenizer(seq_dims, d_model)
        self.ns_tokenizer = AutoSplitNSTokenizer(ns_input_dim, d_model, L_NS)

        # Positional embeddings — sized for max possible sequence length
        self.pos_emb = nn.Embedding(max_seq_len + L_NS, d_model)

        self.blocks = nn.ModuleList([
            OneTransBlock(d_model, n_heads, L_NS, ffn_expand)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)

        # Task heads on mean-pooled final NS-token states
        self.heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(num_tasks)])

    def forward(
        self,
        sequences: list[Tensor],
        ns_features: Tensor,
        type_ids: Optional[list[int]] = None,
    ) -> Tensor:
        """
        Args:
            sequences:   list of (B, L_i, raw_dim_i) behavior sequence tensors.
            ns_features: (B, ns_input_dim) non-sequential features.
            type_ids:    optional sequence type labels for type embeddings.
        Returns:
            logits: (B, num_tasks)
        """
        s_tokens  = self.s_tokenizer(sequences, type_ids)  # (B, L_S, d)
        ns_tokens = self.ns_tokenizer(ns_features)          # (B, L_NS, d)

        B, L_S_init, D = s_tokens.shape

        # Positional embeddings over current full sequence
        pos_ids = torch.arange(L_S_init + self.L_NS, device=s_tokens.device)
        pos     = self.pos_emb(pos_ids).unsqueeze(0)  # (1, L_S+L_NS, d)
        x       = torch.cat([s_tokens, ns_tokens], dim=1) + pos  # (B, L_S+L_NS, d)

        # Pyramid stack: compute per-layer query counts from actual L_S_init
        pyramid = _pyramid_query_counts(
            L_S_init, self.L_NS, self.n_layers, self.pyramid_mult
        )
        L_S_cur = L_S_init
        for l, block in enumerate(self.blocks):
            target  = min(pyramid[l], L_S_cur)
            q_start = L_S_cur - target
            x       = block(x, L_S=L_S_cur, query_start=q_start)
            L_S_cur = target

        x      = self.final_norm(x)
        ns_out = x[:, -self.L_NS:]     # (B, L_NS, d) — final NS-token states
        pooled = ns_out.mean(dim=1)    # (B, d)
        return torch.cat([h(pooled) for h in self.heads], dim=-1)  # (B, num_tasks)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B = 4

    # Two behavior sequences: clicks (L=20, dim=32) and purchases (L=10, dim=32)
    sequences   = [torch.randn(B, 20, 32), torch.randn(B, 10, 32)]
    ns_features = torch.randn(B, 128)   # 128-dim non-sequential features

    model = OneTrans(
        seq_dims=[32, 32],
        ns_input_dim=128,
        d_model=64,
        n_heads=4,
        n_layers=4,
        L_NS=8,
        max_seq_len=64,     # >= sum(L_i) + n_seqs - 1 SEP tokens = 31
        num_tasks=2,        # CTR + CVR
        ffn_expand=2,
        pyramid_mult=8,
    )

    logits = model(sequences, ns_features, type_ids=[0, 1])
    print(f"logits:   {logits.shape}   {logits}")

    total = sum(p.numel() for p in model.parameters())
    print(f"params:   {total:,}")

    L_S_example = 20 + 1 + 10  # 31 S-tokens
    pyramid = _pyramid_query_counts(L_S_example, 8, 4, 8)
    print(f"pyramid:  {pyramid}  (S-token query counts per layer)")

    logits.sum().backward()
    print("backward: OK")


if __name__ == "__main__":
    _smoke_test()
