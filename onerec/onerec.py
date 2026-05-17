"""OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment.

Reference PyTorch implementation.
Paper: https://arxiv.org/abs/2502.18965  (KuaiShou Inc., Feb 2025)

Architecture overview:
    User behavior sequences H_u  →  ResidualQuantizer (Balanced K-means)
                                  →  OneRecEncoder  (T5-like, N/2 layers)
                                  →  H = encoder output
    Target session S             →  OneRecDecoder  (N/2 layers: causal-attn + cross-attn + MoE)
                                  →  NTP loss over semantic ID codes

Post-training — Iterative Preference Alignment (IPA):
    1. Train RewardModel (swt / vtr / ltr towers) on engagement labels.
    2. For each user, beam-search N candidate sessions, score with RM.
    3. Pick winner (max reward) and loser (min reward) as DPO preference pair.
    4. Apply DPO loss jointly with NTP loss; iterate M_t → M_{t+1}.

Key numbers from the paper:
    L=3 codebook levels, K=8192 codebook size per level
    Encoder / Decoder: N/2 = 6 layers each for the 1B model
    MoE: N_MoE=24 experts, K_MoE=2 active per token (top-k)
    Session size m=5 items, history n=256 items
    DPO sample ratio r_DPO=1%, beam size=128
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


# ---------------------------------------------------------------------------
# Balanced K-means Semantic Tokenizer (Algorithm 1 in paper)
# ---------------------------------------------------------------------------

class BalancedKMeans:
    """Offline balanced K-means clustering for item semantic tokenization.

    Unlike standard K-means, each cluster is forced to hold exactly
    w = |V| / K items, preventing the hourglass phenomenon where a few
    large clusters dominate the codebook.

    Usage (offline, before model training):
        bkm = BalancedKMeans(K=8192, max_iter=100)
        bkm.fit(embeddings)          # (N, d) numpy array
        codes = bkm.encode(embeddings)  # (N,) integer cluster IDs
    """

    def __init__(self, K: int = 8192, max_iter: int = 100,
                 tol: float = 1e-4, seed: int = 0) -> None:
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.centroids: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "BalancedKMeans":
        rng = np.random.default_rng(self.seed)
        N, d = X.shape
        w = N // self.K
        idx = rng.choice(N, self.K, replace=False)
        C = X[idx].copy()  # (K, d)

        for _ in range(self.max_iter):
            # Sort each point by distance to its nearest centroid
            dists = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=-1)  # (N, K)
            order = np.argsort(dists.ravel())  # ascending distance, flat index

            assignment = np.full(N, -1, dtype=np.int64)
            counts = np.zeros(self.K, dtype=np.int64)
            for flat_idx in order:
                i, k = divmod(flat_idx, self.K)
                if assignment[i] == -1 and counts[k] < w:
                    assignment[i] = k
                    counts[k] += 1

            # Any unassigned items (remainder) go to first cluster with room
            for i in np.where(assignment == -1)[0]:
                for k in range(self.K):
                    if counts[k] < w + 1:
                        assignment[i] = k
                        counts[k] += 1
                        break

            # Recompute centroids
            C_new = np.zeros_like(C)
            for k in range(self.K):
                mask = assignment == k
                if mask.any():
                    C_new[k] = X[mask].mean(0)
            if np.linalg.norm(C_new - C) < self.tol:
                break
            C = C_new

        self.centroids = C
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        assert self.centroids is not None, "Call fit() first."
        dists = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=-1)
        return dists.argmin(axis=-1)


class ResidualQuantizer(nn.Module):
    """Multi-level residual quantization with frozen codebooks.

    Encodes item embedding e_i into L integer codes (s^1, ..., s^L) via
    hierarchical nearest-centroid lookup with residual subtraction.
    Codebooks are trained offline with BalancedKMeans and loaded as buffers.

    Args:
        num_levels:   L, number of codebook levels.
        codebook_size: K, entries per codebook level.
        embed_dim:    d, item embedding dimension.
    """

    def __init__(self, num_levels: int = 3, codebook_size: int = 8192,
                 embed_dim: int = 256) -> None:
        super().__init__()
        self.L = num_levels
        self.K = codebook_size
        # Learnable codebook embeddings (can be initialised from offline BKM)
        self.codebooks = nn.Parameter(
            torch.randn(num_levels, codebook_size, embed_dim) * 0.02
        )

    def encode(self, e: Tensor) -> Tensor:
        """
        Args:
            e: (N, d) item embeddings.
        Returns:
            codes: (N, L) integer code indices.
        """
        codes = []
        r = e
        for l in range(self.L):
            C = self.codebooks[l]                            # (K, d)
            dists = torch.cdist(r.unsqueeze(0), C.unsqueeze(0)).squeeze(0)  # (N, K)
            idx = dists.argmin(dim=-1)                       # (N,)
            codes.append(idx)
            r = r - C[idx]                                   # residual
        return torch.stack(codes, dim=-1)                    # (N, L)

    def decode(self, codes: Tensor) -> Tensor:
        """
        Args:
            codes: (..., L) integer code indices.
        Returns:
            reconstructed embeddings: (..., d)
        """
        out = torch.zeros(*codes.shape[:-1], self.codebooks.shape[-1],
                          device=codes.device, dtype=self.codebooks.dtype)
        for l in range(self.L):
            out = out + self.codebooks[l][codes[..., l]]
        return out

    def forward(self, e: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (codes, reconstructed_embeddings) with straight-through gradient."""
        codes = self.encode(e)
        e_q   = self.decode(codes)
        # Straight-through: gradients flow through e unchanged
        e_st  = e + (e_q - e).detach()
        return codes, e_st


# ---------------------------------------------------------------------------
# Attention & FFN building blocks
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard multi-head attention (fully visible or causal)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5
        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        B, Tq, _ = q.shape
        Tk = k.shape[1]
        H, Dh = self.n_heads, self.d_head

        def split(t: Tensor, T: int) -> Tensor:
            return t.view(B, T, H, Dh).transpose(1, 2)

        Q = split(self.q_proj(q), Tq)
        K = split(self.k_proj(k), Tk)
        V = split(self.v_proj(v), Tk)

        scores = (Q @ K.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)
        out  = (attn @ V).transpose(1, 2).reshape(B, Tq, H * Dh)
        return self.out_proj(out)


class FFN(nn.Module):
    """Standard SwiGLU FFN."""

    def __init__(self, d_model: int, expand: int = 4) -> None:
        super().__init__()
        hidden = d_model * expand
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up   = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden,  d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoELayer(nn.Module):
    """Sparse Mixture-of-Experts FFN layer (top-k routing).

    Replaces the dense FFN in decoder layers to scale model capacity
    without proportionally increasing active FLOPs per token.

    During inference, only K_MoE / N_MoE ≈ 8% of parameters are active
    (K_MoE=2, N_MoE=24 in the paper 1B model).
    """

    def __init__(self, d_model: int, num_experts: int = 24, top_k: int = 2,
                 expand: int = 4) -> None:
        super().__init__()
        self.top_k  = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([FFN(d_model, expand) for _ in range(num_experts)])

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        x_flat   = x.reshape(B * T, D)
        logits   = self.router(x_flat)                                # (BT, E)
        topk_w, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        gate_w   = F.softmax(topk_w, dim=-1)                          # (BT, k)

        all_out  = torch.stack([e(x_flat) for e in self.experts], dim=1)  # (BT, E, D)
        idx_exp  = topk_idx.unsqueeze(-1).expand(-1, -1, D)           # (BT, k, D)
        selected = all_out.gather(1, idx_exp)                          # (BT, k, D)
        out      = (gate_w.unsqueeze(-1) * selected).sum(dim=1)        # (BT, D)
        return out.reshape(B, T, D)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """T5-like encoder layer: pre-norm, fully-visible self-attention + FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_expand: int = 4) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = MultiHeadAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = FFN(d_model, ffn_expand)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OneRecEncoder(nn.Module):
    """Encode user historical behavior sequences H_u into H.

    Input:  token IDs from semantic quantization, shape (B, n_hist, L)
    Output: H = encoded representation, shape (B, n_hist * L, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 codebook_size: int, num_levels: int,
                 ffn_expand: int = 4) -> None:
        super().__init__()
        vocab_size = codebook_size * num_levels + 2  # +2 for SEP, PAD
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.layers  = nn.ModuleList([
            EncoderLayer(d_model, n_heads, ffn_expand) for _ in range(n_layers)
        ])
        self.norm    = RMSNorm(d_model)
        self.SEP_ID  = vocab_size - 2
        self.num_levels = num_levels

    def forward(self, token_ids: Tensor) -> Tensor:
        """
        Args:
            token_ids: (B, T) flat token sequence including SEP tokens.
        Returns:
            H: (B, T, d_model) encoded features.
        """
        x = self.embed(token_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """OneRec decoder layer: causal self-attn + cross-attn + MoE FFN."""

    def __init__(self, d_model: int, n_heads: int,
                 num_experts: int = 24, top_k: int = 2,
                 ffn_expand: int = 4) -> None:
        super().__init__()
        self.norm1      = RMSNorm(d_model)
        self.self_attn  = MultiHeadAttention(d_model, n_heads)
        self.norm2      = RMSNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.norm3      = RMSNorm(d_model)
        self.moe        = MoELayer(d_model, num_experts, top_k, ffn_expand)

    def forward(self, x: Tensor, enc_out: Tensor,
                causal_mask: Optional[Tensor] = None) -> Tensor:
        # Causal self-attention
        normed = self.norm1(x)
        x = x + self.self_attn(normed, normed, normed, causal_mask)
        # Cross-attention over encoder output
        normed = self.norm2(x)
        x = x + self.cross_attn(normed, enc_out, enc_out)
        # MoE FFN
        x = x + self.moe(self.norm3(x))
        return x


class OneRecDecoder(nn.Module):
    """Autoregressive decoder that generates session semantic IDs.

    Session token layout (eq. 3 in paper):
        [BOS] s^1_1 s^2_1 ... s^L_1 [BOS] s^1_2 ... s^L_m

    Each item contributes (1 + L) tokens; full sequence length = m * (1 + L).
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 codebook_size: int, num_levels: int,
                 num_experts: int = 24, top_k: int = 2,
                 ffn_expand: int = 4) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        vocab_size  = codebook_size * num_levels + 2   # level-offset tokens + BOS + PAD
        self.BOS_ID = vocab_size - 2
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, num_experts, top_k, ffn_expand)
            for _ in range(n_layers)
        ])
        self.norm   = RMSNorm(d_model)
        # Separate head for each code level (each predicts over K entries)
        self.heads  = nn.ModuleList([
            nn.Linear(d_model, codebook_size, bias=False) for _ in range(num_levels)
        ])

    def _causal_mask(self, T: int, device: torch.device) -> Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, tgt_ids: Tensor, enc_out: Tensor) -> list[Tensor]:
        """
        Args:
            tgt_ids:  (B, T_dec) decoder input token IDs (teacher-forced).
            enc_out:  (B, T_enc, d_model) encoder output.
        Returns:
            logits_per_level: list of L tensors, each (B, T_dec, K) —
                              predictions at each code-level position.
        """
        B, T = tgt_ids.shape
        mask = self._causal_mask(T, tgt_ids.device)

        x = self.embed(tgt_ids)
        for layer in self.layers:
            x = layer(x, enc_out, mask)
        x = self.norm(x)  # (B, T, d)

        # Compute logits for each level (positions cycle: BOS, level-0, level-1, ..., level-L-1, BOS, ...)
        logits = [head(x) for head in self.heads]
        return logits  # each (B, T, K)

    @torch.no_grad()
    def greedy_decode(self, enc_out: Tensor, m: int) -> Tensor:
        """Greedy generation of m-item session.

        Returns:
            codes: (B, m, L) generated semantic code indices.
        """
        B = enc_out.size(0)
        L = self.num_levels
        device = enc_out.device
        # Start with BOS for first item
        generated = torch.full((B, 1), self.BOS_ID, dtype=torch.long, device=device)
        item_codes = []
        for item_idx in range(m):
            item_level_codes = []
            for level in range(L):
                logits = self.forward(generated, enc_out)   # list of L tensors
                next_code = logits[level][:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
                item_level_codes.append(next_code)
                # Offset token by level to distinguish code levels in vocabulary
                token = next_code + level * self.codebook_size
                generated = torch.cat([generated, token], dim=1)
            item_codes.append(torch.cat(item_level_codes, dim=-1))  # (B, L)
            # Append BOS for next item (except after last)
            if item_idx < m - 1:
                bos = torch.full((B, 1), self.BOS_ID, dtype=torch.long, device=device)
                generated = torch.cat([generated, bos], dim=1)
        return torch.stack(item_codes, dim=1)  # (B, m, L)


# ---------------------------------------------------------------------------
# Reward Model (§3.3.1)
# ---------------------------------------------------------------------------

class RewardTower(nn.Module):
    """Single sigmoid tower for one reward objective (swt / vtr / ltr)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.mlp(x)).squeeze(-1)


class RewardModel(nn.Module):
    """Session-wise reward model R(u, S).

    Given a session S = {v1,...,vm} with user u, computes target-aware
    item representations e_i = v_i ⊙ u (Hadamard product), applies
    self-attention to capture cross-item interactions, then predicts
    swt (session watch time), vtr (view-through rate), ltr (like-through rate).

    All three towers share the same self-attention backbone h_f.
    """

    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        self.user_proj  = nn.Linear(d_model, d_model, bias=False)
        self.item_proj  = nn.Linear(d_model, d_model, bias=False)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.norm = RMSNorm(d_model)
        self.tower_swt  = RewardTower(d_model)
        self.tower_vtr  = RewardTower(d_model)
        self.tower_ltr  = RewardTower(d_model)

    def forward(self, user_emb: Tensor, item_embs: Tensor) -> dict[str, Tensor]:
        """
        Args:
            user_emb:   (B, d_model) user representation.
            item_embs:  (B, m, d_model) session item embeddings.
        Returns:
            dict with 'swt', 'vtr', 'ltr' each (B,) predicted rewards.
        """
        u = self.user_proj(user_emb).unsqueeze(1)  # (B, 1, d)
        v = self.item_proj(item_embs)               # (B, m, d)
        e = v * u                                   # (B, m, d) target-aware repr (eq. 5)
        h = self.norm(self.attn(e, e, e) + e)       # self-attention fusion
        pooled = h.sum(dim=1)                        # (B, d) sum-pool over session items
        return {
            "swt": self.tower_swt(pooled),
            "vtr": self.tower_vtr(pooled),
            "ltr": self.tower_ltr(pooled),
        }

    def score(self, user_emb: Tensor, item_embs: Tensor) -> Tensor:
        """Returns a single scalar reward per session (sum of all towers)."""
        preds = self.forward(user_emb, item_embs)
        return preds["swt"] + preds["vtr"] + preds["ltr"]  # (B,)


# ---------------------------------------------------------------------------
# Full OneRec model
# ---------------------------------------------------------------------------

def _build_decoder_input(codes: Tensor, BOS_ID: int, codebook_size: int) -> Tensor:
    """Converts (B, m, L) session codes to (B, m*(1+L)) decoder token IDs.

    Layout per item: [BOS, code_level0, code_level1, ..., code_level_L-1]
    Each code at level l is offset by l * codebook_size to give it a unique
    token ID distinct from codes at other levels.
    """
    B, m, L = codes.shape
    device = codes.device
    # Level offset: code at level l gets ID = code + l * K
    offsets = torch.arange(L, device=device) * codebook_size  # (L,)
    codes_off = codes + offsets.unsqueeze(0).unsqueeze(0)      # (B, m, L)
    bos = torch.full((B, m, 1), BOS_ID, dtype=torch.long, device=device)
    tokens = torch.cat([bos, codes_off], dim=-1)               # (B, m, 1+L)
    return tokens.reshape(B, m * (1 + L))                      # (B, m*(1+L))


def ntp_loss(logits_per_level: list[Tensor], target_codes: Tensor,
             codebook_size: int) -> Tensor:
    """Next-token prediction loss over session semantic IDs (eq. 4 in paper).

    Args:
        logits_per_level: list of L tensors each (B, T_dec, K) — decoder output.
        target_codes:     (B, m, L) ground-truth code indices.
        codebook_size:    K.
    Returns:
        scalar NTP loss.
    """
    B, m, L = target_codes.shape
    T_dec = m * (1 + L)
    loss = torch.tensor(0.0, device=target_codes.device)
    count = 0
    # Position mapping: token positions that are code positions (not BOS):
    # for item i, position i*(1+L)+1+l is code level l of item i.
    for level in range(L):
        logit = logits_per_level[level]  # (B, T_dec, K)
        # Collect predictions at code positions for this level.
        # Prediction at position p predicts the *next* token, so we predict
        # s^{l+1} from the token at position p.  We shift by -1 (predict next).
        for i in range(m):
            pred_pos  = i * (1 + L) + level       # position whose output predicts level+1
            tgt_level = level                      # target code level
            if pred_pos < T_dec:
                logit_slice = logit[:, pred_pos, :]   # (B, K)
                target      = target_codes[:, i, tgt_level]  # (B,)
                loss  = loss + F.cross_entropy(logit_slice, target)
                count += 1
    return loss / max(count, 1)


def dpo_loss(model: "OneRec", ref_model: "OneRec",
             enc_input_w: Tensor, dec_input_w: Tensor, codes_w: Tensor,
             enc_input_l: Tensor, dec_input_l: Tensor, codes_l: Tensor,
             beta: float = 0.1) -> Tensor:
    """DPO loss for preference alignment (eq. 10 in paper).

    Computes log-probability ratio of winner vs. loser sessions relative to
    the reference (previous checkpoint) model.

    Args:
        model / ref_model: current and reference OneRec models.
        *_w / *_l: encoder inputs, decoder inputs, and codes for winner / loser.
        beta: KL penalty coefficient.
    Returns:
        scalar DPO loss.
    """
    def log_prob(m: "OneRec", enc_in: Tensor, dec_in: Tensor, codes: Tensor) -> Tensor:
        enc_out = m.encoder(enc_in)
        logits  = m.decoder(dec_in, enc_out)
        # Sum log-prob over all code positions
        B, sess_m, L = codes.shape
        lp = torch.zeros(B, device=codes.device)
        for level in range(L):
            for i in range(sess_m):
                pred_pos = i * (1 + L) + level
                lp = lp + F.log_softmax(logits[level][:, pred_pos, :], dim=-1
                                        ).gather(1, codes[:, i, level:level+1]).squeeze(-1)
        return lp

    with torch.no_grad():
        ref_lp_w = log_prob(ref_model, enc_input_w, dec_input_w, codes_w)
        ref_lp_l = log_prob(ref_model, enc_input_l, dec_input_l, codes_l)

    lp_w = log_prob(model, enc_input_w, dec_input_w, codes_w)
    lp_l = log_prob(model, enc_input_l, dec_input_l, codes_l)

    ratio = beta * ((lp_w - ref_lp_w) - (lp_l - ref_lp_l))
    return -F.logsigmoid(ratio).mean()


class OneRec(nn.Module):
    """OneRec: end-to-end generative session recommendation.

    Args:
        d_model:        hidden dimension.
        n_heads:        attention heads (d_model must be divisible by n_heads).
        n_enc_layers:   encoder depth (paper: N/2 = 6 for 1B model).
        n_dec_layers:   decoder depth.
        codebook_size:  K, entries per codebook level.
        num_levels:     L, residual quantization levels.
        num_experts:    N_MoE, total experts in MoE layers.
        top_k:          K_MoE, active experts per token.
        ffn_expand:     SwiGLU hidden expansion factor.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_enc_layers: int = 6,
        n_dec_layers: int = 6,
        codebook_size: int = 8192,
        num_levels: int = 3,
        num_experts: int = 24,
        top_k: int = 2,
        ffn_expand: int = 4,
    ) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.num_levels    = num_levels

        self.encoder = OneRecEncoder(
            d_model, n_heads, n_enc_layers,
            codebook_size, num_levels, ffn_expand,
        )
        self.decoder = OneRecDecoder(
            d_model, n_heads, n_dec_layers,
            codebook_size, num_levels,
            num_experts, top_k, ffn_expand,
        )
        self.BOS_ID = self.decoder.BOS_ID

    def forward(self, src_ids: Tensor, tgt_codes: Tensor) -> Tensor:
        """Teacher-forced forward pass returning NTP loss.

        Args:
            src_ids:    (B, T_enc) encoder input token IDs (history).
            tgt_codes:  (B, m, L) ground-truth session code indices.
        Returns:
            scalar NTP loss.
        """
        enc_out  = self.encoder(src_ids)
        dec_in   = _build_decoder_input(tgt_codes, self.BOS_ID, self.codebook_size)
        logits   = self.decoder(dec_in, enc_out)
        return ntp_loss(logits, tgt_codes, self.codebook_size)

    @torch.no_grad()
    def generate(self, src_ids: Tensor, m: int = 5) -> Tensor:
        """Greedy session generation.

        Args:
            src_ids: (B, T_enc) encoder input token IDs.
            m:       number of items to generate per session.
        Returns:
            (B, m, L) session code indices.
        """
        enc_out = self.encoder(src_ids)
        return self.decoder.greedy_decode(enc_out, m)


# ---------------------------------------------------------------------------
# Iterative Preference Alignment (Algorithm 2)
# ---------------------------------------------------------------------------

class IterativePreferenceAlignment:
    """IPA trainer: wraps OneRec + RewardModel for iterative DPO fine-tuning.

    Training loop (Algorithm 2):
      For each epoch t in 1..T:
        For each training sample:
          With probability r_DPO:
            Generate N sessions via beam search (approximated by repeated greedy here).
            Score with RM; select winner (max) and loser (min).
            Compute L = L_NTP + λ * L_DPO.
          Otherwise:
            Compute L = L_NTP only.
        Save M_{t+1} from M_t.

    Args:
        model:          seed OneRec model M_t.
        reward_model:   pre-trained RewardModel R(u, S).
        beta:           DPO KL coefficient.
        lam:            DPO loss weight λ (paper uses 1.0; tune with ablation).
        r_dpo:          fraction of samples that use DPO (paper: 0.01).
        n_candidates:   N candidates per DPO sample (paper: 128; reduced here for speed).
    """

    def __init__(self, model: OneRec, reward_model: RewardModel,
                 beta: float = 0.1, lam: float = 1.0,
                 r_dpo: float = 0.01, n_candidates: int = 8) -> None:
        self.model        = model
        self.reward_model = reward_model
        self.beta         = beta
        self.lam          = lam
        self.r_dpo        = r_dpo
        self.n_candidates = n_candidates
        # Reference model (frozen snapshot of M_t)
        import copy
        self.ref_model = copy.deepcopy(model)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    def step(
        self,
        src_ids: Tensor,       # (B, T_enc)
        tgt_codes: Tensor,     # (B, m, L) ground-truth session
        user_embs: Tensor,     # (B, d) for reward model
        item_emb_fn,           # callable: codes (B, m, L) → item_embs (B, m, d)
    ) -> Tensor:
        """Single training step (one mini-batch).

        Returns:
            total loss scalar.
        """
        # Always compute NTP loss
        loss = self.model(src_ids, tgt_codes)

        if torch.rand(1).item() < self.r_dpo:
            # Generate N candidate sessions for each sample in the batch
            device    = src_ids.device
            B         = src_ids.size(0)
            enc_out   = self.model.encoder(src_ids)
            m         = tgt_codes.size(1)

            best_codes  = None
            best_scores = torch.full((B,), float("-inf"), device=device)
            worst_codes = None
            worst_scores = torch.full((B,), float("inf"),  device=device)

            for _ in range(self.n_candidates):
                cand_codes = self.model.decoder.greedy_decode(enc_out, m)  # (B, m, L)
                item_embs  = item_emb_fn(cand_codes)                        # (B, m, d)
                scores     = self.reward_model.score(user_embs, item_embs)  # (B,)

                better = scores > best_scores
                if better.any():
                    best_scores = torch.where(better, scores, best_scores)
                    best_codes  = cand_codes if best_codes is None else torch.where(
                        better[:, None, None].expand_as(cand_codes), cand_codes, best_codes)

                worse = scores < worst_scores
                if worse.any():
                    worst_scores = torch.where(worse, scores, worst_scores)
                    worst_codes  = cand_codes if worst_codes is None else torch.where(
                        worse[:, None, None].expand_as(cand_codes), cand_codes, worst_codes)

            if best_codes is not None and worst_codes is not None:
                dec_w = _build_decoder_input(best_codes,  self.model.BOS_ID,
                                             self.model.codebook_size)
                dec_l = _build_decoder_input(worst_codes, self.model.BOS_ID,
                                             self.model.codebook_size)
                loss_dpo = dpo_loss(
                    self.model, self.ref_model,
                    src_ids, dec_w, best_codes,
                    src_ids, dec_l, worst_codes,
                    self.beta,
                )
                loss = loss + self.lam * loss_dpo

        return loss

    def update_reference(self) -> None:
        """Snapshot current model as the new reference for the next epoch."""
        import copy
        self.ref_model = copy.deepcopy(self.model)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B         = 2
    T_enc     = 20    # encoder input length (history of 10 items × 2 codes each, simplified)
    m         = 3     # session size (paper uses 5)
    L         = 3     # code levels
    K         = 64    # codebook size (paper uses 8192)
    d_model   = 64
    n_heads   = 4
    n_layers  = 2
    n_experts = 4
    top_k     = 2

    model = OneRec(
        d_model=d_model,
        n_heads=n_heads,
        n_enc_layers=n_layers,
        n_dec_layers=n_layers,
        codebook_size=K,
        num_levels=L,
        num_experts=n_experts,
        top_k=top_k,
        ffn_expand=2,
    )

    # Encoder input: flat sequence of semantic token IDs
    vocab_size = K * L + 2
    src_ids   = torch.randint(0, vocab_size - 2, (B, T_enc))

    # Target session: (B, m, L) code indices
    tgt_codes = torch.randint(0, K, (B, m, L))

    # Forward / NTP loss
    loss = model(src_ids, tgt_codes)
    print(f"NTP loss:  {loss.item():.4f}")

    # Backward
    loss.backward()
    print("backward:  OK")

    # Greedy generation
    model.eval()
    with torch.no_grad():
        codes = model.generate(src_ids, m=m)
    print(f"generated: {codes.shape}  (B, m, L) = {list(codes.shape)}")

    total = sum(p.numel() for p in model.parameters())
    print(f"params:    {total:,}")

    # Reward model smoke test
    rm = RewardModel(d_model)
    user_emb  = torch.randn(B, d_model)
    item_embs = torch.randn(B, m, d_model)
    rewards   = rm.forward(user_emb, item_embs)
    print(f"RM swt:    {rewards['swt'].shape}   {rewards['swt'].tolist()}")
    score     = rm.score(user_emb, item_embs)
    print(f"RM score:  {score.tolist()}")


if __name__ == "__main__":
    _smoke_test()
