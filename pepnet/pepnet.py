"""PEPNet: Parameter and Embedding Personalized Network.

Reference PyTorch implementation.
Paper: https://arxiv.org/abs/2302.01115  (KDD 2023, Kuaishou Technology)

Architecture overview:

  Input features
    ├─ Sparse + Dense  → Embedding Layer (shared)  → E ∈ (B, emb_flat)
    ├─ Domain ID, stats → domain_emb               ∈ (B, 2·d)
    └─ User/Item/Author → O_prior                  ∈ (B, 3·d)

  EPNet – Embedding Personalized Network (§2.2.2):
    δ_domain = GateNU_0(domain_emb ⊕ ∅(E))       ∅ = stop gradient
    O_ep     = δ_domain ⊗ E                        element-wise gate

  PPNet – Parameter Personalized Network (§2.2.3):
    δ_task^(l) = GateNU_l(O_prior ⊕ ∅(O_ep)),   l = 1..L
    δ_task^(l) ∈ ℝ^(h_l · T) → split into T task gates each ∈ ℝ^(h_l)

  T Task Towers (personalized per layer by PPNet):
    H_t^(1)    = O_ep                                  initial hidden state
    O_pp_t^(l) = δ_task_t^(l) ⊗ H_t^(l)               gate scales H before linear
    H_t^(l+1)  = ReLU(O_pp_t^(l) W_t^(l) + b_t^(l))  l < L
    ŷ_t        = O_pp_t^(L) W_t^(L) + b_t^(L)         output (no activation)

  Gate Neural Unit (§2.2.1):
    x'  = ReLU(x W + b)
    δ   = γ · Sigmoid(x' W' + b'),   δ ∈ [0, γ],  γ = 2

Production (Kuaishou, 300M DAU): +1.08–2.11% Like, +1.43–2.23% Follow,
  +1.31–1.55% Forward, +1.25–2.12% Watch Time across three domains.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Gate Neural Unit   §2.2.1
# ---------------------------------------------------------------------------

class GateNU(nn.Module):
    """Gate Neural Unit: two-layer MLP producing personalized scaling gates (§2.2.1).

    Inspired by LHUC from speech recognition. Takes prior information x and
    outputs a gate vector δ ∈ [0, γ] that adaptively scales downstream activations:

        x'  = ReLU(x W + b)
        δ   = γ · Sigmoid(x' W' + b'),   δ ∈ [0, γ]

    γ = 2 (default): the output is centered at 1 with range [0, 2], enabling
    both amplification (δ > 1) and suppression (δ < 1). This doubles the
    effective signal strength for important features without hard clipping.

    Args:
        input_dim:  dimension of prior information input x.
        output_dim: dimension of output gate vector δ.
        gamma:      scaling factor (default 2.0).
    """

    def __init__(self, input_dim: int, output_dim: int,
                 gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Returns gate δ ∈ [0, γ] of shape (..., output_dim)."""
        return self.gamma * torch.sigmoid(self.fc2(F.relu(self.fc1(x))))


# ---------------------------------------------------------------------------
# EPNet – Embedding Personalized Network   §2.2.2
# ---------------------------------------------------------------------------

class EPNet(nn.Module):
    """Embedding Personalized Network: domain-specific embedding gating (§2.2.2).

    Injects domain-specific personalization into the shared embedding layer using
    domain-side features (domain ID + domain statistics). The gate scales individual
    embedding dimensions to align feature importance for different domains.

    Stop gradient ∅ on E prevents the Gate NU from interfering with the embedding
    learning dynamics — EPNet adjusts the output of E, not E itself:

        δ_domain = GateNU(E(F_d) ⊕ ∅(E))
        O_ep     = δ_domain ⊗ E

    Args:
        domain_feat_dim: dimension of domain feature embedding E(F_d).
        emb_flat_dim:    flattened general embedding dimension.
        gamma:           Gate NU scaling factor.
    """

    def __init__(self, domain_feat_dim: int, emb_flat_dim: int,
                 gamma: float = 2.0) -> None:
        super().__init__()
        self.gate = GateNU(domain_feat_dim + emb_flat_dim, emb_flat_dim, gamma)

    def forward(self, domain_emb: Tensor, E: Tensor) -> Tensor:
        """
        Args:
            domain_emb: (B, domain_feat_dim) domain feature embedding E(F_d).
            E:          (B, emb_flat_dim) general shared embedding.
        Returns:
            O_ep: (B, emb_flat_dim) domain-personalized embedding.
        """
        gate_in = torch.cat([domain_emb, E.detach()], dim=-1)
        return self.gate(gate_in) * E


# ---------------------------------------------------------------------------
# PPNet – Parameter Personalized Network   §2.2.3
# ---------------------------------------------------------------------------

class PPNet(nn.Module):
    """Parameter Personalized Network: per-layer DNN parameter gating (§2.2.3).

    Generates task-specific gates for each hidden layer of the T task towers.
    One GateNU per layer; stop gradient ∅ on O_ep avoids interfering with EPNet.

        δ_task^(l) = GateNU_l(O_prior ⊕ ∅(O_ep)),   l = 1..L
        δ_task^(l) ∈ ℝ^(h_l · T)  split into T task gates each ∈ ℝ^(h_l)

    gate_dims[l] = h_l = the dimension of H^(l) being gated at layer l.
    Specifically:
        gate_dims[0]  = emb_flat  (H^(1) = O_ep, the personalized embedding)
        gate_dims[l]  = dnn_hidden[l-1]  for l ≥ 1  (output of hidden layer l-1)

    Args:
        prior_dim:  dimension of O_prior (user+item+author embeddings).
        ep_dim:     dimension of O_ep (= emb_flat).
        gate_dims:  list of h_l, one per tower linear layer (len = L).
        n_tasks:    number of prediction tasks T.
        gamma:      Gate NU scaling factor.
    """

    def __init__(self, prior_dim: int, ep_dim: int, gate_dims: list[int],
                 n_tasks: int, gamma: float = 2.0) -> None:
        super().__init__()
        self.n_tasks = n_tasks
        self.gate_dims = gate_dims
        gate_in_dim = prior_dim + ep_dim
        self.gates = nn.ModuleList([
            GateNU(gate_in_dim, h * n_tasks, gamma)
            for h in gate_dims
        ])

    def forward(self, O_prior: Tensor, O_ep: Tensor) -> list[Tensor]:
        """
        Args:
            O_prior: (B, prior_dim) prior feature embedding.
            O_ep:    (B, ep_dim)    personalized embedding (stop-grad applied).
        Returns:
            gates: list of L tensors each (B, n_tasks, h_l).
        """
        gate_in = torch.cat([O_prior, O_ep.detach()], dim=-1)
        result = []
        for gate_nu, h in zip(self.gates, self.gate_dims):
            delta = gate_nu(gate_in)                        # (B, h * n_tasks)
            result.append(delta.view(-1, self.n_tasks, h))  # (B, n_tasks, h)
        return result


# ---------------------------------------------------------------------------
# Full PEPNet model
# ---------------------------------------------------------------------------

class PEPNet(nn.Module):
    """PEPNet: Parameter and Embedding Personalized Network (KDD 2023, Kuaishou).

    Plug-and-play architecture for multi-domain and multi-task recommendation.
    Addresses the "imperfectly double seesaw" phenomenon:
      - Domain seesaw: optimizing one domain degrades others.
      - Task seesaw: optimizing one task degrades others.

    EPNet resolves the domain seesaw by personalizing embeddings per domain.
    PPNet resolves the task seesaw by personalizing DNN hidden units per task.
    Both use stop gradients to avoid interfering with the shared bottom layers.

    The full forward pass:
      E       = shared embedding (sparse + dense features)
      O_ep    = EPNet(domain features, E)         domain-personalized embedding
      O_prior = concat(user_emb, item_emb, author_emb)
      gates   = PPNet(O_prior, O_ep)              per-layer, per-task gates
      ŷ_t     = task_tower_t(O_ep, gates)         gated prediction per task

    Args:
        sparse_vocab_sizes: vocabulary sizes for general sparse features.
        dense_input_dims:   input dims for each dense feature group.
        d_embed:            embedding dimension for all features.
        domain_vocab_size:  domain ID vocabulary size (for EPNet).
        n_domain_stats:     number of domain statistics features (for EPNet).
        user_vocab_size:    user ID vocabulary (for PPNet prior).
        item_vocab_size:    item ID vocabulary (for PPNet prior).
        author_vocab_size:  author ID vocabulary (for PPNet prior).
        dnn_hidden:         hidden dims for task towers, e.g. [256, 128].
        n_tasks:            number of prediction tasks T.
        gamma:              Gate NU scaling factor (default 2.0).
    """

    def __init__(
        self,
        sparse_vocab_sizes: list[int],
        dense_input_dims:   list[int],
        d_embed:            int,
        domain_vocab_size:  int,
        n_domain_stats:     int,
        user_vocab_size:    int,
        item_vocab_size:    int,
        author_vocab_size:  int,
        dnn_hidden:         list[int],
        n_tasks:            int,
        gamma:              float = 2.0,
    ) -> None:
        super().__init__()
        self.n_tasks   = n_tasks
        self.dnn_hidden = dnn_hidden

        # --- Shared embedding layer ---
        self.sparse_embs = nn.ModuleList([
            nn.Embedding(v, d_embed) for v in sparse_vocab_sizes
        ])
        self.dense_projs = nn.ModuleList([
            nn.Linear(dim, d_embed, bias=False) for dim in dense_input_dims
        ])
        n_tokens = len(sparse_vocab_sizes) + len(dense_input_dims)
        emb_flat = n_tokens * d_embed

        # --- Domain features for EPNet ---
        self.domain_id_emb    = nn.Embedding(domain_vocab_size, d_embed)
        self.domain_stats_proj = nn.Linear(n_domain_stats, d_embed, bias=False)
        domain_feat_dim       = 2 * d_embed   # domain_id_emb + stats_emb

        # --- Prior features for PPNet ---
        self.user_emb   = nn.Embedding(user_vocab_size, d_embed)
        self.item_emb   = nn.Embedding(item_vocab_size, d_embed)
        self.author_emb = nn.Embedding(author_vocab_size, d_embed)
        prior_dim = 3 * d_embed

        # --- EPNet ---
        self.epnet = EPNet(domain_feat_dim, emb_flat, gamma)

        # --- PPNet ---
        # gate_dims[l] = dim of H^(l) being gated at linear layer l:
        #   l=0: H^(1) = O_ep (dim emb_flat); l>0: output of hidden layer l-1
        gate_dims = [emb_flat] + dnn_hidden
        self.ppnet = PPNet(prior_dim, emb_flat, gate_dims, n_tasks, gamma)

        # --- T task towers ---
        # Each tower: emb_flat → h1 → h2 → ... → 1 (one linear per layer)
        all_dims = [emb_flat] + dnn_hidden + [1]
        self.towers = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(all_dims[i], all_dims[i + 1])
                for i in range(len(all_dims) - 1)
            ])
            for _ in range(n_tasks)
        ])

    def forward(
        self,
        sparse_feats: list[Tensor],   # list of (B,) int
        dense_feats:  list[Tensor],   # list of (B, dim_i) float
        domain_id:    Tensor,         # (B,) int
        domain_stats: Tensor,         # (B, n_domain_stats) float
        user_id:      Tensor,         # (B,) int
        item_id:      Tensor,         # (B,) int
        author_id:    Tensor,         # (B,) int
    ) -> Tensor:                      # (B, n_tasks)
        # 1. Shared embedding layer
        tokens = (
            [emb(f).unsqueeze(1) for emb, f in zip(self.sparse_embs, sparse_feats)] +
            [proj(f).unsqueeze(1) for proj, f in zip(self.dense_projs, dense_feats)]
        )
        E = torch.cat(tokens, dim=1).flatten(1)                 # (B, emb_flat)

        # 2. Domain features for EPNet
        domain_emb = torch.cat([
            self.domain_id_emb(domain_id),
            self.domain_stats_proj(domain_stats),
        ], dim=-1)                                              # (B, 2*d)

        # 3. EPNet: personalize embedding per domain
        O_ep = self.epnet(domain_emb, E)                        # (B, emb_flat)

        # 4. Prior features for PPNet
        O_prior = torch.cat([
            self.user_emb(user_id),
            self.item_emb(item_id),
            self.author_emb(author_id),
        ], dim=-1)                                              # (B, 3*d)

        # 5. PPNet: generate per-layer, per-task gates
        pp_gates = self.ppnet(O_prior, O_ep)                    # list of (B, T, h_l)

        # 6. T task towers with PPNet-personalized parameters
        n_layers = len(self.towers[0])
        outputs = []
        for t, tower in enumerate(self.towers):
            h = O_ep                                            # H_t^(1) = O_ep
            for l, linear in enumerate(tower):
                gate_t = pp_gates[l][:, t, :]                  # (B, h_l)
                h = gate_t * h                                  # O_pp_t^(l)
                h = linear(h)                                   # H_t^(l+1)
                if l < n_layers - 1:
                    h = F.relu(h)
            outputs.append(h)                                   # (B, 1)

        return torch.cat(outputs, dim=-1)                       # (B, n_tasks)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B = 4

    sparse_vocab_sizes = [1000, 500, 200]
    dense_input_dims   = [8, 16]
    d_embed            = 16

    model = PEPNet(
        sparse_vocab_sizes = sparse_vocab_sizes,
        dense_input_dims   = dense_input_dims,
        d_embed            = d_embed,
        domain_vocab_size  = 10,
        n_domain_stats     = 4,
        user_vocab_size    = 1000,
        item_vocab_size    = 500,
        author_vocab_size  = 200,
        dnn_hidden         = [64, 32],
        n_tasks            = 3,
    )

    sparse_feats = [torch.randint(0, v, (B,)) for v in sparse_vocab_sizes]
    dense_feats  = [torch.randn(B, d) for d in dense_input_dims]
    domain_id    = torch.randint(0, 10, (B,))
    domain_stats = torch.randn(B, 4)
    user_id      = torch.randint(0, 1000, (B,))
    item_id      = torch.randint(0, 500, (B,))
    author_id    = torch.randint(0, 200, (B,))

    logits = model(sparse_feats, dense_feats, domain_id, domain_stats,
                   user_id, item_id, author_id)
    print(f"logits:   {logits.shape}  {logits.tolist()}")

    labels = torch.zeros_like(logits)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    print(f"loss:     {loss.item():.4f}")
    print("backward: OK")

    total = sum(p.numel() for p in model.parameters())
    print(f"params:   {total:,}")

    print(f"\nGate NU γ = {model.epnet.gate.gamma} → δ ∈ [0, {model.epnet.gate.gamma}]")
    print(f"PPNet gate dims: {model.ppnet.gate_dims}  ({model.n_tasks} tasks each)")
    print(f"Task towers: {model.n_tasks} towers × {len(model.towers[0])} layers each")
    print("\nEPNet: personalizes embedding per domain (resolves domain seesaw)")
    print("PPNet: personalizes DNN hidden units per task (resolves task seesaw)")


if __name__ == "__main__":
    _smoke_test()
