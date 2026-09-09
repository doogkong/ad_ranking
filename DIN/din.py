"""DIN: Deep Interest Network for Click-Through Rate Prediction.

Reference PyTorch implementation of DIN, from *"Deep Interest Network for
Click-Through Rate Prediction"* (Zhou et al., Alibaba, KDD 2018),
arXiv:1706.06978.

Prior Embedding&MLP CTR models compress a user's entire behavior history
into one fixed-length vector (via sum/average pooling) before it ever sees
the candidate ad — a bottleneck, since a user's interests are diverse and
only a small, ad-dependent subset of their history is actually relevant to
any one candidate. DIN's fix is architecturally small but effective: a
**local activation unit** computes an ad-conditioned relevance weight for
each historical behavior, and pools the behavior sequence with those
weights instead of uniformly — so the user representation varies per
candidate ad instead of staying fixed. Two training techniques make this
practical at industrial scale:

  Eq. 1     Base model pooling            -> sum_pooling, average_pooling
  Eq. 3     Local activation unit          -> LocalActivationUnit,
                                              activation_weighted_pooling
  Eq. 4-7   Mini-batch aware regularization -> mini_batch_aware_l2_penalty
  Eq. 8-9   Dice activation function        -> Dice
  Eq. 10-11 Evaluation (weighted AUC, RelaImpr) -> user_weighted_auc, rel_impr

`DINModel` assembles these into the full network of Fig. 2: it can run
either as the plain Embedding&MLP "base model" (uniform pooling) or as DIN
(local-activation-weighted pooling) via a single flag, matching how the
paper itself reports both as an ablation.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Eq. 8-9 — Dice: a data-adaptive generalization of PReLU
# ---------------------------------------------------------------------------

class Dice(nn.Module):
    """Dice activation (Eq. 8-9): generalizes PReLU by replacing its hard
    rectification point (fixed at 0) with a data-adaptive one, centered on
    the running mean of the input:

        f(s) = p(s)*s + (1 - p(s))*alpha*s,
        p(s) = sigmoid((s - E[s]) / sqrt(Var[s] + eps))

    E[s]/Var[s] are computed per mini-batch during training (like BatchNorm)
    and tracked via running averages for use at inference. When E[s]=Var[s]=0,
    Dice degenerates exactly into PReLU.

    Args:
        num_channels: size of the last dimension of the input.
        alpha_init: initial value for the learnable per-channel `alpha`.
        eps: numerical-stability constant (paper uses 1e-8).
        momentum: running-average momentum for E[s]/Var[s] at inference.
    """

    def __init__(self, num_channels: int, alpha_init: float = 0.25, eps: float = 1e-8, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.alpha = nn.Parameter(torch.full((num_channels,), alpha_init))
        self.register_buffer("running_mean", torch.zeros(num_channels))
        self.register_buffer("running_var", torch.ones(num_channels))

    def forward(self, s: Tensor) -> Tensor:
        if self.training:
            dims = tuple(range(s.dim() - 1))
            batch_mean = s.mean(dim=dims)
            batch_var = s.var(dim=dims, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * batch_mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * batch_var)
            mean, var = batch_mean, batch_var
        else:
            mean, var = self.running_mean, self.running_var

        p = torch.sigmoid((s - mean) / torch.sqrt(var + self.eps))
        return p * s + (1 - p) * self.alpha * s


def _make_activation(kind: str, num_channels: int) -> nn.Module:
    if kind == "dice":
        return Dice(num_channels)
    if kind == "prelu":
        return nn.PReLU(num_channels)
    raise ValueError(f"unknown activation kind: {kind!r}")


# ---------------------------------------------------------------------------
# Eq. 1 — Base-model pooling (uniform, ad-independent)
# ---------------------------------------------------------------------------

def sum_pooling(embeddings: Tensor, mask: Tensor) -> Tensor:
    """Sum-pools a padded behavior sequence over valid positions.
    embeddings: (B, H, D), mask: (B, H) bool. Returns (B, D)."""
    return (embeddings * mask.unsqueeze(-1).float()).sum(dim=1)


def average_pooling(embeddings: Tensor, mask: Tensor) -> Tensor:
    """Average-pools a padded behavior sequence over valid positions."""
    mask_f = mask.unsqueeze(-1).float()
    return (embeddings * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


# ---------------------------------------------------------------------------
# Eq. 3 — Local Activation Unit (DIN's core contribution)
# ---------------------------------------------------------------------------

class LocalActivationUnit(nn.Module):
    """a(e_j, v_A) (Eq. 3, Fig. 2 right): a small feed-forward network that
    scores the relevance of one historical behavior embedding `e_j` to a
    candidate ad embedding `v_A`. Its input is the concatenation of the two
    embeddings plus their element-wise ("out") product — an explicit
    interaction term the paper adds to help relevance modeling.

    Unlike softmax attention, weights are NOT normalized to sum to 1: the
    paper explicitly relaxes that constraint so that sum(weights) can serve
    as an approximation of the overall intensity of a user's activated
    interests, not just their relative ordering.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 36, activation: str = "dice") -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim * 3, hidden_dim)
        self.act = _make_activation(activation, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, behaviors: Tensor, ad: Tensor) -> Tensor:
        """
        Args:
            behaviors: (B, H, D) historical behavior embeddings e_1..e_H.
            ad: (B, D) candidate ad embedding v_A.
        Returns:
            (B, H) unnormalized activation weights.
        """
        ad_expanded = ad.unsqueeze(1).expand_as(behaviors)
        features = torch.cat([behaviors, ad_expanded, behaviors * ad_expanded], dim=-1)
        h = self.act(self.fc1(features))
        return self.fc2(h).squeeze(-1)


def activation_weighted_pooling(behaviors: Tensor, weights: Tensor, mask: Tensor) -> Tensor:
    """Eq. 3: v_U(A) = sum_j a(e_j, v_A) * e_j — the (unnormalized) weighted
    sum pooling that replaces Eq. 1's uniform pooling. Padded positions are
    masked out of both the weights and the sum.
    """
    masked_weights = weights.masked_fill(~mask, 0.0)
    return (behaviors * masked_weights.unsqueeze(-1)).sum(dim=1)


# ---------------------------------------------------------------------------
# Eq. 4-7 — Mini-batch aware regularization
# ---------------------------------------------------------------------------

def mini_batch_aware_l2_penalty(
    embedding_weight: Tensor,
    batch_feature_ids: Tensor,
    feature_occurrence_counts: Tensor,
    lam: float,
) -> Tensor:
    """Eq. 4-7: an approximate L2 regularizer over an embedding table that
    only touches the rows for features actually present in the current
    mini-batch, each scaled by 1/(number of times that feature occurs across
    the whole training set) — avoiding the prohibitive cost of computing the
    L2 norm over the full (potentially billion-row) embedding table on every
    step, while still regularizing every embedding roughly equally often
    relative to its overall frequency.

    Args:
        embedding_weight: (K, D) the full embedding table.
        batch_feature_ids: (N,) feature ids (into the table's first dim)
            appearing anywhere in the current mini-batch — duplicates are
            fine, only the unique ids' rows are penalized once each.
        feature_occurrence_counts: (K,) precomputed total occurrence count
            n_j for every feature id (must be > 0 for ids that appear).
        lam: regularization strength.
    Returns:
        scalar penalty to add to the mini-batch loss.
    """
    unique_ids = torch.unique(batch_feature_ids)
    rows = embedding_weight[unique_ids]
    counts = feature_occurrence_counts[unique_ids].clamp(min=1.0)
    return lam * (rows.pow(2).sum(dim=-1) / counts).sum()


# ---------------------------------------------------------------------------
# Full model (Fig. 2): Base Model (Embedding&MLP) and DIN
# ---------------------------------------------------------------------------

class MLPTower(nn.Module):
    """The fully-connected tower on top of the concatenated feature vector
    (Fig. 2's "Concat & Flatten" -> hidden layers -> output). Outputs a
    single logit; this is functionally equivalent to the paper's 2-way
    softmax + negative log-likelihood (Eq. 2) for a binary click label, and
    is trained the same way as the rest of this repo, via
    `binary_cross_entropy_with_logits`.
    """

    def __init__(self, in_dim: int, hidden_dims: List[int], activation: str = "dice") -> None:
        super().__init__()
        dims = [in_dim] + hidden_dims
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(_make_activation(activation, dims[i + 1]))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class DINModel(nn.Module):
    """The full network of Fig. 2. With `use_local_activation=False` this is
    the plain Embedding&MLP base model (uniform average pooling over
    behaviors, Eq. 1); with `use_local_activation=True` it is DIN (Eq. 3):
    only the behavior-pooling mechanism differs, every other structure is
    shared, matching how the paper itself frames DIN as a drop-in
    replacement for the base model's pooling layer.

    Args:
        profile_cardinalities: vocab size of each one-hot user-profile field
            (e.g. [2, 10] for a binary gender field + a 10-level age field).
        behavior_vocab_size: vocab size of the shared "goods" embedding table
            used for both historical behaviors and the candidate ad (Fig. 2:
            goods_id/shop_id/cate_id embeddings for one item are concatenated
            upstream into a single D-dim vector; this module treats that as
            already given per behavior/ad entry).
        context_cardinalities: vocab size of each one-hot context field.
        embed_dim: D, shared embedding width.
        mlp_hidden_dims: hidden layer widths of the top MLP tower.
        use_local_activation: True for DIN, False for the base model.
        activation: "dice" or "prelu", used throughout the activation unit
            and MLP tower.
    """

    def __init__(
        self,
        profile_cardinalities: List[int],
        behavior_vocab_size: int,
        context_cardinalities: List[int],
        embed_dim: int = 18,
        mlp_hidden_dims: Optional[List[int]] = None,
        use_local_activation: bool = True,
        activation: str = "dice",
    ) -> None:
        super().__init__()
        self.use_local_activation = use_local_activation

        self.profile_embeddings = nn.ModuleList([nn.Embedding(c, embed_dim) for c in profile_cardinalities])
        self.goods_embedding = nn.Embedding(behavior_vocab_size, embed_dim)  # shared: behaviors + candidate ad
        self.context_embeddings = nn.ModuleList([nn.Embedding(c, embed_dim) for c in context_cardinalities])

        if use_local_activation:
            self.activation_unit = LocalActivationUnit(embed_dim, activation=activation)

        concat_dim = embed_dim * (len(profile_cardinalities) + len(context_cardinalities) + 2)  # +2: pooled behavior, ad
        self.mlp = MLPTower(concat_dim, mlp_hidden_dims or [200, 80], activation)

    def forward(
        self,
        profile_ids: List[Tensor],
        behavior_ids: Tensor,
        behavior_mask: Tensor,
        ad_id: Tensor,
        context_ids: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            profile_ids: one (B,) LongTensor per profile field.
            behavior_ids: (B, H) LongTensor of historical goods ids (padded).
            behavior_mask: (B, H) bool, True at real (non-padding) positions.
            ad_id: (B,) LongTensor, the candidate ad's goods id.
            context_ids: one (B,) LongTensor per context field.
        Returns:
            logits: (B,) click-probability logits.
            activation_weights: (B, H) — the pooling weights actually used
                (uniform 1/H_valid if use_local_activation=False, else Eq. 3's
                unnormalized a(e_j, v_A)), exposed for inspection/visualization
                as in the paper's Fig. 5.
        """
        profile_emb = [emb(ids) for emb, ids in zip(self.profile_embeddings, profile_ids)]
        context_emb = [emb(ids) for emb, ids in zip(self.context_embeddings, context_ids)]
        behavior_emb = self.goods_embedding(behavior_ids)
        ad_emb = self.goods_embedding(ad_id)

        if self.use_local_activation:
            weights = self.activation_unit(behavior_emb, ad_emb)
            weights = weights.masked_fill(~behavior_mask, 0.0)
            pooled = (behavior_emb * weights.unsqueeze(-1)).sum(dim=1)
        else:
            mask_f = behavior_mask.unsqueeze(-1).float()
            pooled = (behavior_emb * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
            weights = mask_f.squeeze(-1) / mask_f.sum(dim=1).clamp(min=1.0)

        x = torch.cat(profile_emb + [pooled, ad_emb] + context_emb, dim=-1)
        logits = self.mlp(x)
        return logits, weights


# ---------------------------------------------------------------------------
# Eq. 10-11 — Evaluation: (user-weighted) AUC and RelaImpr
# ---------------------------------------------------------------------------

def pairwise_auc(scores: Tensor, labels: Tensor) -> float:
    """Binary AUC via pairwise concordance: P(score_pos > score_neg) +
    0.5 * P(score_pos == score_neg). Returns NaN if either class is absent.
    """
    labels = labels.bool()
    pos, neg = scores[labels], scores[~labels]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    return ((diff > 0).float().mean() + 0.5 * (diff == 0).float().mean()).item()


def user_weighted_auc(user_ids: Sequence, scores: Tensor, labels: Tensor) -> float:
    """Eq. 10: AUC = sum_i(#impression_i * AUC_i) / sum_i(#impression_i) —
    the impression-weighted average of each user's own (intra-user) AUC,
    which the paper adopts as more reflective of online ranking quality than
    a single pooled AUC across all users. Users for whom AUC is undefined
    (all-same-label impressions) are skipped, per standard practice.
    """
    groups: Dict[object, List[int]] = defaultdict(list)
    for idx, uid in enumerate(user_ids):
        groups[uid].append(idx)

    total_weight, weighted_sum = 0.0, 0.0
    for idx_list in groups.values():
        idx = torch.tensor(idx_list)
        auc_i = pairwise_auc(scores[idx], labels[idx])
        if math.isnan(auc_i):
            continue
        weight = len(idx_list)
        weighted_sum += weight * auc_i
        total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else float("nan")


def rel_impr(auc_model: float, auc_base: float) -> float:
    """Eq. 11: RelaImpr, relative improvement over a base model, normalized
    against the AUC of a random guesser (0.5). Returned as a fraction
    (multiply by 100 for a percentage).
    """
    return (auc_model - 0.5) / (auc_base - 0.5) - 1.0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B, H, D = 8, 20, 18

    model = DINModel(
        profile_cardinalities=[2, 10],  # gender, age_level
        behavior_vocab_size=1000,
        context_cardinalities=[10],  # e.g. pid
        embed_dim=D,
        mlp_hidden_dims=[64, 32],
        use_local_activation=True,
        activation="dice",
    )

    profile_ids = [torch.randint(0, 2, (B,)), torch.randint(0, 10, (B,))]
    behavior_ids = torch.randint(1, 1000, (B, H))
    lengths = torch.randint(5, H + 1, (B,))
    behavior_mask = torch.arange(H).unsqueeze(0) < lengths.unsqueeze(1)
    ad_id = torch.randint(0, 1000, (B,))
    context_ids = [torch.randint(0, 10, (B,))]
    labels = torch.randint(0, 2, (B,)).float()

    logits, weights = model(profile_ids, behavior_ids, behavior_mask, ad_id, context_ids)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    print(f"DIN logits: {logits.shape}, loss: {loss.item():.4f}, backward OK")
    print(f"activation weights (unnormalized) sum per example: {weights.sum(dim=1)[:4].tolist()}")

    print("\n--- DIN varies its user representation per candidate ad ---")
    model.eval()
    with torch.no_grad():
        ad_a = torch.full((B,), 5, dtype=torch.long)
        ad_b = torch.full((B,), 500, dtype=torch.long)
        _, w_a = model(profile_ids, behavior_ids, behavior_mask, ad_a, context_ids)
        _, w_b = model(profile_ids, behavior_ids, behavior_mask, ad_b, context_ids)
    print(f"weights differ across candidate ads: {not torch.allclose(w_a, w_b)}")

    print("\n--- Base model (uniform pooling) for comparison ---")
    base_model = DINModel(
        profile_cardinalities=[2, 10], behavior_vocab_size=1000, context_cardinalities=[10],
        embed_dim=D, mlp_hidden_dims=[64, 32], use_local_activation=False, activation="prelu",
    )
    base_logits, base_weights = base_model(profile_ids, behavior_ids, behavior_mask, ad_id, context_ids)
    print(f"base model logits: {base_logits.shape}, weights sum to 1 per example: {base_weights.sum(dim=1)[:4].tolist()}")

    print("\n--- Mini-batch aware regularization (Eq. 4-7) ---")
    counts = torch.randint(1, 100, (1000,)).float()
    penalty = mini_batch_aware_l2_penalty(model.goods_embedding.weight, behavior_ids.reshape(-1), counts, lam=0.01)
    print(f"mini-batch aware L2 penalty: {penalty.item():.4f}")

    print("\n--- Evaluation metrics (Eq. 10-11) ---")
    user_ids = [i % 3 for i in range(20)]
    scores = torch.rand(20)
    eval_labels = torch.randint(0, 2, (20,))
    auc = user_weighted_auc(user_ids, scores, eval_labels)
    print(f"user-weighted AUC: {auc:.4f}, RelaImpr vs base_model=0.6: {rel_impr(auc, 0.6):.4f}")


if __name__ == "__main__":
    _smoke_test()
