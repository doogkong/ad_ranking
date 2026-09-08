"""GR2: Generative Reasoning Re-Ranker.

Reference PyTorch implementation of the algorithmic components of GR2, from
the *"GR2 Technical Report"* (Meta AI, Jul 2026), arXiv:2606.31984.

Unlike RankMixer/HSTU, GR2 is not a single encoder architecture — it is a
four-stage *training recipe* for turning a pretrained LLM into an industrial
re-ranker: (1) mid-train on tokenized Semantic IDs, (2) activate reasoning via
distillation from a stronger teacher, (3) sharpen ranking behavior with RL on
verifiable rewards, (4) shrink serving cost. This module implements the parts
of that recipe that are pure algorithm — independent of which base LLM is
used — so they can be exercised and tested without a real multi-billion
parameter model:

  Sec 2.1  Semantic-ID tokenization      -> RQKMeansTokenizer
  Sec 3.2  Targeted / rejection sampling -> build_targeted_prompt,
                                             build_rejection_prompt, rejection_sample
  Sec 3.3  SFT loss (Eq. 3)              -> sft_loss
  Sec 3.4  On-Policy Distillation (Eq.4) -> opd_loss
  Sec 4.1  Ranking rewards (Eq. 5-6)     -> auc_reward, ndcg_reward
  Sec 4.2  De-hacked reward (Eq. 7)      -> conditional_reward
  Sec 4.3  DAPO objective (Eq. 8-9)      -> group_relative_advantage,
                                             dynamic_sampling_filter, dapo_loss
  Sec 5.1  Context-compressor reward     -> compressor_reward
           (Eq. 10-11)

The paper's central lesson, reflected throughout this module, is that in
re-ranking RL the *reward function itself* is most of the engineering effort:
naive rewards let a policy "win" by preserving the incoming order or
exploiting position bias rather than by actually reasoning about items, so
several of these functions exist specifically to close a reward-hacking path
the authors identified (see `conditional_reward`).
"""

import math
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Sec 2.1 — Semantic-ID Tokenization
# ---------------------------------------------------------------------------

class RQKMeansTokenizer:
    """Residual multi-level K-means quantizer producing Semantic IDs (SIDs).

    Tokenizer(x) = (z_1, ..., z_K) in {1..C}^K: each item embedding is
    quantized against a codebook, the quantization residual is passed to the
    next level, and the process repeats for K levels. The paper's production
    tokenizer core is an RQ-VAE; this is the simpler, gradient-free RQ-KMeans
    alternative (fit via Lloyd's algorithm per level), sufficient to
    reproduce the paper's central tokenizer-quality metric: SID uniqueness
    across the catalog (the paper targets >=99%).
    """

    def __init__(self, num_levels: int, codebook_size: int, dim: int, n_iter: int = 25, seed: int = 0) -> None:
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.dim = dim
        self.n_iter = n_iter
        self.seed = seed
        self.codebooks: List[Tensor] = []

    @staticmethod
    def _nearest(x: Tensor, centroids: Tensor) -> Tensor:
        return torch.cdist(x, centroids).argmin(dim=1)

    def _kmeans(self, x: Tensor, generator: torch.Generator) -> Tensor:
        n = x.shape[0]
        init_idx = torch.randperm(n, generator=generator)[: self.codebook_size]
        centroids = x[init_idx].clone()
        for _ in range(self.n_iter):
            assignments = self._nearest(x, centroids)
            for c in range(self.codebook_size):
                mask = assignments == c
                if mask.any():
                    centroids[c] = x[mask].mean(dim=0)
        return centroids

    def fit(self, embeddings: Tensor) -> "RQKMeansTokenizer":
        """embeddings: (N, dim) item embeddings for the whole catalog."""
        generator = torch.Generator().manual_seed(self.seed)
        residual = embeddings.clone()
        self.codebooks = []
        for _ in range(self.num_levels):
            centroids = self._kmeans(residual, generator)
            self.codebooks.append(centroids)
            codes = self._nearest(residual, centroids)
            residual = residual - centroids[codes]
        return self

    def encode(self, embeddings: Tensor) -> Tensor:
        """Returns (N, num_levels) integer Semantic IDs."""
        if not self.codebooks:
            raise RuntimeError("Call fit() before encode().")
        residual = embeddings.clone()
        codes = []
        for centroids in self.codebooks:
            c = self._nearest(residual, centroids)
            codes.append(c)
            residual = residual - centroids[c]
        return torch.stack(codes, dim=1)

    @staticmethod
    def uniqueness(codes: Tensor) -> float:
        """Fraction of rows whose full SID tuple is unique across the batch."""
        rows = [tuple(row.tolist()) for row in codes]
        return len(set(rows)) / len(rows)


# ---------------------------------------------------------------------------
# Sec 3.2 — Reasoning-trace generation: targeted & rejection sampling
# ---------------------------------------------------------------------------

def build_targeted_prompt(history: List, candidates: List, target) -> dict:
    """Eq. 1: P_targeted(x, y, z) — asks a teacher to explain why the
    ground-truth target `z` would be the user's next interaction, given
    history `x` and candidate list `y`. The target is revealed to the
    teacher, so traces built this way are always "correct" but may encode
    post-hoc, label-aware shortcuts rather than causal reasoning.
    """
    return {
        "history": list(history),
        "candidates": list(candidates),
        "target": target,
        "instruction": (
            "Given the user's history and the candidate list, explain step by "
            "step why the target item would be the user's next interaction."
        ),
    }


def build_rejection_prompt(history: List, candidates: List) -> dict:
    """Eq. 2: P_rejection(x, y) — asks a teacher which candidate the user is
    most likely to interact with next, WITHOUT revealing the ground truth.
    """
    return {
        "history": list(history),
        "candidates": list(candidates),
        "instruction": (
            "Given the user's history, reason step by step about which "
            "candidate the user is most likely to interact with next."
        ),
    }


def rejection_sample(
    history: List,
    candidates: List,
    target_index: int,
    teacher_fn: Callable[[dict], Tuple[str, int]],
    max_attempts: int = 8,
) -> Tuple[str, int, bool]:
    """Eq. 2: repeatedly query `teacher_fn` on the (target-blind) rejection
    prompt until its predicted index matches the ground-truth target, or
    `max_attempts` is exhausted — in which case the example is discarded, as
    in the paper (this silently wastes teacher compute and drops precisely
    the hardest examples, which is one of the two failure modes On-Policy
    Distillation, `opd_loss` below, is designed to avoid).

    Args:
        teacher_fn: callable(prompt) -> (reasoning_trace, predicted_index).
    Returns:
        (trace, predicted_index, accepted).
    """
    prompt = build_rejection_prompt(history, candidates)
    trace, predicted_index = "", -1
    for _ in range(max_attempts):
        trace, predicted_index = teacher_fn(prompt)
        if predicted_index == target_index:
            return trace, predicted_index, True
    return trace, predicted_index, False


# ---------------------------------------------------------------------------
# Sec 3.3 — Supervised fine-tuning loss (Eq. 3)
# ---------------------------------------------------------------------------

def sft_loss(
    reasoning_logprobs: Tensor,
    ranking_logprobs: Tensor,
    reasoning_mask: Tensor,
    ranking_mask: Tensor,
    lambda_r: float,
    lambda_o: float,
) -> Tensor:
    """Eq. 3: L_SFT = -lambda_r * sum(log P(reasoning tokens))
                       -lambda_o * sum(log P(ranking tokens))

    Reasoning and ranking segments are weighted separately (lambda_r <
    lambda_o in the paper) so the model is pushed harder to get the final
    ranked list right than to phrase the rationale a particular way.

    Args:
        reasoning_logprobs, ranking_logprobs: (B, T) per-token log-probs.
        reasoning_mask, ranking_mask: (B, T) bool, True at valid (non-pad) positions.
    Returns:
        scalar loss, averaged over the batch.
    """
    r_term = -(reasoning_logprobs * reasoning_mask).sum(dim=-1)
    o_term = -(ranking_logprobs * ranking_mask).sum(dim=-1)
    return (lambda_r * r_term + lambda_o * o_term).mean()


# ---------------------------------------------------------------------------
# Sec 3.4 — On-Policy Distillation loss (Eq. 4)
# ---------------------------------------------------------------------------

def opd_loss(
    logp_student: Tensor,
    logp_student_old: Tensor,
    logp_teacher: Tensor,
    advantages: Tensor,
    mask: Tensor,
    eps_lo: float = 0.2,
    eps_hi: float = 0.2,
    beta: float = 0.1,
) -> Tensor:
    """Eq. 4: a GRPO-style clipped surrogate on the student's own rollouts,
    plus a per-token reverse-KL anchor pulling the student toward a frozen
    teacher. The student samples the trajectories (on-policy); the teacher
    only ever contributes token log-probabilities, never labels — regularizing
    the student's distribution rather than supervising a fixed target
    sequence. This is what lets OPD avoid both failure modes of trace-based
    SFT: it never trains on states the student won't visit at deployment, and
    every prompt contributes gradient regardless of whether the teacher could
    solve it.

    The KL term is estimated from the single sampled token via the standard
    unbiased "k3" estimator (Schulman, *Approximating KL Divergence*, 2020):
    for x sampled from the student, KL(student || teacher) is estimated as
    exp(logratio) - logratio - 1, where logratio = logp_teacher - logp_student.

    Args:
        logp_student: (G, T) current policy's log-prob of the sampled tokens.
        logp_student_old: (G, T) rollout-time policy's log-prob of the same tokens.
        logp_teacher: (G, T) teacher's log-prob of the same tokens.
        advantages: (G,) or (G, T) group-relative advantage (see `group_relative_advantage`).
        mask: (G, T) bool, True at valid (non-pad) token positions.
        eps_lo, eps_hi: PPO-style clip range.
        beta: distillation strength (KL weight).
    Returns:
        scalar loss.
    """
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)

    ratios = torch.exp(logp_student - logp_student_old)
    surrogate = torch.min(
        ratios * advantages,
        torch.clamp(ratios, 1 - eps_lo, 1 + eps_hi) * advantages,
    )
    denom = mask.sum().clamp(min=1)
    pg_loss = -(surrogate * mask).sum() / denom

    logratio = logp_teacher - logp_student
    kl_hat = torch.exp(logratio) - logratio - 1
    kl_term = (kl_hat * mask).sum() / denom

    return pg_loss + beta * kl_term


# ---------------------------------------------------------------------------
# Sec 4.1 — Ranking rewards (Eq. 5-6)
# ---------------------------------------------------------------------------

def auc_reward(ranks: Tensor, labels: Tensor) -> float:
    """Eq. 5: per-impression AUC of a predicted permutation against binary
    engagement labels. AUC handles multi-positive slates natively (unlike a
    single-target rank-delta), is bounded in [0, 1], and is invariant to
    relabeling within each class, giving a stable RL signal.

    Args:
        ranks: (K,) rank position assigned to each item by the permutation
            (1 = top). labels: (K,) in {0, 1}.
    Returns:
        R_AUC in [0, 1].
    Raises:
        ValueError if the slate has no positives or no negatives (per the
        paper, such slates are filtered out at data-loading, not scored).
    """
    labels = labels.bool()
    pos_ranks = ranks[labels]
    neg_ranks = ranks[~labels]
    if pos_ranks.numel() == 0 or neg_ranks.numel() == 0:
        raise ValueError("auc_reward requires at least one positive and one negative label.")
    concordant = (pos_ranks.unsqueeze(1) < neg_ranks.unsqueeze(0)).float()
    return concordant.mean().item()


def ndcg_reward(ranks: Tensor, grades: Tensor) -> float:
    """Eq. 6: NDCG@K with graded relevance (e.g. {0: none, 1: click, 2: click+conversion}).

    Args:
        ranks: (K,) rank position assigned to each item (1 = top).
        grades: (K,) non-negative integer/float relevance grades.
    Returns:
        R_NDCG in [0, 1]; 0.0 if every grade is 0 (no signal to rank against).
    """
    discounts = 1.0 / torch.log2(ranks.float() + 1)
    dcg = ((2.0 ** grades - 1) * discounts).sum()

    ideal_ranks = torch.arange(1, grades.numel() + 1, dtype=torch.float32)
    ideal_discounts = 1.0 / torch.log2(ideal_ranks + 1)
    ideal_grades, _ = torch.sort(grades.float(), descending=True)
    ideal_dcg = ((2.0 ** ideal_grades - 1) * ideal_discounts).sum()

    if ideal_dcg.item() == 0.0:
        return 0.0
    return (dcg / ideal_dcg).item()


# ---------------------------------------------------------------------------
# Sec 4.2 — Conditional (de-hacked) reward (Eq. 7)
# ---------------------------------------------------------------------------

def conditional_reward(
    ranks: Tensor,
    labels: Tensor,
    r_fmt: float,
    alpha: float,
    valid_format: bool,
) -> float:
    """Eq. 7: gates the ranking reward against two reward-hacking paths the
    paper identifies:

    1. An invalid output could still earn a non-trivial ranking reward from a
       partial parse, so R_rank is zeroed whenever `valid_format` is False.
    2. If the model just re-emits the input order verbatim (the "identity
       permutation") *and* that input order was not already optimal, the
       policy is harvesting format reward while dodging the re-ranking task
       entirely — so R_rank is zeroed in that specific case too. If the
       input order happens to already be optimal, emitting it back is the
       correct answer and is rewarded normally.

    Args:
        ranks: (K,) predicted rank position per item (1 = top).
        labels: (K,) binary engagement labels.
        r_fmt: the (already-computed) format reward Omega(o).
        alpha: format-reward weight.
        valid_format: whether the output was parseable into a valid permutation.
    Returns:
        R = R_rank + alpha * R_fmt, or alpha * R_fmt alone if either
        reward-hacking condition is detected.
    """
    if not valid_format:
        return alpha * r_fmt

    k = ranks.numel()
    identity = torch.arange(1, k + 1, dtype=ranks.dtype, device=ranks.device)
    is_identity = torch.equal(ranks, identity)

    if is_identity:
        identity_is_optimal = auc_reward(identity, labels) == 1.0
        if not identity_is_optimal:
            return alpha * r_fmt

    r_rank = auc_reward(ranks, labels)
    return r_rank + alpha * r_fmt


# ---------------------------------------------------------------------------
# Sec 4.3 — DAPO training objective (Eq. 8-9)
# ---------------------------------------------------------------------------

def group_relative_advantage(rewards: Tensor) -> Tensor:
    """Eq. 9: Â_i = (R_i - mean(R)) / std(R) over a group of G rollouts for
    the same prompt. Returns zeros if the group has zero variance (handled
    separately by `dynamic_sampling_filter`, which should drop such groups
    before they reach the loss — a zero-variance group carries no learning
    signal either way).
    """
    std = rewards.std(unbiased=False)
    if std.item() == 0.0:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / std


def dynamic_sampling_filter(rewards: Tensor, eps: float = 1e-8) -> Tensor:
    """DAPO's dynamic sampling: drop prompts whose group of G rollouts are
    all correct or all incorrect (reward std ~ 0), since such groups yield a
    zero policy gradient — "oversamples and filters out prompts with the
    accuracy equal to 1 and 0" so the training batch is spent on prompts that
    actually carry a learning signal.

    Args:
        rewards: (num_prompts, G).
    Returns:
        (num_prompts,) bool mask, True = keep.
    """
    return rewards.std(dim=-1, unbiased=False) > eps


def dapo_loss(ratios: Tensor, advantages: Tensor, mask: Tensor, eps_lo: float = 0.2, eps_hi: float = 0.28) -> Tensor:
    """Eq. 8: the DAPO objective (decoupled clip, dynamic sampling). Unlike
    GRPO's per-sequence-then-average normalization, DAPO normalizes the
    clipped surrogate by the TOTAL number of valid tokens across the whole
    group (the "decoupled" token-level normalization that fixes GRPO's
    rollout-length bias, where longer sequences would otherwise be
    under-weighted).

    Args:
        ratios: (G, T) per-token importance ratio pi_theta / pi_theta_old.
        advantages: (G,) or (G, T) group-relative advantage.
        mask: (G, T) bool, True at valid (non-pad) token positions.
        eps_lo, eps_hi: asymmetric clip range (DAPO's "Clip-Higher").
    Returns:
        scalar loss (negative of the objective to maximize).
    """
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(-1)
    surrogate = torch.min(
        ratios * advantages,
        torch.clamp(ratios, 1 - eps_lo, 1 + eps_hi) * advantages,
    )
    denom = mask.sum().clamp(min=1)
    return -(surrogate * mask).sum() / denom


# ---------------------------------------------------------------------------
# Sec 5.1 — Context-compressor reward (Eq. 10-11)
# ---------------------------------------------------------------------------

def compressor_reward(
    solvable: bool,
    info_preservation: float,
    ranking_quality: float,
    original_len: int,
    compressed_len: int,
    ellipsis_penalty: float = 1.0,
    alpha: float = 0.5,
) -> float:
    """Eq. 10-11: reward for a context compressor, trained with an
    LLM-as-a-judge that scores compressed input along solvability (can the
    ranking task still be solved from it?), information preservation, and
    ranking quality.

    When solvable, ranking quality is weighted far more than raw information
    preservation (0.8 vs 0.2) — the paper finds preservation correlates only
    weakly with downstream ranking, while more aggressive compression can
    actually suppress noise and help.

    Args:
        solvable: judge's solvability verdict s in {0,1}.
        info_preservation, ranking_quality: judge scores in [1, 10].
        original_len, compressed_len: token counts, for the compression-ratio term.
        ellipsis_penalty: in [0, 1], penalizes truncation shortcuts (e.g. "...").
        alpha: trade-off between compression ratio and the judge term.
    Returns:
        scalar reward.
    """
    p_bar = info_preservation / 10.0
    q_bar = ranking_quality / 10.0
    r_judge = (0.2 * p_bar + 0.8 * q_bar) if solvable else (0.8 * p_bar + 0.2 * q_bar)
    r_comp = max(0.0, 1.0 - compressed_len / original_len)
    return (alpha * r_comp + (1 - alpha) * r_judge) * ellipsis_penalty


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)

    print("--- Semantic-ID tokenizer (Sec 2.1) ---")
    catalog = torch.randn(500, 16)
    tokenizer = RQKMeansTokenizer(num_levels=3, codebook_size=32, dim=16, n_iter=10)
    tokenizer.fit(catalog)
    codes = tokenizer.encode(catalog)
    print(f"codes shape: {codes.shape}, uniqueness: {tokenizer.uniqueness(codes):.4f}")

    print("\n--- Rejection sampling (Sec 3.2) ---")
    history, candidates, target_index = ["itemA", "itemB"], ["itemC", "itemD", "itemE"], 1
    calls = {"n": 0}

    def stub_teacher(prompt):
        calls["n"] += 1
        predicted = target_index if calls["n"] >= 3 else 0
        return f"reasoning attempt {calls['n']}", predicted

    trace, predicted_index, accepted = rejection_sample(history, candidates, target_index, stub_teacher)
    print(f"accepted={accepted} after {calls['n']} attempts, predicted_index={predicted_index}")

    print("\n--- SFT / OPD losses (Sec 3.3-3.4) ---")
    B, T, G = 4, 10, 4
    reasoning_lp = torch.randn(B, T, requires_grad=True)
    ranking_lp = torch.randn(B, T, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.bool)
    loss_sft = sft_loss(reasoning_lp, ranking_lp, mask, mask, lambda_r=0.5, lambda_o=1.0)
    loss_sft.backward()
    print(f"sft_loss: {loss_sft.item():.4f}, backward OK")

    logp_student = torch.randn(G, T, requires_grad=True)
    logp_student_old = logp_student.detach() + 0.01 * torch.randn(G, T)
    logp_teacher = torch.randn(G, T)
    rewards = torch.tensor([0.9, 0.2, 0.9, 0.1])
    advantages = group_relative_advantage(rewards)
    loss_opd = opd_loss(logp_student, logp_student_old, logp_teacher, advantages, mask[:G])
    loss_opd.backward()
    print(f"opd_loss:  {loss_opd.item():.4f}, backward OK")

    print("\n--- Ranking rewards + de-hacked reward (Sec 4.1-4.2) ---")
    ranks = torch.tensor([1, 2, 3, 4, 5])
    labels = torch.tensor([1, 0, 1, 0, 0])
    grades = torch.tensor([2, 0, 1, 0, 0])
    print(f"auc_reward:  {auc_reward(ranks, labels):.4f}")
    print(f"ndcg_reward: {ndcg_reward(ranks, grades):.4f}")

    identity = torch.arange(1, 6)
    r = conditional_reward(identity, labels, r_fmt=1.0, alpha=0.1, valid_format=True)
    print(f"conditional_reward (identity, suboptimal input order): {r:.4f}  (should equal alpha*r_fmt = 0.1)")

    print("\n--- DAPO objective (Sec 4.3) ---")
    prompt_rewards = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.9, 0.2, 0.9, 0.1], [0.0, 0.0, 0.0, 0.0]])
    keep = dynamic_sampling_filter(prompt_rewards)
    print(f"dynamic_sampling_filter keep-mask: {keep.tolist()}  (drops all-1 and all-0 groups)")
    ratios = torch.exp(torch.randn(G, T) * 0.05)
    loss_dapo = dapo_loss(ratios, advantages, mask[:G])
    print(f"dapo_loss: {loss_dapo.item():.4f}")

    print("\n--- Context-compressor reward (Sec 5.1) ---")
    r = compressor_reward(solvable=True, info_preservation=6, ranking_quality=9, original_len=1000, compressed_len=180)
    print(f"compressor_reward: {r:.4f}")


if __name__ == "__main__":
    _smoke_test()
