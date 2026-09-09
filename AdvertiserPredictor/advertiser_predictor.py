"""AdvertiserPredictor: Fine-Tuned LLM as a Complementary Predictor Improving Ads System.

Reference implementation of the algorithmic components of the system
described in *"Fine-Tuned LLM as a Complementary Predictor Improving Ads
System"* (Pinterest, May 2026), arXiv:2605.27856.

Like GR2, LoopFM and ExFM, this paper's contribution is a *training and
serving recipe*, not a single encoder architecture: a fine-tuned open-source
LLM is used not as a ranker, but as an ads-specific ancillary predictor that
forecasts likely next advertisers (and user interests) from a structured
summary of a user's profile and behavior, whose outputs then feed both
retrieval (as an additional candidate generator) and ranking (as features
for downstream conversion models). This module implements the pipeline's
concretely algorithmic pieces:

  Sec 3.2  Data pipeline               -> select_active_users,
                                           construct_next_advertiser_label,
                                           select_users_for_incremental_inference
  Sec 3.3  Stage-specific prompting    -> build_advertiser_prediction_prompt
  Sec 3.4  GRPO reward (Eq. 1-5)       -> rank_match_reward, length_penalty,
                                           grpo_reward
  Sec 3.5  Semantic-ID two-phase       -> freeze_all_but_sid_embeddings,
           pretraining                   unfreeze_all
  Sec 3.6  Downstream integration      -> blend_candidates,
                                           advertiser_concentration
  Sec 4    Evaluation (Recall@K)       -> rank_of_advertiser, recall_at_k,
                                           mean_recall_at_k
  --       Structured output parsing   -> parse_structured_output

Prompt templates (`build_advertiser_prediction_prompt`) are an original,
functionally-equivalent re-implementation of the paper's stage-specific
prompt *structure* (role instruction, structured user context, quantity and
output-format requirements) — not a reproduction of the paper's own prompt
wording (Appendix A). GRPO/RL training itself, the actual LLM fine-tuning,
and the vLLM+Ray serving stack (prefix caching, paged-attention KV cache,
continuous batching) are out of scope: this module is the LLM-agnostic
scaffolding around them (reward, parsing, data selection, evaluation).
"""

import re
from collections import Counter
from datetime import timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sec 3.2 — Data pipeline
# ---------------------------------------------------------------------------

def select_active_users(users: List[dict], as_of_date, lookback_days: int = 90) -> List[dict]:
    """Sec 3.2.1 (User Selection): restricts daily inference to users with a
    conversion within the last `lookback_days` — the paper's serving
    population is active users with a past conversion in the previous 90
    days, keeping inference focused on users with meaningful commercial
    intent while bounding daily inference volume.

    Args:
        users: dicts with at least a "last_conversion_date" field.
        as_of_date: the reference date inference is being run for.
    """
    cutoff = as_of_date - timedelta(days=lookback_days)
    return [u for u in users if u["last_conversion_date"] >= cutoff]


def construct_next_advertiser_label(
    conversions: Sequence[Tuple[object, object]], window_start, window_end
) -> Optional[object]:
    """Sec 3.2.2 (Label Construction): the label is the FIRST advertiser a
    user converts with inside the prediction window — a next-advertiser
    formulation, not a multi-label one, chosen to match how the prediction
    is actually consumed downstream (a single set of advertiser priors).

    Args:
        conversions: (timestamp, advertiser_id) pairs, any order.
        window_start, window_end: inclusive window bounds.
    Returns:
        The first (by timestamp) advertiser_id inside the window, or None.
    """
    in_window = sorted((t, a) for t, a in conversions if window_start <= t <= window_end)
    return in_window[0][1] if in_window else None


def select_users_for_incremental_inference(eligible_users: Sequence, users_with_new_activity: Sequence) -> List:
    """Sec 3.4.2 (tail): only users with newly observed activity since the
    last inference pass are re-inferred, substantially cutting daily
    inference volume relative to re-scoring every eligible user.
    """
    new_activity_set = set(users_with_new_activity)
    return [u for u in eligible_users if u in new_activity_set]


# ---------------------------------------------------------------------------
# Sec 3.3 — Stage-specific prompt design
# ---------------------------------------------------------------------------

def build_advertiser_prediction_prompt(
    profile: dict,
    behavior: Dict[str, Sequence],
    active_advertisers: Sequence[str],
    preset_advertiser_pool: Sequence[str],
    stage: str = "sft",
    num_advertisers: int = 1,
    num_interests: int = 0,
    sid_sequences: Optional[Sequence] = None,
) -> str:
    """Builds a structured prompt over a user's profile and behavior,
    following the paper's stage-specific design (Table 1, Sec 3.3): SFT asks
    for a single free-text advertiser; GRPO/inference ask for a ranked list
    of `num_advertisers` advertisers plus up to `num_interests` interests in
    a structured, deterministically-parseable block. Optionally includes
    Semantic ID sequences (Sec 3.5) alongside the textual context.

    This is an original, functionally-equivalent template — not the paper's
    own prompt wording (see Appendix A there for the exact text used).

    Args:
        profile: e.g. {"age":..., "gender":..., "state":...}.
        behavior: named behavior-sequence summaries, e.g.
            {"onsite_searches": [...], "offsite_urls": [...], ...}.
        active_advertisers: advertisers the user has past conversions with.
        preset_advertiser_pool: fallback high-revenue advertiser pool.
        stage: "sft", "grpo", or "inference" (grpo and inference share format).
        num_advertisers: how many ranked advertisers to request (1 for SFT;
            20 in the paper's GRPO/inference stages, to widen reward variance).
        num_interests: how many interests to request (0 to omit entirely).
        sid_sequences: optional recent Semantic ID sequences (Sec 3.5).
    """
    lines = [
        "You are an ads-matching assistant for a commerce platform.",
        f"Task: identify the {num_advertisers} advertiser(s) the user is most likely to convert with next"
        + (f", and up to {num_interests} of the user's interests." if num_interests else "."),
        "",
        "User profile:",
        f"  age={profile.get('age')}, gender={profile.get('gender')}, state={profile.get('state')}",
        "Behavior summary:",
    ]
    for key, values in behavior.items():
        lines.append(f"  {key}: {list(values)}")
    if sid_sequences:
        lines.append(f"Recent Semantic ID sequences: {list(sid_sequences)}")
    lines += [
        f"Active advertisers with past conversions: {list(active_advertisers)}",
        f"Preset advertiser pool: {list(preset_advertiser_pool)}",
        "",
    ]

    if stage == "sft":
        lines.append("Respond with exactly one advertiser name, chosen from the lists above where possible.")
    else:
        lines.append(
            f"Respond with EXACTLY {num_advertisers} ranked advertiser names (most likely first)"
            + (f" and up to {num_interests} interests" if num_interests else "")
            + ", as the following structured block:"
        )
        interests_part = "<interests>[i1|i2|...]</interests>" if num_interests else ""
        lines.append(f"<answer>{interests_part}<advertiser_names>a1|a2|...</advertiser_names></answer>")

    return "\n".join(lines)


def parse_structured_output(text: str) -> Dict[str, List[str]]:
    """Extracts the pipe-delimited advertiser/interest lists from the model's
    structured output block (e.g. `<advertiser_names>A|B|C</advertiser_names>`),
    enabling deterministic downstream parsing and scoring.
    """
    result: Dict[str, List[str]] = {"advertisers": [], "interests": []}
    for tag, key in [("advertiser_names", "advertisers"), ("interests", "interests")]:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        if match:
            raw = match.group(1).strip().strip("[]")
            result[key] = [item.strip() for item in raw.split("|") if item.strip()]
    return result


# ---------------------------------------------------------------------------
# Sec 3.4 — GRPO reward (Eq. 1-5)
# ---------------------------------------------------------------------------

def rank_match_reward(rank: Optional[int]) -> float:
    """Eq. 2-4: R_match(i) = R_base(i) + R_bonus(i), where i is the
    1-indexed rank of the ground-truth advertiser within the predicted list
    (rank 1 = top). If the advertiser is absent from the list (`rank=None`),
    the match reward is 0.

        R_base(i)  = 0.1 * (20 - i)
        R_bonus(i) = 2.0 if i <= 4 else 0.0
    """
    if rank is None:
        return 0.0
    base = 0.1 * (20 - rank)
    bonus = 2.0 if rank <= 4 else 0.0
    return base + bonus


def length_penalty(n: int, n_target: int) -> float:
    """Eq. 5: P_len(n, n*). Zero exactly at the target count; otherwise a
    penalty growing (capped at 1.0) with the distance from the target, plus
    a flat +1.0 — discouraging both over- and under-generating relative to
    the requested advertiser/interest count.
    """
    if n == n_target:
        return 0.0
    return min(0.1 * abs(n - n_target), 1.0) + 1.0


def grpo_reward(
    rank: Optional[int],
    num_advertisers_predicted: int,
    num_interests_predicted: int,
    advertiser_target: int = 20,
    interest_target: int = 5,
) -> float:
    """Eq. 1: R_total = R_match - P_adv_len - P_interest_len — rewards
    ranking the ground-truth advertiser highly while penalizing deviation
    from the requested advertiser/interest counts.
    """
    r_match = rank_match_reward(rank)
    p_adv = length_penalty(num_advertisers_predicted, advertiser_target)
    p_interest = length_penalty(num_interests_predicted, interest_target)
    return r_match - p_adv - p_interest


# ---------------------------------------------------------------------------
# Sec 3.5 — Semantic-ID two-phase pretraining
# ---------------------------------------------------------------------------

def freeze_all_but_sid_embeddings(model: nn.Module, embedding: nn.Embedding, sid_token_ids: Sequence[int]):
    """Phase 1 (Sec 3.5): a preliminary alignment step that updates ONLY the
    embedding rows for `sid_token_ids`, freezing every other parameter
    (including the rest of the embedding table) — installing a rough mapping
    between the new Semantic ID tokens and the model's existing text-token
    space before touching anything else.

    Returns a hook handle; call `.remove()` on it (or `unfreeze_all`) before
    Phase 2's full-parameter pretraining.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    embedding.weight.requires_grad_(True)

    mask = torch.zeros(embedding.num_embeddings, 1)
    mask[list(sid_token_ids)] = 1.0

    def _mask_grad(grad):
        return grad * mask.to(grad.device)

    return embedding.weight.register_hook(_mask_grad)


def unfreeze_all(model: nn.Module) -> None:
    """Phase 2 (Sec 3.5): unfreezes every parameter for full-parameter
    pretraining that incorporates recommendation (SID) knowledge into the
    whole model, mixed with general-domain data to mitigate catastrophic
    forgetting.
    """
    for p in model.parameters():
        p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Sec 3.6 — Downstream integration: candidate blending
# ---------------------------------------------------------------------------

def blend_candidates(main_candidates: Sequence[str], llm_candidates: Sequence[str], llm_quota: int) -> List[str]:
    """Sec 3.6.1: blends the LLM-based Candidate Generator's output into the
    main retrieval candidate list. Takes the LLM-CG's top `llm_quota`
    candidates and appends any not already present in `main_candidates`
    (which keeps its original order and priority), producing a deduplicated
    combined list.

    The paper warns `llm_quota` must be tuned carefully: because the LLM-CG
    targets high-conversion-intent advertisers, too large a quota lets it
    dominate the blended, de-duplicated list — see `advertiser_concentration`.
    """
    combined = list(main_candidates)
    seen = set(main_candidates)
    for cand in llm_candidates[:llm_quota]:
        if cand not in seen:
            combined.append(cand)
            seen.add(cand)
    return combined


def advertiser_concentration(candidates: Sequence[str]) -> float:
    """The share of `candidates` occupied by its single most-frequent entry —
    a simple proxy for the over-concentration risk (Sec 4.5.2) of a too-large
    LLM-CG quota dominating the blended candidate list and hurting advertiser
    diversity after de-duplication.
    """
    if not candidates:
        return 0.0
    counts = Counter(candidates)
    return max(counts.values()) / len(candidates)


# ---------------------------------------------------------------------------
# Sec 4 — Evaluation: Recall@K
# ---------------------------------------------------------------------------

def rank_of_advertiser(predicted: Sequence, ground_truth) -> Optional[int]:
    """1-indexed rank of `ground_truth` within `predicted`, or None if absent."""
    for i, p in enumerate(predicted, start=1):
        if p == ground_truth:
            return i
    return None


def recall_at_k(predicted: Sequence, ground_truth, k: int) -> float:
    """1.0 if the ground-truth advertiser appears in the top-k of `predicted`, else 0.0."""
    return 1.0 if ground_truth in list(predicted)[:k] else 0.0


def mean_recall_at_k(pairs: Sequence[Tuple[Sequence, object]], k: int) -> float:
    """Mean Recall@k (Table 2/3's Recall@1, Recall@5, Recall@20) over a
    batch of (predicted_list, ground_truth) pairs.
    """
    if not pairs:
        return 0.0
    return sum(recall_at_k(pred, gt, k) for pred, gt in pairs) / len(pairs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    from datetime import date

    print("--- Sec 3.2: Data pipeline ---")
    users = [
        {"user_id": "u1", "last_conversion_date": date(2026, 4, 1)},
        {"user_id": "u2", "last_conversion_date": date(2026, 1, 1)},
    ]
    active = select_active_users(users, as_of_date=date(2026, 5, 1), lookback_days=90)
    print(f"active users (of {len(users)}): {[u['user_id'] for u in active]}")

    conversions = [(date(2026, 5, 3), "AdvB"), (date(2026, 5, 2), "AdvA"), (date(2026, 5, 10), "AdvC")]
    label = construct_next_advertiser_label(conversions, window_start=date(2026, 5, 2), window_end=date(2026, 5, 8))
    print(f"next-advertiser label: {label}")

    reinfer = select_users_for_incremental_inference(["u1", "u2", "u3"], ["u2"])
    print(f"users needing re-inference: {reinfer}")

    print("\n--- Sec 3.3: Prompt design ---")
    prompt = build_advertiser_prediction_prompt(
        profile={"age": 34, "gender": "F", "state": "CA"},
        behavior={"onsite_searches": ["running shoes", "yoga mats"], "offsite_urls": ["nike.com"]},
        active_advertisers=["Nike", "Lululemon"],
        preset_advertiser_pool=["Amazon", "Target"],
        stage="grpo",
        num_advertisers=20,
        num_interests=5,
    )
    print(prompt[:200] + "...")

    output = "<answer><interests>[fitness|running]</interests><advertiser_names>Nike|Lululemon|Amazon</advertiser_names></answer>"
    parsed = parse_structured_output(output)
    print(f"parsed: {parsed}")

    print("\n--- Sec 3.4: GRPO reward (Eq. 1-5) ---")
    rank = rank_of_advertiser(parsed["advertisers"], "Lululemon")
    reward = grpo_reward(rank, num_advertisers_predicted=len(parsed["advertisers"]), num_interests_predicted=len(parsed["interests"]))
    print(f"rank={rank}, reward={reward:.4f}")

    print("\n--- Sec 3.5: SID two-phase pretraining ---")
    vocab_size, d_model = 100, 16
    embedding = nn.Embedding(vocab_size, d_model)
    toy_model = nn.Sequential(embedding, nn.Linear(d_model, d_model))
    sid_ids = list(range(90, 100))
    hook = freeze_all_but_sid_embeddings(toy_model, embedding, sid_ids)
    x = torch.tensor([0, 1, 2, 95, 96, 97, 98, 99])  # mix of non-SID (0-2) and SID (95-99) rows
    toy_model(x).sum().backward()
    non_sid_row_grad = embedding.weight.grad[0].abs().sum().item()
    sid_row_grad = embedding.weight.grad[95].abs().sum().item()
    print(f"phase 1 grad on non-SID row: {non_sid_row_grad:.6f}, on SID row: {sid_row_grad:.6f}")
    hook.remove()
    unfreeze_all(toy_model)
    print(f"phase 2: all params trainable = {all(p.requires_grad for p in toy_model.parameters())}")

    print("\n--- Sec 3.6: Candidate blending ---")
    main_candidates = ["A1", "A2", "A3", "A4", "A5"]
    llm_candidates = ["A9", "A9", "A9", "A2", "A10"]
    blended = blend_candidates(main_candidates, llm_candidates, llm_quota=3)
    print(f"blended: {blended}, concentration: {advertiser_concentration(blended):.2f}")

    print("\n--- Sec 4: Recall@K ---")
    pairs = [(["A1", "A2", "A3"], "A2"), (["A5", "A6"], "A9")]
    print(f"Recall@1: {mean_recall_at_k(pairs, 1):.2f}, Recall@3: {mean_recall_at_k(pairs, 3):.2f}")


if __name__ == "__main__":
    _smoke_test()
