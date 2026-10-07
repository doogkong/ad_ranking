"""REA: Ranking Engineer Agent -- autonomous ML experimentation for ads ranking.

Reference implementation of the system described in Meta's engineering post
"Ranking Engineer Agent (REA): The Autonomous AI Agent Accelerating Meta's Ads
Ranking Innovation" (Kumar, Gao, Levi, Yadawad, Wong, Iyer, Sunkara; Meta,
March 17 2026).

REA is an *agent system*, not a neural network: it drives the ML-experimentation
loop (hypothesize -> plan -> launch training -> debug failures -> learn ->
iterate) over days/weeks with humans only at strategic decision points. The
post discloses the architecture but not the code, so every component below is a
faithful-in-spirit *reference design*; details the post does not specify
(persistence format, scoring rules, runbook contents, ...) are our own choices.

  Post section                          Here
  ------------------------------------  -------------------------------------
  Hibernate-and-wake                    REAgent.step / run, run_with_hibernation
  REA Planner                           Planner (+ Plan, approve callback)
  REA Executor (agent loop + wait)      REAgent (phases, Job collection)
  Historical insights database          InsightsDB
  ML research agent                     ResearchAgent / HeuristicResearchAgent
  Dual-source hypothesis engine         HypothesisGenerator
  Three-phase planning                  VALIDATION -> COMBINATION -> EXPLOITATION
  Failure-pattern runbook + guardrails  FailureRunbook, Guardrails
  Compute budgets (halt/pause)          BudgetGuard
  Preflight access checklist            PreflightChecklist
  Experiment logger -> insights loop    REAgent._log_outcome

The training cluster is abstracted behind a tiny scheduler interface
(submit / poll / advance). `SimulatedCluster` is a deterministic synthetic
workload (response surface + OOM / loss-explosion / transient-infra failures)
so the whole agent can run and be tested without GPUs.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import zlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Set, Tuple

Config = Dict[str, Any]

# Phases of one experimentation round (post: Validation -> Combination -> Exploitation).
PLAN = "plan"
VALIDATION = "validation"
COMBINATION = "combination"
EXPLOITATION = "exploitation"
PHASE_ORDER = (VALIDATION, COMBINATION, EXPLOITATION)


class Status(str, Enum):
    RUNNING = "running"
    HIBERNATED = "hibernated"   # waiting on jobs; agent process may exit
    DONE = "done"
    HALTED = "halted"           # budget exhausted / plan rejected / preflight failed
    ESCALATED = "escalated"     # failure outside the runbook: a human is needed


TERMINAL = {Status.DONE.value, Status.HALTED.value, Status.ESCALATED.value}


def config_key(cfg: Config) -> str:
    return json.dumps(cfg, sort_keys=True)


def diff(cfg: Config, baseline: Config) -> Config:
    """The knobs of `cfg` whose value differs from `baseline`."""
    return {k: v for k, v in cfg.items() if baseline.get(k) != v}


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------

@dataclass
class Hypothesis:
    """A proposed change to the baseline configuration."""
    label: str
    changes: Config
    source: str            # "history" | "research"
    category: str          # "architecture" | "training_efficiency" | "mixed"
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Hypothesis":
        return Hypothesis(**d)


def make_label(changes: Config) -> str:
    return "+".join(f"{k}={v}" for k, v in sorted(changes.items()))


def categorize(changes: Config, knob_category: Dict[str, str]) -> str:
    cats = {knob_category.get(k, "architecture") for k in changes}
    return cats.pop() if len(cats) == 1 else "mixed"


# ---------------------------------------------------------------------------
# Historical insights database  (post: "curated repository of past experiments")
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRecord:
    label: str
    changes: Config
    config_key: str            # full configuration the experiment ran
    status: str                # "success" | "failed" | "excluded"
    delta: Optional[float]     # metric - baseline metric (success only)
    failure: Optional[str]     # "oom" | "loss_explosion" | "infra" | ...
    gpu_hours: float
    round_idx: int
    phase: str


class InsightsDB:
    """Persistent memory of every experiment the agent has run.

    Supports the two uses the post describes: pattern recognition over past
    successes (`effect`) and over past failures (`known_bad`), plus dedup
    (`has_config`) so the agent never repeats an experiment.
    """

    def __init__(self, records: Optional[List[ExperimentRecord]] = None):
        self.records: List[ExperimentRecord] = list(records or [])

    def add(self, rec: ExperimentRecord) -> None:
        self.records.append(rec)

    def has_config(self, cfg: Config) -> bool:
        key = config_key(cfg)
        return any(r.config_key == key for r in self.records)

    def effect(self, knob: str, value: Any) -> Tuple[float, int]:
        """Mean per-knob metric delta of successful runs that set knob=value.

        A multi-knob experiment's delta is attributed equally across its knobs.
        Returns (mean_delta, n_observations).
        """
        ds = [
            r.delta / len(r.changes)
            for r in self.records
            if r.status == "success" and r.delta is not None and r.changes.get(knob) == value
        ]
        return (sum(ds) / len(ds), len(ds)) if ds else (0.0, 0)

    def known_bad(self, knob: str, value: Any) -> bool:
        """True if changing only knob->value previously OOM'd or diverged."""
        return any(
            r.status in ("failed", "excluded")
            and r.failure in ("oom", "loss_explosion")
            and r.changes == {knob: value}
            for r in self.records
        )

    def top_patterns(self, k: int = 5) -> List[Tuple[str, Any, float, int]]:
        seen = {(kn, v) for r in self.records for kn, v in r.changes.items()}
        rows = []
        for kn, v in seen:
            m, n = self.effect(kn, v)
            if n:
                rows.append((kn, v, m, n))
        return sorted(rows, key=lambda t: -t[2])[:k]

    def to_dict(self) -> List[Dict[str, Any]]:
        return [asdict(r) for r in self.records]

    @staticmethod
    def from_dict(rows: List[Dict[str, Any]]) -> "InsightsDB":
        return InsightsDB([ExperimentRecord(**r) for r in rows])


# ---------------------------------------------------------------------------
# Dual-source hypothesis engine  (post: insights DB + ML research agent)
# ---------------------------------------------------------------------------

class ResearchAgent(Protocol):
    def propose(
        self, baseline: Config, insights: InsightsDB, n: int, round_idx: int
    ) -> List[Hypothesis]: ...


class HeuristicResearchAgent:
    """Stand-in for the post's "deep ML research agent".

    The real component investigates baseline model configurations and proposes
    novel strategies (an LLM-driven agent). This one enumerates untried
    single-knob moves and *architecture x training-efficiency* pairs -- the
    combination the post credits for REA's most impactful wins -- preferring
    moves to adjacent values, with seeded random tie-breaking.
    """

    def __init__(self, search_space: Dict[str, List[Any]], knob_category: Dict[str, str], seed: int = 0):
        self.space = search_space
        self.category = knob_category
        self.seed = seed

    def propose(self, baseline: Config, insights: InsightsDB, n: int, round_idx: int) -> List[Hypothesis]:
        rng = random.Random(self.seed * 10007 + round_idx)
        moves: List[Tuple[float, Config]] = []
        for knob, vals in self.space.items():
            i = vals.index(baseline[knob])
            for j, v in enumerate(vals):
                if v != baseline[knob] and not insights.known_bad(knob, v):
                    moves.append((abs(j - i), {knob: v}))
        arch = [k for k in self.space if self.category.get(k) == "architecture"]
        eff = [k for k in self.space if self.category.get(k) == "training_efficiency"]
        for ka, ke in itertools.product(arch, eff):
            ia, ie = self.space[ka].index(baseline[ka]), self.space[ke].index(baseline[ke])
            for ja, va in enumerate(self.space[ka]):
                for je, ve in enumerate(self.space[ke]):
                    if (abs(ja - ia) == 1 and abs(je - ie) == 1
                            and not insights.known_bad(ka, va) and not insights.known_bad(ke, ve)):
                        moves.append((abs(ja - ia) + abs(je - ie) + 0.5, {ka: va, ke: ve}))
        scored = sorted(moves, key=lambda m: (m[0], rng.random()))
        out = []
        for _, ch in scored:
            if insights.has_config({**baseline, **ch}):
                continue
            out.append(Hypothesis(
                label=make_label(ch), changes=ch, source="research",
                category=categorize(ch, self.category),
                rationale="untried move near the current baseline",
            ))
            if len(out) >= n:
                break
        return out


class HypothesisGenerator:
    """Interleaves hypotheses mined from history with research-agent proposals."""

    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        knob_category: Dict[str, str],
        insights: InsightsDB,
        research_agent: Optional[ResearchAgent] = None,
        seed: int = 0,
    ):
        self.space = search_space
        self.category = knob_category
        self.insights = insights
        self.research = research_agent or HeuristicResearchAgent(search_space, knob_category, seed)

    def from_history(self, baseline: Config, n: int) -> List[Hypothesis]:
        """Knob values that helped in past experiments and are not yet applied."""
        rows = []
        for knob, vals in self.space.items():
            for v in vals:
                if v == baseline[knob] or self.insights.known_bad(knob, v):
                    continue
                mean, cnt = self.insights.effect(knob, v)
                if cnt and mean > 0 and not self.insights.has_config({**baseline, knob: v}):
                    rows.append((mean, cnt, knob, v))
        rows.sort(key=lambda t: -t[0])
        return [
            Hypothesis(
                label=make_label({k: v}), changes={k: v}, source="history",
                category=categorize({k: v}, self.category),
                rationale=f"mean delta {m:+.4f} over {c} prior run(s)",
            )
            for m, c, k, v in rows[:n]
        ]

    def generate(self, baseline: Config, n: int, round_idx: int = 0) -> List[Hypothesis]:
        hist = self.from_history(baseline, n)
        res = self.research.propose(baseline, self.insights, 2 * n, round_idx)
        out: List[Hypothesis] = []
        seen: Set[str] = set()
        for h in (x for pair in itertools.zip_longest(hist, res) for x in pair if x is not None):
            key = config_key(h.changes)
            if key in seen or self.insights.has_config({**baseline, **h.changes}):
                continue
            seen.add(key)
            out.append(h)
            if len(out) >= n:
                break
        return out


# ---------------------------------------------------------------------------
# Budget, safeguards, failure runbook
# ---------------------------------------------------------------------------

class BudgetGuard:
    """Tracks committed GPU-hours; refuses work that would exceed the budget.

    `reserve(est)` books the estimated cost when a job is launched;
    `settle(est, actual)` replaces the estimate with the real cost when it ends
    (failed jobs usually cost far less than planned).
    """

    def __init__(self, total: float, committed: float = 0.0):
        self.total = total
        self.committed = committed

    @property
    def remaining(self) -> float:
        return self.total - self.committed

    def can_reserve(self, est: float) -> bool:
        return self.committed + est <= self.total + 1e-9

    def reserve(self, est: float) -> bool:
        if not self.can_reserve(est):
            return False
        self.committed += est
        return True

    def settle(self, est: float, actual: float) -> None:
        self.committed += actual - est

    def to_dict(self) -> Dict[str, float]:
        return {"total": self.total, "committed": self.committed}


class Action(str, Enum):
    RETRY = "retry"
    EXCLUDE = "exclude"
    ESCALATE = "escalate"


@dataclass
class Guardrails:
    """Limits inside which the agent adapts on its own instead of asking a human."""
    max_attempts_per_job: int = 3     # first try + 2 retries
    max_total_retries: int = 8


class FailureRunbook:
    """Known failure patterns -> autonomous response (post: "runbook of common
    failure patterns" with prioritization such as excluding jobs with clear OOM
    or loss-explosion signals). Unknown errors are escalated to a human."""

    def __init__(self, guardrails: Optional[Guardrails] = None):
        self.guardrails = guardrails or Guardrails()

    def decide(self, error: str, attempt: int, retries_used: int) -> Tuple[Action, str]:
        if error == "oom":
            return Action.EXCLUDE, "out-of-memory: configuration does not fit; excluded"
        if error == "loss_explosion":
            return Action.EXCLUDE, "training instability (loss explosion); excluded"
        if error == "infra":
            if attempt + 1 >= self.guardrails.max_attempts_per_job:
                return Action.EXCLUDE, "infrastructure failure persisted past retry limit"
            if retries_used >= self.guardrails.max_total_retries:
                return Action.EXCLUDE, "global retry budget exhausted"
            return Action.RETRY, "transient infrastructure failure; retrying"
        return Action.ESCALATE, f"unknown failure '{error}'; escalating to an engineer"


@dataclass
class PreflightChecklist:
    """Engineer-granted access controls, verified before anything runs
    (post: "engineers grant explicit access controls through preflight
    checklist reviews"; REA "works exclusively on Meta's ads ranking model
    codebase")."""
    items: Dict[str, bool] = field(default_factory=lambda: {
        "codebase_access_reviewed": True,
        "compute_budget_confirmed": True,
    })
    allowed_knobs: Optional[Set[str]] = None
    allowed_codebase: str = "ads_ranking_models"

    def verify(self, hypotheses: Sequence[Hypothesis], codebase: str) -> List[str]:
        failures = [f"checklist item not granted: {k}" for k, ok in self.items.items() if not ok]
        if codebase != self.allowed_codebase:
            failures.append(f"codebase '{codebase}' is outside the allowed '{self.allowed_codebase}'")
        if self.allowed_knobs is not None:
            for h in hypotheses:
                bad = set(h.changes) - self.allowed_knobs
                if bad:
                    failures.append(f"hypothesis '{h.label}' touches disallowed knobs {sorted(bad)}")
        return failures


# ---------------------------------------------------------------------------
# REA Planner  (post: exploration strategy + GPU cost estimate + engineer approval)
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    round_idx: int
    hypotheses: List[Hypothesis]
    dropped: List[str]                 # labels that did not fit the validation budget
    round_budget: float                # GPU-hours
    phase_fractions: Tuple[float, float, float]
    est_validation_cost: float

    def summary(self) -> str:
        lines = [
            f"Round {self.round_idx}: budget {self.round_budget:.1f} GPU-h "
            f"(validation/combination/exploitation = "
            + "/".join(f"{f:.0%}" for f in self.phase_fractions) + ")",
            f"  validate {len(self.hypotheses)} hypotheses, est {self.est_validation_cost:.1f} GPU-h",
        ]
        lines += [f"    - [{h.source}/{h.category}] {h.label}" for h in self.hypotheses]
        if self.dropped:
            lines.append(f"  dropped (over validation budget): {', '.join(self.dropped)}")
        return "\n".join(lines)


class Planner:
    def __init__(
        self,
        cost_model: Callable[[Config], float],
        phase_fractions: Tuple[float, float, float] = (0.4, 0.3, 0.3),
    ):
        assert abs(sum(phase_fractions) - 1.0) < 1e-9
        self.cost_model = cost_model
        self.fractions = phase_fractions

    def cumulative_caps(self, round_budget: float) -> Dict[str, float]:
        """Cumulative per-round spend ceilings at the end of each phase."""
        caps, run = {}, 0.0
        for phase, f in zip(PHASE_ORDER, self.fractions):
            run += f
            caps[phase] = run * round_budget
        return caps

    def make_plan(
        self,
        hypotheses: Sequence[Hypothesis],
        baseline: Config,
        round_budget: float,
        round_idx: int,
        reserved: float = 0.0,
    ) -> Plan:
        """Keep hypotheses, in priority order, until the validation budget is full.

        `reserved` is spent before any hypothesis (the baseline measurement).
        """
        cap = self.fractions[0] * round_budget - reserved
        kept, dropped, spent = [], [], 0.0
        for h in hypotheses:
            c = self.cost_model({**baseline, **h.changes})
            if spent + c <= cap + 1e-9:
                kept.append(h)
                spent += c
            else:
                dropped.append(h.label)
        return Plan(round_idx, kept, dropped, round_budget, self.fractions, spent + reserved)

    def combination_candidates(
        self,
        validated: Sequence[Dict[str, Any]],
        baseline: Config,
        budget: float,
        k: int,
    ) -> List[Tuple[str, Config]]:
        """Combine validated-positive hypotheses that touch disjoint knobs,
        ranked by the sum of their individual gains (the additive prior)."""
        pos = sorted((v for v in validated if v["delta"] > 0), key=lambda v: -v["delta"])
        combos = []
        for r in (2, 3):
            for subset in itertools.combinations(pos, r):
                knobs = [kn for v in subset for kn in v["changes"]]
                if len(knobs) != len(set(knobs)):
                    continue   # conflicting knobs
                merged: Config = {}
                for v in subset:
                    merged.update(v["changes"])
                combos.append((sum(v["delta"] for v in subset), merged))
        combos.sort(key=lambda t: -t[0])
        out, spent = [], 0.0
        for _, merged in combos:
            c = self.cost_model({**baseline, **merged})
            if spent + c > budget + 1e-9:
                continue
            out.append((make_label(merged), merged))
            spent += c
            if len(out) >= k:
                break
        return out

    def exploitation_candidates(
        self,
        best_cfg: Config,
        baseline: Config,
        search_space: Dict[str, List[Any]],
        insights: InsightsDB,
        tried_keys: Set[str],
        budget: float,
        k: int,
    ) -> List[Tuple[str, Config]]:
        """Local search around the best configuration found so far: step each
        knob to its adjacent values, favouring moves the insights DB says help."""
        cands = []
        for order, (knob, vals) in enumerate(search_space.items()):
            i = vals.index(best_cfg[knob])
            for j in (i - 1, i + 1):
                if not 0 <= j < len(vals):
                    continue
                cfg2 = {**best_cfg, knob: vals[j]}
                if config_key(cfg2) in tried_keys or insights.has_config(cfg2):
                    continue
                mean, n = insights.effect(knob, vals[j])
                cands.append((-(mean if n else 0.0), order, cfg2))
        cands.sort(key=lambda t: (t[0], t[1]))
        out, spent = [], 0.0
        for _, _, cfg2 in cands:
            c = self.cost_model(cfg2)
            if spent + c > budget + 1e-9:
                continue
            ch = diff(cfg2, baseline)
            out.append(("exploit:" + make_label(diff(cfg2, best_cfg)), ch))
            spent += c
            if len(out) >= k:
                break
        return out


# ---------------------------------------------------------------------------
# Scheduler abstraction + synthetic cluster
# ---------------------------------------------------------------------------

@dataclass
class JobResult:
    state: str                 # "running" | "succeeded" | "failed"
    metric: Optional[float] = None
    error: Optional[str] = None
    gpu_hours: float = 0.0


class Cluster(Protocol):
    def submit(self, cfg: Config, attempt: int = 0) -> str: ...
    def poll(self, job_id: str) -> JobResult: ...
    def advance(self) -> None: ...


def default_search_space() -> Dict[str, List[Any]]:
    return {
        "num_layers": [2, 4, 6, 8],
        "hidden_dim": [128, 256, 512, 1024],
        "seq_len": [64, 128, 256, 512],
        "feature_interaction": ["dot", "cross", "mixer"],
        "lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
        "warmup_steps": [0, 500, 2000],
        "use_bf16": [False, True],
    }


def default_knob_category() -> Dict[str, str]:
    return {
        "num_layers": "architecture", "hidden_dim": "architecture",
        "seq_len": "architecture", "feature_interaction": "architecture",
        "lr": "training_efficiency", "warmup_steps": "training_efficiency",
        "use_bf16": "training_efficiency",
    }


def default_baseline() -> Config:
    return {
        "num_layers": 4, "hidden_dim": 256, "seq_len": 128,
        "feature_interaction": "cross", "lr": 1e-3, "warmup_steps": 0, "use_bf16": False,
    }


def memory_gb(cfg: Config) -> float:
    m = cfg["num_layers"] * cfg["hidden_dim"] ** 2 / 1e5
    if cfg["use_bf16"]:
        m *= 0.5
    return m + cfg["seq_len"] * cfg["hidden_dim"] / 1e4


def simulated_cost(cfg: Config) -> float:
    """Estimated GPU-hours for one training run of `cfg`."""
    c = cfg["num_layers"] * cfg["hidden_dim"] ** 2 * cfg["seq_len"] / 2e7
    if cfg["use_bf16"]:
        c *= 0.6
    return 0.5 + c


_EFFECTS = {
    "num_layers": {2: -0.020, 4: 0.0, 6: 0.015, 8: 0.020},
    "hidden_dim": {128: -0.020, 256: 0.0, 512: 0.012, 1024: 0.018},
    "seq_len": {64: -0.015, 128: 0.0, 256: 0.010, 512: 0.016},
    "feature_interaction": {"dot": -0.010, "cross": 0.0, "mixer": 0.012},
    "lr": {1e-4: -0.008, 3e-4: 0.004, 1e-3: 0.0, 3e-3: 0.006, 1e-2: 0.0},
    "warmup_steps": {0: 0.0, 500: 0.003, 2000: 0.004},
    "use_bf16": {False: 0.0, True: 0.0},
}
SYNERGY_BONUS = 0.008   # mixer x bf16: architecture + training-efficiency synergy


def simulated_metric(cfg: Config) -> float:
    """Synthetic model-quality response surface (higher is better)."""
    m = 0.60 + sum(_EFFECTS[k][cfg[k]] for k in _EFFECTS)
    if cfg["feature_interaction"] == "mixer" and cfg["use_bf16"]:
        m += SYNERGY_BONUS
    noise = ((zlib.crc32(config_key(cfg).encode()) % 1000) / 1000.0 - 0.5) * 0.001
    return m + noise


class SimulatedCluster:
    """Deterministic fake training cluster with a logical clock.

    Failure modes (checked in this order): OOM when the memory model exceeds
    `mem_limit_gb`; loss explosion when lr > 3e-3 with < 500 warmup steps;
    a transient infra failure on the *first* attempt of ~1/`infra_mod` configs.
    """

    def __init__(self, mem_limit_gb: float = 80.0, infra_mod: int = 7):
        self.mem_limit_gb = mem_limit_gb
        self.infra_mod = infra_mod
        self.clock = 0
        self._next = 0
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def is_infra_flaky(self, cfg: Config) -> bool:
        return zlib.crc32(config_key(cfg).encode()) % self.infra_mod == 0

    def _outcome(self, cfg: Config, attempt: int) -> Dict[str, Any]:
        cost = simulated_cost(cfg)
        if memory_gb(cfg) > self.mem_limit_gb:
            return {"state": "failed", "error": "oom", "gpu_hours": 0.1 * cost, "ticks": 1}
        if cfg["lr"] > 3e-3 and cfg["warmup_steps"] < 500:
            return {"state": "failed", "error": "loss_explosion", "gpu_hours": 0.3 * cost,
                    "ticks": max(1, math.ceil(0.3 * cost))}
        if attempt == 0 and self.is_infra_flaky(cfg):
            return {"state": "failed", "error": "infra", "gpu_hours": 0.2 * cost,
                    "ticks": max(1, math.ceil(0.2 * cost))}
        return {"state": "succeeded", "metric": simulated_metric(cfg), "gpu_hours": cost,
                "ticks": max(1, math.ceil(cost))}

    def submit(self, cfg: Config, attempt: int = 0) -> str:
        job_id = f"job-{self._next}"
        self._next += 1
        out = self._outcome(cfg, attempt)
        self._jobs[job_id] = {"finish": self.clock + out["ticks"], "outcome": out}
        return job_id

    def poll(self, job_id: str) -> JobResult:
        job = self._jobs[job_id]
        if self.clock < job["finish"]:
            return JobResult("running")
        o = job["outcome"]
        return JobResult(o["state"], o.get("metric"), o.get("error"), o["gpu_hours"])

    def running(self) -> List[str]:
        return [j for j, v in self._jobs.items() if self.clock < v["finish"]]

    def advance(self) -> None:
        """Move the clock to the next job completion (the wake-up event)."""
        pending = [self._jobs[j]["finish"] for j in self.running()]
        if pending:
            self.clock = min(pending)


# ---------------------------------------------------------------------------
# Agent state (everything that must survive hibernation)
# ---------------------------------------------------------------------------

@dataclass
class JobRec:
    job_id: str
    label: str
    changes: Config
    config: Config
    round_idx: int
    phase: str
    est_cost: float
    attempt: int = 0
    status: str = "running"    # running | succeeded | failed | excluded
    metric: Optional[float] = None
    failure: Optional[str] = None
    actual_cost: float = 0.0


@dataclass
class AgentState:
    status: str = Status.RUNNING.value
    halt_reason: str = ""
    round_idx: int = 0
    phase: str = PLAN
    phase_submitted: bool = False
    baseline_config: Config = field(default_factory=dict)
    baseline_metric: Optional[float] = None
    best: Optional[Dict[str, Any]] = None       # {config, changes, metric, label}
    jobs: List[JobRec] = field(default_factory=list)
    plan: Optional[Dict[str, Any]] = None
    validated: List[Dict[str, Any]] = field(default_factory=list)
    guard: Dict[str, float] = field(default_factory=dict)
    round_budget: float = 0.0
    round_start_committed: float = 0.0
    retries_used: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    insights: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# REA: Planner + Executor with hibernate-and-wake
# ---------------------------------------------------------------------------

class REAgent:
    """The agent loop. `step()` runs until it must wait on training jobs, then
    checkpoints and reports HIBERNATED; a background monitor (`run`,
    `run_with_hibernation`) advances the cluster and wakes it, and the agent
    resumes exactly where it left off from the checkpoint.
    """

    def __init__(
        self,
        *,
        baseline: Config,
        search_space: Dict[str, List[Any]],
        cluster: Cluster,
        cost_model: Callable[[Config], float] = simulated_cost,
        total_budget: float,
        knob_category: Optional[Dict[str, str]] = None,
        rounds: int = 1,
        n_hypotheses: int = 6,
        k_combination: int = 3,
        k_exploitation: int = 3,
        approve: Optional[Callable[[Plan], bool]] = None,
        research_agent: Optional[ResearchAgent] = None,
        runbook: Optional[FailureRunbook] = None,
        checklist: Optional[PreflightChecklist] = None,
        planner: Optional[Planner] = None,
        state_path: Optional[str] = None,
        codebase: str = "ads_ranking_models",
        seed: int = 0,
    ):
        self.space = search_space
        self.category = knob_category or {k: "architecture" for k in search_space}
        self.cluster = cluster
        self.cost_model = cost_model
        self.rounds = rounds
        self.n_hypotheses = n_hypotheses
        self.k_combination = k_combination
        self.k_exploitation = k_exploitation
        self.approve = approve or (lambda plan: True)
        self.runbook = runbook or FailureRunbook()
        self.checklist = checklist or PreflightChecklist()
        self.planner = planner or Planner(cost_model)
        self.state_path = state_path
        self.codebase = codebase
        self.seed = seed
        self.research_agent = research_agent

        self.state = AgentState(baseline_config=dict(baseline), guard=BudgetGuard(total_budget).to_dict())
        self.insights = InsightsDB()
        self.guard = BudgetGuard(**self.state.guard)
        self.generator = HypothesisGenerator(
            self.space, self.category, self.insights, research_agent, seed)

    # ---- persistence (hibernate / wake) ----------------------------------

    def save(self) -> None:
        self.state.guard = self.guard.to_dict()
        self.state.insights = self.insights.to_dict()
        if self.state_path:
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(asdict(self.state), f, indent=1)
            os.replace(tmp, self.state_path)

    @classmethod
    def resume(cls, state_path: str, **kwargs: Any) -> "REAgent":
        """Rebuild an agent from its checkpoint (same kwargs as the original)."""
        agent = cls(state_path=state_path, **kwargs)
        with open(state_path) as f:
            d = json.load(f)
        d["jobs"] = [JobRec(**j) for j in d["jobs"]]
        agent.state = AgentState(**d)
        agent.insights = InsightsDB.from_dict(agent.state.insights)
        agent.guard = BudgetGuard(**agent.state.guard)
        agent.generator = HypothesisGenerator(
            agent.space, agent.category, agent.insights, agent.research_agent, agent.seed)
        return agent

    def _log(self, msg: str) -> None:
        self.state.events.append(f"[r{self.state.round_idx}/{self.state.phase}] {msg}")

    def _stop(self, status: Status, reason: str) -> None:
        self.state.status = status.value
        self.state.halt_reason = reason
        self._log(f"{status.value}: {reason}")

    # ---- main loop --------------------------------------------------------

    def step(self) -> Status:
        """Advance until the agent must wait (HIBERNATED) or is finished."""
        st = self.state
        if st.status == Status.HIBERNATED.value:
            st.status = Status.RUNNING.value   # woken up
        while True:
            if st.status in TERMINAL:
                self.save()
                return Status(st.status)
            if st.phase == PLAN:
                self._begin_round()
            else:
                if not st.phase_submitted:
                    self._submit_phase(st.phase)
                    st.phase_submitted = True
                self._collect()
                if st.status in TERMINAL:
                    continue
                if any(j.status == "running" for j in self._phase_jobs(st.phase)):
                    st.status = Status.HIBERNATED.value
                    self._log("waiting on training jobs; hibernating")
                    self.save()
                    return Status.HIBERNATED
                self._finish_phase(st.phase)

    def run(self, max_wakes: int = 100_000) -> Status:
        """Drive the agent in-process: hibernate, let the cluster progress, wake."""
        for _ in range(max_wakes):
            s = self.step()
            if s != Status.HIBERNATED:
                return s
            self.cluster.advance()
        raise RuntimeError("max_wakes exceeded")

    # ---- round / phase machinery -----------------------------------------

    def _phase_jobs(self, phase: str) -> List[JobRec]:
        return [j for j in self.state.jobs
                if j.round_idx == self.state.round_idx and j.phase == phase]

    def _begin_round(self) -> None:
        st = self.state
        if self.guard.remaining <= 1e-9:
            return self._stop(Status.HALTED, "compute budget exhausted")
        st.round_budget = self.guard.remaining / (self.rounds - st.round_idx)
        st.round_start_committed = self.guard.committed

        hyps = self.generator.generate(st.baseline_config, self.n_hypotheses, st.round_idx)
        reserved = self.cost_model(st.baseline_config) if st.baseline_metric is None else 0.0
        plan = self.planner.make_plan(hyps, st.baseline_config, st.round_budget, st.round_idx, reserved)
        st.plan = {**asdict(plan), "hypotheses": [h.to_dict() for h in plan.hypotheses]}
        if not plan.hypotheses:
            return self._stop(Status.DONE, "no untried hypotheses fit the remaining budget")

        failures = self.checklist.verify(plan.hypotheses, self.codebase)
        if failures:
            return self._stop(Status.HALTED, "preflight failed: " + "; ".join(failures))
        if not self.approve(plan):
            return self._stop(Status.HALTED, "plan not approved by engineer")

        self._log(f"plan approved: {len(plan.hypotheses)} hypotheses")
        st.validated = []
        st.phase, st.phase_submitted = VALIDATION, False

    def _plan_hypotheses(self) -> List[Hypothesis]:
        return [Hypothesis.from_dict(h) for h in self.state.plan["hypotheses"]]

    def _phase_candidates(self, phase: str) -> List[Tuple[str, Config]]:
        st = self.state
        caps = self.planner.cumulative_caps(st.round_budget)
        budget = caps[phase] - (self.guard.committed - st.round_start_committed)
        if phase == VALIDATION:
            cands: List[Tuple[str, Config]] = []
            if st.baseline_metric is None:
                cands.append(("baseline", {}))
            return cands + [(h.label, h.changes) for h in self._plan_hypotheses()]
        if phase == COMBINATION:
            return self.planner.combination_candidates(
                st.validated, st.baseline_config, budget, self.k_combination)
        if not st.best or not st.best["changes"]:
            return []   # nothing beat the baseline: nothing to exploit
        tried = {config_key(j.config) for j in st.jobs}
        return self.planner.exploitation_candidates(
            st.best["config"], st.baseline_config, self.space, self.insights, tried, budget,
            self.k_exploitation)

    def _submit_phase(self, phase: str) -> None:
        st = self.state
        caps = self.planner.cumulative_caps(st.round_budget)
        for label, changes in self._phase_candidates(phase):
            cfg = {**st.baseline_config, **changes}
            est = self.cost_model(cfg)
            spent = self.guard.committed - st.round_start_committed
            if label != "baseline" and spent + est > caps[phase] + 1e-9:
                self._log(f"skip {label}: over {phase} budget cap")
                continue
            if not self.guard.reserve(est):
                self._log(f"skip {label}: over total compute budget")
                continue
            job_id = self.cluster.submit(cfg, 0)
            st.jobs.append(JobRec(job_id, label, changes, cfg, st.round_idx, phase, est))
            self._log(f"launched {label} ({job_id}, est {est:.1f} GPU-h)")

    def _collect(self) -> None:
        """Poll running jobs; apply the runbook to failures without a human."""
        st = self.state
        for rec in self._phase_jobs(st.phase):
            if rec.status != "running":
                continue
            res = self.cluster.poll(rec.job_id)
            if res.state == "running":
                continue
            self.guard.settle(rec.est_cost, res.gpu_hours)
            rec.actual_cost += res.gpu_hours
            if res.state == "succeeded":
                rec.status, rec.metric = "succeeded", res.metric
                continue
            rec.failure = res.error
            action, why = self.runbook.decide(res.error or "unknown", rec.attempt, st.retries_used)
            self._log(f"{rec.label}: {res.error} -> {action.value} ({why})")
            if action == Action.RETRY and self.guard.reserve(rec.est_cost):
                st.retries_used += 1
                rec.attempt += 1
                rec.job_id = self.cluster.submit(rec.config, rec.attempt)
            elif action == Action.ESCALATE:
                rec.status = "failed"
                self._stop(Status.ESCALATED, why)
                return
            else:
                rec.status = "excluded"

    def _log_outcome(self, rec: JobRec, delta: Optional[float]) -> None:
        """Experiment logger: outcomes flow back into the insights DB, which
        the next round's hypothesis generator reads (the knowledge loop)."""
        self.insights.add(ExperimentRecord(
            label=rec.label, changes=rec.changes, config_key=config_key(rec.config),
            status="success" if rec.status == "succeeded" else rec.status,
            delta=delta, failure=rec.failure,
            gpu_hours=rec.actual_cost, round_idx=rec.round_idx, phase=rec.phase,
        ))

    def _finish_phase(self, phase: str) -> None:
        st = self.state
        jobs = self._phase_jobs(phase)

        for rec in jobs:
            if rec.label == "baseline":
                if rec.status != "succeeded":
                    return self._stop(Status.ESCALATED, "baseline run failed; cannot measure deltas")
                st.baseline_metric = rec.metric
                st.best = {"config": dict(st.baseline_config), "changes": {}, "metric": rec.metric,
                           "label": "baseline"}
        for rec in jobs:
            if rec.label == "baseline":
                continue
            if rec.status == "succeeded":
                delta = rec.metric - st.baseline_metric
                if phase == VALIDATION:
                    st.validated.append({"label": rec.label, "changes": rec.changes, "delta": delta})
                if rec.metric > st.best["metric"]:
                    st.best = {"config": rec.config, "changes": rec.changes,
                               "metric": rec.metric, "label": rec.label}
                self._log_outcome(rec, delta)
            else:
                self._log_outcome(rec, None)

        if phase == EXPLOITATION:
            return self._end_round()
        st.phase = PHASE_ORDER[PHASE_ORDER.index(phase) + 1]
        st.phase_submitted = False

    def _end_round(self) -> None:
        st = self.state
        improved = st.best["metric"] > st.baseline_metric + 1e-12
        st.history.append({
            "round": st.round_idx, "baseline_metric": st.baseline_metric,
            "best_metric": st.best["metric"], "best_label": st.best["label"],
            "spent": self.guard.committed - st.round_start_committed,
        })
        if improved:
            st.baseline_config = dict(st.best["config"])
            st.baseline_metric = st.best["metric"]
            st.best = {"config": dict(st.baseline_config), "changes": {}, "metric": st.baseline_metric,
                       "label": "baseline"}
        st.round_idx += 1
        st.phase_submitted = False
        if st.round_idx >= self.rounds:
            self._stop(Status.DONE, "all rounds complete")
        else:
            st.phase = PLAN

    # ---- reporting --------------------------------------------------------

    def report(self) -> str:
        st = self.state
        first = st.history[0]["baseline_metric"] if st.history else st.baseline_metric
        lines = [
            f"status={st.status} ({st.halt_reason})",
            f"budget: {self.guard.committed:.1f}/{self.guard.total:.1f} GPU-h, retries={st.retries_used}",
            f"metric: {first} -> {st.baseline_metric}",
            f"config: {st.baseline_config}",
        ]
        for h in st.history:
            lines.append(f"  round {h['round']}: best={h['best_label']} ({h['best_metric']:.4f}), "
                         f"spent {h['spent']:.1f} GPU-h")
        return "\n".join(lines)


def run_with_hibernation(
    make_agent: Callable[[bool], REAgent], max_wakes: int = 100_000
) -> REAgent:
    """Run an agent while *discarding it* at every wait, to demonstrate that only
    the checkpoint and the external cluster carry state across a hibernation.

    `make_agent(resume)` must build a fresh REAgent, from the checkpoint when
    `resume` is True (e.g. `REAgent.resume(path, **kwargs)`).
    """
    agent = make_agent(False)
    for _ in range(max_wakes):
        s = agent.step()
        if s != Status.HIBERNATED:
            return agent
        cluster = agent.cluster
        del agent                      # "shut down to conserve resources"
        cluster.advance()              # background system waits for a completion
        agent = make_agent(True)       # ...and wakes a new agent from the checkpoint
    raise RuntimeError("max_wakes exceeded")


if __name__ == "__main__":
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), "rea_state.json")
    kwargs = dict(
        baseline=default_baseline(), search_space=default_search_space(),
        knob_category=default_knob_category(), cluster=SimulatedCluster(),
        total_budget=120.0, rounds=2, seed=1,
        approve=lambda plan: (print(plan.summary()) or True),
    )
    final = run_with_hibernation(
        lambda resume: REAgent.resume(path, **kwargs) if resume else REAgent(state_path=path, **kwargs))
    print(final.report())
    print("top insights:", final.insights.top_patterns(3))
