# REA

Reference implementation of **Ranking Engineer Agent (REA): The Autonomous AI Agent Accelerating Meta's Ads Ranking Innovation** (Kumar, Gao, Levi, Yadawad, Wong, Iyer, Sunkara; Meta Engineering, March 17 2026).

Source: https://engineering.fb.com/2026/03/17/developer-tools/ranking-engineer-agent-rea-autonomous-ai-system-accelerating-meta-ads-ranking-innovation/

In its first production rollout across six ads-ranking models, REA-driven iterations **doubled average model accuracy** over baseline, and three engineers delivered improvement proposals for **eight models** — work that previously took two engineers per model (~5x engineering output).

> **Read this first — what this is and isn't.** REA is an *agent system*, not a neural network, and the source is an engineering **blog post**, not a paper: Meta discloses the architecture but no code, models, prompts, or numbers beyond the headline results. This folder implements the **described architecture** as a runnable reference design. The LLM-driven parts (the ML research agent) are replaced by a deterministic heuristic behind a pluggable interface, the training cluster is a deterministic simulator, and everything the post leaves unspecified (checkpoint format, scoring rules, runbook contents, budget split) is our own design choice. Numbers from the simulator say nothing about Meta's results.

---

## Summary

Optimizing a mature ads ranking model is a slow, sequential loop: craft a hypothesis, design an experiment, launch training, debug failures across a complex codebase, analyze, iterate. One cycle spans days to weeks. Existing AI tools are reactive and session-bound — they help with single steps, but an engineer still decides what to do next and babysits long-running jobs.

REA instead runs the **end-to-end experimentation lifecycle autonomously**, across multi-week workflows, and asks a human only at key strategic points. It rests on three ideas:

| Challenge | REA's answer | Here |
|---|---|---|
| Training jobs run for hours/days, longer than any session | **Hibernate-and-wake**: delegate the wait to a background system, shut down, resume where it left off | `REAgent.step/run`, `run_with_hibernation` |
| Experiment quality is capped by hypothesis quality | **Dual-source hypothesis engine**: historical insights DB + ML research agent | `InsightsDB`, `HypothesisGenerator`, `ResearchAgent` |
| Real infra fails; compute is finite | **Three-phase planning inside an approved budget** + a failure runbook that adapts within guardrails | `Planner`, `BudgetGuard`, `FailureRunbook`, `Guardrails` |

---

## Architecture

```
            ┌────────────┐  export plan   ┌──────────────────────────┐
 Engineer ◀▶│ REA Planner│ ─────────────▶ │       REA Executor       │
 (approves  │ hypotheses │                │  agent loop ⇄ wait state │
  plan &    │ + GPU-cost │ ◀───────────── │  async job execution     │
  budget)   │  estimate  │  resume w/     └──────────────────────────┘
            └────────────┘  results                  ▲   │ launch / poll
                   ▲                                 │   ▼
                   │            ┌───────────────────────────────────┐
                   └────────────│  Skill / Knowledge / Tool system  │
                                │  insights DB · schedulers · code  │
                                └───────────────────────────────────┘
```

* **REA Planner** — an engineer collaborates with the hypothesis generator to produce an experiment plan; the planner estimates total GPU cost and gets the engineer's confirmation *before anything runs*.
* **REA Executor** — runs the exported plan through an agent loop with a **wait state**: launch jobs, hibernate, wake with results.
* **Skill, Knowledge and Tool System** — shared ML skills, the historical experiment data, and integrations (job scheduler, experiment tracking, codebase tools). The post builds this on Meta's internal agent framework *Confucius*.

Two flows run through it:

* **Execution flow** (long-horizon autonomy): plan → executor → hibernate during training → resume with results.
* **Knowledge flow** (hypothesis quality): an experiment logger records outcomes, metrics and configurations into the insights DB; the hypothesis generator reads it to propose better hypotheses next round, so the system's intelligence compounds.

---

## Key Ideas

### Hibernate-and-wake

```
step():  … launch jobs → jobs still running? → checkpoint, return HIBERNATED
                                                        │  (process may exit)
background monitor: wait for a job to finish ───────────┘
resume():  load checkpoint → poll jobs → continue exactly where it left off
```

All agent state — phase, job table, budget ledger, insights DB, plan — lives in a JSON checkpoint. `run_with_hibernation` *deletes the agent object at every wait* and rebuilds it from disk, and a test asserts the result is identical to an uninterrupted run.

### Dual-source hypothesis engine

* **Historical insights database** — curated past experiments enabling in-context learning and pattern recognition over successes *and* failures. `InsightsDB.effect(knob, value)` gives the mean metric delta of past runs; `known_bad` remembers clean OOM / divergence failures.
* **ML research agent** — a deep-research component proposing novel strategies from baseline configs. Here a `ResearchAgent` protocol with a `HeuristicResearchAgent` default (untried moves near the baseline, plus **architecture × training-efficiency pairs**, the combination the post credits for REA's most impactful wins). Swap in an LLM agent by implementing `propose(baseline, insights, n, round_idx)`.

`HypothesisGenerator` interleaves both sources, de-duplicates, and never re-proposes an experiment already run.

### Three-phase planning within a compute budget

Each round splits its GPU-hour budget 40 / 30 / 30 (cumulative caps enforced):

1. **Validation** — test individual hypotheses from both sources *in parallel* to establish quality baselines.
2. **Combination** — combine validated-positive hypotheses touching disjoint knobs, ranked by summed individual gain (the additive prior).
3. **Exploitation** — intensive local search around the best configuration found, favouring moves history says help.

### Resilient execution

When a job fails, the executor consults a **runbook** and adapts within guardrails instead of escalating:

| Failure | Response |
|---|---|
| `oom` | exclude the configuration (and remember it) |
| `loss_explosion` | exclude (training instability) |
| `infra` (transient) | retry, up to `max_attempts_per_job` and `max_total_retries` |
| anything unknown | **escalate to a human** and stop |

Failed jobs settle at their *actual* (usually much smaller) cost, so a crashed run refunds budget.

### Safeguards

* `PreflightChecklist` — engineer-granted access controls verified before launch; the agent works only in its allowed codebase and allowed knobs.
* `BudgetGuard` — reserves estimated cost at launch, settles actual cost at completion; refuses work that would exceed the budget.
* Human approval of the plan (`approve` callback) before any job starts; the agent stops rather than proceeds if rejected.

---

## Key Components

| Component | Post section |
|---|---|
| `REAgent` (`step`, `run`, `save`, `resume`) | REA Executor; Hibernate-and-wake |
| `run_with_hibernation` | Hibernate-and-wake (drops and rebuilds the agent at each wait) |
| `Planner`, `Plan` | REA Planner; Three-Phase Planning |
| `HypothesisGenerator`, `ResearchAgent`, `HeuristicResearchAgent` | Dual-Source Hypothesis Engine |
| `InsightsDB`, `ExperimentRecord` | Historical Insights Database; knowledge flow |
| `REAgent._log_outcome` | Experiment logger |
| `FailureRunbook`, `Guardrails` | Resilient execution |
| `BudgetGuard` | Compute budgets; halt/pause at thresholds |
| `PreflightChecklist` | Preflight checklist reviews; codebase restriction |
| `SimulatedCluster`, `simulated_metric/cost`, `memory_gb` | *(ours)* synthetic workload so the agent runs without GPUs |

---

## Usage

```python
from rea import (REAgent, SimulatedCluster, default_baseline, default_search_space,
                 default_knob_category)

agent = REAgent(
    baseline=default_baseline(),
    search_space=default_search_space(),
    knob_category=default_knob_category(),
    cluster=SimulatedCluster(),
    total_budget=120.0,          # GPU-hours
    rounds=2,
    approve=lambda plan: (print(plan.summary()) or True),   # engineer sign-off
    state_path="rea_state.json", # enables real hibernate/resume
)
agent.run()
print(agent.report())
```

Hibernate for real — the agent is discarded at every wait and rebuilt from the checkpoint:

```python
from rea import REAgent, run_with_hibernation

make = lambda resume: REAgent.resume(path, **kw) if resume else REAgent(state_path=path, **kw)
final = run_with_hibernation(make)
```

### Plugging in a real system

Only two interfaces touch the outside world:

* `Cluster` — `submit(cfg, attempt) -> job_id`, `poll(job_id) -> JobResult`, `advance()` (block until something finishes). Implement against your scheduler to run real training.
* `ResearchAgent` — `propose(baseline, insights, n, round_idx) -> [Hypothesis]`. Implement with an LLM to replace the heuristic.

Also replace `cost_model` with your GPU-hour estimator and `search_space` with your real config knobs.

---

## Running

```bash
python3 rea.py                         # end-to-end demo (2 rounds, hibernating at every wait)
python3 -m pytest test_rea.py -v       # full test suite
```

Pure standard library — no PyTorch or other dependencies.

---

## What the post does not specify

Our choices, not Meta's: the model(s) behind the research agent and hypothesis generator; the persistence format and wake trigger; how hypotheses are scored and prioritized; the runbook's contents beyond "OOM" and "loss explosion"; the 40/30/30 budget split; and the synthetic response surface. Treat these as placeholders to replace.
