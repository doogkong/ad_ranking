"""Tests for the REA (Ranking Engineer Agent) implementation.

Run with:
    pytest test_rea.py -v
"""

import json

import pytest

from rea import (
    COMBINATION,
    EXPLOITATION,
    PHASE_ORDER,
    VALIDATION,
    Action,
    BudgetGuard,
    ExperimentRecord,
    FailureRunbook,
    Guardrails,
    Hypothesis,
    HypothesisGenerator,
    InsightsDB,
    JobResult,
    Planner,
    PreflightChecklist,
    REAgent,
    SimulatedCluster,
    Status,
    config_key,
    default_baseline,
    default_knob_category,
    default_search_space,
    diff,
    memory_gb,
    run_with_hibernation,
    simulated_cost,
    simulated_metric,
)

SPACE = default_search_space()
CAT = default_knob_category()
BASE = default_baseline()


def hyp(**changes):
    return Hypothesis(
        label="+".join(f"{k}={v}" for k, v in sorted(changes.items())),
        changes=changes, source="research", category="architecture")


class StubResearch:
    """A research agent that proposes a fixed list of hypotheses."""

    def __init__(self, hyps):
        self.hyps = hyps

    def propose(self, baseline, insights, n, round_idx):
        return [h for h in self.hyps if not insights.has_config({**baseline, **h.changes})][:n]


def make_kw(**over):
    kw = dict(baseline=default_baseline(), search_space=default_search_space(),
              knob_category=default_knob_category(), cluster=SimulatedCluster(),
              total_budget=120.0, rounds=1, seed=0)
    kw.update(over)
    return kw


def rec(changes, status="success", delta=0.01, failure=None, cfg=None):
    return ExperimentRecord(
        label=str(changes), changes=changes, config_key=config_key(cfg or {**BASE, **changes}),
        status=status, delta=delta, failure=failure, gpu_hours=1.0, round_idx=0, phase=VALIDATION)


# ---------------------------------------------------------------------------
# Insights database
# ---------------------------------------------------------------------------

class TestInsightsDB:
    def test_effect_mean_and_count(self):
        db = InsightsDB([rec({"num_layers": 6}, delta=0.02), rec({"num_layers": 6}, delta=0.04,
                                                                   cfg={**BASE, "num_layers": 6, "lr": 3e-4})])
        mean, n = db.effect("num_layers", 6)
        assert n == 2 and mean == pytest.approx(0.03)

    def test_effect_attributes_multi_knob_delta_equally(self):
        db = InsightsDB([rec({"num_layers": 6, "lr": 3e-4}, delta=0.04)])
        assert db.effect("num_layers", 6)[0] == pytest.approx(0.02)
        assert db.effect("lr", 3e-4)[0] == pytest.approx(0.02)

    def test_effect_ignores_failures(self):
        db = InsightsDB([rec({"num_layers": 8}, status="failed", delta=None, failure="oom")])
        assert db.effect("num_layers", 8) == (0.0, 0)

    def test_known_bad_only_for_clean_single_knob_failures(self):
        db = InsightsDB([
            rec({"lr": 1e-2}, status="excluded", delta=None, failure="loss_explosion"),
            rec({"num_layers": 8, "hidden_dim": 1024}, status="excluded", delta=None, failure="oom"),
            rec({"use_bf16": True}, status="excluded", delta=None, failure="infra"),
        ])
        assert db.known_bad("lr", 1e-2)
        assert not db.known_bad("num_layers", 8)        # multi-knob: not attributable
        assert not db.known_bad("use_bf16", True)       # infra failures are not the config's fault

    def test_has_config(self):
        db = InsightsDB([rec({"num_layers": 6})])
        assert db.has_config({**BASE, "num_layers": 6})
        assert not db.has_config({**BASE, "num_layers": 2})

    def test_top_patterns_sorted(self):
        db = InsightsDB([rec({"num_layers": 6}, delta=0.01),
                         rec({"hidden_dim": 512}, delta=0.03, cfg={**BASE, "hidden_dim": 512})])
        top = db.top_patterns(2)
        assert top[0][:2] == ("hidden_dim", 512)
        assert top[0][2] > top[1][2]

    def test_roundtrip_through_json(self):
        db = InsightsDB([rec({"num_layers": 6}), rec({"lr": 1e-2}, "failed", None, "loss_explosion")])
        db2 = InsightsDB.from_dict(json.loads(json.dumps(db.to_dict())))
        assert db2.to_dict() == db.to_dict()


# ---------------------------------------------------------------------------
# Budget / runbook / preflight
# ---------------------------------------------------------------------------

class TestBudgetGuard:
    def test_reserve_and_refuse(self):
        g = BudgetGuard(10.0)
        assert g.reserve(6.0)
        assert not g.reserve(5.0)
        assert g.committed == 6.0 and g.remaining == 4.0

    def test_settle_refunds_overestimate(self):
        g = BudgetGuard(10.0)
        g.reserve(6.0)
        g.settle(6.0, 1.0)          # job failed early: only 1 GPU-h used
        assert g.committed == pytest.approx(1.0)
        assert g.reserve(9.0)


class TestFailureRunbook:
    def test_oom_and_explosion_excluded(self):
        rb = FailureRunbook()
        assert rb.decide("oom", 0, 0)[0] == Action.EXCLUDE
        assert rb.decide("loss_explosion", 0, 0)[0] == Action.EXCLUDE

    def test_infra_retried_then_excluded(self):
        rb = FailureRunbook(Guardrails(max_attempts_per_job=3, max_total_retries=8))
        assert rb.decide("infra", 0, 0)[0] == Action.RETRY
        assert rb.decide("infra", 1, 1)[0] == Action.RETRY
        assert rb.decide("infra", 2, 2)[0] == Action.EXCLUDE   # out of attempts

    def test_global_retry_guardrail(self):
        rb = FailureRunbook(Guardrails(max_attempts_per_job=5, max_total_retries=2))
        assert rb.decide("infra", 0, 2)[0] == Action.EXCLUDE

    def test_unknown_error_escalates(self):
        assert FailureRunbook().decide("segfault", 0, 0)[0] == Action.ESCALATE


class TestPreflight:
    def test_passes_by_default(self):
        assert PreflightChecklist().verify([hyp(num_layers=6)], "ads_ranking_models") == []

    def test_ungranted_item_fails(self):
        c = PreflightChecklist(items={"compute_budget_confirmed": False})
        assert any("compute_budget_confirmed" in f for f in c.verify([], "ads_ranking_models"))

    def test_wrong_codebase_fails(self):
        assert PreflightChecklist().verify([], "some_other_repo")

    def test_disallowed_knob_fails(self):
        c = PreflightChecklist(allowed_knobs={"lr"})
        assert c.verify([hyp(num_layers=6)], "ads_ranking_models")
        assert c.verify([hyp(lr=3e-4)], "ads_ranking_models") == []


# ---------------------------------------------------------------------------
# Simulated cluster
# ---------------------------------------------------------------------------

class TestSimulatedCluster:
    def test_metric_deterministic(self):
        assert simulated_metric(BASE) == simulated_metric(dict(BASE))

    def test_bf16_cheaper_and_lighter(self):
        bf = {**BASE, "use_bf16": True}
        assert simulated_cost(bf) < simulated_cost(BASE)
        assert memory_gb(bf) < memory_gb(BASE)

    def test_synergy_bonus(self):
        a = simulated_metric({**BASE, "feature_interaction": "mixer"})
        b = simulated_metric({**BASE, "use_bf16": True})
        both = simulated_metric({**BASE, "feature_interaction": "mixer", "use_bf16": True})
        assert both - simulated_metric(BASE) > (a - simulated_metric(BASE)) + (b - simulated_metric(BASE)) + 0.005

    def test_job_runs_until_clock_advances(self):
        cl = SimulatedCluster()
        cfg = {**BASE, "num_layers": 6}
        j = cl.submit(cfg, attempt=1)       # attempt>0 skips the transient-infra failure
        assert cl.poll(j).state == "running"
        cl.advance()
        assert cl.poll(j).state == "succeeded"

    def test_oom(self):
        cl = SimulatedCluster()
        j = cl.submit({**BASE, "num_layers": 8, "hidden_dim": 1024})
        cl.advance()
        r = cl.poll(j)
        assert (r.state, r.error) == ("failed", "oom")

    def test_loss_explosion_needs_warmup(self):
        cl = SimulatedCluster()
        bad = cl.submit({**BASE, "lr": 1e-2, "warmup_steps": 0})
        ok = cl.submit({**BASE, "lr": 1e-2, "warmup_steps": 2000}, attempt=1)
        cl.advance(); cl.advance(); cl.advance(); cl.advance()
        assert cl.poll(bad).error == "loss_explosion"
        assert cl.poll(ok).state in ("running", "succeeded")

    def test_infra_failure_is_transient(self):
        cl = SimulatedCluster()
        flaky = next(c for c in ({**BASE, "num_layers": 6, "lr": lr, "seq_len": s}
                                 for lr in SPACE["lr"][:3] for s in SPACE["seq_len"])
                     if cl.is_infra_flaky(c) and memory_gb(c) < 80)
        j0 = cl.submit(flaky, attempt=0)
        j1 = cl.submit(flaky, attempt=1)
        for _ in range(5):
            cl.advance()
        assert cl.poll(j0).error == "infra"
        assert cl.poll(j1).state == "succeeded"

    def test_failed_jobs_cost_less(self):
        cl = SimulatedCluster()
        cfg = {**BASE, "num_layers": 8, "hidden_dim": 1024}
        j = cl.submit(cfg)
        cl.advance()
        assert cl.poll(j).gpu_hours < 0.2 * simulated_cost(cfg)


# ---------------------------------------------------------------------------
# Hypothesis engine
# ---------------------------------------------------------------------------

class TestHypothesisGenerator:
    def test_fresh_db_gives_research_only(self):
        gen = HypothesisGenerator(SPACE, CAT, InsightsDB(), seed=0)
        hs = gen.generate(BASE, 6)
        assert hs and all(h.source == "research" for h in hs)

    def test_history_source_uses_positive_past_effects(self):
        db = InsightsDB([rec({"num_layers": 6}, delta=0.02),
                         rec({"hidden_dim": 512}, delta=-0.01, cfg={**BASE, "hidden_dim": 512})])
        gen = HypothesisGenerator(SPACE, CAT, db, seed=0)
        hist = gen.from_history({**BASE, "num_layers": 4}, 5)
        # num_layers=6 already *tried* at this baseline, so excluded; nothing else was positive
        assert hist == []
        hist2 = gen.from_history({**BASE, "lr": 3e-4}, 5)   # different baseline: num_layers=6 is new
        assert [h.changes for h in hist2] == [{"num_layers": 6}]
        assert hist2[0].source == "history"

    def test_dual_sources_interleave(self):
        db = InsightsDB([rec({"num_layers": 6}, delta=0.02, cfg={**BASE, "num_layers": 6})])
        gen = HypothesisGenerator(SPACE, CAT, db, seed=0)
        hs = gen.generate({**BASE, "lr": 3e-4}, 6)
        assert {"history", "research"} <= {h.source for h in hs}

    def test_skips_tried_configs(self):
        db = InsightsDB([rec({"num_layers": 6})])
        hs = HypothesisGenerator(SPACE, CAT, db, seed=0).generate(BASE, 50)
        assert all(h.changes != {"num_layers": 6} for h in hs)

    def test_skips_known_bad(self):
        db = InsightsDB([rec({"lr": 1e-2}, "excluded", None, "loss_explosion", cfg={**BASE, "lr": 1e-2})])
        hs = HypothesisGenerator(SPACE, CAT, db, seed=0).generate(BASE, 100)
        assert all(h.changes.get("lr") != 1e-2 for h in hs)

    def test_no_duplicates_and_respects_n(self):
        hs = HypothesisGenerator(SPACE, CAT, InsightsDB(), seed=3).generate(BASE, 8)
        assert len(hs) == 8
        assert len({config_key(h.changes) for h in hs}) == 8

    def test_deterministic_given_seed(self):
        a = HypothesisGenerator(SPACE, CAT, InsightsDB(), seed=5).generate(BASE, 6)
        b = HypothesisGenerator(SPACE, CAT, InsightsDB(), seed=5).generate(BASE, 6)
        c = HypothesisGenerator(SPACE, CAT, InsightsDB(), seed=6).generate(BASE, 6)
        assert [h.label for h in a] == [h.label for h in b]
        assert [h.label for h in a] != [h.label for h in c]

    def test_research_proposes_arch_x_efficiency_pairs(self):
        hs = HypothesisGenerator(SPACE, CAT, InsightsDB(), seed=0).research.propose(BASE, InsightsDB(), 500, 0)
        assert any(h.category == "mixed" for h in hs)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class TestPlanner:
    def test_phase_caps_are_cumulative(self):
        caps = Planner(simulated_cost).cumulative_caps(100.0)
        assert caps[VALIDATION] == pytest.approx(40.0)
        assert caps[COMBINATION] == pytest.approx(70.0)
        assert caps[EXPLOITATION] == pytest.approx(100.0)

    def test_plan_truncates_to_validation_budget(self):
        hs = [hyp(num_layers=6), hyp(hidden_dim=1024), hyp(lr=3e-4)]
        p = Planner(simulated_cost).make_plan(hs, BASE, round_budget=20.0, round_idx=0)
        assert p.est_validation_cost <= 0.4 * 20.0 + 1e-9
        assert "hidden_dim=1024" in p.dropped          # too expensive to validate
        assert {h.label for h in p.hypotheses} | set(p.dropped) == {h.label for h in hs}

    def test_reserved_cost_shrinks_the_plan(self):
        hs = [hyp(num_layers=6), hyp(lr=3e-4)]
        a = Planner(simulated_cost).make_plan(hs, BASE, 40.0, 0, reserved=0.0)
        b = Planner(simulated_cost).make_plan(hs, BASE, 40.0, 0, reserved=15.5)
        assert len(b.hypotheses) < len(a.hypotheses)

    def test_combination_only_positive_and_disjoint(self):
        validated = [
            {"label": "a", "changes": {"num_layers": 6}, "delta": 0.02},
            {"label": "b", "changes": {"num_layers": 8}, "delta": 0.03},    # conflicts with a
            {"label": "c", "changes": {"lr": 3e-4}, "delta": 0.01},
            {"label": "d", "changes": {"seq_len": 64}, "delta": -0.02},     # negative
        ]
        out = Planner(simulated_cost).combination_candidates(validated, BASE, 1000.0, 10)
        changes = [c for _, c in out]
        assert {"num_layers": 6, "lr": 3e-4} in changes
        assert {"num_layers": 8, "lr": 3e-4} in changes
        assert not any("seq_len" in c for c in changes)
        assert not any(c.get("num_layers") in (6, 8) and len(c) == 1 for c in changes)
        assert all(not ({"num_layers": 6} .items() <= c.items() and {"num_layers": 8}.items() <= c.items())
                   for c in changes)

    def test_combination_ranked_by_summed_gain_and_budgeted(self):
        validated = [
            {"label": "a", "changes": {"num_layers": 6}, "delta": 0.02},
            {"label": "c", "changes": {"lr": 3e-4}, "delta": 0.01},
            {"label": "e", "changes": {"seq_len": 256}, "delta": 0.03},
        ]
        out = Planner(simulated_cost).combination_candidates(validated, BASE, 1000.0, 4)
        gain = {"num_layers": 0.02, "lr": 0.01, "seq_len": 0.03}
        sums = [sum(gain[k] for k in ch) for _, ch in out]
        assert sums == sorted(sums, reverse=True)                      # best predicted gain first
        assert out[0][1] == {"num_layers": 6, "lr": 3e-4, "seq_len": 256}   # the 3-way combo
        assert out[1][1] == {"num_layers": 6, "seq_len": 256}               # best pair next
        assert len(Planner(simulated_cost).combination_candidates(validated, BASE, 1000.0, 2)) == 2
        assert Planner(simulated_cost).combination_candidates(validated, BASE, 0.1, 5) == []

    def test_exploitation_steps_to_adjacent_untried_values(self):
        best = {**BASE, "num_layers": 6}
        out = Planner(simulated_cost).exploitation_candidates(
            best, BASE, SPACE, InsightsDB(), {config_key(best)}, 1000.0, 50)
        cfgs = [{**BASE, **ch} for _, ch in out]
        assert all(sum(best[k] != c[k] for k in best) == 1 for c in cfgs)   # one knob moved
        assert {**best, "num_layers": 8} in cfgs and {**best, "num_layers": 4} in cfgs
        assert {**best, "num_layers": 2} not in cfgs                        # not adjacent

    def test_exploitation_skips_tried(self):
        best = {**BASE, "num_layers": 6}
        tried = {config_key(best), config_key({**best, "num_layers": 8})}
        out = Planner(simulated_cost).exploitation_candidates(
            best, BASE, SPACE, InsightsDB(), tried, 1000.0, 50)
        assert {**best, "num_layers": 8} not in [{**BASE, **ch} for _, ch in out]

    def test_exploitation_prefers_moves_history_likes(self):
        best = {**BASE, "num_layers": 6}
        db = InsightsDB([rec({"seq_len": 256}, delta=0.05, cfg={**BASE, "seq_len": 256})])
        out = Planner(simulated_cost).exploitation_candidates(
            best, BASE, SPACE, db, {config_key(best)}, 1000.0, 1)
        assert out[0][1]["seq_len"] == 256


# ---------------------------------------------------------------------------
# End-to-end agent
# ---------------------------------------------------------------------------

class TestAgentEndToEnd:
    def test_improves_over_baseline_and_finishes(self):
        agent = REAgent(**make_kw())
        assert agent.run() == Status.DONE
        st = agent.state
        assert st.baseline_metric > st.history[0]["baseline_metric"]
        assert st.events

    @pytest.mark.parametrize("budget,seed", [(30.0, 0), (60.0, 1), (120.0, 2), (250.0, 3)])
    def test_never_exceeds_compute_budget(self, budget, seed):
        agent = REAgent(**make_kw(total_budget=budget, rounds=2, seed=seed))
        agent.run()
        assert agent.guard.committed <= budget + 1e-6

    def test_committed_matches_actual_job_costs(self):
        agent = REAgent(**make_kw(rounds=2))
        agent.run()
        assert agent.guard.committed == pytest.approx(sum(j.actual_cost for j in agent.state.jobs))

    def test_phases_run_in_order_each_round(self):
        agent = REAgent(**make_kw(rounds=2, total_budget=150.0))
        agent.run()
        for r in range(2):
            idx = [PHASE_ORDER.index(j.phase) for j in agent.state.jobs if j.round_idx == r]
            assert idx == sorted(idx)

    def test_combination_built_only_from_validated_positives(self):
        agent = REAgent(**make_kw())
        agent.run()
        positive = [v["changes"] for v in agent.state.validated if v["delta"] > 0]
        combos = [j for j in agent.state.jobs if j.phase == COMBINATION]
        assert combos, "expected the combination phase to run"
        for j in combos:
            for k, v in j.changes.items():
                assert any(p.get(k) == v for p in positive)
            assert len(j.changes) >= 2

    def test_never_repeats_an_experiment(self):
        agent = REAgent(**make_kw(rounds=3, total_budget=200.0))
        agent.run()
        keys = [config_key(j.config) for j in agent.state.jobs]
        assert len(keys) == len(set(keys))

    def test_every_finished_job_is_logged_to_insights(self):
        agent = REAgent(**make_kw(rounds=2))
        agent.run()
        non_baseline = [j for j in agent.state.jobs if j.label != "baseline" and j.status != "running"]
        assert len(agent.insights.records) == len(non_baseline)
        assert agent.insights.top_patterns(1)

    def test_baseline_metric_never_decreases_across_rounds(self):
        agent = REAgent(**make_kw(rounds=3, total_budget=200.0))
        agent.run()
        ms = [h["baseline_metric"] for h in agent.state.history]
        assert ms == sorted(ms)

    def test_later_rounds_use_the_history_source(self):
        plans = []
        agent = REAgent(**make_kw(rounds=2, seed=1, approve=lambda p: plans.append(p) or True))
        agent.run()
        assert len(plans) == 2
        assert all(h.source == "research" for h in plans[0].hypotheses)
        assert any(h.source == "history" for h in plans[1].hypotheses)

    # ---- failure handling -------------------------------------------------

    def test_oom_hypothesis_is_excluded_and_remembered(self):
        oom = hyp(num_layers=8, hidden_dim=1024)
        agent = REAgent(**make_kw(total_budget=300.0, research_agent=StubResearch([oom, hyp(num_layers=6)])))
        assert agent.run() == Status.DONE
        job = next(j for j in agent.state.jobs if j.label == oom.label)
        assert job.status == "excluded" and job.failure == "oom"
        assert agent.state.best["label"] != oom.label
        assert any(r.failure == "oom" and r.status == "excluded" for r in agent.insights.records)

    def test_loss_explosion_is_excluded_and_known_bad(self):
        agent = REAgent(**make_kw(research_agent=StubResearch([hyp(lr=1e-2), hyp(num_layers=6)])))
        agent.run()
        job = next(j for j in agent.state.jobs if j.changes == {"lr": 1e-2})
        assert job.status == "excluded" and job.failure == "loss_explosion"
        assert agent.insights.known_bad("lr", 1e-2)

    def test_transient_infra_failure_is_retried_automatically(self):
        cl = SimulatedCluster()
        flaky = next(h for h in (hyp(num_layers=6, seq_len=s, lr=lr)
                                 for s in SPACE["seq_len"] for lr in SPACE["lr"][:3])
                     if cl.is_infra_flaky({**BASE, **h.changes}) and memory_gb({**BASE, **h.changes}) < 80)
        agent = REAgent(**make_kw(cluster=cl, research_agent=StubResearch([flaky])))
        assert agent.run() == Status.DONE
        job = next(j for j in agent.state.jobs if j.label == flaky.label)
        assert job.attempt >= 1 and job.status == "succeeded"
        assert agent.state.retries_used >= 1

    def test_unknown_failure_escalates_to_human(self):
        class Crashy(SimulatedCluster):
            def _outcome(self, cfg, attempt):
                if cfg["num_layers"] == 6:
                    return {"state": "failed", "error": "segfault", "gpu_hours": 0.1, "ticks": 1}
                return super()._outcome(cfg, attempt)

        agent = REAgent(**make_kw(cluster=Crashy(), research_agent=StubResearch([hyp(num_layers=6)])))
        assert agent.run() == Status.ESCALATED
        assert "segfault" in agent.state.halt_reason

    def test_failed_baseline_escalates(self):
        agent = REAgent(**make_kw(cluster=SimulatedCluster(mem_limit_gb=1.0)))
        assert agent.run() == Status.ESCALATED
        assert "baseline" in agent.state.halt_reason

    # ---- human oversight --------------------------------------------------

    def test_engineer_rejection_halts_before_any_job(self):
        agent = REAgent(**make_kw(approve=lambda plan: False))
        assert agent.run() == Status.HALTED
        assert agent.state.jobs == [] and "not approved" in agent.state.halt_reason

    def test_approval_sees_a_cost_estimate(self):
        seen = []
        REAgent(**make_kw(approve=lambda plan: seen.append(plan) or False)).run()
        assert seen[0].est_validation_cost > 0 and "GPU-h" in seen[0].summary()

    def test_preflight_failure_halts_before_any_job(self):
        bad = PreflightChecklist(items={"codebase_access_reviewed": False})
        agent = REAgent(**make_kw(checklist=bad))
        assert agent.run() == Status.HALTED
        assert agent.state.jobs == [] and "preflight" in agent.state.halt_reason

    def test_agent_refuses_a_codebase_outside_its_allowlist(self):
        agent = REAgent(**make_kw(codebase="payments_service"))
        assert agent.run() == Status.HALTED and agent.state.jobs == []

    def test_disallowed_knob_blocks_the_plan(self):
        agent = REAgent(**make_kw(checklist=PreflightChecklist(allowed_knobs={"lr"})))
        assert agent.run() == Status.HALTED

    def test_tiny_budget_degrades_gracefully(self):
        agent = REAgent(**make_kw(total_budget=1.0))
        assert agent.run() in (Status.DONE, Status.HALTED)
        assert agent.guard.committed <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Hibernate-and-wake
# ---------------------------------------------------------------------------

class TestHibernateAndWake:
    def test_step_hibernates_and_checkpoints(self, tmp_path):
        path = str(tmp_path / "state.json")
        agent = REAgent(state_path=path, **make_kw())
        assert agent.step() == Status.HIBERNATED
        with open(path) as f:
            d = json.load(f)
        assert d["status"] == "hibernated"
        assert d["jobs"] and all(j["status"] == "running" for j in d["jobs"])

    def test_resume_from_checkpoint_finishes_the_run(self, tmp_path):
        path = str(tmp_path / "state.json")
        kw = make_kw()
        first = REAgent(state_path=path, **kw)
        assert first.step() == Status.HIBERNATED
        del first                                   # the agent process is gone
        kw["cluster"].advance()
        agent = REAgent.resume(path, **kw)
        assert agent.state.jobs and agent.guard.committed > 0
        assert agent.run() == Status.DONE

    def test_hibernating_at_every_wait_matches_a_continuous_run(self, tmp_path):
        def summary(a):
            return ([(j.label, j.status, j.metric, j.attempt) for j in a.state.jobs],
                    a.state.best["label"], a.state.baseline_metric, round(a.guard.committed, 9))

        cont = REAgent(**make_kw(rounds=2, seed=4))
        cont.run()

        path = str(tmp_path / "state.json")
        kw = make_kw(rounds=2, seed=4)
        hib = run_with_hibernation(
            lambda resume: REAgent.resume(path, **kw) if resume else REAgent(state_path=path, **kw))
        assert hib.state.status == Status.DONE.value
        assert summary(hib) == summary(cont)

    def test_checkpoint_roundtrips_insights(self, tmp_path):
        path = str(tmp_path / "state.json")
        kw = make_kw(rounds=2)
        hib = run_with_hibernation(
            lambda resume: REAgent.resume(path, **kw) if resume else REAgent(state_path=path, **kw))
        again = REAgent.resume(path, **make_kw(rounds=2))
        assert again.insights.to_dict() == hib.insights.to_dict()
        assert again.state.status == Status.DONE.value

    def test_no_state_path_still_works_in_memory(self):
        assert REAgent(**make_kw()).run() == Status.DONE


def test_diff_helper():
    assert diff({"a": 1, "b": 2}, {"a": 1, "b": 3}) == {"b": 2}
