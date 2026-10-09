"""Tests for DeepRetrieval implementation.

Run with:
    pytest test_deepretrieval.py -v
"""

import random
import sqlite3

import pytest
import torch

from deepretrieval import (
    ANSWER_CLOSE, ANSWER_OPEN, BOS, EOS, FLOOR_REWARD, FORMAT_BAD, FORMAT_OK, PAD,
    THINK_CLOSE, THINK_OPEN,
    BooleanBM25Engine, DeepRetrievalTrainer, PPOConfig, QuerySyntaxError, ToyIRWorld,
    ToyRetrievalTask, Vocab,
    compose_reward, compute_gae, evidence_seeking_reward, execute_sql, execution_accuracy,
    extract_query, first_answer_rank, generate, grpo_advantages, kl_k3, left_pad,
    literature_search_reward, make_critic, make_policy, masked_whiten, ndcg_at_k, ndcg_reward,
    parse_boolean_query, parse_response, ppo_policy_loss, ppo_value_loss, pretrain_format,
    query_terms, random_format_response, recall_at_k, response_logprobs, response_values,
    sql_reward,
)


# ---------------------------------------------------------------------------
# Output protocol
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_well_formed(self):
        p = parse_response("<think> reason here </think> <answer> a OR b </answer>")
        assert p.well_formed and p.think == "reason here" and p.answer == "a OR b"

    def test_leading_think_tag_is_optional(self):
        # The prompt ends with "<think>", so the model's text starts after it.
        p = parse_response("reason </think> <answer> a </answer>")
        assert p.well_formed and p.think == "reason"

    @pytest.mark.parametrize("text", [
        "",
        "<think> x </think>",                                           # no answer
        "<think> x </think> <answer> </answer>",                       # empty answer
        "<think> x </think> <answer> a </answer> trailing",            # junk after
        "<answer> a </answer> <think> x </think>",                     # wrong order
        "<think> x </think> <answer> a </answer> <answer> b </answer>",  # duplicated tag
        "<think> x </think> <answer> a",                               # unterminated (truncated)
    ])
    def test_malformed(self, text):
        assert not parse_response(text).well_formed

    def test_extract_query_json_and_raw(self):
        assert extract_query('{"query": "a AND b"}') == "a AND b"
        assert extract_query("  a AND b ") == "a AND b"
        assert extract_query('{"query": ') == ""
        assert extract_query('{"q": "x"}') == ""
        assert extract_query('{"query": 5}') == ""


# ---------------------------------------------------------------------------
# Boolean query language + engine
# ---------------------------------------------------------------------------

DOCS = [
    "aspirin headache pain",       # 0
    "aspirin heart attack",        # 1
    "ibuprofen headache",          # 2
    "heart surgery recovery",      # 3
    "pain management ibuprofen",   # 4
]


class TestBooleanQuery:
    def test_terms_and_lowercase(self):
        assert query_terms(parse_boolean_query("Aspirin AND (heart OR pain)")) == ["aspirin", "heart", "pain"]

    def test_and_binds_tighter_than_or(self):
        node = parse_boolean_query("a OR b AND c")
        assert node[0] == "or" and node[1][1][0] == "and"

    def test_juxtaposition_is_or(self):
        assert parse_boolean_query("a b")[0] == "or"

    @pytest.mark.parametrize("q", ["", "   ", "a AND", "OR a", "(a", "a)", "()", "a AND OR b", "(a OR)"])
    def test_syntax_errors(self, q):
        with pytest.raises(QuerySyntaxError):
            parse_boolean_query(q)

    def test_depth_and_length_limits(self):
        with pytest.raises(QuerySyntaxError):
            parse_boolean_query("(" * 100 + "a" + ")" * 100)
        with pytest.raises(QuerySyntaxError):
            parse_boolean_query(" ".join(["a"] * 1000))


class TestEngine:
    def setup_method(self):
        self.e = BooleanBM25Engine(DOCS)

    def test_and(self):
        assert set(self.e.search("aspirin AND headache", 10)) == {0}

    def test_or(self):
        assert set(self.e.search("aspirin OR ibuprofen", 10)) == {0, 1, 2, 4}

    def test_parentheses(self):
        assert set(self.e.search("(aspirin OR ibuprofen) AND headache", 10)) == {0, 2}

    def test_unknown_term_matches_nothing(self):
        assert self.e.search("zzz", 10) == []
        assert set(self.e.search("zzz OR heart", 10)) == {1, 3}
        assert self.e.search("zzz AND heart", 10) == []

    def test_top_k_truncates_and_ranks_by_bm25(self):
        res = self.e.search("headache pain", 10)
        assert res[0] == 0                       # matches both terms
        assert len(self.e.search("headache pain", 1)) == 1

    def test_deterministic(self):
        assert self.e.search("heart OR pain", 3) == self.e.search("heart OR pain", 3)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_recall(self):
        assert recall_at_k([1, 2, 3, 4], {1, 3, 9}, 4) == pytest.approx(2 / 3)
        assert recall_at_k([1, 2, 3, 4], {1, 3, 9}, 2) == pytest.approx(1 / 3)   # cutoff matters
        assert recall_at_k([], {1}, 5) == 0.0
        assert recall_at_k([1], set(), 5) == 0.0

    def test_first_answer_rank(self):
        docs = ["nothing here", "Anthony Kennedy served", "other"]
        assert first_answer_rank([0, 1, 2], docs, ["anthony kennedy"]) == 2   # case-insensitive
        assert first_answer_rank([0, 2], docs, ["anthony kennedy"]) is None
        assert first_answer_rank([1], docs, ["", "kennedy"]) == 1            # empty answer ignored

    def test_ndcg_perfect_and_worse(self):
        qrels = {1: 2.0, 2: 1.0}
        assert ndcg_at_k([1, 2, 9], qrels, 3) == pytest.approx(1.0)
        assert ndcg_at_k([2, 1, 9], qrels, 3) < 1.0
        assert ndcg_at_k([9, 8, 7], qrels, 3) == 0.0
        assert ndcg_at_k([1], {}, 3) == 0.0

    def test_ndcg_cutoff(self):
        assert ndcg_at_k([9, 1], {1: 1.0}, 1) == 0.0
        assert ndcg_at_k([9, 1], {1: 1.0}, 2) > 0.0

    def test_ndcg_reward_is_the_metric(self):
        assert ndcg_reward(0.37) == 0.37


# ---------------------------------------------------------------------------
# Reward tables (paper Table 8)
# ---------------------------------------------------------------------------

class TestRewards:
    @pytest.mark.parametrize("recall,expected", [
        (1.0, 5.0), (0.7, 5.0), (0.699, 4.0), (0.5, 4.0), (0.499, 3.0), (0.4, 3.0),
        (0.399, 1.0), (0.3, 1.0), (0.299, 0.5), (0.1, 0.5), (0.099, 0.1), (0.05, 0.1),
        (0.049, -3.5), (0.0, -3.5),
    ])
    def test_literature_tiers(self, recall, expected):
        assert literature_search_reward(recall) == expected

    @pytest.mark.parametrize("rank,expected", [
        (1, 5.0), (5, 5.0), (6, 4.0), (20, 4.0), (21, 2.0), (50, 2.0), (51, 1.0), (100, 1.0),
        (101, 0.5), (1000, 0.5), (1001, 0.1), (3000, 0.1), (3001, -3.5), (None, -3.5),
    ])
    def test_evidence_tiers(self, rank, expected):
        assert evidence_seeking_reward(rank) == expected

    def test_literature_reward_monotone(self):
        grid = [i / 100 for i in range(101)]
        r = [literature_search_reward(x) for x in grid]
        assert all(a <= b for a, b in zip(r, r[1:]))

    def test_compose_adds_format_term(self):
        assert compose_reward(4.0, True) == (4.0 + FORMAT_OK, 4.0, FORMAT_OK)
        assert compose_reward(4.0, False)[0] == 4.0 + FORMAT_BAD

    def test_malformed_never_beats_wellformed_failure(self):
        # A broken answer must not out-score a well-formed answer that retrieves nothing.
        assert compose_reward(FLOOR_REWARD, False)[0] < compose_reward(FLOOR_REWARD, True)[0]


class TestSQL:
    def setup_method(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.executescript(
            "CREATE TABLE t (id INTEGER, name TEXT, v INTEGER);"
            "INSERT INTO t VALUES (1,'a',10),(2,'b',20),(3,'c',30);")
        self.gold = "SELECT name FROM t WHERE v > 15"

    def test_correct_even_if_written_differently(self):
        assert execution_accuracy(self.conn, "SELECT name FROM t WHERE v >= 20 ORDER BY id DESC", self.gold) == 1.0

    def test_wrong_result(self):
        assert execution_accuracy(self.conn, "SELECT name FROM t", self.gold) == 0.0

    @pytest.mark.parametrize("sql", ["SELEC name FROM t", "SELECT x FROM nope", "", "DROP TABLE t"])
    def test_errors_score_zero(self, sql):
        assert execution_accuracy(self.conn, sql, self.gold) == 0.0

    def test_executable_bonus_only_when_it_runs(self):
        assert sql_reward(self.conn, "SELECT name FROM t", self.gold, 0.3) == pytest.approx(0.3)
        assert sql_reward(self.conn, self.gold, self.gold, 0.3) == pytest.approx(1.3)
        assert sql_reward(self.conn, "SELECT x FROM nope", self.gold, 0.3) == 0.0
        assert sql_reward(self.conn, "SELECT name FROM t", self.gold) == 0.0   # no bonus by default

    def test_model_sql_cannot_mutate_db(self):
        assert execute_sql(self.conn, "DROP TABLE t") is None
        assert execute_sql(self.conn, "WITH x AS (SELECT 1) DELETE FROM t") is None
        assert len(execute_sql(self.conn, "SELECT * FROM t")) == 3
        assert execute_sql(self.conn, "SELECT 1; DROP TABLE t") is None   # multi-statement rejected


# ---------------------------------------------------------------------------
# Toy world / task
# ---------------------------------------------------------------------------

class TestToyWorld:
    def setup_method(self):
        self.world = ToyIRWorld(seed=0)
        self.task = ToyRetrievalTask(self.world)

    def test_topics_partition_the_corpus(self):
        all_ids = set().union(*self.world.topic_docs)
        assert all_ids == set(range(len(self.world.docs)))
        assert sum(len(s) for s in self.world.topic_docs) == len(self.world.docs)

    def test_original_query_is_underspecified(self):
        rng = random.Random(0)
        recalls = [self.task.original_query_recall(e) for e in self.task.sample_batch(100, rng)]
        assert 0.2 < sum(recalls) / len(recalls) < 0.8

    def test_oracle_expansion_reaches_full_recall(self):
        ex = self.world.sample_example(random.Random(1))
        q = " OR ".join(self.world.core[ex.topic])
        assert recall_at_k(self.world.engine.search(q, self.world.top_k), ex.relevant, self.world.top_k) > 0.95

    def test_reward_end_to_end(self):
        ex = self.world.sample_example(random.Random(2))
        good = " OR ".join(self.world.core[ex.topic])
        info = self.task.reward(ex, f"hmm {THINK_CLOSE} {ANSWER_OPEN} {good} {ANSWER_CLOSE}")
        assert info.well_formed and info.metric > 0.95 and info.total == 5.0 + FORMAT_OK
        assert info.query == good and info.query_len == len(good.split())

    def test_reward_json_answer(self):
        ex = self.world.sample_example(random.Random(2))
        good = " OR ".join(self.world.core[ex.topic])
        info = self.task.reward(ex, f'x {THINK_CLOSE} {ANSWER_OPEN} {{"query": "{good}"}} {ANSWER_CLOSE}')
        assert info.well_formed and info.metric > 0.95

    def test_reward_bad_query_syntax_is_format_failure(self):
        ex = self.world.sample_example(random.Random(2))
        info = self.task.reward(ex, f"x {THINK_CLOSE} {ANSWER_OPEN} ( a OR {ANSWER_CLOSE}")
        assert not info.well_formed and info.format == FORMAT_BAD and info.retrieval == FLOOR_REWARD

    def test_reward_truncated_response(self):
        ex = self.world.sample_example(random.Random(2))
        info = self.task.reward(ex, "")
        assert not info.well_formed and info.total == FLOOR_REWARD + FORMAT_BAD

    def test_vocab_roundtrip(self):
        v = self.task.vocab
        toks = ["<bos>", "OR", self.world.words[0]]
        assert v.decode(v.encode(toks)).split() == toks
        assert v.stoi["<pad>"] == PAD and v.stoi["<bos>"] == BOS and v.stoi["<eos>"] == EOS

    def test_prompt_ends_with_think(self):
        ex = self.world.sample_example(random.Random(0))
        ids = self.task.prompt_ids(ex)
        assert self.task.vocab.itos[ids[-1]] == THINK_OPEN and ids[0] == BOS


# ---------------------------------------------------------------------------
# Policy / generation / log-probs
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    world = ToyIRWorld(seed=0)
    task = ToyRetrievalTask(world)
    policy = make_policy(len(task.vocab), 64)
    pretrain_format(policy, task, steps=400)
    return world, task, policy


class TestGeneration:
    def _prompts(self, task, n=6):
        return left_pad([task.prompt_ids(e) for e in task.sample_batch(n, random.Random(0))])

    def test_shapes_and_mask(self, setup):
        _, task, policy = setup
        prompts = self._prompts(task)
        resp, mask = generate(policy, prompts, 30)
        assert resp.shape == mask.shape and resp.size(0) == 6 and resp.size(1) <= 30
        for row, m in zip(resp, mask):
            n = int(m.sum())
            assert (row[n:] == PAD).all()
            ended = n > 0 and int(row[n - 1]) == EOS
            assert ended or n == resp.size(1)
            if n > 1:
                assert (row[:n - 1] != EOS).all()    # <eos> only as the last real token

    def test_greedy_is_deterministic(self, setup):
        _, task, policy = setup
        prompts = self._prompts(task)
        a, _ = generate(policy, prompts, 30, greedy=True)
        b, _ = generate(policy, prompts, 30, greedy=True)
        assert torch.equal(a, b)

    def test_logprobs_match_sampling_distribution(self, setup):
        _, task, policy = setup
        prompts = self._prompts(task)
        resp, mask = generate(policy, prompts, 30, greedy=True)
        logp, ent = response_logprobs(policy, prompts, resp)
        assert logp.shape == resp.shape and (logp <= 1e-6).all() and (ent >= -1e-6).all()
        # Greedy tokens are argmax, so their probability is >= 1/V everywhere.
        assert ((logp * mask) >= -torch.log(torch.tensor(float(len(task.vocab)))) * mask - 1e-4).all()

    def test_logprobs_causal(self, setup):
        # Changing a later response token must not change earlier log-probs.
        _, task, policy = setup
        prompts = self._prompts(task, 2)
        resp, _ = generate(policy, prompts, 12, greedy=True)
        other = resp.clone()
        other[:, -1] = (other[:, -1] + 1) % len(task.vocab)
        a, _ = response_logprobs(policy, prompts, resp)
        b, _ = response_logprobs(policy, prompts, other)
        assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)

    def test_temperature_scales_entropy(self, setup):
        _, task, policy = setup
        prompts = self._prompts(task)
        resp, _ = generate(policy, prompts, 10, greedy=True)
        _, hot = response_logprobs(policy, prompts, resp, temperature=2.0)
        _, cold = response_logprobs(policy, prompts, resp, temperature=0.5)
        assert hot.mean() > cold.mean()

    def test_critic_values_shape(self, setup):
        _, task, policy = setup
        prompts = self._prompts(task)
        resp, _ = generate(policy, prompts, 10, greedy=True)
        v = response_values(make_critic(len(task.vocab), 64), prompts, resp)
        assert v.shape == resp.shape


class TestPretrain:
    def test_random_format_response_is_valid(self):
        world = ToyIRWorld()
        task = ToyRetrievalTask(world)
        rng = random.Random(0)
        for _ in range(200):
            toks = random_format_response(world.words, rng, ["t0w0", "n1"])
            assert toks[-1] == "<eos>"
            info = task.reward(world.sample_example(rng), " ".join(toks[:-1]))
            assert info.well_formed

    def test_policy_learns_the_format_but_not_the_task(self, setup):
        world, task, policy = setup
        tr = DeepRetrievalTrainer(policy, task, PPOConfig(max_new_tokens=40))
        out = tr.evaluate(n=100, greedy=False)
        assert out["valid_rate"] > 0.8
        assert out["metric"] < out["original_query_metric"]    # knows format, not retrieval


# ---------------------------------------------------------------------------
# RL math
# ---------------------------------------------------------------------------

class TestGAE:
    def test_matches_hand_computation(self):
        rewards = torch.tensor([[0.0, 0.0, 1.0]])
        values = torch.tensor([[0.5, 0.4, 0.3]])
        mask = torch.ones(1, 3)
        adv, ret = compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)
        # lam=1 => advantage = (reward-to-go) - V
        assert torch.allclose(adv, torch.tensor([[0.5, 0.6, 0.7]]))
        assert torch.allclose(ret, torch.tensor([[1.0, 1.0, 1.0]]))

    def test_lambda_zero_is_td_error(self):
        rewards = torch.tensor([[0.1, 0.2, 1.0]])
        values = torch.tensor([[0.5, 0.4, 0.3]])
        adv, _ = compute_gae(rewards, values, torch.ones(1, 3), gamma=0.9, lam=0.0)
        expected = torch.tensor([[0.1 + 0.9 * 0.4 - 0.5, 0.2 + 0.9 * 0.3 - 0.4, 1.0 - 0.3]])
        assert torch.allclose(adv, expected, atol=1e-6)

    def test_padding_is_ignored(self):
        rewards = torch.tensor([[0.0, 1.0, 9.0, 9.0]])
        values = torch.tensor([[0.2, 0.3, 7.0, 7.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        adv, _ = compute_gae(rewards, values, mask, 1.0, 1.0)
        short, _ = compute_gae(rewards[:, :2], values[:, :2], torch.ones(1, 2), 1.0, 1.0)
        assert torch.allclose(adv[:, :2], short) and (adv[:, 2:] == 0).all()

    def test_whiten(self):
        x = torch.randn(4, 6)
        mask = torch.ones(4, 6)
        mask[:, 4:] = 0
        w = masked_whiten(x, mask)
        valid = w[mask.bool()]
        assert abs(valid.mean()) < 1e-5 and abs(valid.std(unbiased=False) - 1) < 1e-3
        assert (w[:, 4:] == 0).all()


class TestGRPO:
    def test_group_normalised(self):
        scores = torch.tensor([1.0, 3.0, 5.0, 10.0, 10.0, 10.0])
        adv = grpo_advantages(scores, group_size=3)
        assert abs(adv[:3].mean()) < 1e-6 and abs(adv[:3].std() - 1) < 1e-3
        assert torch.allclose(adv[3:], torch.zeros(3), atol=1e-3)   # identical samples: no signal
        assert adv[2] > adv[1] > adv[0]


class TestPPOLoss:
    def test_on_policy_ratio_one(self):
        lp = torch.randn(3, 5)
        adv = torch.randn(3, 5)
        mask = torch.ones(3, 5)
        loss, clipfrac = ppo_policy_loss(lp, lp.clone(), adv, mask)
        assert loss.item() == pytest.approx(-(adv.mean().item()), abs=1e-5) and clipfrac == 0

    def test_clipping_blocks_gradient_for_positive_advantage(self):
        old = torch.zeros(1, 1)
        lp = torch.full((1, 1), 1.0, requires_grad=True)       # ratio e >> 1.2
        loss, clipfrac = ppo_policy_loss(lp, old, torch.ones(1, 1), torch.ones(1, 1), 0.2)
        loss.backward()
        assert lp.grad.abs().item() == 0 and clipfrac == 1

    def test_no_clipping_when_it_would_help_the_objective(self):
        # negative advantage and ratio > 1+eps: the unclipped (more pessimistic) term is used.
        old = torch.zeros(1, 1)
        lp = torch.full((1, 1), 1.0, requires_grad=True)
        loss, _ = ppo_policy_loss(lp, old, -torch.ones(1, 1), torch.ones(1, 1), 0.2)
        loss.backward()
        assert lp.grad.abs().item() > 0

    def test_mask_excludes_padding(self):
        lp = torch.zeros(1, 2)
        adv = torch.tensor([[1.0, 100.0]])
        loss, _ = ppo_policy_loss(lp, lp.clone(), adv, torch.tensor([[1.0, 0.0]]))
        assert loss.item() == pytest.approx(-1.0)

    def test_value_loss_zero_at_target_and_clipped(self):
        t = torch.ones(1, 3)
        m = torch.ones(1, 3)
        assert ppo_value_loss(t, t, t, m).item() == 0
        v = torch.full((1, 1), 5.0, requires_grad=True)
        loss = ppo_value_loss(v, torch.zeros(1, 1), torch.zeros(1, 1), torch.ones(1, 1), clip_range=0.5)
        assert loss.item() == pytest.approx(0.5 * 25)       # the max() keeps the larger, unclipped error

    def test_kl_k3_nonnegative_and_zero_when_equal(self):
        a = torch.randn(100)
        assert (kl_k3(a, a) == 0).all()
        assert (kl_k3(a, torch.randn(100)) >= 0).all()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def _fresh_trainer(seed=1, steps_pretrain=300, **cfg):
    torch.manual_seed(seed)
    task = ToyRetrievalTask(ToyIRWorld(seed=0))
    policy = make_policy(len(task.vocab))
    pretrain_format(policy, task, steps=steps_pretrain, seed=seed)
    return DeepRetrievalTrainer(policy, task, PPOConfig(seed=seed, **cfg))


class TestTrainer:
    def test_reference_policy_is_frozen_copy(self):
        tr = _fresh_trainer(steps_pretrain=20)
        assert all(not p.requires_grad for p in tr.ref.parameters())
        for a, b in zip(tr.policy.parameters(), tr.ref.parameters()):
            assert torch.equal(a, b)

    def test_step_updates_actor_critic_not_ref(self):
        tr = _fresh_trainer(steps_pretrain=20, batch_size=16, mini_batch_size=8)
        ref0 = [p.clone() for p in tr.ref.parameters()]
        actor0 = [p.clone() for p in tr.policy.parameters()]
        critic0 = [p.clone() for p in tr.critic.parameters()]
        row = tr.step()
        assert any(not torch.equal(a, b) for a, b in zip(actor0, tr.policy.parameters()))
        assert any(not torch.equal(a, b) for a, b in zip(critic0, tr.critic.parameters()))
        assert all(torch.equal(a, b) for a, b in zip(ref0, tr.ref.parameters()))
        for k in ("reward", "metric", "valid_rate", "think_len", "query_len", "kl", "clipfrac", "value_loss"):
            assert k in row

    def test_kl_is_zero_before_any_update(self):
        tr = _fresh_trainer(steps_pretrain=20, batch_size=16, mini_batch_size=16, actor_lr=0.0)
        row = tr.step()
        assert row["kl"] == pytest.approx(0.0, abs=1e-5)

    def test_grpo_has_no_critic(self):
        tr = _fresh_trainer(steps_pretrain=20, batch_size=16, mini_batch_size=8,
                            advantage="grpo", group_size=4)
        assert tr.critic is None
        row = tr.step()
        assert "value_loss" not in row

    def test_ppo_epochs_multiplies_updates(self):
        tr = _fresh_trainer(steps_pretrain=20, batch_size=16, mini_batch_size=8, ppo_epochs=2)
        tr.step()
        assert len(tr.history) == 1

    def test_rl_improves_retrieval_reward_without_reference_queries(self):
        # Core claim of the paper: retrieval reward alone teaches query expansion.
        tr = _fresh_trainer(seed=1, steps_pretrain=300)
        before = tr.evaluate(n=128, greedy=False, seed=7)
        tr.train(250)
        after = tr.evaluate(n=128, greedy=False, seed=7)
        assert after["reward"] > before["reward"] + 1.0
        assert after["metric"] > before["metric"] + 0.15
        assert after["valid_rate"] > 0.9

    def test_grpo_also_improves(self):
        tr = _fresh_trainer(seed=1, steps_pretrain=300, advantage="grpo", group_size=8)
        before = tr.evaluate(n=128, greedy=False, seed=7)
        tr.train(250)
        after = tr.evaluate(n=128, greedy=False, seed=7)
        assert after["metric"] > before["metric"] + 0.1
