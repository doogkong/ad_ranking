"""DeepRetrieval: RL-trained LLM query generation with retrieval metrics as reward.

Reference implementation of "DeepRetrieval: Hacking Real Search Engines and
Retrievers with Large Language Models via Reinforcement Learning"
(Jiang, Lin, Cao, Tian, Kang, Wang, Sun, Han; arXiv 2503.00223).

The paper trains Qwen2.5-3B-Instruct with PPO (verl) so that it rewrites a user
query into an augmented query q'. The reward is the retrieval metric achieved by
a *real* search engine / retriever / database on q' (Eq. 1):

    r(q, q') = r_retrieval(q, q') + r_format(q')

and the policy maximises (Eq. 2)  E[ r(q,q') - beta * log pi(q'|q)/pi_ref(q'|q) ]
with PPO (Eqs. 3-4: clipped surrogate + value loss + entropy bonus, GAE).
The model first reasons in <think>...</think>, then emits the query in
<answer>...</answer>.

What is here (everything is PyTorch / stdlib; no GPU, no LLM, no network):

  * Output protocol:   parse_response, extract_query, format reward.
  * Retrieval stack:   boolean-query parser + BM25 ranker (BooleanBM25Engine)
                       standing in for PubMed / BM25 / dense retrievers.
  * Metrics+rewards:   Recall@K, H@N, NDCG@K, execution accuracy, and the exact
                       tiered reward tables of the paper's Table 8 (+0.3 SQL
                       executability bonus for BIRD).
  * RL machinery:      actor + separate critic, token-level KL-in-reward, GAE,
                       PPO clipped objective (Eq. 3/4) with entropy bonus and
                       clipped value loss, plus a GRPO advantage option.
  * A toy world:       a synthetic corpus where the user's query is under-
                       specified, so the policy must learn to *expand* it.
                       A tiny GRU LM (the "base model") is first taught the
                       output format only, then trained purely from retrieval
                       reward -- no reference queries anywhere.

What is NOT here: the 3B LLM, vLLM/verl infrastructure, real PubMed /
ClinicalTrials.gov / BEIR / BIRD / Spider data. Numbers from the toy world say
nothing about the paper's results; they only show the mechanism learns.
"""

from __future__ import annotations

import json
import math
import random
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Output protocol: <think> ... </think> <answer> ... </answer>
# ---------------------------------------------------------------------------

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"


@dataclass
class ParsedResponse:
    think: str
    answer: str
    well_formed: bool


def parse_response(text: str) -> ParsedResponse:
    """Split a model response into reasoning and answer.

    Well-formed means exactly one <think>..</think> followed by exactly one
    <answer>..</answer>, nothing after it, and a non-empty answer. The prompt in
    the paper ends with "<think>" so callers may pass the text with or without
    the leading tag; it is added if missing.
    """
    text = text.strip()
    if not text.startswith(THINK_OPEN):
        text = THINK_OPEN + " " + text
    tags = [THINK_OPEN, THINK_CLOSE, ANSWER_OPEN, ANSWER_CLOSE]
    if any(text.count(t) != 1 for t in tags):
        return ParsedResponse("", "", False)
    i_to, i_tc = text.index(THINK_OPEN), text.index(THINK_CLOSE)
    i_ao, i_ac = text.index(ANSWER_OPEN), text.index(ANSWER_CLOSE)
    if not (i_to < i_tc < i_ao < i_ac) or text[i_ac + len(ANSWER_CLOSE):].strip():
        return ParsedResponse("", "", False)
    think = text[i_to + len(THINK_OPEN):i_tc].strip()
    answer = text[i_ao + len(ANSWER_OPEN):i_ac].strip()
    return ParsedResponse(think, answer, bool(answer))


def extract_query(answer: str) -> str:
    """The paper's prompts ask for `{"query": "..."}`; raw text is also accepted."""
    answer = answer.strip()
    if answer.startswith("{"):
        try:
            obj = json.loads(answer)
        except json.JSONDecodeError:
            return ""
        q = obj.get("query", "") if isinstance(obj, dict) else ""
        return q.strip() if isinstance(q, str) else ""
    return answer


# ---------------------------------------------------------------------------
# 2. Boolean query language + BM25 engine (stand-in for the real search engine)
# ---------------------------------------------------------------------------

class QuerySyntaxError(ValueError):
    pass


_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")
MAX_QUERY_TOKENS = 256
MAX_NESTING = 32


class _Parser:
    """Grammar:  or  := and ((OR)? and)*      (juxtaposition = implicit OR, as in BM25)
                 and := atom (AND atom)*
                 atom := TERM | '(' or ')'
    """

    def __init__(self, toks: List[str]):
        self.toks, self.pos, self.depth = toks, 0, 0

    def peek(self) -> Optional[str]:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def parse(self):
        if not self.toks:
            raise QuerySyntaxError("empty query")
        node = self.or_expr()
        if self.pos != len(self.toks):
            raise QuerySyntaxError("unbalanced ')'")
        return node

    def or_expr(self):
        parts = [self.and_expr()]
        while self.peek() is not None and self.peek() != ")":
            if self.peek() == "OR":
                self.pos += 1
            parts.append(self.and_expr())
        return parts[0] if len(parts) == 1 else ("or", parts)

    def and_expr(self):
        parts = [self.atom()]
        while self.peek() == "AND":
            self.pos += 1
            parts.append(self.atom())
        return parts[0] if len(parts) == 1 else ("and", parts)

    def atom(self):
        t = self.peek()
        if t is None:
            raise QuerySyntaxError("unexpected end of query")
        self.pos += 1
        if t == "(":
            self.depth += 1
            if self.depth > MAX_NESTING:
                raise QuerySyntaxError("nesting too deep")
            node = self.or_expr()
            if self.peek() != ")":
                raise QuerySyntaxError("missing ')'")
            self.pos += 1
            self.depth -= 1
            return node
        if t in (")", "AND", "OR"):
            raise QuerySyntaxError(f"unexpected {t!r}")
        return ("term", t.lower())


def parse_boolean_query(query: str):
    toks = _TOKEN_RE.findall(query)
    if len(toks) > MAX_QUERY_TOKENS:
        raise QuerySyntaxError("query too long")
    return _Parser(toks).parse()


def query_terms(node) -> List[str]:
    if node[0] == "term":
        return [node[1]]
    return [t for child in node[1] for t in query_terms(child)]


class BooleanBM25Engine:
    """Boolean match (AND/OR/parentheses) over an inverted index, ranked by BM25.

    Mirrors how a real engine treats a generated query: the boolean structure
    decides *which* documents match, BM25 decides their order, and only the
    top-K are returned (the paper's Recall@K / NDCG@K are computed on that list).
    """

    def __init__(self, docs: Sequence[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.doc_len: List[int] = []
        self.postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        for i, d in enumerate(docs):
            toks = d.lower().split()
            self.doc_len.append(len(toks))
            for t in toks:
                self.postings[t][i] = self.postings[t].get(i, 0) + 1
        self.avg_len = sum(self.doc_len) / max(self.N, 1)

    def _match(self, node) -> Set[int]:
        kind = node[0]
        if kind == "term":
            return set(self.postings.get(node[1], ()))
        sets = [self._match(c) for c in node[1]]
        return set.intersection(*sets) if kind == "and" else set.union(*sets)

    def _idf(self, term: str) -> float:
        n = len(self.postings.get(term, ()))
        return math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))

    def _bm25(self, doc: int, terms: Set[str]) -> float:
        s = 0.0
        for t in terms:
            tf = self.postings.get(t, {}).get(doc, 0)
            if tf:
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[doc] / self.avg_len)
                s += self._idf(t) * tf * (self.k1 + 1) / norm
        return s

    def search(self, query: str, k: int) -> List[int]:
        """Top-k doc ids. Raises QuerySyntaxError on malformed queries."""
        node = parse_boolean_query(query)
        matched = self._match(node)
        terms = set(query_terms(node))
        ranked = sorted(matched, key=lambda d: (-self._bm25(d, terms), d))
        return ranked[:k]


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: Sequence[int], relevant: Set[int], k: int) -> float:
    """Fraction of ground-truth documents found in the top-k (literature search)."""
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def first_answer_rank(retrieved: Sequence[int], docs: Sequence[str],
                      answers: Sequence[str]) -> Optional[int]:
    """1-based rank of the first retrieved doc containing any answer span (H@N)."""
    answers = [a.lower() for a in answers if a]
    for rank, d in enumerate(retrieved, start=1):
        text = docs[d].lower()
        if any(a in text for a in answers):
            return rank
    return None


def ndcg_at_k(retrieved: Sequence[int], qrels: Dict[int, float], k: int) -> float:
    """NDCG@k with linear gain and log2 discount (classic IR / BEIR)."""
    dcg = sum(qrels.get(d, 0.0) / math.log2(i + 2) for i, d in enumerate(retrieved[:k]))
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def execute_sql(conn: sqlite3.Connection, sql: str) -> Optional[List[tuple]]:
    """Run a single read-only SELECT; None on any error (syntax, missing table...)."""
    sql = sql.strip().rstrip(";")
    if not sql.lower().startswith(("select", "with")):
        return None
    try:
        conn.execute("PRAGMA query_only = ON")   # model-written SQL must never mutate the DB
        return conn.execute(sql).fetchall()
    except (sqlite3.Error, ValueError):
        return None


def execution_accuracy(conn: sqlite3.Connection, pred_sql: str, gold_sql: str) -> float:
    """1.0 if the predicted SQL returns the same result set as the gold SQL."""
    pred, gold = execute_sql(conn, pred_sql), execute_sql(conn, gold_sql)
    if pred is None or gold is None:
        return 0.0
    return float(set(pred) == set(gold))


# ---------------------------------------------------------------------------
# 4. Reward functions (Table 8 of the paper) + format reward
# ---------------------------------------------------------------------------

FLOOR_REWARD = -3.5

# (threshold, reward): first row whose threshold is met wins.
LITERATURE_TIERS: Tuple[Tuple[float, float], ...] = (
    (0.7, 5.0), (0.5, 4.0), (0.4, 3.0), (0.3, 1.0), (0.1, 0.5), (0.05, 0.1),
)
# (max rank, reward): first row whose rank bound is satisfied wins.
EVIDENCE_TIERS: Tuple[Tuple[int, float], ...] = (
    (5, 5.0), (20, 4.0), (50, 2.0), (100, 1.0), (1000, 0.5), (3000, 0.1),
)
SQL_EXECUTABLE_BONUS = 0.3   # BIRD only: valid SQL (no syntax error / missing table)

FORMAT_OK, FORMAT_BAD = 0.0, -1.0   # r_format: paper shows only a check/cross; values are ours


def literature_search_reward(recall: float) -> float:
    for thr, r in LITERATURE_TIERS:
        if recall >= thr:
            return r
    return FLOOR_REWARD


def evidence_seeking_reward(rank: Optional[int]) -> float:
    if rank is not None:
        for bound, r in EVIDENCE_TIERS:
            if rank <= bound:
                return r
    return FLOOR_REWARD


def ndcg_reward(ndcg: float) -> float:
    """Sparse/dense retrieval: the reward is the metric itself."""
    return ndcg


def sql_reward(conn: sqlite3.Connection, pred_sql: str, gold_sql: str,
               executable_bonus: float = 0.0) -> float:
    """Execution accuracy, plus the optional 0.3 bonus when the SQL merely runs."""
    r = execution_accuracy(conn, pred_sql, gold_sql)
    if executable_bonus and execute_sql(conn, pred_sql) is not None:
        r += executable_bonus
    return r


@dataclass
class RewardInfo:
    total: float
    retrieval: float
    format: float
    metric: float            # the underlying task metric (e.g. Recall@K)
    well_formed: bool
    query: str = ""
    think_len: int = 0       # whitespace tokens, for Figure-4 style monitoring
    query_len: int = 0


def compose_reward(retrieval: float, well_formed: bool) -> Tuple[float, float, float]:
    """Eq. (1): r = r_retrieval + r_format. Returns (total, retrieval, format)."""
    fmt = FORMAT_OK if well_formed else FORMAT_BAD
    return retrieval + fmt, retrieval, fmt


# ---------------------------------------------------------------------------
# 5. Toy retrieval world (so the RL loop runs anywhere, in seconds)
# ---------------------------------------------------------------------------

@dataclass
class ToyExample:
    prompt_words: List[str]     # the (under-specified) user query
    relevant: Set[int]          # ground-truth doc ids
    topic: int


class ToyIRWorld:
    """Synthetic literature-search task.

    `n_topics` topics each own `core_per_topic` vocabulary words; a document is
    `doc_core` random words of its topic plus one noise word. The user query is
    one core word of the target topic (the "cue") plus one noise word, so the
    original query reaches only the ~half of the topic's documents containing
    the cue. To do better the policy must learn which other words to OR in --
    knowledge that only the search engine's feedback reveals.
    """

    def __init__(self, n_topics: int = 3, core_per_topic: int = 6, docs_per_topic: int = 20,
                 doc_core: int = 3, n_noise: int = 8, top_k: int = 20, seed: int = 0):
        rng = random.Random(seed)
        self.n_topics, self.top_k = n_topics, top_k
        self.core = [[f"t{t}w{j}" for j in range(core_per_topic)] for t in range(n_topics)]
        self.noise = [f"n{i}" for i in range(n_noise)]
        self.docs: List[str] = []
        self.topic_docs: List[Set[int]] = [set() for _ in range(n_topics)]
        for t in range(n_topics):
            for _ in range(docs_per_topic):
                words = rng.sample(self.core[t], doc_core) + [rng.choice(self.noise)]
                rng.shuffle(words)
                self.topic_docs[t].add(len(self.docs))
                self.docs.append(" ".join(words))
        self.engine = BooleanBM25Engine(self.docs)

    @property
    def words(self) -> List[str]:
        return [w for topic in self.core for w in topic] + self.noise

    def sample_example(self, rng: random.Random) -> ToyExample:
        t = rng.randrange(self.n_topics)
        return ToyExample([rng.choice(self.core[t]), rng.choice(self.noise)],
                          set(self.topic_docs[t]), t)


# ---------------------------------------------------------------------------
# 6. Task: prompt construction + reward for the toy world
# ---------------------------------------------------------------------------

SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>", THINK_OPEN, THINK_CLOSE, ANSWER_OPEN,
            ANSWER_CLOSE, "AND", "OR", "(", ")"]
PAD, BOS, SEP, EOS = 0, 1, 2, 3


class Vocab:
    def __init__(self, words: Sequence[str]):
        self.itos = SPECIALS + [w for w in words if w not in SPECIALS]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, toks: Sequence[str]) -> List[int]:
        return [self.stoi[t] for t in toks]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self.itos[i] for i in ids)


class ToyRetrievalTask:
    """Prompt = `<bos> user query <sep> <think>` (the paper's prompts also end in <think>)."""

    def __init__(self, world: ToyIRWorld):
        self.world = world
        self.vocab = Vocab(world.words)

    def sample_batch(self, n: int, rng: random.Random) -> List[ToyExample]:
        return [self.world.sample_example(rng) for _ in range(n)]

    def prompt_ids(self, ex: ToyExample) -> List[int]:
        return self.vocab.encode(["<bos>", *ex.prompt_words, "<sep>", THINK_OPEN])

    def reward(self, ex: ToyExample, response: str) -> RewardInfo:
        parsed = parse_response(response)
        query, recall, ok = "", 0.0, parsed.well_formed
        if ok:
            query = extract_query(parsed.answer)
            try:
                retrieved = self.world.engine.search(query, self.world.top_k)
                recall = recall_at_k(retrieved, ex.relevant, self.world.top_k)
            except QuerySyntaxError:
                ok = False
        total, ret, fmt = compose_reward(literature_search_reward(recall), ok)
        return RewardInfo(total, ret, fmt, recall, ok, query,
                          len(parsed.think.split()), len(query.split()))

    def original_query_recall(self, ex: ToyExample) -> float:
        retrieved = self.world.engine.search(" ".join(ex.prompt_words), self.world.top_k)
        return recall_at_k(retrieved, ex.relevant, self.world.top_k)


# ---------------------------------------------------------------------------
# 7. Policy ("LLM") and critic
# ---------------------------------------------------------------------------

class GRULM(nn.Module):
    """Tiny causal LM standing in for Qwen2.5-3B-Instruct; also used as the critic body."""

    def __init__(self, vocab_size: int, d_model: int = 96, out_dim: Optional[int] = None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.head = nn.Linear(d_model, out_dim or vocab_size)

    def forward(self, ids: torch.Tensor, h: Optional[torch.Tensor] = None):
        x, h = self.gru(self.emb(ids), h)
        return self.head(x), h


def make_policy(vocab_size: int, d_model: int = 96) -> GRULM:
    return GRULM(vocab_size, d_model)


def make_critic(vocab_size: int, d_model: int = 96) -> GRULM:
    """Scalar value per position. (Paper: critic is initialised from the same LLM.)"""
    return GRULM(vocab_size, d_model, out_dim=1)


def left_pad(seqs: Sequence[Sequence[int]]) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    return torch.tensor([[PAD] * (width - len(s)) + list(s) for s in seqs], dtype=torch.long)


@torch.no_grad()
def generate(policy: GRULM, prompts: torch.Tensor, max_new: int, temperature: float = 1.0,
             greedy: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample responses until <eos> or max_new.

    Returns (responses[B,R], mask[B,R]); mask is 1 for every generated token
    including the <eos>, 0 for padding after it.
    """
    B = prompts.size(0)
    logits, h = policy(prompts)
    last = logits[:, -1]
    alive = torch.ones(B, dtype=torch.bool)
    toks, masks = [], []
    for _ in range(max_new):
        if greedy:
            tok = last.argmax(-1)
        else:
            tok = torch.multinomial(F.softmax(last / temperature, dim=-1), 1).squeeze(1)
        tok = torch.where(alive, tok, torch.full_like(tok, PAD))
        toks.append(tok)
        masks.append(alive.clone())
        alive = alive & (tok != EOS)
        if not alive.any():
            break
        logits, h = policy(tok.unsqueeze(1), h)
        last = logits[:, -1]
    return torch.stack(toks, 1), torch.stack(masks, 1).float()


def response_logprobs(policy: GRULM, prompts: torch.Tensor, responses: torch.Tensor,
                      temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token log pi(token | prefix) and entropy of the response positions, both [B,R]."""
    P, R = prompts.size(1), responses.size(1)
    full = torch.cat([prompts, responses], 1)
    logits, _ = policy(full[:, :-1])
    logp_all = F.log_softmax(logits[:, P - 1:P - 1 + R] / temperature, dim=-1)
    logp = logp_all.gather(-1, responses.unsqueeze(-1)).squeeze(-1)
    entropy = -(logp_all.exp() * logp_all).sum(-1)
    return logp, entropy


def response_values(critic: GRULM, prompts: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
    """V(state before emitting token t) for each response position, [B,R]."""
    P, R = prompts.size(1), responses.size(1)
    full = torch.cat([prompts, responses], 1)
    v, _ = critic(full[:, :-1])
    return v[:, P - 1:P - 1 + R, 0]


# ---------------------------------------------------------------------------
# 8. RL: GAE, PPO objective (Eqs. 3-4), GRPO advantages
# ---------------------------------------------------------------------------

def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def masked_whiten(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask)
    return (x - mean) * torch.rsqrt(var + eps) * mask


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor,
                gamma: float = 1.0, lam: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation over variable-length responses.

    rewards/values/mask: [B,R]. The value after the last valid token is 0.
    Returns (advantages, returns) with returns = advantages + values.
    """
    B, R = rewards.shape
    adv = torch.zeros_like(rewards)
    running = torch.zeros(B)
    for t in reversed(range(R)):
        if t + 1 < R:
            nxt_mask = mask[:, t + 1]
            next_v, carry = values[:, t + 1] * nxt_mask, nxt_mask
        else:
            next_v, carry = torch.zeros(B), torch.zeros(B)
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        running = (delta + gamma * lam * carry * running) * mask[:, t]
        adv[:, t] = running
    return adv, adv + values


def grpo_advantages(scores: torch.Tensor, group_size: int, eps: float = 1e-6) -> torch.Tensor:
    """Group-relative advantage: standardise each prompt's G samples. scores: [B*G] grouped."""
    g = scores.view(-1, group_size)
    return ((g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + eps)).view(-1)


def ppo_policy_loss(logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor,
                    mask: torch.Tensor, clip_eps: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """-L^CLIP of Eq. (4) (to minimise) and the fraction of clipped tokens."""
    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    loss = -masked_mean(torch.min(unclipped, clipped), mask)
    clipfrac = masked_mean((clipped < unclipped).float(), mask)
    return loss, clipfrac


def ppo_value_loss(values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor,
                   mask: torch.Tensor, clip_range: float = 0.5) -> torch.Tensor:
    """L^VF of Eq. (3) with the usual value clipping."""
    v_clip = old_values + torch.clamp(values - old_values, -clip_range, clip_range)
    return 0.5 * masked_mean(torch.max((values - returns) ** 2, (v_clip - returns) ** 2), mask)


def kl_k3(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Low-variance non-negative KL(pi || pi_ref) estimator used for GRPO's loss term."""
    d = ref_logp - logp
    return torch.exp(d) - d - 1.0


@dataclass
class PPOConfig:
    # Paper values (Appendix B.2): actor lr 1e-6, critic lr 1e-5, KL coef 0.001,
    # temperature 0.6, batch 64, PPO mini-batch 16, clip 0.2. The learning rates
    # below are larger because our policy is a from-scratch toy, not a 3B LLM.
    actor_lr: float = 5e-4
    critic_lr: float = 3e-3
    kl_coef: float = 0.001
    clip_eps: float = 0.2
    value_clip: float = 0.5
    entropy_coef: float = 0.001      # c2 in Eq. (3)
    gamma: float = 1.0
    lam: float = 1.0
    temperature: float = 1.0
    batch_size: int = 64
    mini_batch_size: int = 16
    ppo_epochs: int = 2
    max_new_tokens: int = 40
    advantage: str = "gae"           # "gae" (PPO) or "grpo"
    group_size: int = 4              # only for grpo
    max_grad_norm: float = 1.0
    seed: int = 0


class DeepRetrievalTrainer:
    """PPO (or GRPO) over a task that exposes sample_batch / prompt_ids / reward / vocab."""

    def __init__(self, policy: GRULM, task, cfg: PPOConfig, critic: Optional[GRULM] = None):
        self.cfg, self.task, self.policy = cfg, task, policy
        self.ref = make_policy(len(task.vocab), policy.emb.embedding_dim)
        self.ref.load_state_dict(policy.state_dict())
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.critic = None
        if cfg.advantage == "gae":
            self.critic = critic or make_critic(len(task.vocab), policy.emb.embedding_dim)
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.actor_opt = torch.optim.Adam(policy.parameters(), lr=cfg.actor_lr)
        self.rng = random.Random(cfg.seed)
        self.history: List[Dict[str, float]] = []

    # -- rollout ------------------------------------------------------------
    def _rollout(self):
        cfg = self.cfg
        if cfg.advantage == "grpo":
            base = self.task.sample_batch(cfg.batch_size // cfg.group_size, self.rng)
            examples = [ex for ex in base for _ in range(cfg.group_size)]
        else:
            examples = self.task.sample_batch(cfg.batch_size, self.rng)
        prompts = left_pad([self.task.prompt_ids(ex) for ex in examples])
        responses, mask = generate(self.policy, prompts, cfg.max_new_tokens, cfg.temperature)
        infos = []
        for ex, row, m in zip(examples, responses, mask):
            ids = [int(t) for t, keep in zip(row, m) if keep and int(t) != EOS]
            ended = bool(m.sum() > 0 and int(row[int(m.sum()) - 1]) == EOS)
            text = self.task.vocab.decode(ids)
            info = self.task.reward(ex, text) if ended else self._truncated_reward(ex)
            infos.append(info)
        return prompts, responses, mask, infos

    def _truncated_reward(self, ex) -> RewardInfo:
        """Responses that never emit <eos> count as malformed (no answer was committed)."""
        return self.task.reward(ex, "")

    # -- one PPO/GRPO iteration ---------------------------------------------
    def step(self) -> Dict[str, float]:
        cfg = self.cfg
        prompts, responses, mask, infos = self._rollout()
        scores = torch.tensor([i.total for i in infos])
        B, R = responses.shape

        with torch.no_grad():
            old_logp, _ = response_logprobs(self.policy, prompts, responses, cfg.temperature)
            ref_logp, _ = response_logprobs(self.ref, prompts, responses, cfg.temperature)
            kl_tok = (old_logp - ref_logp) * mask
            if cfg.advantage == "gae":
                old_values = response_values(self.critic, prompts, responses) * mask
                token_rewards = -cfg.kl_coef * kl_tok            # KL penalty in the reward (Eq. 2)
                last_idx = mask.sum(1).long() - 1
                token_rewards[torch.arange(B), last_idx] += scores
                adv, returns = compute_gae(token_rewards, old_values, mask, cfg.gamma, cfg.lam)
                adv = masked_whiten(adv, mask)
            else:
                adv = grpo_advantages(scores, cfg.group_size).unsqueeze(1).expand(B, R) * mask
                old_values = returns = None

        stats = defaultdict(list)
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(B)
            for s in range(0, B, cfg.mini_batch_size):
                idx = perm[s:s + cfg.mini_batch_size]
                m = mask[idx]
                logp, ent = response_logprobs(self.policy, prompts[idx], responses[idx], cfg.temperature)
                pl, clipfrac = ppo_policy_loss(logp, old_logp[idx], adv[idx], m, cfg.clip_eps)
                loss = pl - cfg.entropy_coef * masked_mean(ent, m)
                if cfg.advantage == "grpo":
                    loss = loss + cfg.kl_coef * masked_mean(kl_k3(logp, ref_logp[idx]), m)
                self.actor_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.actor_opt.step()
                stats["clipfrac"].append(float(clipfrac))

                if self.critic is not None:
                    v = response_values(self.critic, prompts[idx], responses[idx])
                    vl = ppo_value_loss(v, old_values[idx], returns[idx], m, cfg.value_clip)
                    self.critic_opt.zero_grad()
                    vl.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                    self.critic_opt.step()
                    stats["value_loss"].append(float(vl.detach()))

        n = len(infos)
        row = {
            "reward": float(scores.mean()),
            "metric": sum(i.metric for i in infos) / n,
            "valid_rate": sum(i.well_formed for i in infos) / n,
            "think_len": sum(i.think_len for i in infos) / n,
            "query_len": sum(i.query_len for i in infos) / n,
            "kl": float(masked_mean(kl_tok, mask)),
            **{k: sum(v) / len(v) for k, v in stats.items()},
        }
        self.history.append(row)
        return row

    def train(self, steps: int, log_every: int = 0) -> List[Dict[str, float]]:
        for i in range(steps):
            row = self.step()
            if log_every and (i + 1) % log_every == 0:
                print(f"step {i + 1:4d}  reward {row['reward']:+.2f}  recall {row['metric']:.3f}  "
                      f"valid {row['valid_rate']:.2f}  think {row['think_len']:.1f}  "
                      f"query {row['query_len']:.1f}  kl {row['kl']:.3f}")
        return self.history

    # -- evaluation ---------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, n: int = 128, greedy: bool = True, seed: int = 123) -> Dict[str, float]:
        rng = random.Random(seed)
        examples = self.task.sample_batch(n, rng)
        prompts = left_pad([self.task.prompt_ids(ex) for ex in examples])
        responses, mask = generate(self.policy, prompts, self.cfg.max_new_tokens,
                                   self.cfg.temperature, greedy=greedy)
        infos = []
        for ex, row, m in zip(examples, responses, mask):
            ids = [int(t) for t, keep in zip(row, m) if keep and int(t) != EOS]
            ended = bool(m.sum() > 0 and int(row[int(m.sum()) - 1]) == EOS)
            infos.append(self.task.reward(ex, self.task.vocab.decode(ids)) if ended
                         else self._truncated_reward(ex))
        out = {
            "reward": sum(i.total for i in infos) / n,
            "metric": sum(i.metric for i in infos) / n,
            "valid_rate": sum(i.well_formed for i in infos) / n,
        }
        if hasattr(self.task, "original_query_recall"):
            out["original_query_metric"] = sum(self.task.original_query_recall(e) for e in examples) / n
        return out


# ---------------------------------------------------------------------------
# 9. "Instruct model" stage: teach format only (no retrieval knowledge)
# ---------------------------------------------------------------------------

def _random_boolean_tokens(words: Sequence[str], rng: random.Random,
                           copy_from: Sequence[str] = ()) -> List[str]:
    """A random syntactically valid boolean query. Like an instruct model, it
    often copies words from the user's query, otherwise picks words at random."""
    def word() -> str:
        return rng.choice(copy_from) if copy_from and rng.random() < 0.5 else rng.choice(words)

    def group() -> List[str]:
        n = rng.randint(1, 3)
        toks = [word()]
        for _ in range(n - 1):
            toks += [rng.choice(["OR", "OR", "AND", ""]), word()]
        return [t for t in toks if t]
    g1 = group()
    if rng.random() < 0.5:
        return g1
    g2 = group()
    return ["("] + g1 + [")", rng.choice(["OR", "AND"]), "("] + g2 + [")"]


def random_format_response(words: Sequence[str], rng: random.Random,
                           copy_from: Sequence[str] = ()) -> List[str]:
    think = [rng.choice(words) for _ in range(rng.randint(1, 4))]
    return think + [THINK_CLOSE, ANSWER_OPEN, *_random_boolean_tokens(words, rng, copy_from),
                    ANSWER_CLOSE, "<eos>"]


def pretrain_format(policy: GRULM, task: ToyRetrievalTask, steps: int = 300, batch_size: int = 64,
                    lr: float = 3e-3, seed: int = 0) -> float:
    """Supervised warm-up on *random* well-formed responses.

    Stands in for starting from an instruction-tuned model: it can follow the
    <think>/<answer> protocol (and copy some user-query words) but has zero
    knowledge of which extra terms retrieve well. Returns the final cross-entropy.
    """
    rng = random.Random(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    words, vocab, loss_val = task.world.words, task.vocab, float("nan")
    for _ in range(steps):
        exs = task.sample_batch(batch_size, rng)
        prompts = left_pad([task.prompt_ids(e) for e in exs])
        resp = [vocab.encode(random_format_response(words, rng, e.prompt_words)) for e in exs]
        R = max(len(r) for r in resp)
        targets = torch.tensor([r + [PAD] * (R - len(r)) for r in resp])
        mask = (targets != PAD).float()
        logp, _ = response_logprobs(policy, prompts, targets)
        loss = -masked_mean(logp, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_val = float(loss.detach())
    return loss_val


# ---------------------------------------------------------------------------
# 10. Smoke test
# ---------------------------------------------------------------------------

def main(rl_steps: int = 500, seed: int = 1) -> Dict[str, Dict[str, float]]:
    torch.manual_seed(seed)
    world = ToyIRWorld(seed=seed)
    task = ToyRetrievalTask(world)
    policy = make_policy(len(task.vocab))
    ce = pretrain_format(policy, task, seed=seed)
    print(f"format warm-up done (cross-entropy {ce:.2f}); vocab={len(task.vocab)}, docs={world.engine.N}")

    trainer = DeepRetrievalTrainer(policy, task, PPOConfig(seed=seed))
    before = trainer.evaluate()
    print(f"original query    Recall@{world.top_k}: {before['original_query_metric']:.3f}")
    print(f"before RL (greedy) Recall@{world.top_k}: {before['metric']:.3f}  "
          f"valid {before['valid_rate']:.2f}  reward {before['reward']:+.2f}")
    trainer.train(rl_steps, log_every=max(rl_steps // 10, 1))
    after = trainer.evaluate()
    print(f"after  RL (greedy) Recall@{world.top_k}: {after['metric']:.3f}  "
          f"valid {after['valid_rate']:.2f}  reward {after['reward']:+.2f}")
    return {"before": before, "after": after}


if __name__ == "__main__":
    main()
