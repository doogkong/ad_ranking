"""End-to-end CoGR demo on a synthetic many-to-many retrieval task.

Run:  python3 demo.py
"""
import random

import torch

from cogr import (InvertedIndex, ToyKeywordGenerator, build_sft_query_targets,
                  co_evolve, evaluate, sft_train)


def make_data(n_topics=6, items_per_topic=12, queries_per_topic=8, seed=0):
    rnd = random.Random(seed)
    topic_kw = {t: [f"t{t}_k{j}" for j in range(4)] for t in range(n_topics)}
    noise = [f"noise{j}" for j in range(10)]
    items, init_kws, item_topic = [], {}, {}
    for t in range(n_topics):
        for j in range(items_per_topic):
            name = f"item_t{t}_{j}"
            items.append(name)
            item_topic[name] = t
            # the "base LLM" keywords: partly on-topic, partly noise
            init_kws[name] = rnd.sample(topic_kw[t], 2) + rnd.sample(noise, 2)
    queries, rel = [], {}
    for t in range(n_topics):
        for j in range(queries_per_topic):
            q = f"query about topic {t} variant {j}"
            queries.append(q)
            rel[q] = {i for i in items if item_topic[i] == t}
    vocab = sorted({k for ks in init_kws.values() for k in ks} | {k for v in topic_kw.values() for k in v})
    return queries, items, rel, init_kws, vocab


def main():
    random.seed(0)
    torch.manual_seed(0)
    queries, items, rel, init_kws, vocab = make_data()
    random.shuffle(queries)
    train_q, val_q = queries[:-12], queries[-12:]

    # Phase 1: SFT (Alg. 1)
    targets = build_sft_query_targets(train_q, init_kws, rel, top_n=6)
    q_gen = ToyKeywordGenerator(vocab, seed=1)
    i_gen = ToyKeywordGenerator(vocab, seed=2)
    sft_train(q_gen, train_q, [targets[q] for q in train_q], epochs=30)
    sft_train(i_gen, items, [init_kws[i] for i in items], epochs=30)

    def report(tag):
        idx = InvertedIndex(i_gen.generate(items, 30))
        m = evaluate(val_q, q_gen.generate(val_q, 30), idx, rel, k=20)
        print(f"{tag:>14}: P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} MRR@20={m['MRR@20']:.3f}")

    report("after SFT")
    # Phase 2: co-evolving RL (Alg. 2)
    co_evolve(q_gen, i_gen, train_q, items, rel, rounds=3, query_epochs=4, item_epochs=3,
              log=lambda s: print(f"  round {s['round']}: R_q={s['query_reward']:.3f} R_i={s['item_reward']:.3f}"))
    report("after co-evolve")


if __name__ == "__main__":
    main()
