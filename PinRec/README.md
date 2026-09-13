# PinRec

PyTorch reference implementation of **PinRec: Unified Generative Retrieval for Pinterest Recommender Systems** (Pinterest, KDD 2026).

Paper: https://arxiv.org/abs/2504.10507

Deployed as a candidate generator across Pinterest's Home Feed, Search, and Related Pins surfaces: **+4% search saves**, **+16% product outbound clicks** for outcome-conditioned generation targeting that action, and **+71.4%** recall from multi-step autoregressive generation (16 steps vs. 1).

---

## Summary

Generative retrieval (TIGER, HSTU) reframes retrieval as sequence generation over a transformer, and typically trains one such model per surface against one generic engagement target. PinRec instead builds **one unified generative retrieval model for all of Pinterest's surfaces** — Home Feed, Search, and Related Pins — via the same pretrain-then-fine-tune paradigm as [`../PinFM`](../PinFM), but adds the machinery needed to make a *single* model useful when every surface defines success differently (saves vs. outbound clicks vs. session depth):

1. **Outcome-Conditioned Generation** (Sec 4.1) — condition the output head on a desired outcome (an action type, a surface constraint) so the *same* pretrained model can be steered toward whichever metric a surface cares about at generation time, instead of retraining per objective.
2. **Cross-surface pretrain/fine-tune with impression negatives** (Sec 4.3) — pretrained once on lifelong, cross-surface action sequences; fine-tuned per surface on that surface's impression logs, where shown-but-not-acted-on items become extra hard negatives.
3. **Multi-step, multi-embedding autoregressive generation with budget control** (Sec 4.4) — because outcome-conditioned generation produces many more candidate embeddings than a two-tower model, PinRec needs to allocate retrieval budget across outcomes and compress near-duplicate generated embeddings before ANN search.

---

## Key Ideas

### Outcome-Conditioned Generation (Eq. 1)

```
i_hat_{u,t} = O(h_{u,t}, c_1, c_2, ..., c_k)
```

The output head `O` (`OutcomeConditionedHead`) maps a transformer hidden state, optionally summed with one or more learned outcome-condition embeddings (`OutcomeEmbedding` — e.g. "target action = Save", "surface = Search"), to a predicted item representation. With no conditions this is **PinRec-UC** (unconditioned); with conditions, **PinRec-OC**. The paper's ablation (Table 2) finds removing outcome conditioning is the single most damaging change, causing -7.5% to -12.6% recall degradation across surfaces — more than removing fine-tuning entirely.

### Sampled softmax with popularity bias correction (Eq. 2)

```
s(i_hat, i_c) = lambda * i_hat^T i_c - Q(i_c)
```

`Q(i_c)` corrects for the fact that popular items are sampled disproportionately often as in-batch negatives; without it, they'd be over-penalized relative to their true prevalence. `Q` is estimated online via a **Count-Min Sketch** (`CountMinSketch`) frequency counter over item ids — the same style of correction used in PinnerFormer, made explicit here as `log_bias`.

### Pretraining vs. fine-tuning loss (Eq. 3-4)

| Stage | Sequence | Negatives |
|---|---|---|
| Pretraining (`pretraining_next_item_loss`) | lifelong, cross-surface, action-only history `H(u, t_max)` | in-batch positives only |
| Fine-tuning (`finetuning_next_item_loss`) | `H(u,t)` concatenated with the surface's action-only feed view `F(u,s,t)` | in-batch positives **+ impression-only items** (shown, not acted on) as extra hard negatives |

Impression negatives give a small but consistent additional lift on Home Feed, and the paper finds fine-tuning provides a further +2.0-4.5% recall on top of outcome conditioning — the two techniques are complementary, not redundant (Sec 5.2.2).

### Multi-step outcome-conditioned generation (Sec 4.4, Fig. 3)

Unlike single-step next-item generation, PinRec generates a whole sequence of candidate embeddings autoregressively. At each step, **one forward pass produces one candidate embedding per outcome** (multi-task-head style); one outcome's embedding is sampled and fed back into the sequence for the next step, so generation adapts jointly across outcomes while every outcome's embedding at every step remains a valid, separately-tracked retrieval candidate (`generate_outcome_conditioned`). The paper reports up to **+71.4%** recall from 16-step vs. 1-step generation (Table 4), confirming the gain comes from genuinely adapting to evolving intent, not just producing more embeddings.

Since outcome-conditioned, multi-step generation produces many more candidate embeddings than a single query vector, two additional mechanisms are needed at retrieval time:

- **Budget Allocation** (`allocate_budget`): splits the total retrieval budget `N` across outcomes proportional to specified target fractions, so e.g. a "Save"-targeting branch and an "Outbound Click"-targeting branch each get their designated share of retrieved candidates.
- **Embedding Compression** (`compress_embeddings`): merges a newly generated embedding into a previously *uncompressed* one (in generation order) if their cosine similarity exceeds a threshold, summing budgets — otherwise near-duplicate autoregressive steps would retrieve heavily overlapping candidate sets and hurt diversity.

---

## Architecture

```
Interaction tuple (item_j, action_j, surface_j, t_j) for j=1..m
    │
ItemEmbedder (f_tau: pretrained OmniSage/OmniSearchSage feature -> MLP -> L2 norm)
    │
PinRecInputEncoder: x_j = item_repr_j + action_emb(action_j) + surface_emb(surface_j) + TemporalEncoder(t_j)
    │
CausalTransformer (causal self-attention stack, + learned position embeddings)
    │
h_{u,t} = TransformerStack(x_{u,1:t})
    │
OutcomeConditionedHead: i_hat_{u,t} = O(h_{u,t}, c_1, ..., c_k)     [Eq. 1]
    │
sampled_softmax_loss (Eq. 2, Count-Min-Sketch bias-corrected)
    ├── pretraining_next_item_loss   (Eq. 3, lifelong cross-surface sequence)
    └── finetuning_next_item_loss    (Eq. 4, + impression-only hard negatives)

Generation (Fig. 3):
  generate_unconditional            -> (B, num_steps, d)
  generate_outcome_conditioned      -> {outcome: (B, num_steps, d)}
        │
  allocate_budget(total_budget, outcome_fractions)   -> per-outcome candidate quota
  compress_embeddings(embeddings, budgets, threshold) -> deduplicated candidates for ANN search
```

---

## Usage

```python
from pinrec import (
    ItemEmbedder, PinRecInputEncoder, CausalTransformer,
    OutcomeEmbedding, OutcomeConditionedHead, CountMinSketch,
    pretraining_next_item_loss, finetuning_next_item_loss,
    generate_unconditional, generate_outcome_conditioned,
    allocate_budget, compress_embeddings,
)

# --- Build model ---
pin_embedder = ItemEmbedder(raw_dim=256, d_model=768)          # OmniSage -> shared space
input_encoder = PinRecInputEncoder(d_model=768, num_actions=8, num_surfaces=4)
backbone = CausalTransformer(d_model=768, n_layers=12, n_heads=12)
head = OutcomeConditionedHead(d_model=768)
outcome_emb = OutcomeEmbedding(num_outcomes=8, d_model=768)

# --- Pretraining ---
item_repr = pin_embedder(raw_pin_features)                      # (B, m, d)
x = input_encoder(item_repr, action_ids, surface_ids, abs_time, prev_time)
h = backbone(x)
H = head(h)                                                      # unconditioned, all positions

cms = CountMinSketch()
cms.update(item_ids)
loss = pretraining_next_item_loss(H, item_repr, item_ids, cms)
loss.backward()

# --- Fine-tuning: add impression negatives ---
cms.update(impression_ids)
loss = finetuning_next_item_loss(H, item_repr, item_ids, impression_embeds, impression_ids, cms)

# --- Outcome-conditioned autoregressive generation ---
save_cond = [outcome_emb(torch.tensor([SAVE_ACTION_ID] * B))]
click_cond = [outcome_emb(torch.tensor([CLICK_ACTION_ID] * B))]
generated = generate_outcome_conditioned(
    backbone, head, context_emb=x, outcome_condition_sets=[save_cond, click_cond], num_steps=16,
)
# generated[0]: (B, 16, d) "Save"-targeted candidates, generated[1]: "Click"-targeted candidates

# --- Budget + compression before ANN retrieval ---
allocation = allocate_budget(total_budget=100, outcome_fractions={0: 0.6, 1: 0.4})
all_embeds = torch.cat([generated[0][0], generated[1][0]], dim=0)   # one user, both outcomes
all_budgets = torch.cat([
    torch.full((16,), allocation[0] / 16.0), torch.full((16,), allocation[1] / 16.0),
])
final_embeds, final_budgets = compress_embeddings(all_embeds, all_budgets, threshold=0.9)
```

---

## Files

```
PinRec/
├── pinrec.py         # full implementation + smoke test
│   ├── ItemEmbedder                 # f_tau: pretrained feature -> shared space (Sec 4.2)
│   ├── TemporalEncoder              # sinusoidal absolute + log-scale relative time
│   ├── PinRecInputEncoder           # item + action + surface + temporal -> x_{u,j}
│   ├── CausalTransformer            # causal self-attention backbone M
│   ├── OutcomeEmbedding             # learned c_1..c_k outcome-condition embeddings
│   ├── OutcomeConditionedHead       # O(h, c_1..c_k) -> item repr (Eq. 1)
│   ├── CountMinSketch                # popularity frequency estimator for Q(i_c)
│   ├── sampled_softmax_loss         # Eq. 2 contrastive core
│   ├── pretraining_next_item_loss   # Eq. 3
│   ├── finetuning_next_item_loss    # Eq. 4, + impression negatives
│   ├── generate_unconditional       # PinRec-UC autoregressive generation
│   ├── generate_outcome_conditioned # PinRec-OC autoregressive generation (Fig. 3)
│   ├── allocate_budget              # Sec 4.4 Budget Allocation
│   └── compress_embeddings          # Sec 4.4 Embedding Compression
├── test_pinrec.py    # pytest test suite (39 tests)
└── README.md
```

---

## Running

### Smoke test

```bash
python3 pinrec.py
```

Expected output (values vary slightly by seed):

```
--- Input encoding ---
input sequence x: torch.Size([4, 20, 32])

--- Backbone + Outcome-Conditioned Head ---
H (UC): torch.Size([4, 32]), H (OC): torch.Size([4, 32])

--- Sampled softmax loss with CMS bias correction ---
L_pretrain=4.4077  L_finetune=4.7707  backward: OK

--- Autoregressive generation ---
unconditional generation: torch.Size([4, 3, 32])
outcome 0 generation: torch.Size([4, 3, 32])
outcome 1 generation: torch.Size([4, 3, 32])

--- Budget allocation & embedding compression ---
budget allocation: {0: 70, 1: 30}
compressed 6 -> 6 embeddings, budgets sum to 6 (should equal 6)
```

### Test suite

```bash
python3 -m pytest test_pinrec.py -v
```

Expected output:

```
collected 39 items
...
39 passed in ~1s
```

Useful variants:

```bash
python3 -m pytest test_pinrec.py::TestGeneration -v
python3 -m pytest test_pinrec.py -x
```
