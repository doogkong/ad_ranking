"""Tests for PEPNet: Parameter and Embedding Personalized Network."""

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from pepnet import GateNU, EPNet, PPNet, PEPNet

B = 4
D = 16   # d_embed
T = 20   # sequence length (not used in PEPNet directly, kept for convention)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pepnet(
    d_embed: int = D,
    dnn_hidden: list[int] | None = None,
    n_tasks: int = 2,
    n_domain_stats: int = 4,
    gamma: float = 2.0,
) -> PEPNet:
    if dnn_hidden is None:
        dnn_hidden = [32, 16]
    return PEPNet(
        sparse_vocab_sizes=[100, 50],
        dense_input_dims=[8, 4],
        d_embed=d_embed,
        domain_vocab_size=5,
        n_domain_stats=n_domain_stats,
        user_vocab_size=100,
        item_vocab_size=50,
        author_vocab_size=20,
        dnn_hidden=dnn_hidden,
        n_tasks=n_tasks,
        gamma=gamma,
    )


def _make_inputs(model: PEPNet, batch: int = B):
    sparse_feats = [torch.randint(0, 100, (batch,)),
                    torch.randint(0, 50, (batch,))]
    dense_feats  = [torch.randn(batch, 8), torch.randn(batch, 4)]
    domain_id    = torch.randint(0, 5, (batch,))
    domain_stats = torch.randn(batch, 4)
    user_id      = torch.randint(0, 100, (batch,))
    item_id      = torch.randint(0, 50, (batch,))
    author_id    = torch.randint(0, 20, (batch,))
    return sparse_feats, dense_feats, domain_id, domain_stats, user_id, item_id, author_id


# ---------------------------------------------------------------------------
# GateNU tests
# ---------------------------------------------------------------------------

class TestGateNU:
    def test_output_shape(self):
        gate = GateNU(input_dim=10, output_dim=20)
        x = torch.randn(B, 10)
        out = gate(x)
        assert out.shape == (B, 20)

    def test_output_range_default_gamma(self):
        gate = GateNU(input_dim=8, output_dim=16, gamma=2.0)
        x = torch.randn(100, 8)
        out = gate(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 2.0

    def test_output_range_custom_gamma(self):
        gamma = 3.0
        gate = GateNU(input_dim=8, output_dim=16, gamma=gamma)
        x = torch.randn(100, 8)
        out = gate(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= gamma + 1e-6

    def test_gamma_stored(self):
        gate = GateNU(10, 10, gamma=1.5)
        assert gate.gamma == 1.5

    def test_gradient_flows(self):
        gate = GateNU(8, 8)
        x = torch.randn(B, 8, requires_grad=True)
        out = gate(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_batch_independence(self):
        gate = GateNU(6, 6)
        x = torch.randn(B, 6)
        out_full = gate(x)
        out_single = gate(x[:1])
        assert torch.allclose(out_full[:1], out_single, atol=1e-5)

    def test_two_linear_layers(self):
        gate = GateNU(4, 8)
        assert isinstance(gate.fc1, nn.Linear)
        assert isinstance(gate.fc2, nn.Linear)
        assert gate.fc1.in_features == 4
        assert gate.fc1.out_features == 8
        assert gate.fc2.in_features == 8
        assert gate.fc2.out_features == 8

    def test_output_centered_near_one(self):
        # With large enough batch, mean should be near γ/2 = 1.0 due to sigmoid
        torch.manual_seed(0)
        gate = GateNU(32, 32, gamma=2.0)
        x = torch.randn(1000, 32)
        with torch.no_grad():
            out = gate(x)
        # mean should be between 0.5 and 1.5
        assert 0.5 < out.mean().item() < 1.5


# ---------------------------------------------------------------------------
# EPNet tests
# ---------------------------------------------------------------------------

class TestEPNet:
    def test_output_shape(self):
        ep = EPNet(domain_feat_dim=8, emb_flat_dim=32)
        domain_emb = torch.randn(B, 8)
        E = torch.randn(B, 32)
        out = ep(domain_emb, E)
        assert out.shape == (B, 32)

    def test_stop_gradient_on_E(self):
        ep = EPNet(domain_feat_dim=8, emb_flat_dim=16)
        domain_emb = torch.randn(B, 8)
        E = torch.randn(B, 16, requires_grad=True)
        # EPNet uses E.detach() in gate input — the gate path doesn't flow to E
        out = ep(domain_emb, E)
        out.sum().backward()
        # E should still get gradient through the element-wise product O_ep = gate * E
        assert E.grad is not None

    def test_domain_emb_gradient_flows(self):
        ep = EPNet(domain_feat_dim=8, emb_flat_dim=16)
        domain_emb = torch.randn(B, 8, requires_grad=True)
        E = torch.randn(B, 16)
        out = ep(domain_emb, E)
        out.sum().backward()
        assert domain_emb.grad is not None

    def test_gate_scales_embedding(self):
        ep = EPNet(domain_feat_dim=4, emb_flat_dim=8, gamma=2.0)
        domain_emb = torch.randn(B, 4)
        E = torch.ones(B, 8)
        with torch.no_grad():
            out = ep(domain_emb, E)
        # output = gate * E; since E=1, output = gate ∈ [0, 2]
        assert out.min().item() >= 0.0
        assert out.max().item() <= 2.0 + 1e-5

    def test_different_domains_different_output(self):
        ep = EPNet(domain_feat_dim=8, emb_flat_dim=16)
        E = torch.randn(1, 16).expand(2, -1)
        domain_a = torch.randn(1, 8)
        domain_b = torch.randn(1, 8) + 5.0   # very different domain
        with torch.no_grad():
            out_a = ep(domain_a, E[:1])
            out_b = ep(domain_b, E[:1])
        assert not torch.allclose(out_a, out_b)

    def test_output_same_shape_as_E(self):
        for emb_dim in [16, 32, 64]:
            ep = EPNet(domain_feat_dim=4, emb_flat_dim=emb_dim)
            out = ep(torch.randn(B, 4), torch.randn(B, emb_dim))
            assert out.shape == (B, emb_dim)


# ---------------------------------------------------------------------------
# PPNet tests
# ---------------------------------------------------------------------------

class TestPPNet:
    def test_output_num_gates(self):
        gate_dims = [32, 16, 8]
        pp = PPNet(prior_dim=12, ep_dim=32, gate_dims=gate_dims, n_tasks=3)
        O_prior = torch.randn(B, 12)
        O_ep = torch.randn(B, 32)
        gates = pp(O_prior, O_ep)
        assert len(gates) == len(gate_dims)

    def test_gate_shapes(self):
        gate_dims = [32, 16, 8]
        n_tasks = 3
        pp = PPNet(prior_dim=12, ep_dim=32, gate_dims=gate_dims, n_tasks=n_tasks)
        gates = pp(torch.randn(B, 12), torch.randn(B, 32))
        for i, (g, h) in enumerate(zip(gates, gate_dims)):
            assert g.shape == (B, n_tasks, h), f"gate {i} shape mismatch"

    def test_stop_gradient_on_O_ep(self):
        pp = PPNet(prior_dim=8, ep_dim=16, gate_dims=[16], n_tasks=2)
        O_prior = torch.randn(B, 8, requires_grad=True)
        O_ep = torch.randn(B, 16, requires_grad=True)
        gates = pp(O_prior, O_ep)
        gates[0].sum().backward()
        # O_prior should have gradient (used in gate computation)
        assert O_prior.grad is not None
        # O_ep should NOT have gradient through PPNet (detach)
        assert O_ep.grad is None

    def test_n_gates_equals_n_layers(self):
        gate_dims = [10, 8, 4, 2]
        pp = PPNet(prior_dim=6, ep_dim=10, gate_dims=gate_dims, n_tasks=1)
        assert len(pp.gates) == len(gate_dims)

    def test_gate_range(self):
        pp = PPNet(prior_dim=8, ep_dim=16, gate_dims=[16, 8], n_tasks=2, gamma=2.0)
        gates = pp(torch.randn(B, 8), torch.randn(B, 16))
        for g in gates:
            assert g.min().item() >= 0.0
            assert g.max().item() <= 2.0 + 1e-5

    def test_task_gates_differ(self):
        pp = PPNet(prior_dim=8, ep_dim=16, gate_dims=[16], n_tasks=3)
        gates = pp(torch.randn(B, 8), torch.randn(B, 16))
        g = gates[0]   # (B, 3, 16)
        # Gates for different tasks should differ (different output_dim slices)
        assert not torch.allclose(g[:, 0, :], g[:, 1, :])


# ---------------------------------------------------------------------------
# PEPNet forward tests
# ---------------------------------------------------------------------------

class TestPEPNet:
    def test_output_shape_single_task(self):
        model = _make_pepnet(n_tasks=1)
        inputs = _make_inputs(model)
        out = model(*inputs)
        assert out.shape == (B, 1)

    def test_output_shape_multi_task(self):
        model = _make_pepnet(n_tasks=4)
        inputs = _make_inputs(model)
        out = model(*inputs)
        assert out.shape == (B, 4)

    def test_backward(self):
        model = _make_pepnet(n_tasks=2)
        inputs = _make_inputs(model)
        out = model(*inputs)
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_different_domains_different_output(self):
        model = _make_pepnet(n_tasks=2)
        sf = [torch.randint(0, 100, (1,)), torch.randint(0, 50, (1,))]
        df = [torch.randn(1, 8), torch.randn(1, 4)]
        stats = torch.randn(1, 4)
        uid = torch.randint(0, 100, (1,))
        iid = torch.randint(0, 50, (1,))
        aid = torch.randint(0, 20, (1,))
        with torch.no_grad():
            out_d0 = model(sf, df, torch.tensor([0]), stats, uid, iid, aid)
            out_d1 = model(sf, df, torch.tensor([1]), stats, uid, iid, aid)
        assert not torch.allclose(out_d0, out_d1)

    def test_different_users_different_output(self):
        model = _make_pepnet(n_tasks=2)
        sf = [torch.randint(0, 100, (1,)), torch.randint(0, 50, (1,))]
        df = [torch.randn(1, 8), torch.randn(1, 4)]
        did = torch.tensor([0])
        stats = torch.randn(1, 4)
        iid = torch.randint(0, 50, (1,))
        aid = torch.randint(0, 20, (1,))
        with torch.no_grad():
            out_u0 = model(sf, df, did, stats, torch.tensor([0]), iid, aid)
            out_u5 = model(sf, df, did, stats, torch.tensor([5]), iid, aid)
        assert not torch.allclose(out_u0, out_u5)

    def test_batch_independence(self):
        torch.manual_seed(0)
        model = _make_pepnet(n_tasks=2)
        model.eval()
        inputs = _make_inputs(model, batch=B)
        with torch.no_grad():
            out_batch = model(*inputs)
            # Run first sample independently
            single_inputs = tuple(
                [x[:1] for x in inp] if isinstance(inp, list) else inp[:1]
                for inp in inputs
            )
            out_single = model(*single_inputs)
        assert torch.allclose(out_batch[:1], out_single, atol=1e-5)

    def test_no_task_seesaw_gradients_isolated(self):
        # Gradients for task 0's tower should not directly come from task 1's tower
        model = _make_pepnet(n_tasks=2)
        inputs = _make_inputs(model)
        out = model(*inputs)
        # Backward on task 0 only
        out[:, 0].sum().backward()
        # Task 0 tower params should have gradient
        for name, param in model.named_parameters():
            if "towers.0" in name:
                assert param.grad is not None
        model.zero_grad()
        # Backward on task 1 only
        out2 = model(*inputs)
        out2[:, 1].sum().backward()
        for name, param in model.named_parameters():
            if "towers.1" in name:
                assert param.grad is not None

    def test_epnet_gate_applied(self):
        # Verify EPNet is in the computation graph
        model = _make_pepnet(n_tasks=2)
        inputs = _make_inputs(model)
        out = model(*inputs)
        out.sum().backward()
        # EPNet gate parameters should have gradients
        assert model.epnet.gate.fc1.weight.grad is not None
        assert model.epnet.gate.fc2.weight.grad is not None

    def test_ppnet_gates_applied(self):
        model = _make_pepnet(n_tasks=2)
        inputs = _make_inputs(model)
        out = model(*inputs)
        out.sum().backward()
        for i, gate_nu in enumerate(model.ppnet.gates):
            assert gate_nu.fc1.weight.grad is not None, f"PPNet gate {i} no grad"

    def test_single_sample(self):
        model = _make_pepnet(n_tasks=2)
        inputs = _make_inputs(model, batch=1)
        out = model(*inputs)
        assert out.shape == (1, 2)

    def test_deep_tower(self):
        model = _make_pepnet(dnn_hidden=[64, 32, 16, 8], n_tasks=2)
        inputs = _make_inputs(model)
        out = model(*inputs)
        assert out.shape == (B, 2)

    def test_shallow_tower_single_hidden(self):
        model = _make_pepnet(dnn_hidden=[32], n_tasks=2)
        inputs = _make_inputs(model)
        out = model(*inputs)
        assert out.shape == (B, 2)

    def test_param_count_reasonable(self):
        model = _make_pepnet()
        total = sum(p.numel() for p in model.parameters())
        assert total > 0
        assert total < 10_000_000   # sanity upper bound

    def test_train_eval_same_output(self):
        # PEPNet has no dropout/batchnorm so train/eval should match
        torch.manual_seed(0)
        model = _make_pepnet(n_tasks=2)
        inputs = _make_inputs(model)
        model.train()
        with torch.no_grad():
            out_train = model(*inputs)
        model.eval()
        with torch.no_grad():
            out_eval = model(*inputs)
        assert torch.allclose(out_train, out_eval)


# ---------------------------------------------------------------------------
# Integration: multi-domain multi-task
# ---------------------------------------------------------------------------

class TestMultiDomainMultiTask:
    def test_all_domains_produce_output(self):
        n_domains = 5
        model = _make_pepnet(n_tasks=3)
        model.eval()
        sf = [torch.randint(0, 100, (1,)), torch.randint(0, 50, (1,))]
        df = [torch.randn(1, 8), torch.randn(1, 4)]
        stats = torch.randn(1, 4)
        uid = torch.tensor([0])
        iid = torch.tensor([0])
        aid = torch.tensor([0])
        with torch.no_grad():
            outs = [
                model(sf, df, torch.tensor([d]), stats, uid, iid, aid)
                for d in range(n_domains)
            ]
        assert all(o.shape == (1, 3) for o in outs)
        # Different domains → different outputs
        assert not all(torch.allclose(outs[0], outs[i]) for i in range(1, n_domains))

    def test_training_with_per_task_loss(self):
        model = _make_pepnet(n_tasks=3)
        inputs = _make_inputs(model)
        out = model(*inputs)   # (B, 3)
        labels = torch.zeros_like(out)
        # Compute separate BCE loss per task and sum
        losses = [
            torch.nn.functional.binary_cross_entropy_with_logits(
                out[:, t], labels[:, t]
            )
            for t in range(3)
        ]
        total_loss = sum(losses)
        total_loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No grad: {name}"

    def test_epnet_domain_personalization_scale(self):
        # EPNet output is O_ep = gate * E where gate ∈ [0, γ]
        # The scale must be within [0, γ] applied elementwise
        model = _make_pepnet()
        sf = [torch.randint(0, 100, (B,)), torch.randint(0, 50, (B,))]
        df = [torch.randn(B, 8), torch.randn(B, 4)]
        domain_id = torch.zeros(B, dtype=torch.long)
        domain_stats = torch.randn(B, 4)
        # Get embedding E by hacking the forward
        tokens = (
            [emb(f).unsqueeze(1) for emb, f in zip(model.sparse_embs, sf)] +
            [proj(f).unsqueeze(1) for proj, f in zip(model.dense_projs, df)]
        )
        E = torch.cat(tokens, dim=1).flatten(1)
        domain_emb = torch.cat([
            model.domain_id_emb(domain_id),
            model.domain_stats_proj(domain_stats),
        ], dim=-1)
        with torch.no_grad():
            O_ep = model.epnet(domain_emb, E)
        # O_ep = gate * E; since gate ∈ [0, 2], |O_ep| ≤ 2 * |E|
        assert (O_ep.abs() <= E.abs() * 2.0 + 1e-4).all()

    def test_ppnet_gates_per_task_differ(self):
        model = _make_pepnet(n_tasks=3)
        inputs = _make_inputs(model)
        sf, df, did, ds, uid, iid, aid = inputs
        O_prior = torch.cat([
            model.user_emb(uid),
            model.item_emb(iid),
            model.author_emb(aid),
        ], dim=-1)
        tokens = (
            [emb(f).unsqueeze(1) for emb, f in zip(model.sparse_embs, sf)] +
            [proj(f).unsqueeze(1) for proj, f in zip(model.dense_projs, df)]
        )
        E = torch.cat(tokens, dim=1).flatten(1)
        domain_emb = torch.cat([
            model.domain_id_emb(did),
            model.domain_stats_proj(ds),
        ], dim=-1)
        with torch.no_grad():
            O_ep = model.epnet(domain_emb, E)
            gates = model.ppnet(O_prior, O_ep)
        for l, g in enumerate(gates):
            # Each gate has shape (B, n_tasks, h_l); tasks should differ
            assert g.shape[1] == 3
            assert not torch.allclose(g[:, 0, :], g[:, 1, :]), \
                f"Gate {l}: task 0 == task 1"

    def test_gamma_effect(self):
        # Higher gamma → larger gate range → more extreme personalization
        model_g2 = _make_pepnet(gamma=2.0)
        model_g4 = _make_pepnet(gamma=4.0)
        inputs = _make_inputs(model_g2)
        sf, df, did, ds, uid, iid, aid = inputs
        for m in (model_g2, model_g4):
            tokens = (
                [emb(f).unsqueeze(1) for emb, f in zip(m.sparse_embs, sf)] +
                [proj(f).unsqueeze(1) for proj, f in zip(m.dense_projs, df)]
            )
            E = torch.cat(tokens, dim=1).flatten(1)
            domain_emb = torch.cat([
                m.domain_id_emb(did),
                m.domain_stats_proj(ds),
            ], dim=-1)
        gate_g2 = model_g2.epnet.gate
        gate_g4 = model_g4.epnet.gate
        x = torch.randn(100, gate_g2.fc1.in_features)
        with torch.no_grad():
            out_g2 = gate_g2(x)
            out_g4 = gate_g4(x)
        assert out_g2.max().item() <= 2.0 + 1e-5
        assert out_g4.max().item() <= 4.0 + 1e-5
