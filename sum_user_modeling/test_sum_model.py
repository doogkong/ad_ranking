"""Unit tests for sum_model.py."""

import unittest

import torch
import torch.nn as nn

from sum_model import (
    MLPExtractor,
    DotCompressionWithAttention,
    MLPMixerExtractor,
    DeepCrossExtractor,
    InteractionModule,
    UserTower,
    MixTower,
    multi_task_loss,
    SOAPFeatureStore,
    SOAPClient,
    SUMModel,
    SIMRetriever,
    RecencyWeightedPooling,
    HybridUserTower,
    GradientSurgery,
)


class TestExtractors(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.B, self.L, self.D = 4, 10, 32

    def test_mlp_extractor_shape(self):
        m = MLPExtractor(in_dim=self.D, hidden_dim=64, out_dim=16)
        x = torch.randn(self.B, self.D)
        out = m(x)
        self.assertEqual(out.shape, (self.B, 16))

    def test_dot_compression_shape(self):
        m = DotCompressionWithAttention(
            seq_dim=self.D, dense_dim=16, d_k=8, n_heads=4, out_dim=20
        )
        x_seq   = torch.randn(self.B, self.L, self.D)
        x_dense = torch.randn(self.B, 16)
        out = m(x_seq, x_dense)
        self.assertEqual(out.shape, (self.B, 20))

    def test_mlp_mixer_shape(self):
        m = MLPMixerExtractor(
            seq_len=self.L, in_dim=self.D, hidden_token=20, hidden_channel=64, out_dim=16
        )
        x = torch.randn(self.B, self.L, self.D)
        out = m(x)
        self.assertEqual(out.shape, (self.B, 16))

    def test_deep_cross_shape(self):
        m = DeepCrossExtractor(in_dim=self.D, n_cross_layers=3, out_dim=16)
        x = torch.randn(self.B, self.D)
        out = m(x)
        self.assertEqual(out.shape, (self.B, 16))

    def test_extractors_backprop(self):
        for cls, args, kwargs, x in [
            (MLPExtractor,
             [], dict(in_dim=self.D, hidden_dim=64, out_dim=16),
             torch.randn(self.B, self.D, requires_grad=True)),
            (DeepCrossExtractor,
             [], dict(in_dim=self.D, n_cross_layers=2, out_dim=16),
             torch.randn(self.B, self.D, requires_grad=True)),
        ]:
            m = cls(**kwargs)
            loss = m(x).sum()
            loss.backward()
            self.assertIsNotNone(x.grad)


class TestInteractionModule(unittest.TestCase):

    def test_output_shape(self):
        torch.manual_seed(0)
        B, L, D, DENSE = 3, 8, 32, 16
        m = InteractionModule(in_dim=D, dense_dim=DENSE, seq_len=L, extractor_out=24, out_dim=D)
        x_seq   = torch.randn(B, L, D)
        x_dense = torch.randn(B, DENSE)
        out = m(x_seq, x_dense)
        self.assertEqual(out.shape, (B, L, D))

    def test_residual_dim_change(self):
        torch.manual_seed(0)
        B, L = 2, 5
        m = InteractionModule(in_dim=32, dense_dim=16, seq_len=L, extractor_out=16, out_dim=64)
        out = m(torch.randn(B, L, 32), torch.randn(B, 16))
        self.assertEqual(out.shape, (B, L, 64))


class TestUserTower(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.B, self.L = 4, 12

    def test_n_embeddings(self):
        tower = UserTower(input_dim=32, dense_dim=16, seq_len=self.L,
                          n_layers=3, n_outputs=2, user_emb_dim=48, extractor_out=32, hidden_dim=64)
        embs = tower(torch.randn(self.B, self.L, 32), torch.randn(self.B, 16))
        self.assertEqual(len(embs), 2)
        for e in embs:
            self.assertEqual(e.shape, (self.B, 48))

    def test_single_output(self):
        tower = UserTower(input_dim=32, dense_dim=16, seq_len=self.L,
                          n_layers=2, n_outputs=1, user_emb_dim=24, extractor_out=16, hidden_dim=32)
        embs = tower(torch.randn(self.B, self.L, 32), torch.randn(self.B, 16))
        self.assertEqual(len(embs), 1)

    def test_gradient_flows(self):
        tower = UserTower(input_dim=32, dense_dim=16, seq_len=self.L,
                          n_layers=2, n_outputs=2, user_emb_dim=24, extractor_out=16, hidden_dim=32)
        embs = tower(torch.randn(self.B, self.L, 32), torch.randn(self.B, 16))
        loss = sum(e.sum() for e in embs)
        loss.backward()
        for p in tower.parameters():
            self.assertIsNotNone(p.grad)


class TestMixTower(unittest.TestCase):

    def test_shape(self):
        torch.manual_seed(0)
        B = 4
        mix = MixTower(user_emb_dim=48, n_user_embs=2, ad_feat_dim=24, n_tasks=3, hidden_dim=64)
        user_embs = [torch.randn(B, 48), torch.randn(B, 48)]
        ad_feats  = torch.randn(B, 24)
        out = mix(user_embs, ad_feats)
        self.assertEqual(out.shape, (B, 3))


class TestMultiTaskLoss(unittest.TestCase):

    def test_shape_and_positive(self):
        torch.manual_seed(0)
        logits = torch.randn(8, 4)
        labels = torch.randint(0, 2, (8, 4)).float()
        loss = multi_task_loss(logits, labels)
        self.assertEqual(loss.shape, ())
        self.assertGreater(loss.item(), 0)

    def test_weighted_vs_uniform(self):
        torch.manual_seed(1)
        logits = torch.randn(8, 3)
        labels = torch.randint(0, 2, (8, 3)).float()
        uniform  = multi_task_loss(logits, labels)
        weighted = multi_task_loss(logits, labels, task_weights=torch.tensor([1.0, 2.0, 3.0]))
        self.assertNotAlmostEqual(uniform.item(), weighted.item(), places=4)

    def test_zero_loss_perfect_predictions(self):
        # sigmoid(10) ≈ 1, sigmoid(-10) ≈ 0 → near-zero BCE
        logits = torch.tensor([[10.0, -10.0]])
        labels = torch.tensor([[1.0, 0.0]])
        loss = multi_task_loss(logits, labels)
        self.assertLess(loss.item(), 1e-3)


class TestSOAP(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.tower = UserTower(
            input_dim=16, dense_dim=8, seq_len=5, n_layers=2,
            n_outputs=2, user_emb_dim=24, extractor_out=16, hidden_dim=32
        )
        self.store  = SOAPFeatureStore(window_size=3)
        self.client = SOAPClient(self.tower, self.store)

    def test_cold_start_returns_new_embedding(self):
        x_seq   = torch.randn(5, 16)
        x_dense = torch.randn(8)
        emb = self.client.serve('new_user', x_seq, x_dense)
        self.assertEqual(emb.ndim, 1)

    def test_warm_returns_previous(self):
        x_seq   = torch.randn(5, 16)
        x_dense = torch.randn(8)
        emb1 = self.client.serve('u1', x_seq, x_dense)
        # Second call returns the embedding written by first call
        emb2 = self.client.serve('u1', x_seq, x_dense)
        # emb1 was the "new" embedding from cycle 0; emb2 is the cached value
        self.assertEqual(emb1.shape, emb2.shape)

    def test_rolling_average_window(self):
        store = SOAPFeatureStore(window_size=2)
        store.write('u', torch.ones(4))
        store.write('u', torch.full((4,), 3.0))
        avg = store.read('u')
        self.assertTrue(torch.allclose(avg, torch.full((4,), 2.0)))

    def test_window_evicts_oldest(self):
        store = SOAPFeatureStore(window_size=2)
        store.write('u', torch.zeros(4))
        store.write('u', torch.ones(4))
        store.write('u', torch.full((4,), 4.0))
        avg = store.read('u')
        # Window contains [1.0, 4.0] → mean = 2.5
        self.assertTrue(torch.allclose(avg, torch.full((4,), 2.5)))


class TestSUMModel(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.B, self.L = 4, 10
        self.model = SUMModel(
            input_dim=32, dense_dim=16, seq_len=self.L,
            ad_feat_dim=24, n_tasks=3, n_layers=2,
            extractor_out=32, hidden_dim=64, user_emb_dim=48, n_user_embs=2,
        )

    def test_forward_shapes(self):
        x_seq    = torch.randn(self.B, self.L, 32)
        x_dense  = torch.randn(self.B, 16)
        ad_feats = torch.randn(self.B, 24)
        embs, logits = self.model(x_seq, x_dense, ad_feats)
        self.assertEqual(len(embs), 2)
        self.assertEqual(logits.shape, (self.B, 3))

    def test_backward(self):
        x_seq    = torch.randn(self.B, self.L, 32)
        x_dense  = torch.randn(self.B, 16)
        ad_feats = torch.randn(self.B, 24)
        labels   = torch.randint(0, 2, (self.B, 3)).float()
        _, logits = self.model(x_seq, x_dense, ad_feats)
        loss = self.model.compute_loss(logits, labels)
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in self.model.parameters() if p.grad is not None]
        self.assertGreater(len(grad_norms), 0)
        self.assertTrue(all(g > 0 for g in grad_norms))


# ---------------------------------------------------------------------------
# Tests for limitation solutions
# ---------------------------------------------------------------------------

class TestSIMRetriever(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.B, self.L, self.D = 3, 50, 32

    def test_output_shape(self):
        sim = SIMRetriever(item_dim=self.D, target_dim=self.D, top_k=10)
        history      = torch.randn(self.B, self.L, self.D)
        history_cats = torch.randint(0, 5, (self.B, self.L))
        target_emb   = torch.randn(self.B, self.D)
        target_cat   = torch.randint(0, 5, (self.B,))
        out = sim(history, history_cats, target_emb, target_cat)
        self.assertEqual(out.shape, (self.B, self.D))

    def test_top_k_limits_retrieval(self):
        # top_k=5 with L=50; output should still be (B, D)
        sim = SIMRetriever(item_dim=self.D, target_dim=self.D, top_k=5)
        out = sim(
            torch.randn(self.B, self.L, self.D),
            torch.zeros(self.B, self.L, dtype=torch.long),
            torch.randn(self.B, self.D),
            torch.zeros(self.B, dtype=torch.long),
        )
        self.assertEqual(out.shape, (self.B, self.D))

    def test_gradient_flows(self):
        sim = SIMRetriever(item_dim=self.D, target_dim=self.D, top_k=10)
        history = torch.randn(self.B, self.L, self.D, requires_grad=True)
        out = sim(
            history,
            torch.randint(0, 4, (self.B, self.L)),
            torch.randn(self.B, self.D),
            torch.randint(0, 4, (self.B,)),
        )
        out.sum().backward()
        self.assertIsNotNone(history.grad)


class TestRecencyWeightedPooling(unittest.TestCase):

    def test_output_shape(self):
        rw = RecencyWeightedPooling(decay_lambda=0.1)
        x = torch.randn(4, 20, 32)
        out = rw(x)
        self.assertEqual(out.shape, (4, 32))

    def test_uniform_when_lambda_zero(self):
        # λ=0 → all weights equal → should match simple mean
        rw = RecencyWeightedPooling(decay_lambda=0.0)
        x = torch.randn(3, 10, 16)
        rw_out   = rw(x)
        mean_out = x.mean(dim=1)
        self.assertTrue(torch.allclose(rw_out, mean_out, atol=1e-5))

    def test_recent_items_weighted_higher(self):
        # Sequence: all zeros except last item = [1,1,...,1]
        # With decay, last item should dominate
        rw = RecencyWeightedPooling(decay_lambda=1.0)
        x = torch.zeros(1, 10, 4)
        x[0, -1] = 1.0
        out = rw(x)
        self.assertGreater(out.mean().item(), 0.5)

    def test_weights_sum_to_one(self):
        # Verify weights are normalised
        rw = RecencyWeightedPooling(decay_lambda=0.2)
        x = torch.ones(1, 8, 1)
        out = rw(x)
        self.assertAlmostEqual(out.item(), 1.0, places=5)


class TestHybridUserTower(unittest.TestCase):

    def test_output_shape(self):
        hybrid = HybridUserTower(lt_dim=96, item_dim=32, st_hidden=24, out_dim=64)
        lt_emb     = torch.randn(4, 96)
        recent_seq = torch.randn(4, 5, 32)
        out = hybrid(lt_emb, recent_seq)
        self.assertEqual(out.shape, (4, 64))

    def test_gradient_flows(self):
        hybrid = HybridUserTower(lt_dim=48, item_dim=16, st_hidden=16, out_dim=32)
        lt_emb     = torch.randn(2, 48, requires_grad=True)
        recent_seq = torch.randn(2, 3, 16)
        out = hybrid(lt_emb, recent_seq)
        out.sum().backward()
        self.assertIsNotNone(lt_emb.grad)


class TestGradientSurgery(unittest.TestCase):

    def test_output_length(self):
        grads = [torch.randn(32) for _ in range(3)]
        proj  = GradientSurgery.pcgrad(grads)
        self.assertEqual(len(proj), 3)

    def test_conflicting_gradients_projected(self):
        # g1 and g2 perfectly anti-parallel → after surgery g1 should be zero
        g1 = torch.ones(4)
        g2 = -torch.ones(4)
        proj = GradientSurgery.pcgrad([g1, g2])
        # g1 projected away from g2: g1 - (g1·g2/|g2|²) g2 = 0
        self.assertTrue(torch.allclose(proj[0], torch.zeros(4), atol=1e-5))

    def test_non_conflicting_gradients_unchanged(self):
        g1 = torch.tensor([1.0, 0.0])
        g2 = torch.tensor([0.0, 1.0])
        proj = GradientSurgery.pcgrad([g1, g2])
        # Orthogonal → dot = 0 → no projection
        self.assertTrue(torch.allclose(proj[0], g1, atol=1e-5))
        self.assertTrue(torch.allclose(proj[1], g2, atol=1e-5))

    def test_same_direction_unchanged(self):
        g1 = torch.tensor([1.0, 1.0])
        g2 = torch.tensor([2.0, 2.0])
        proj = GradientSurgery.pcgrad([g1, g2])
        self.assertTrue(torch.allclose(proj[0], g1, atol=1e-5))


if __name__ == '__main__':
    unittest.main(verbosity=2)
