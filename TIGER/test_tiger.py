"""Tests for the TIGER implementation.

Run with:
    pytest test_tiger.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from tiger import (
    ResidualQuantizer,
    rqvae_quant_loss,
    RQVAE,
    resolve_collisions,
    stable_hash,
    SemanticIDVocab,
    TigerTransformer,
    generate_semantic_id,
    SemanticIDLookup,
    category_entropy,
)

B, DIM, LEVELS, K = 6, 16, 3, 8


# ---------------------------------------------------------------------------
# ResidualQuantizer
# ---------------------------------------------------------------------------

class TestResidualQuantizer:
    def _make(self):
        return ResidualQuantizer(num_levels=LEVELS, codebook_size=K, dim=DIM)

    def test_output_shapes(self):
        rq = self._make()
        z = torch.randn(B, DIM)
        quantized_st, codes, residuals, quant_vectors = rq(z)
        assert quantized_st.shape == (B, DIM)
        assert codes.shape == (B, LEVELS)
        assert len(residuals) == len(quant_vectors) == LEVELS

    def test_codes_within_codebook_range(self):
        rq = self._make()
        _, codes, _, _ = rq(torch.randn(B, DIM))
        assert codes.min() >= 0
        assert codes.max() < K

    def test_quantized_equals_sum_of_quant_vectors(self):
        rq = self._make()
        z = torch.randn(B, DIM)
        quantized_st, _, _, quant_vectors = rq(z)
        expected = torch.stack(quant_vectors, dim=0).sum(dim=0)
        assert torch.allclose(quantized_st, expected, atol=1e-5)

    def test_straight_through_gradient_reaches_z(self):
        rq = self._make()
        z = torch.randn(B, DIM, requires_grad=True)
        quantized_st, _, _, _ = rq(z)
        quantized_st.sum().backward()
        assert z.grad is not None
        assert torch.allclose(z.grad, torch.ones_like(z))  # identity gradient via straight-through

    def test_kmeans_init_reduces_quantization_error(self):
        torch.manual_seed(0)
        z = torch.randn(100, DIM)

        def total_error(rq):
            _, _, residuals, quant_vectors = rq(z)
            return sum(F.mse_loss(r, e).item() for r, e in zip(residuals, quant_vectors))

        rq_random = self._make()
        error_random = total_error(rq_random)

        rq_kmeans = self._make()
        rq_kmeans.kmeans_init(z, n_iter=15)
        error_kmeans = total_error(rq_kmeans)

        assert error_kmeans < error_random


class TestRQVAEQuantLoss:
    def test_zero_when_residual_matches_codeword(self):
        r = [torch.randn(4, DIM)]
        e = [r[0].clone()]
        loss = rqvae_quant_loss(r, e, beta=0.25)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_nonnegative(self):
        r = [torch.randn(4, DIM) for _ in range(3)]
        e = [torch.randn(4, DIM) for _ in range(3)]
        loss = rqvae_quant_loss(r, e)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# RQVAE
# ---------------------------------------------------------------------------

class TestRQVAE:
    def _make(self):
        return RQVAE(input_dim=DIM, hidden_dims=[12, 10], latent_dim=6, num_levels=LEVELS, codebook_size=K)

    def test_forward_shapes(self):
        model = self._make()
        x = torch.randn(B, DIM)
        x_hat, codes, residuals, quant_vectors = model(x)
        assert x_hat.shape == (B, DIM)
        assert codes.shape == (B, LEVELS)

    def test_loss_dict_keys(self):
        model = self._make()
        losses, codes = model.loss(torch.randn(B, DIM))
        assert set(losses.keys()) == {"recon", "rqvae", "total"}
        assert torch.allclose(losses["total"], losses["recon"] + losses["rqvae"])

    def test_encode_codes_shape_and_range(self):
        model = self._make()
        codes = model.encode_codes(torch.randn(20, DIM))
        assert codes.shape == (20, LEVELS)
        assert codes.min() >= 0 and codes.max() < K

    def test_encode_codes_no_grad(self):
        model = self._make()
        codes = model.encode_codes(torch.randn(5, DIM))
        assert not codes.requires_grad

    def test_training_reduces_reconstruction_loss(self):
        torch.manual_seed(0)
        model = self._make()
        x = torch.randn(50, DIM)
        model.init_codebooks_kmeans(x, n_iter=10)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        first_loss = None
        for step in range(150):
            opt.zero_grad()
            losses, _ = model.loss(x)
            losses["total"].backward()
            opt.step()
            if step == 0:
                first_loss = losses["recon"].item()
        final_loss = model.loss(x)[0]["recon"].item()
        assert final_loss < first_loss

    def test_init_codebooks_kmeans_changes_codebook(self):
        model = self._make()
        before = model.quantizer.codebooks[0].data.clone()
        model.init_codebooks_kmeans(torch.randn(50, DIM), n_iter=5)
        after = model.quantizer.codebooks[0].data
        assert not torch.allclose(before, after)


# ---------------------------------------------------------------------------
# resolve_collisions
# ---------------------------------------------------------------------------

class TestResolveCollisions:
    def test_no_collisions_all_get_zero(self):
        codes = torch.tensor([[1, 2], [3, 4], [5, 6]])
        resolved = resolve_collisions(codes)
        assert resolved.shape == (3, 3)
        assert torch.all(resolved[:, -1] == 0)

    def test_duplicates_get_incrementing_extra_token(self):
        codes = torch.tensor([[1, 2], [1, 2], [1, 2], [3, 4]])
        resolved = resolve_collisions(codes)
        assert resolved[:, -1].tolist() == [0, 1, 2, 0]

    def test_result_is_fully_unique(self):
        codes = torch.tensor([[1, 2], [1, 2], [3, 4], [1, 2], [3, 4]])
        resolved = resolve_collisions(codes)
        rows = [tuple(r.tolist()) for r in resolved]
        assert len(set(rows)) == len(rows)


# ---------------------------------------------------------------------------
# stable_hash
# ---------------------------------------------------------------------------

class TestStableHash:
    def test_deterministic(self):
        assert stable_hash("user_1", 100) == stable_hash("user_1", 100)

    def test_within_bucket_range(self):
        for key in ["a", "b", "user_42", 12345]:
            h = stable_hash(key, 50)
            assert 0 <= h < 50

    def test_different_keys_can_differ(self):
        hashes = {stable_hash(f"user_{i}", 1000) for i in range(20)}
        assert len(hashes) > 1


# ---------------------------------------------------------------------------
# SemanticIDVocab
# ---------------------------------------------------------------------------

class TestSemanticIDVocab:
    def _make(self):
        return SemanticIDVocab(num_levels=LEVELS, codebook_size=K, num_user_buckets=50)

    def test_vocab_size_formula(self):
        vocab = self._make()
        assert vocab.vocab_size == vocab.num_special + LEVELS * K + 50

    def test_level_ranges_are_disjoint(self):
        vocab = self._make()
        ranges = [vocab.level_range(l) for l in range(LEVELS)]
        for i, (s1, e1) in enumerate(ranges):
            for j, (s2, e2) in enumerate(ranges):
                if i != j:
                    assert e1 <= s2 or e2 <= s1

    def test_item_token_codes_roundtrip(self):
        vocab = self._make()
        codes = [3, 5, 1]
        tokens = vocab.item_tokens(codes)
        assert vocab.codes_from_tokens(tokens) == codes

    def test_item_tokens_below_user_block(self):
        vocab = self._make()
        tokens = vocab.item_tokens([K - 1] * LEVELS)
        assert all(t < vocab.user_block_start for t in tokens)

    def test_user_token_within_user_block(self):
        vocab = self._make()
        tok = vocab.user_token("some_user")
        assert vocab.user_block_start <= tok < vocab.vocab_size

    def test_build_encoder_input_structure(self):
        vocab = self._make()
        history = [[1, 2, 3], [4, 5, 6]]
        tokens = vocab.build_encoder_input("u1", history)
        assert tokens[0] == vocab.user_token("u1")
        assert tokens[1:] == vocab.item_tokens([1, 2, 3]) + vocab.item_tokens([4, 5, 6])
        assert len(tokens) == 1 + LEVELS * len(history)


# ---------------------------------------------------------------------------
# TigerTransformer
# ---------------------------------------------------------------------------

class TestTigerTransformer:
    def _make(self, vocab_size=64):
        return TigerTransformer(
            vocab_size=vocab_size, d_model=16, n_head=2,
            num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32, max_len=32,
        )

    def test_forward_shape(self):
        model = self._make()
        enc_ids = torch.randint(3, 60, (2, 10))
        dec_ids = torch.randint(3, 60, (2, 4))
        logits = model(enc_ids, dec_ids)
        assert logits.shape == (2, 4, 64)

    def test_gradient(self):
        model = self._make()
        enc_ids = torch.randint(3, 60, (2, 10))
        dec_ids = torch.randint(3, 60, (2, 4))
        logits = model(enc_ids, dec_ids)
        logits.sum().backward()
        assert model.token_emb.weight.grad is not None

    def test_decoder_is_causal(self):
        model = self._make()
        model.eval()
        enc_ids = torch.randint(3, 60, (1, 10))
        dec_ids = torch.randint(3, 60, (1, 5))
        dec_ids_perturbed = dec_ids.clone()
        dec_ids_perturbed[0, -1] = (dec_ids_perturbed[0, -1] + 1) % 60

        with torch.no_grad():
            memory = model.encode(enc_ids)
            out1 = model.decode(dec_ids, memory)
            out2 = model.decode(dec_ids_perturbed, memory)
        assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5)


# ---------------------------------------------------------------------------
# generate_semantic_id
# ---------------------------------------------------------------------------

class TestGenerateSemanticId:
    def _setup(self):
        vocab = SemanticIDVocab(num_levels=LEVELS, codebook_size=K, num_user_buckets=20)
        model = TigerTransformer(
            vocab_size=vocab.vocab_size, d_model=16, n_head=2,
            num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=32, max_len=32,
        )
        enc_ids = torch.tensor([vocab.build_encoder_input("u1", [[1, 2, 3], [4, 5, 6]])])
        return vocab, model, enc_ids

    def test_output_shape(self):
        vocab, model, enc_ids = self._setup()
        codes = generate_semantic_id(model, enc_ids, vocab, temperature=0.0)
        assert codes.shape == (1, LEVELS)

    def test_codes_within_per_level_range(self):
        vocab, model, enc_ids = self._setup()
        codes = generate_semantic_id(model, enc_ids, vocab, temperature=1.0)
        assert torch.all(codes >= 0) and torch.all(codes < K)

    def test_greedy_is_deterministic(self):
        vocab, model, enc_ids = self._setup()
        model.eval()
        c1 = generate_semantic_id(model, enc_ids, vocab, temperature=0.0)
        c2 = generate_semantic_id(model, enc_ids, vocab, temperature=0.0)
        assert torch.equal(c1, c2)

    def test_sampling_is_reproducible_with_generator(self):
        vocab, model, enc_ids = self._setup()
        g1 = torch.Generator().manual_seed(0)
        g2 = torch.Generator().manual_seed(0)
        c1 = generate_semantic_id(model, enc_ids, vocab, temperature=1.0, generator=g1)
        c2 = generate_semantic_id(model, enc_ids, vocab, temperature=1.0, generator=g2)
        assert torch.equal(c1, c2)


# ---------------------------------------------------------------------------
# SemanticIDLookup
# ---------------------------------------------------------------------------

class TestSemanticIDLookup:
    def test_retrieves_known_item(self):
        codes = torch.tensor([[1, 2, 3], [4, 5, 6]])
        lookup = SemanticIDLookup(item_ids=["itemA", "itemB"], codes=codes)
        result = lookup.retrieve(torch.tensor([[4, 5, 6]]))
        assert result == ["itemB"]

    def test_unknown_id_returns_none(self):
        codes = torch.tensor([[1, 2, 3]])
        lookup = SemanticIDLookup(item_ids=["itemA"], codes=codes)
        result = lookup.retrieve(torch.tensor([[9, 9, 9]]))
        assert result == [None]

    def test_batch_retrieval(self):
        codes = torch.tensor([[1, 2], [3, 4], [5, 6]])
        lookup = SemanticIDLookup(item_ids=[10, 20, 30], codes=codes)
        result = lookup.retrieve(torch.tensor([[3, 4], [9, 9], [1, 2]]))
        assert result == [20, None, 10]


# ---------------------------------------------------------------------------
# category_entropy
# ---------------------------------------------------------------------------

class TestCategoryEntropy:
    def test_single_category_is_zero(self):
        assert category_entropy(["A", "A", "A"]) == pytest.approx(0.0)

    def test_uniform_distribution_matches_log2(self):
        cats = ["A", "B", "C", "D"] * 5
        assert category_entropy(cats) == pytest.approx(2.0)  # log2(4)

    def test_empty_is_zero(self):
        assert category_entropy([]) == 0.0

    def test_more_diverse_has_higher_entropy(self):
        low = ["A"] * 9 + ["B"]
        high = ["A", "B", "C", "D"] * 2 + ["E", "F"]
        assert category_entropy(high) > category_entropy(low)
