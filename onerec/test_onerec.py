"""Tests for OneRec implementation.

Run with:
    pytest test_onerec.py -v
"""

import copy

import pytest
import torch
import torch.nn as nn

from onerec import (
    RMSNorm,
    BalancedKMeans,
    ResidualQuantizer,
    MultiHeadAttention,
    FFN,
    MoELayer,
    EncoderLayer,
    OneRecEncoder,
    DecoderLayer,
    OneRecDecoder,
    RewardModel,
    OneRec,
    _build_decoder_input,
    ntp_loss,
    dpo_loss,
    IterativePreferenceAlignment,
)

B   = 2    # batch size
D   = 32   # model dim
K   = 16   # codebook size (small for tests)
L   = 3    # code levels
m   = 3    # session size
T_E = 12   # encoder sequence length
H   = 4    # attention heads
NE  = 4    # num experts
TK  = 2    # top-k experts


def _vocab():
    return K * L + 2   # vocab_size used in encoder/decoder


def _src(b=B):
    return torch.randint(0, _vocab() - 2, (b, T_E))


def _codes(b=B):
    return torch.randint(0, K, (b, m, L))


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_shape(self):
        assert RMSNorm(D)(torch.randn(B, 5, D)).shape == (B, 5, D)

    def test_unit_rms(self):
        n = RMSNorm(D)
        n.weight.data.fill_(1.0)
        x = torch.randn(B, 8, D)
        rms = n(x).pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_gradient(self):
        x = torch.randn(B, 4, D, requires_grad=True)
        RMSNorm(D)(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# BalancedKMeans (offline, numpy)
# ---------------------------------------------------------------------------

class TestBalancedKMeans:
    def test_fit_encode(self):
        import numpy as np
        X = np.random.randn(100, 8).astype(np.float32)
        bkm = BalancedKMeans(K=10, max_iter=10)
        bkm.fit(X)
        codes = bkm.encode(X)
        assert codes.shape == (100,)
        assert codes.min() >= 0 and codes.max() < 10

    def test_balanced_clusters(self):
        import numpy as np
        N, Kc = 100, 10
        X = np.random.randn(N, 4).astype(np.float32)
        bkm = BalancedKMeans(K=Kc, max_iter=20)
        bkm.fit(X)
        codes = bkm.encode(X)
        counts = [(codes == k).sum() for k in range(Kc)]
        # Each cluster should be close to N/K = 10
        for c in counts:
            assert abs(c - N // Kc) <= 2


# ---------------------------------------------------------------------------
# ResidualQuantizer
# ---------------------------------------------------------------------------

class TestResidualQuantizer:
    def _make(self):
        return ResidualQuantizer(num_levels=L, codebook_size=K, embed_dim=D)

    def test_encode_shape(self):
        rq = self._make()
        e = torch.randn(B, D)
        codes = rq.encode(e)
        assert codes.shape == (B, L)
        assert codes.min() >= 0 and codes.max() < K

    def test_decode_shape(self):
        rq = self._make()
        codes = torch.randint(0, K, (B, L))
        out = rq.decode(codes)
        assert out.shape == (B, D)

    def test_forward_straight_through(self):
        rq = self._make()
        e = torch.randn(B, D, requires_grad=True)
        codes, e_st = rq(e)
        e_st.sum().backward()
        assert e.grad is not None

    def test_round_trip_shape(self):
        rq = self._make()
        e = torch.randn(5, D)
        codes, _ = rq(e)
        recon = rq.decode(codes)
        assert recon.shape == (5, D)


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

class TestMultiHeadAttention:
    def test_self_attn_shape(self):
        attn = MultiHeadAttention(D, H)
        x = torch.randn(B, 10, D)
        assert attn(x, x, x).shape == (B, 10, D)

    def test_cross_attn_shape(self):
        attn = MultiHeadAttention(D, H)
        q = torch.randn(B, 5, D)
        kv = torch.randn(B, 8, D)
        assert attn(q, kv, kv).shape == (B, 5, D)

    def test_causal_mask(self):
        # With a mask that blocks all keys, output should be zero (softmax → -inf → 0)
        attn = MultiHeadAttention(D, H)
        x = torch.randn(B, 4, D)
        mask = torch.ones(4, 4, dtype=torch.bool)
        out = attn(x, x, x, mask)
        assert torch.all(out == 0) or out.abs().max() < 1e-4  # nan_to_num → zeros

    def test_gradient(self):
        attn = MultiHeadAttention(D, H)
        x = torch.randn(B, 6, D, requires_grad=True)
        attn(x, x, x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# MoELayer
# ---------------------------------------------------------------------------

class TestMoELayer:
    def test_shape(self):
        moe = MoELayer(D, num_experts=NE, top_k=TK, expand=2)
        x = torch.randn(B, 5, D)
        assert moe(x).shape == (B, 5, D)

    def test_top_k_less_than_experts(self):
        moe = MoELayer(D, num_experts=6, top_k=2, expand=2)
        x = torch.randn(B, 3, D)
        assert moe(x).shape == (B, 3, D)

    def test_gradient(self):
        moe = MoELayer(D, num_experts=NE, top_k=TK, expand=2)
        x = torch.randn(B, 4, D, requires_grad=True)
        moe(x).sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# OneRecEncoder
# ---------------------------------------------------------------------------

class TestOneRecEncoder:
    def _make(self):
        return OneRecEncoder(D, H, n_layers=2, codebook_size=K,
                             num_levels=L, ffn_expand=2)

    def test_output_shape(self):
        enc = self._make()
        src = _src()
        out = enc(src)
        assert out.shape == (B, T_E, D)

    def test_gradient(self):
        enc = self._make()
        src = _src()
        enc(src).sum().backward()
        assert enc.embed.weight.grad is not None


# ---------------------------------------------------------------------------
# OneRecDecoder
# ---------------------------------------------------------------------------

class TestOneRecDecoder:
    def _make(self):
        return OneRecDecoder(D, H, n_layers=2, codebook_size=K, num_levels=L,
                             num_experts=NE, top_k=TK, ffn_expand=2)

    def _enc_out(self):
        return torch.randn(B, T_E, D)

    def test_logits_shape(self):
        dec = self._make()
        codes = _codes()
        dec_in = _build_decoder_input(codes, dec.BOS_ID, K)
        logits = dec(dec_in, self._enc_out())
        assert len(logits) == L
        T_dec = m * (1 + L)
        for lg in logits:
            assert lg.shape == (B, T_dec, K)

    def test_greedy_decode_shape(self):
        dec = self._make()
        codes = dec.greedy_decode(self._enc_out(), m=m)
        assert codes.shape == (B, m, L)
        assert codes.min() >= 0 and codes.max() < K

    def test_gradient(self):
        dec = self._make()
        codes = _codes()
        dec_in = _build_decoder_input(codes, dec.BOS_ID, K)
        enc_out = self._enc_out().requires_grad_(True)
        logits = dec(dec_in, enc_out)
        logits[0].sum().backward()
        assert enc_out.grad is not None


# ---------------------------------------------------------------------------
# _build_decoder_input
# ---------------------------------------------------------------------------

class TestBuildDecoderInput:
    def test_shape(self):
        codes = _codes()
        BOS = K * L
        out = _build_decoder_input(codes, BOS, K)
        assert out.shape == (B, m * (1 + L))

    def test_bos_positions(self):
        codes = _codes()
        BOS = K * L
        out = _build_decoder_input(codes, BOS, K)
        for i in range(m):
            assert (out[:, i * (1 + L)] == BOS).all()


# ---------------------------------------------------------------------------
# NTP loss
# ---------------------------------------------------------------------------

class TestNTPLoss:
    def test_positive(self):
        # Random logits → loss should be positive
        logits = [torch.randn(B, m * (1 + L), K) for _ in range(L)]
        codes  = _codes()
        loss   = ntp_loss(logits, codes, K)
        assert loss.item() > 0

    def test_perfect_prediction(self):
        # One-hot logits exactly at target → near-zero loss
        codes  = _codes()
        T_dec  = m * (1 + L)
        logits = []
        for level in range(L):
            lg = torch.full((B, T_dec, K), -1e9)
            for i in range(m):
                pos = i * (1 + L) + level
                lg[:, pos, :].scatter_(-1, codes[:, i, level:level+1], 1e9)
            logits.append(lg)
        loss = ntp_loss(logits, codes, K)
        assert loss.item() < 0.01

    def test_gradient(self):
        logits = [torch.randn(B, m * (1 + L), K, requires_grad=True) for _ in range(L)]
        codes  = _codes()
        ntp_loss(logits, codes, K).backward()
        for lg in logits:
            assert lg.grad is not None


# ---------------------------------------------------------------------------
# RewardModel
# ---------------------------------------------------------------------------

class TestRewardModel:
    def _make(self):
        return RewardModel(D, n_heads=H)

    def test_output_shapes(self):
        rm = self._make()
        u  = torch.randn(B, D)
        v  = torch.randn(B, m, D)
        out = rm(u, v)
        for key in ("swt", "vtr", "ltr"):
            assert out[key].shape == (B,)

    def test_score_shape(self):
        rm = self._make()
        assert rm.score(torch.randn(B, D), torch.randn(B, m, D)).shape == (B,)

    def test_gradient(self):
        rm = self._make()
        u  = torch.randn(B, D, requires_grad=True)
        v  = torch.randn(B, m, D, requires_grad=True)
        rm.score(u, v).sum().backward()
        assert u.grad is not None and v.grad is not None

    def test_different_sessions_different_scores(self):
        rm = self._make()
        u  = torch.randn(B, D)
        s1 = torch.randn(B, m, D)
        s2 = torch.randn(B, m, D)
        assert not torch.allclose(rm.score(u, s1), rm.score(u, s2))


# ---------------------------------------------------------------------------
# OneRec (full model)
# ---------------------------------------------------------------------------

class TestOneRec:
    def _make(self):
        return OneRec(d_model=D, n_heads=H, n_enc_layers=2, n_dec_layers=2,
                      codebook_size=K, num_levels=L,
                      num_experts=NE, top_k=TK, ffn_expand=2)

    def test_ntp_loss_positive(self):
        model = self._make()
        loss  = model(_src(), _codes())
        assert loss.item() > 0

    def test_backward(self):
        model = self._make()
        model(_src(), _codes()).backward()
        first_p = next(model.parameters())
        assert first_p.grad is not None

    def test_grad_all_parameters(self):
        model = self._make()
        model(_src(), _codes()).backward()
        no_grad = [n for n, p in model.named_parameters()
                   if p.requires_grad and p.grad is None]
        assert no_grad == [], f"Missing grad: {no_grad}"

    def test_generate_shape(self):
        model = self._make()
        model.eval()
        with torch.no_grad():
            codes = model.generate(_src(), m=m)
        assert codes.shape == (B, m, L)

    def test_generate_valid_codes(self):
        model = self._make()
        model.eval()
        with torch.no_grad():
            codes = model.generate(_src(), m=m)
        assert codes.min() >= 0 and codes.max() < K

    def test_deterministic_with_seed(self):
        model = self._make()
        model.eval()
        src = _src()
        with torch.no_grad():
            torch.manual_seed(0)
            c1 = model.generate(src, m=m)
            torch.manual_seed(0)
            c2 = model.generate(src, m=m)
        assert torch.equal(c1, c2)

    def test_different_history_different_output(self):
        model = self._make()
        model.eval()
        with torch.no_grad():
            c1 = model.generate(_src(), m=m)
            c2 = model.generate(_src(), m=m)
        assert not torch.equal(c1, c2)


# ---------------------------------------------------------------------------
# IterativePreferenceAlignment
# ---------------------------------------------------------------------------

class TestIPA:
    def _make(self):
        model = OneRec(d_model=D, n_heads=H, n_enc_layers=2, n_dec_layers=2,
                       codebook_size=K, num_levels=L,
                       num_experts=NE, top_k=TK, ffn_expand=2)
        rm    = RewardModel(D, n_heads=H)
        ipa   = IterativePreferenceAlignment(model, rm, r_dpo=1.0, n_candidates=3)
        return model, rm, ipa

    def test_step_ntp_only(self):
        model = OneRec(d_model=D, n_heads=H, n_enc_layers=2, n_dec_layers=2,
                       codebook_size=K, num_levels=L,
                       num_experts=NE, top_k=TK, ffn_expand=2)
        rm  = RewardModel(D, n_heads=H)
        ipa = IterativePreferenceAlignment(model, rm, r_dpo=0.0, n_candidates=3)

        def item_emb_fn(codes):
            return torch.randn(codes.size(0), codes.size(1), D)

        loss = ipa.step(_src(), _codes(), torch.randn(B, D), item_emb_fn)
        assert loss.item() > 0

    def test_step_with_dpo(self):
        _, _, ipa = self._make()

        def item_emb_fn(codes):
            return torch.randn(codes.size(0), codes.size(1), D)

        loss = ipa.step(_src(), _codes(), torch.randn(B, D), item_emb_fn)
        assert loss.item() > 0

    def test_update_reference(self):
        model, rm, ipa = self._make()
        # Modify model weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ipa.update_reference()
        # Reference should now match model
        for (_, p_model), (_, p_ref) in zip(
            model.named_parameters(), ipa.ref_model.named_parameters()
        ):
            assert torch.allclose(p_model, p_ref)

    def test_backward_through_dpo_step(self):
        _, _, ipa = self._make()
        opt = torch.optim.Adam(ipa.model.parameters(), lr=1e-3)

        def item_emb_fn(codes):
            return torch.randn(codes.size(0), codes.size(1), D)

        opt.zero_grad()
        loss = ipa.step(_src(), _codes(), torch.randn(B, D), item_emb_fn)
        loss.backward()
        opt.step()
        first_p = next(ipa.model.parameters())
        assert first_p.grad is not None
