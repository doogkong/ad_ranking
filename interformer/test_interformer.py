"""Smoke tests for InterFormer — verifies shapes and forward/backward pass."""

import torch
import torch.optim as optim
import pytest
from interformer import (
    InterFormer,
    MaskNet,
    CrossArch,
    SequenceArch,
    InteractionArch,
    InterFormerBlock,
    binary_cross_entropy_loss,
)

B, T, D = 4, 10, 32
N_DENSE = 3
VOCAB_SIZES = [100, 200, 50]   # 3 sparse features
N_SEQ_ITEMS = 500


def make_inputs(k: int = 1, device: str = "cpu"):
    dense = torch.randn(B, N_DENSE)
    sparse = torch.stack(
        [torch.randint(1, v, (B,)) for v in VOCAB_SIZES], dim=1
    )
    if k == 1:
        seqs = torch.randint(1, N_SEQ_ITEMS, (B, T))
    else:
        seqs = torch.randint(1, N_SEQ_ITEMS, (B, k, T))
    return dense.to(device), sparse.to(device), seqs.to(device)


def make_model(k: int = 1, **kwargs) -> InterFormer:
    return InterFormer(
        n_dense=N_DENSE,
        vocab_sizes=VOCAB_SIZES,
        n_seq_items=N_SEQ_ITEMS,
        d=D,
        T=T,
        k=k,
        n_layers=2,
        n_sum_x=2,
        n_heads=4,
        n_pma_seeds=2,
        n_recent=3,
        n_cross_layers=2,
        dropout=0.0,
        **kwargs,
    )


class TestMaskNet:
    def test_output_shape(self):
        net = MaskNet(k=3, d=D)
        seqs = [torch.randn(B, T, D) for _ in range(3)]
        out = net(seqs)
        assert out.shape == (B, T, D)


class TestCrossArch:
    def setup_method(self):
        self.cross = CrossArch(
            n_nonseq=4, d=D, n_sum_x=2, n_heads=4, n_pma_seeds=2, n_recent=3
        )

    def test_x_sum_shape(self):
        X = torch.randn(B, 4, D)
        x_sum = self.cross.get_x_sum(X)
        assert x_sum.shape == (B, 2, D)

    def test_s_sum_shape(self):
        S = torch.randn(B, T + 1, D)  # +1 for CLS
        s_sum = self.cross.get_s_sum(S)
        assert s_sum.shape == (B, 1 + 2 + 3, D)  # CLS + PMA + recent


class TestSequenceArch:
    def test_output_shape(self):
        arch = SequenceArch(d=D, n_heads=4, dropout=0.0)
        x_sum = torch.randn(B, 2, D)
        S = torch.randn(B, T + 1, D)
        out = arch(x_sum, S)
        assert out.shape == (B, T + 1, D)


class TestInteractionArch:
    def test_output_shape(self):
        arch = InteractionArch(n_nonseq=4, d=D, n_sum_seq=6)
        X = torch.randn(B, 4, D)
        s_sum = torch.randn(B, 6, D)
        out = arch(X, s_sum)
        assert out.shape == (B, 4, D)


class TestInterFormerBlock:
    def test_shapes_preserved(self):
        block = InterFormerBlock(
            n_nonseq=4, d=D, n_sum_x=2, n_heads=4, n_pma_seeds=2, n_recent=3
        )
        X = torch.randn(B, 4, D)
        S = torch.randn(B, T + 1, D)
        X_new, S_new = block(X, S)
        assert X_new.shape == X.shape
        assert S_new.shape == S.shape


class TestInterFormer:
    def test_output_shape(self):
        model = make_model()
        dense, sparse, seqs = make_inputs()
        logits = model(dense, sparse, seqs)
        assert logits.shape == (B, 1)

    def test_sparse_only(self):
        model = make_model()
        _, sparse, seqs = make_inputs()
        logits = model(sparse_feats=sparse, sequences=seqs)
        assert logits.shape == (B, 1)

    def test_multi_sequence(self):
        model = make_model(k=3)
        dense, sparse, seqs = make_inputs(k=3)
        logits = model(dense, sparse, seqs)
        assert logits.shape == (B, 1)

    def test_backward_pass(self):
        model = make_model()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        dense, sparse, seqs = make_inputs()
        labels = torch.randint(0, 2, (B,)).float()

        logits = model(dense, sparse, seqs)
        loss = binary_cross_entropy_loss(logits, labels)
        loss.backward()
        optimizer.step()

        assert not torch.isnan(loss)

    def test_output_range_after_sigmoid(self):
        model = make_model()
        model.eval()
        dense, sparse, seqs = make_inputs()
        with torch.no_grad():
            probs = torch.sigmoid(model(dense, sparse, seqs))
        assert probs.min() >= 0.0 and probs.max() <= 1.0

    def test_param_count(self):
        model = make_model()
        n_params = sum(p.numel() for p in model.parameters())
        # sanity check: should be in the hundreds-of-thousands range for d=32
        assert 100_000 < n_params < 50_000_000

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
    def test_gpu(self):
        model = make_model().cuda()
        dense, sparse, seqs = make_inputs(device="cuda")
        logits = model(dense, sparse, seqs)
        assert logits.device.type == "cuda"


if __name__ == "__main__":
    # Quick manual smoke test
    model = make_model()
    dense, sparse, seqs = make_inputs()
    labels = torch.randint(0, 2, (B,)).float()

    model.train()
    logits = model(dense, sparse, seqs)
    loss = binary_cross_entropy_loss(logits, labels)
    loss.backward()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"logits shape : {logits.shape}")
    print(f"loss         : {loss.item():.4f}")
    print(f"param count  : {n_params:,}")
    print("All checks passed.")
