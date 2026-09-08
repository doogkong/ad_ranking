"""TIGER: Transformer Index for GEnerative Recommenders.

Reference PyTorch implementation of TIGER, from *"Recommender Systems with
Generative Retrieval"* (Rajput et al., Google/Wisconsin-Madison, NeurIPS
2023), arXiv:2305.05065.

TIGER reframes sequential recommendation as generative retrieval: instead of
embedding a user and all candidate items into one vector space and running
approximate nearest-neighbor search, TIGER assigns every item a short,
semantically meaningful **Semantic ID** — a tuple of discrete codewords
derived from the item's content embedding — and trains a sequence-to-sequence
Transformer to autoregressively decode the next item's Semantic ID from a
user's interaction history. Two stages:

  Stage 1  Semantic ID generation  -> RQVAE, ResidualQuantizer, resolve_collisions
  Stage 2  Generative retrieval    -> SemanticIDVocab, TigerTransformer,
                                       generate_semantic_id, SemanticIDLookup

Because items sharing Semantic-ID prefixes are semantically related by
construction (Semantic IDs are hierarchical: coarser codewords first, finer
ones later), this also gives TIGER two capabilities plain atomic-ID sequence
models lack: it can score never-seen items (cold start, Sec 4.3) and its
predictions can be made more or less diverse just by adjusting decoding
temperature over which hierarchy level is being sampled (Sec 4.3).
"""

import hashlib
import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Stage 1 — Semantic ID generation: RQ-VAE (Sec 3.1, Fig. 3)
# ---------------------------------------------------------------------------

class ResidualQuantizer(nn.Module):
    """m-level residual vector quantizer (Fig. 3): quantizes a latent z
    against `num_levels` codebooks, one level at a time. At level d, the
    current residual r_d is mapped to its nearest codeword e_{c_d} in that
    level's codebook; the next residual is r_{d+1} = r_d - e_{c_d}. The
    resulting Semantic ID is the tuple of codeword indices (c_0,...,c_{m-1}).

    Uses a straight-through estimator (Van Den Oord et al., 2017) so
    gradients flow from the decoder back to the encoder as if quantization
    were the identity function — the standard mechanism that makes VQ-/RQ-VAE
    end-to-end trainable.
    """

    def __init__(self, num_levels: int, codebook_size: int, dim: int) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.dim = dim
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, dim) * 0.01) for _ in range(num_levels)
        ])

    @staticmethod
    def _nearest(x: Tensor, codebook: Tensor) -> Tensor:
        return torch.cdist(x, codebook).argmin(dim=-1)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor, List[Tensor], List[Tensor]]:
        """
        Args:
            z: (B, dim) encoder output.
        Returns:
            quantized_st: (B, dim), straight-through quantized representation
                sum_d e_{c_d}, with gradients routed to z.
            codes: (B, num_levels) codeword index per level.
            residuals: list of num_levels tensors (B, dim), r_0,...,r_{m-1}.
            quant_vectors: list of num_levels tensors (B, dim), e_{c_0},...,e_{c_{m-1}}.
        """
        residual = z
        residuals, quant_vectors, codes = [], [], []
        for codebook in self.codebooks:
            code = self._nearest(residual, codebook)
            e = codebook[code]
            residuals.append(residual)
            quant_vectors.append(e)
            codes.append(code)
            residual = residual - e
        codes = torch.stack(codes, dim=1)
        quantized = torch.stack(quant_vectors, dim=0).sum(dim=0)
        quantized_st = z + (quantized - z).detach()
        return quantized_st, codes, residuals, quant_vectors

    def kmeans_init(self, z: Tensor, n_iter: int = 20, generator: Optional[torch.Generator] = None) -> None:
        """K-means-based codebook initialization (Sec 3.1): fit level 0's
        codebook on z via Lloyd's algorithm, then fit each subsequent level's
        codebook on the residual left after the previous levels — this is
        what the paper uses to avoid codebook collapse (most input mapping to
        only a few codebook vectors).
        """
        generator = generator or torch.Generator()
        residual = z.detach()
        with torch.no_grad():
            for codebook in self.codebooks:
                n = residual.shape[0]
                init_idx = torch.randperm(n, generator=generator)[: self.codebook_size]
                centroids = residual[init_idx].clone()
                for _ in range(n_iter):
                    assignments = self._nearest(residual, centroids)
                    for c in range(self.codebook_size):
                        mask = assignments == c
                        if mask.any():
                            centroids[c] = residual[mask].mean(dim=0)
                codebook.data.copy_(centroids)
                code = self._nearest(residual, centroids)
                residual = residual - centroids[code]


def rqvae_quant_loss(residuals: List[Tensor], quant_vectors: List[Tensor], beta: float = 0.25) -> Tensor:
    """L_rqvae = sum_d ||sg[r_d] - e_{c_d}||^2 + beta * ||r_d - sg[e_{c_d}]||^2

    The first (codebook) term pulls each codeword toward the residuals
    assigned to it; the second (commitment) term, scaled by beta, pulls the
    encoder's output toward the codeword it was assigned — the standard
    VQ-VAE-style loss (Van Den Oord et al., 2017), applied per residual level.
    """
    loss = torch.zeros((), device=residuals[0].device)
    for r, e in zip(residuals, quant_vectors):
        loss = loss + F.mse_loss(r.detach(), e) + beta * F.mse_loss(r, e.detach())
    return loss


class RQVAE(nn.Module):
    """Residual-Quantized VAE for Semantic ID generation (Sec 3.1).

    x -> encoder -> z -> ResidualQuantizer -> quantized z_hat -> decoder -> x_hat
    L(x) = L_recon + L_rqvae,   L_recon = ||x - x_hat||^2

    Args:
        input_dim: dimension of the content embedding x (e.g. 768 for
            Sentence-T5), hidden_dims: encoder MLP hidden widths (mirrored by
            the decoder), latent_dim: dimension of z, num_levels: m,
            codebook_size: K, beta: commitment-loss weight.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        num_levels: int,
        codebook_size: int,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.beta = beta

        enc_dims = [input_dim] + hidden_dims + [latent_dim]
        enc_layers: List[nn.Module] = []
        for i in range(len(enc_dims) - 1):
            enc_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            if i < len(enc_dims) - 2:
                enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)

        dec_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        dec_layers: List[nn.Module] = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

        self.quantizer = ResidualQuantizer(num_levels, codebook_size, latent_dim)

    def forward(self, x: Tensor):
        z = self.encoder(x)
        quantized_st, codes, residuals, quant_vectors = self.quantizer(z)
        x_hat = self.decoder(quantized_st)
        return x_hat, codes, residuals, quant_vectors

    def loss(self, x: Tensor) -> Tuple[Dict[str, Tensor], Tensor]:
        x_hat, codes, residuals, quant_vectors = self(x)
        recon = F.mse_loss(x_hat, x)
        rq = rqvae_quant_loss(residuals, quant_vectors, self.beta)
        return {"recon": recon, "rqvae": rq, "total": recon + rq}, codes

    @torch.no_grad()
    def encode_codes(self, x: Tensor) -> Tensor:
        """Returns (B, num_levels) Semantic IDs for a batch of content embeddings."""
        z = self.encoder(x)
        _, codes, _, _ = self.quantizer(z)
        return codes

    def init_codebooks_kmeans(self, x: Tensor, n_iter: int = 20) -> None:
        with torch.no_grad():
            z = self.encoder(x)
        self.quantizer.kmeans_init(z, n_iter=n_iter)


def resolve_collisions(codes: Tensor) -> Tensor:
    """Appends a disambiguating extra token to any items that share the same
    Semantic ID (Sec 3.1, "Handling Collisions"): the k-th item (in input
    order) among those sharing a given (c_0,...,c_{m-1}) tuple gets an extra
    token k, so e.g. two items sharing (12, 24, 52) become (12, 24, 52, 0)
    and (12, 24, 52, 1).

    Args:
        codes: (N, m).
    Returns:
        (N, m + 1), unique across rows.
    """
    seen: Dict[Tuple[int, ...], int] = {}
    extra = []
    for row in codes:
        key = tuple(row.tolist())
        idx = seen.get(key, 0)
        extra.append(idx)
        seen[key] = idx + 1
    extra_t = torch.tensor(extra, dtype=codes.dtype, device=codes.device).unsqueeze(1)
    return torch.cat([codes, extra_t], dim=1)


# ---------------------------------------------------------------------------
# Stage 2 — Generative retrieval: vocabulary, seq2seq model, decoding
# ---------------------------------------------------------------------------

def stable_hash(key, num_buckets: int) -> int:
    """Deterministic feature-hashing (Weinberger et al., 2009), used to map
    raw user IDs onto a small, fixed-size set of vocabulary tokens (Sec 4:
    "we use the Hashing Trick to map the raw user ID to one of the 2000 user
    ID tokens").
    """
    digest = hashlib.md5(str(key).encode("utf-8")).hexdigest()
    return int(digest, 16) % num_buckets


class SemanticIDVocab:
    """Builds the shared token vocabulary the sequence-to-sequence model reads
    and writes: special tokens, then one token per (level, codeword) pair
    (so `num_levels * codebook_size` item tokens), then a fixed-size block of
    hashed user-ID tokens.

    Args:
        num_levels: m (the number of Semantic ID tokens per item, including
            any collision-disambiguation token from `resolve_collisions`).
        codebook_size: K.
        num_user_buckets: size of the hashed user-ID token block.
        pad_id / bos_id / eos_id: special token ids (0, 1, 2 by default).
    """

    def __init__(self, num_levels: int, codebook_size: int, num_user_buckets: int = 2000) -> None:
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.num_user_buckets = num_user_buckets
        self.pad_id, self.bos_id, self.eos_id = 0, 1, 2
        self.num_special = 3
        self.item_vocab_size = num_levels * codebook_size
        self.user_block_start = self.num_special + self.item_vocab_size

    @property
    def vocab_size(self) -> int:
        return self.num_special + self.item_vocab_size + self.num_user_buckets

    def level_range(self, level: int) -> Tuple[int, int]:
        start = self.num_special + level * self.codebook_size
        return start, start + self.codebook_size

    def item_token(self, level: int, code: int) -> int:
        start, _ = self.level_range(level)
        return start + code

    def item_tokens(self, codes: Sequence[int]) -> List[int]:
        return [self.item_token(level, code) for level, code in enumerate(codes)]

    def codes_from_tokens(self, tokens: Sequence[int]) -> List[int]:
        return [tok - self.level_range(level)[0] for level, tok in enumerate(tokens)]

    def user_token(self, user_id) -> int:
        return self.user_block_start + stable_hash(user_id, self.num_user_buckets)

    def build_encoder_input(self, user_id, item_code_history: Sequence[Sequence[int]]) -> List[int]:
        """user token followed by the flattened Semantic ID tokens of the
        user's item interaction history (Fig. 2b's encoder input).
        """
        tokens = [self.user_token(user_id)]
        for codes in item_code_history:
            tokens.extend(self.item_tokens(codes))
        return tokens


# ---------------------------------------------------------------------------
# Seq2seq Transformer (Fig. 2b: bidirectional encoder, autoregressive decoder)
# ---------------------------------------------------------------------------

class TigerTransformer(nn.Module):
    """A T5-style encoder-decoder Transformer over `SemanticIDVocab`'s shared
    vocabulary. The paper's own contribution is the Semantic ID representation
    and the generative-retrieval framing, not a novel attention mechanism —
    so this reuses standard `nn.TransformerEncoder`/`nn.TransformerDecoder`
    building blocks, matching Fig. 2b's "bidirectional encoder, causal
    decoder" setup.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_head: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 64,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)
        dec_layer = nn.TransformerDecoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def _embed(self, ids: Tensor) -> Tensor:
        T = ids.shape[1]
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        return self.token_emb(ids) + self.pos_emb(pos)

    def encode(self, encoder_ids: Tensor, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        return self.encoder(self._embed(encoder_ids), src_key_padding_mask=src_key_padding_mask)

    def decode(
        self,
        decoder_ids: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        T = decoder_ids.shape[1]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(decoder_ids.device)
        out = self.decoder(
            self._embed(decoder_ids), memory, tgt_mask=causal_mask, memory_key_padding_mask=memory_key_padding_mask
        )
        return self.out_proj(out)

    def forward(
        self,
        encoder_ids: Tensor,
        decoder_ids: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        memory = self.encode(encoder_ids, src_key_padding_mask)
        return self.decode(decoder_ids, memory, memory_key_padding_mask)


@torch.no_grad()
def generate_semantic_id(
    model: TigerTransformer,
    encoder_ids: Tensor,
    vocab: SemanticIDVocab,
    temperature: float = 0.0,
    src_key_padding_mask: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Autoregressively decodes one Semantic ID (vocab.num_levels tokens) per
    example (Sec 3.2). At each decoding step, logits are restricted to that
    level's `codebook_size` sub-vocabulary — guaranteeing every generated
    token is at least well-typed for its position, isolating the paper's
    "invalid IDs" phenomenon (Sec 4.5) to level-valid tuples that simply don't
    correspond to any real item, rather than to malformed positions.

    `temperature > 0` samples (enabling the diversity-via-decoding capability
    of Sec 4.3: sampling at an early level draws from coarse-grained
    categories, sampling at a later level stays within one); `temperature ==
    0` decodes greedily.

    Returns:
        (B, vocab.num_levels) generated codes.
    """
    B = encoder_ids.shape[0]
    memory = model.encode(encoder_ids, src_key_padding_mask)
    decoder_ids = torch.full((B, 1), vocab.bos_id, dtype=torch.long, device=encoder_ids.device)

    for level in range(vocab.num_levels):
        logits = model.decode(decoder_ids, memory, src_key_padding_mask)[:, -1, :]
        start, end = vocab.level_range(level)
        masked = torch.full_like(logits, float("-inf"))
        masked[:, start:end] = logits[:, start:end]
        if temperature > 0:
            probs = torch.softmax(masked / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
        else:
            next_id = masked.argmax(dim=-1)
        decoder_ids = torch.cat([decoder_ids, next_id.unsqueeze(1)], dim=1)

    generated_tokens = decoder_ids[:, 1:]
    codes = torch.stack(
        [generated_tokens[:, level] - vocab.level_range(level)[0] for level in range(vocab.num_levels)], dim=1
    )
    return codes


class SemanticIDLookup:
    """Maps generated Semantic ID tuples back to item ids; a generated tuple
    with no match is an "invalid ID" (Sec 4.5) and looks up to None.
    """

    def __init__(self, item_ids: Sequence, codes: Tensor) -> None:
        self.table = {tuple(row.tolist()): item_id for item_id, row in zip(item_ids, codes)}

    def retrieve(self, codes: Tensor) -> List[Optional[object]]:
        return [self.table.get(tuple(row.tolist())) for row in codes]


# ---------------------------------------------------------------------------
# Diversity metric (Sec 4.3: Entropy@K)
# ---------------------------------------------------------------------------

def category_entropy(categories: Sequence) -> float:
    """Shannon entropy (bits) of the ground-truth categories of a set of
    retrieved items — the paper's Entropy@K diversity metric. Higher entropy
    means predictions span more distinct categories.
    """
    n = len(categories)
    if n == 0:
        return 0.0
    counts = Counter(categories)
    probs = [c / n for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)

    print("--- Stage 1: RQ-VAE Semantic ID generation ---")
    num_items = 200
    content_dim, latent_dim, num_levels, codebook_size = 32, 8, 3, 16
    content_embeddings = torch.randn(num_items, content_dim)

    rqvae = RQVAE(input_dim=content_dim, hidden_dims=[24, 16], latent_dim=latent_dim,
                  num_levels=num_levels, codebook_size=codebook_size)
    rqvae.init_codebooks_kmeans(content_embeddings, n_iter=10)

    opt = torch.optim.Adam(rqvae.parameters(), lr=1e-2)
    for _ in range(200):
        opt.zero_grad()
        losses, _ = rqvae.loss(content_embeddings)
        losses["total"].backward()
        opt.step()
    print(f"final RQ-VAE loss: recon={losses['recon'].item():.4f}, rqvae={losses['rqvae'].item():.4f}")

    codes = rqvae.encode_codes(content_embeddings)
    codes_unique = resolve_collisions(codes)
    n_collisions = codes.shape[0] - len({tuple(r.tolist()) for r in codes})
    print(f"codes: {codes.shape}, collisions resolved: {n_collisions}, "
          f"unique after fix: {len({tuple(r.tolist()) for r in codes_unique})}/{num_items}")

    print("\n--- Stage 2: Generative retrieval ---")
    vocab = SemanticIDVocab(num_levels=codes_unique.shape[1], codebook_size=codebook_size, num_user_buckets=50)
    print(f"vocab_size: {vocab.vocab_size}")

    item_code_history = [codes_unique[i].tolist() for i in range(5)]
    encoder_tokens = vocab.build_encoder_input(user_id="user_42", item_code_history=item_code_history)
    encoder_ids = torch.tensor([encoder_tokens])
    target_codes = codes_unique[5].tolist()
    decoder_input = torch.tensor([[vocab.bos_id] + vocab.item_tokens(target_codes[:-1])])
    decoder_target = torch.tensor([vocab.item_tokens(target_codes)])

    model = TigerTransformer(vocab_size=vocab.vocab_size, d_model=32, n_head=4,
                             num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=64, max_len=64)
    logits = model(encoder_ids, decoder_input)
    loss = F.cross_entropy(logits.reshape(-1, vocab.vocab_size), decoder_target.reshape(-1))
    loss.backward()
    print(f"seq2seq logits: {logits.shape}, ce loss: {loss.item():.4f}, backward OK")

    generated = generate_semantic_id(model, encoder_ids, vocab, temperature=0.0)
    print(f"greedy-decoded Semantic ID: {generated.tolist()}")

    lookup = SemanticIDLookup(item_ids=list(range(num_items)), codes=codes_unique)
    retrieved = lookup.retrieve(generated)
    print(f"retrieved item id: {retrieved}")

    print("\n--- Diversity metric ---")
    categories = ["Hair", "Hair", "Skin", "Makeup", "Hair", "Skin"]
    print(f"category_entropy: {category_entropy(categories):.4f} bits")


if __name__ == "__main__":
    _smoke_test()
