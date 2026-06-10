"""Fast CPU shape/gradient tests: `pytest tests/ -q` (no data needed)."""

import torch

from cvq.losses import VQGANLoss, adaptive_gan_weight
from cvq.models.car import CARModel
from cvq.models.tokenizer import CVQTokenizer
from cvq.text import WordTokenizer


def tiny_tokenizer() -> CVQTokenizer:
    return CVQTokenizer(
        resolution=32, in_channels=1, ch=32, ch_mult=(1, 2, 2),
        num_res_blocks=1, z_channels=64, num_codes=128,
    )


def test_tokenizer_roundtrip():
    tok = tiny_tokenizer().eval()
    x = torch.randn(2, 1, 32, 32)
    recon, vq_loss, idx = tok(x)
    assert recon.shape == x.shape
    assert idx.shape == (2, 64)  # one token per channel
    assert vq_loss.ndim == 0
    # encode -> decode through indices alone
    again = tok.decode_indices(tok.encode_indices(x))
    assert again.shape == x.shape


def test_nested_dropout_masks_channels():
    tok = tiny_tokenizer()
    x = torch.randn(2, 1, 32, 32)
    z = tok.encoder(x)
    z_q, _, _ = tok.quantizer(z, c_keep=5)
    assert torch.all(z_q[:, 5:] == 0)
    assert not torch.all(z_q[:, :5] == 0)


def test_adaptive_gan_weight_monotone():
    c = 256
    weights = [adaptive_gan_weight(k, c) for k in (1, c // 2, c)]
    assert weights[0] < weights[1] < weights[2]
    assert abs(weights[1] - 0.5) < 1e-6  # sigmoid midpoint at c/2


def test_vqgan_loss_backward():
    tok = tiny_tokenizer()
    loss_fn = VQGANLoss(in_channels=1, perceptual_weight=0.0, gan_weight=1.0, disc_start=0)
    x = torch.randn(2, 1, 32, 32)
    recon, vq_loss, _ = tok(x, c_keep=3)
    loss, logs = loss_fn.generator_loss(recon, x, vq_loss, c_keep=3, c_total=64, global_step=1)
    loss.backward()
    assert "loss/g" in logs and logs["loss/gan_weight"] < 0.5  # few channels -> low GAN weight
    assert loss_fn.discriminator_loss(recon, x, global_step=1) is not None


def test_car_train_and_generate():
    tok = tiny_tokenizer().eval()
    text_tok = WordTokenizer.for_mnist()
    car = CARModel(
        {"kind": "tiny", "dim": 64, "depth": 2, "heads": 2, "max_seq_len": 128},
        num_codes=128, code_dim=64, num_channels=64, text_vocab_size=len(text_tok),
    )
    car.load_codebook(tok.quantizer.embedding.weight)
    ids = torch.tensor([text_tok.encode("a handwritten digit 3", 8)] * 2)
    x = torch.randn(2, 1, 32, 32)
    loss = car(ids, tok.encode_indices(x))
    loss.backward()
    assert loss.ndim == 0
    out = car.generate(ids, top_k=16, cfg_scale=1.5)
    assert out.shape == (2, 64)
    assert tok.decode_indices(out, c_keep=8).shape == (2, 1, 32, 32)
