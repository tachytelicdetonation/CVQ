"""Nested-dropout ablation test (paper Tables 7 & 8, at MNIST scale).

Trains two small CVQ tokenizers on an MNIST subset — nested channel
dropout alpha=0.25 vs alpha=0 — and asserts the mechanism the paper's
thesis rests on:

  1. Full-channel reconstruction is comparable (dropout is ~free,
     Table 7: rFID 2.60 vs 2.62).
  2. Decoding from only the first c/4 channels is much better WITH
     dropout (the coarse-to-fine ordering exists, Table 8).
  3. With dropout, reconstruction improves monotonically as channels
     are added (the AR-friendly curriculum).

Slow (~minutes; trains two models): run with
    pytest tests/test_ablation_dropout.py -m slow -s
Tune duration with CVQ_ABLATION_STEPS (default 800).

Thresholds are calibrated against a measured run (800 steps, MPS):
truncated-MSE ratio dropout/no-dropout was 0.58-0.61 for c_keep in
{4, 8, 16} and 1.03 at full channels. Note MSE understates the gap on
MNIST: the mostly-black background bounds how badly even a collapsed
decode can score, so a 2x gap (as rFID shows in the paper) is not
reachable on this metric/dataset.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from cvq.models.tokenizer import CVQTokenizer
from cvq.utils import pick_device, set_seed

STEPS = int(os.environ.get("CVQ_ABLATION_STEPS", 800))
BATCH = 128
SUBSET = 8192


def mnist_subset(device: torch.device) -> torch.Tensor:
    from torchvision import datasets, transforms

    tf = transforms.Compose(
        [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
    )
    ds = datasets.MNIST("data", train=True, download=True, transform=tf)
    imgs = torch.stack([ds[i][0] for i in range(SUBSET)])
    return imgs.to(device)


def train_tokenizer(images: torch.Tensor, dropout_ratio: float, device: torch.device) -> CVQTokenizer:
    """Minimal recon + VQ training (no GAN/LPIPS — the ordering property
    comes from nested dropout alone, and this keeps the test fast)."""
    set_seed(0)
    model = CVQTokenizer(
        resolution=32, in_channels=1, ch=32, ch_mult=(1, 2, 2), num_res_blocks=1,
        z_channels=64, num_codes=512, dropout_ratio=dropout_ratio,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.9))
    model.train()
    g = torch.Generator().manual_seed(0)
    for _ in range(STEPS):
        idx = torch.randint(0, images.shape[0], (BATCH,), generator=g)
        batch = images[idx]
        recon, vq_loss, _ = model(batch, c_keep=model.sample_c_keep())
        loss = F.mse_loss(recon, batch) + vq_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def truncated_mse(model: CVQTokenizer, images: torch.Tensor, c_keep: int) -> float:
    """Reconstruction MSE when only the first c_keep channels survive."""
    total, n = 0.0, 0
    for i in range(0, min(images.shape[0], 1024), 256):
        batch = images[i : i + 256]
        recon = model.decode_indices(model.encode_indices(batch), c_keep=c_keep)
        total += F.mse_loss(recon.clamp(-1, 1), batch, reduction="sum").item()
        n += batch.numel()
    return total / n


@pytest.mark.slow
def test_nested_dropout_induces_coarse_to_fine_ordering():
    device = pick_device()
    images = mnist_subset(device)
    c = 64  # total channels

    with_dropout = train_tokenizer(images, dropout_ratio=0.25, device=device)
    no_dropout = train_tokenizer(images, dropout_ratio=0.0, device=device)

    keeps = (c // 8, c // 4, c // 2, c)  # 8, 16, 32, 64
    mse_w = {k: truncated_mse(with_dropout, images, k) for k in keeps}
    mse_wo = {k: truncated_mse(no_dropout, images, k) for k in keeps}
    for k in keeps:
        print(f"\nc_keep={k:>2}: dropout={mse_w[k]:.4f}  no-dropout={mse_wo[k]:.4f}  ratio={mse_w[k]/mse_wo[k]:.2f}")

    # (1) Table 7: dropout leaves full-channel reconstruction roughly
    # intact (measured ratio 1.03; allow 1.5x).
    assert mse_w[c] < mse_wo[c] * 1.5, "nested dropout should not wreck full reconstruction"

    # (2) Table 8 / Table 4: without dropout, channels carry no order, so
    # truncated decoding degrades much faster. Measured ratio ~0.58 at
    # c/8 and c/4; assert < 0.8 for margin.
    for k in (c // 8, c // 4):
        assert mse_w[k] < mse_wo[k] * 0.8, (
            f"truncated decoding at c_keep={k} should be clearly better with "
            f"nested dropout (got {mse_w[k]:.4f} vs {mse_wo[k]:.4f})"
        )

    # (3) Coarse-to-fine monotonicity for the dropout model: more
    # channels -> strictly better reconstruction.
    assert mse_w[c // 8] > mse_w[c // 4] > mse_w[c // 2] > mse_w[c], (
        f"expected monotone improvement with more channels: {mse_w}"
    )
