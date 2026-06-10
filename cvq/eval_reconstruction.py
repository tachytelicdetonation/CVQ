"""Evaluate a trained CVQ tokenizer: PSNR, SSIM, codebook usage, and
(optionally, if torchmetrics is installed) rFID — the metrics of Table 1.

    python -m cvq.eval_reconstruction --config configs/tokenizer_mnist.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import build_loader
from .utils import load_config, load_tokenizer_checkpoint, pick_device


def gaussian_ssim(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM over images in [0, 1]."""
    coords = torch.arange(window, dtype=x.dtype, device=x.device) - window // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).outer(g / g.sum()).expand(x.shape[1], 1, window, window)
    pad, ch = window // 2, x.shape[1]
    mu_x = F.conv2d(x, g, padding=pad, groups=ch)
    mu_y = F.conv2d(y, g, padding=pad, groups=ch)
    sx = F.conv2d(x * x, g, padding=pad, groups=ch) - mu_x**2
    sy = F.conv2d(y * y, g, padding=pad, groups=ch) - mu_y**2
    sxy = F.conv2d(x * y, g, padding=pad, groups=ch) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    return (((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sx + sy + c2))).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None, help="defaults to <out_dir>/tokenizer.pt")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device(cfg.get("device", "auto"))
    ckpt = args.ckpt or Path(cfg["out_dir"]) / "tokenizer.pt"
    model, _ = load_tokenizer_checkpoint(str(ckpt), device)

    loader = build_loader({**cfg["data"], "batch_size": cfg["train"]["batch_size"]}, train=False)

    fid = None
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance

        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    except ImportError:
        print("torchmetrics not installed — skipping rFID (pip install cvq[eval])")

    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    used = torch.zeros(model.quantizer.num_codes, dtype=torch.bool, device=device)
    with torch.no_grad():
        for b, (img, _) in enumerate(tqdm(loader, desc="eval")):
            if args.max_batches and b >= args.max_batches:
                break
            img = img.to(device)
            recon, _, idx = model(img)
            x = (img * 0.5 + 0.5).clamp(0, 1)
            y = (recon * 0.5 + 0.5).clamp(0, 1)
            mse = ((x - y) ** 2).mean(dim=(1, 2, 3))
            psnr_sum += (10 * torch.log10(1.0 / mse.clamp_min(1e-10))).sum().item()
            ssim_sum += gaussian_ssim(y, x).item() * img.shape[0]
            used[idx.unique()] = True
            n += img.shape[0]
            if fid is not None:
                rgb = lambda t: t.repeat(1, 3, 1, 1) if t.shape[1] == 1 else t
                fid.update(rgb(x), real=True)
                fid.update(rgb(y), real=False)

    print(f"PSNR:           {psnr_sum / n:.2f} dB")
    print(f"SSIM:           {ssim_sum / n:.4f}")
    print(f"Codebook usage: {used.float().mean().item() * 100:.1f}%")
    if fid is not None:
        print(f"rFID:           {fid.compute().item():.2f}")


if __name__ == "__main__":
    main()
