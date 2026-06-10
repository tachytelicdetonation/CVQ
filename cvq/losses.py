"""Tokenizer training objective (Sec. 3.1 "Tokenizer Training" + Appendix B.1).

L = L_recon (pixel l2) + L_quant + w_p * L_lpips + lambda_GAN(c_keep) * L_GAN

The GAN weight follows the sigmoid schedule of Eq. 8 so that adversarial
pressure ramps up with the number of active channels — a discriminator on
a 4-channel reconstruction would otherwise destabilize training.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.discriminator import PatchDiscriminator


def adaptive_gan_weight(c_keep: int, c_total: int, lambda0: float = 1.0, eta: float = 0.05) -> float:
    """Eq. 8: lambda_GAN(c_keep) = lambda0 / (1 + exp(-eta * (c_keep - c/2)))."""
    return lambda0 / (1.0 + math.exp(-eta * (c_keep - c_total / 2.0)))


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    return 0.5 * (F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean())


class VQGANLoss(nn.Module):
    """Bundles reconstruction / perceptual / adversarial terms.

    perceptual_weight=0 disables LPIPS entirely (no extra dependency),
    gan_weight=0 disables the discriminator. The MNIST config keeps the
    GAN on (to exercise the full pipeline) but skips LPIPS by default.
    """

    def __init__(
        self,
        in_channels: int = 3,
        perceptual_weight: float = 1.0,
        gan_weight: float = 1.0,
        gan_eta: float = 0.05,
        disc_start: int = 10000,
        disc_ch: int = 64,
        disc_layers: int = 3,
    ):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.gan_weight = gan_weight
        self.gan_eta = gan_eta
        self.disc_start = disc_start

        self.lpips = None
        if perceptual_weight > 0:
            import lpips  # optional dependency: pip install cvq[perceptual]

            self.lpips = lpips.LPIPS(net="vgg")
            self.lpips.requires_grad_(False)

        self.discriminator = PatchDiscriminator(in_channels, disc_ch, disc_layers) if gan_weight > 0 else None

    def perceptual(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.lpips is None:
            return recon.new_zeros(())
        if recon.shape[1] == 1:  # LPIPS' VGG expects 3 channels (e.g. MNIST)
            recon, target = recon.repeat(1, 3, 1, 1), target.repeat(1, 3, 1, 1)
        return self.lpips(recon.clamp(-1, 1), target).mean()

    def generator_loss(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        vq_loss: torch.Tensor,
        c_keep: int | None,
        c_total: int,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        rec = F.mse_loss(recon, target)
        per = self.perceptual(recon, target)
        loss = rec + vq_loss + self.perceptual_weight * per
        logs = {"loss/rec": rec.item(), "loss/vq": vq_loss.item(), "loss/lpips": float(per)}

        if self.discriminator is not None and global_step >= self.disc_start:
            logits_fake = self.discriminator(recon)
            g_loss = -logits_fake.mean()
            lam = adaptive_gan_weight(c_keep if c_keep is not None else c_total, c_total, self.gan_weight, self.gan_eta)
            loss = loss + lam * g_loss
            logs.update({"loss/g": g_loss.item(), "loss/gan_weight": lam})
        return loss, logs

    def discriminator_loss(
        self, recon: torch.Tensor, target: torch.Tensor, global_step: int
    ) -> torch.Tensor | None:
        if self.discriminator is None or global_step < self.disc_start:
            return None
        return hinge_d_loss(self.discriminator(target), self.discriminator(recon.detach()))
