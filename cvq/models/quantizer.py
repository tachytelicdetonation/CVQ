"""Channel-wise Vector Quantization (Sec. 3.1 of arXiv:2605.26089).

Standard VQ treats Z in R^{h x w x c} as h*w spatial vectors of dim c.
CVQ instead treats it as c channel vectors of dim h*w: each channel
z^(k) in R^{h x w} is matched against a codebook of codewords
e_n in R^{h x w} by Frobenius norm (Eq. 4), which is equivalent to
flattening and doing an L2 nearest-neighbour lookup.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelVectorQuantizer(nn.Module):
    """Quantizes each *channel* of the latent feature map.

    Args:
        num_codes: codebook size N (paper: 16,384).
        code_dim: dimensionality of each codeword = h * w of the latent
            (paper: 16*16 = 256 for the 256-token model).
        beta: commitment loss weight (VQGAN default 0.25).
    """

    def __init__(self, num_codes: int, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_codes, code_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    @torch.no_grad()
    def lookup(self, indices: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """indices (B, C) -> quantized latent (B, C, h, w)."""
        codes = self.embedding(indices)  # (B, C, h*w)
        return codes.view(*indices.shape, h, w)

    def quantize_flat(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """flat (M, code_dim) -> (quantized (M, code_dim), indices (M,))."""
        # ||z - e||^2 = ||z||^2 - 2 z.e + ||e||^2; argmin over codebook.
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(dim=1)
        )
        indices = dist.argmin(dim=1)
        return self.embedding(indices), indices

    def forward(
        self, z: torch.Tensor, c_keep: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize z (B, C, h, w) channel-wise.

        c_keep: if set (nested channel dropout, Appendix B), the
            quantization loss is computed only over the first c_keep
            channels; channels >= c_keep are zeroed in the output.

        Returns (z_q, vq_loss, indices) with z_q (B, C, h, w) carrying
        straight-through gradients and indices (B, C).
        """
        B, C, h, w = z.shape
        flat = z.reshape(B * C, h * w)
        quantized, indices = self.quantize_flat(flat)
        z_q = quantized.view(B, C, h, w)
        indices = indices.view(B, C)

        if c_keep is None or c_keep >= C:
            active_q, active_z = z_q, z
        else:
            # Loss exclusively over active channels (Eq. 7 discussion).
            active_q, active_z = z_q[:, :c_keep], z[:, :c_keep]
        codebook_loss = F.mse_loss(active_q, active_z.detach())
        commit_loss = F.mse_loss(active_q.detach(), active_z)
        vq_loss = codebook_loss + self.beta * commit_loss

        # Straight-through estimator (Eq. 5).
        z_q = z + (z_q - z).detach()

        if c_keep is not None and c_keep < C:
            mask = torch.zeros(1, C, 1, 1, device=z.device, dtype=z.dtype)
            mask[:, :c_keep] = 1.0
            z_q = z_q * mask

        return z_q, vq_loss, indices
