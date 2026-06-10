"""CVQ tokenizer = Encoder -> ChannelVectorQuantizer -> Decoder.

Includes nested channel dropout (Appendix B.1): with probability alpha
per iteration, c_keep ~ U{1..c} channels are retained and the rest are
zeroed, forcing a coarse-to-fine ordering of channels that CAR later
exploits as its autoregressive order.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn

from .autoencoder import Decoder, Encoder
from .quantizer import ChannelVectorQuantizer


class CVQTokenizer(nn.Module):
    def __init__(
        self,
        resolution: int = 256,
        in_channels: int = 3,
        ch: int = 128,
        ch_mult: tuple[int, ...] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        z_channels: int = 256,
        num_codes: int = 16384,
        beta: float = 0.25,
        use_mid_attn: bool = True,
        dropout_ratio: float = 0.25,
    ):
        super().__init__()
        f = 2 ** (len(ch_mult) - 1)
        assert resolution % f == 0, f"resolution {resolution} not divisible by downsample factor {f}"
        self.latent_size = resolution // f
        self.num_channels = z_channels  # = number of tokens per image
        self.code_dim = self.latent_size**2
        self.dropout_ratio = dropout_ratio

        self.encoder = Encoder(in_channels, ch, ch_mult, num_res_blocks, z_channels, use_mid_attn)
        self.decoder = Decoder(in_channels, ch, ch_mult, num_res_blocks, z_channels, use_mid_attn)
        self.quantizer = ChannelVectorQuantizer(num_codes, self.code_dim, beta)

    def sample_c_keep(self) -> int | None:
        """Nested channel dropout schedule (Eq. 9): with prob alpha,
        c_keep ~ U{1..c}; otherwise train on the full configuration."""
        if self.training and random.random() < self.dropout_ratio:
            return random.randint(1, self.num_channels)
        return None

    def forward(
        self, x: torch.Tensor, c_keep: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, vq_loss, indices (B, C))."""
        z = self.encoder(x)
        z_q, vq_loss, indices = self.quantizer(z, c_keep=c_keep)
        recon = self.decoder(z_q)
        return recon, vq_loss, indices

    @torch.no_grad()
    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Image -> 1D channel-token sequence (B, C)."""
        z = self.encoder(x)
        _, _, indices = self.quantizer(z)
        return indices

    @torch.no_grad()
    def decode_indices(self, indices: torch.Tensor, c_keep: int | None = None) -> torch.Tensor:
        """Channel-token sequence (B, C) -> image. If c_keep is given,
        only the first c_keep channels are used (progressive decoding,
        Table 8 / Fig. 5)."""
        z_q = self.quantizer.lookup(indices, self.latent_size, self.latent_size)
        if c_keep is not None and c_keep < z_q.shape[1]:
            mask = torch.zeros_like(z_q)
            mask[:, :c_keep] = 1.0
            z_q = z_q * mask
        return self.decoder(z_q)
