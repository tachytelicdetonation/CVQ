"""VQGAN-style convolutional encoder/decoder (Esser et al., 2021).

The paper trains CVQ "following the standard VQGAN approach", so we use
the taming-transformers architecture: ResNet blocks + GroupNorm + SiLU,
with optional self-attention at the lowest resolution. The downsample
factor f = 2^(len(ch_mult)-1) and the latent channel count `z_channels`
are set so that c = z_channels equals the desired token count and
h*w = (resolution/f)^2 equals the codeword dimension.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32 if channels % 32 == 0 else 8, num_channels=channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = normalize(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = normalize(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = normalize(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(1)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2).unsqueeze(1), k.transpose(1, 2).unsqueeze(1), v.transpose(1, 2).unsqueeze(1)
        )
        return x + self.proj(attn.squeeze(1).transpose(1, 2).reshape(B, C, H, W))


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1)))


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        ch: int = 128,
        ch_mult: tuple[int, ...] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        z_channels: int = 256,
        use_mid_attn: bool = True,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, ch, 3, padding=1)
        blocks: list[nn.Module] = []
        cur = ch
        for i, mult in enumerate(ch_mult):
            out = ch * mult
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(cur, out))
                cur = out
            if i != len(ch_mult) - 1:
                blocks.append(Downsample(cur))
        self.down = nn.Sequential(*blocks)
        mid: list[nn.Module] = [ResnetBlock(cur, cur)]
        if use_mid_attn:
            mid.append(AttnBlock(cur))
        mid.append(ResnetBlock(cur, cur))
        self.mid = nn.Sequential(*mid)
        self.norm_out = normalize(cur)
        self.conv_out = nn.Conv2d(cur, z_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mid(self.down(self.conv_in(x)))
        return self.conv_out(F.silu(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        ch: int = 128,
        ch_mult: tuple[int, ...] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        z_channels: int = 256,
        use_mid_attn: bool = True,
    ):
        super().__init__()
        cur = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(z_channels, cur, 3, padding=1)
        mid: list[nn.Module] = [ResnetBlock(cur, cur)]
        if use_mid_attn:
            mid.append(AttnBlock(cur))
        mid.append(ResnetBlock(cur, cur))
        self.mid = nn.Sequential(*mid)
        blocks: list[nn.Module] = []
        for i, mult in reversed(list(enumerate(ch_mult))):
            out = ch * mult
            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(cur, out))
                cur = out
            if i != 0:
                blocks.append(Upsample(cur))
        self.up = nn.Sequential(*blocks)
        self.norm_out = normalize(cur)
        self.conv_out = nn.Conv2d(cur, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.up(self.mid(self.conv_in(z)))
        return self.conv_out(F.silu(self.norm_out(h)))
