"""PatchGAN discriminator (Isola et al., 2017), as used by VQGAN."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, ch: int = 64, num_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, ch, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        mult = 1
        for i in range(1, num_layers + 1):
            prev, mult = mult, min(2**i, 8)
            stride = 2 if i < num_layers else 1
            layers += [
                nn.Conv2d(ch * prev, ch * mult, 4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(ch * mult),
                nn.LeakyReLU(0.2, True),
            ]
        layers.append(nn.Conv2d(ch * mult, 1, 4, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
