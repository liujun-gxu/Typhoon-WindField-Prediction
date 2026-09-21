"""Attention and multi-branch feature-fusion layers used by STL-Net."""

from __future__ import annotations

import torch
from torch import nn


class SEBlock(nn.Module):
    """Channel-wise squeeze-and-excitation recalibration."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        weights = self.gate(self.pool(x).reshape(batch, channels))
        return x * weights.reshape(batch, channels, 1, 1)


class MBFNBlock(nn.Module):
    """Three 1x1/3x3/5x5 branches followed by 1x1 feature fusion."""

    def __init__(self, in_channels: int = 64, out_channels: int = 96) -> None:
        super().__init__()
        branch_channels = out_channels // 3

        def branch(kernel: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=kernel, padding=kernel // 2),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )

        self.branch_1 = branch(1)
        self.branch_3 = branch(3)
        self.branch_5 = branch(5)
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat((self.branch_1(x), self.branch_3(x), self.branch_5(x)), dim=1))
