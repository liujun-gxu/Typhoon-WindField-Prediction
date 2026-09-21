"""Full CNN-LSTM-SE-MBFN (STL-Net) model used in the thesis experiments."""

from __future__ import annotations

import torch
from torch import nn

from .blocks import MBFNBlock, SEBlock


class FullFieldSTLNet(nn.Module):
    """CNN encoder -> SE -> MBFN -> two-layer LSTM -> full 2x31x31 field."""

    def __init__(self, patch_size: int = 31, time_steps: int = 4, in_channels: int = 2) -> None:
        super().__init__()
        self.patch_size, self.time_steps, self.in_channels = patch_size, time_steps, in_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.se = SEBlock(64, reduction=8)
        self.mbfn = MBFNBlock(64, 96)
        self.frame_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, 256), nn.ReLU(inplace=True)
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)
        self.regression_head = nn.Linear(256, 2 * patch_size * patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = x.shape
        expected = (self.time_steps, self.in_channels, self.patch_size, self.patch_size)
        if (steps, channels, height, width) != expected:
            raise ValueError(f"Expected (B,{','.join(map(str, expected))}); got {tuple(x.shape)}")
        frames = x.reshape(batch * steps, channels, height, width)
        features = self.frame_projection(self.mbfn(self.se(self.encoder(frames))))
        temporal, _ = self.lstm(features.reshape(batch, steps, 256))
        return self.regression_head(temporal[:, -1]).reshape(batch, 2, self.patch_size, self.patch_size)
