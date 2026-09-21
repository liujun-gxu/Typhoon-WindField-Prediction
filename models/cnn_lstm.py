"""CNN-LSTM baseline for full local wind-field prediction."""

from __future__ import annotations

import torch
from torch import nn


class CNNLSTM(nn.Module):
    """Encode four u/v patches, model their temporal sequence, and predict a 31x31 u/v field."""

    def __init__(self, patch_size: int = 31, time_steps: int = 4, in_channels: int = 2) -> None:
        super().__init__()
        self.patch_size, self.time_steps, self.in_channels = patch_size, time_steps, in_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 256), nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)
        self.regression_head = nn.Linear(256, 2 * patch_size * patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = x.shape
        if (steps, channels, height, width) != (self.time_steps, self.in_channels, self.patch_size, self.patch_size):
            raise ValueError(f"Expected (B,{self.time_steps},{self.in_channels},{self.patch_size},{self.patch_size}); got {tuple(x.shape)}")
        frames = x.reshape(batch * steps, channels, height, width)
        sequence = self.encoder(frames).reshape(batch, steps, 256)
        temporal, _ = self.lstm(sequence)
        return self.regression_head(temporal[:, -1]).reshape(batch, 2, self.patch_size, self.patch_size)
