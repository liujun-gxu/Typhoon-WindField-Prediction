"""U-Net baseline: encoder-decoder with three double-conv stages (32/64/128), a 256-channel bottleneck, three skip-connected decoder stages, and a 1x1 two-channel field head.

Extracted from the authors' unified fixed-t0 benchmark code used to produce the results reported in: Liu, J.; Cui, J.; Liu, Y. Multi-Horizon Typhoon Wind Field Prediction via a Lightweight CNN-LSTM Network: Error Growth and Cross-Year Robustness at 6-24 h Lead Times. Atmosphere, submitted (Manuscript ID: atmosphere-4519259). See Section 4.1 and Table 6 of the manuscript; the parameter count is verified against Table 5. Run this file directly (`python <filename>.py`) to reproduce the parameter count.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

PATCH_SIZE = 31
TIME_STEPS = 4
TARGET_SHAPE = (2, PATCH_SIZE, PATCH_SIZE)

EXPECTED_PARAMETERS = 1929890

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1, self.down2, self.down3 = DoubleConv(8, 32), DoubleConv(32, 64), DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up3, self.conv3 = nn.ConvTranspose2d(256, 128, 2, stride=2), DoubleConv(256, 128)
        self.up2, self.conv2 = nn.ConvTranspose2d(128, 64, 2, stride=2), DoubleConv(128, 64)
        self.up1, self.conv1 = nn.ConvTranspose2d(64, 32, 2, stride=2), DoubleConv(64, 32)
        self.final = nn.Conv2d(32, 2, 1)

    @staticmethod
    def _join(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((skip, up), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = x.reshape(batch, 8, PATCH_SIZE, PATCH_SIZE)
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        z = self.bottleneck(self.pool(d3))
        z = self.conv3(self._join(self.up3(z), d3))
        z = self.conv2(self._join(self.up2(z), d2))
        z = self.conv1(self._join(self.up1(z), d1))
        return self.final(z)


if __name__ == "__main__":
    model = UNetBaseline()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with torch.no_grad():
        output = model(torch.zeros(1, TIME_STEPS, 2, PATCH_SIZE, PATCH_SIZE))
    assert count == EXPECTED_PARAMETERS, f"{count:,} != {EXPECTED_PARAMETERS:,}"
    assert tuple(output.shape) == (1, *TARGET_SHAPE)
    print(f"trainable parameters: {count:,}  (Table 5: 1.930 M)")
    print(f"forward check: (1, 4, 2, 31, 31) -> {tuple(output.shape)}")
