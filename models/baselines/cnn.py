"""CNN baseline: four-frame channel-stacked spatial CNN with full-field regression head.

Extracted from the authors' unified fixed-t0 benchmark code used to produce the results reported in: Liu, J.; Cui, J.; Liu, Y. Multi-Horizon Typhoon Wind Field Prediction via a Lightweight CNN-LSTM Network: Error Growth and Cross-Year Robustness at 6-24 h Lead Times. Atmosphere, submitted (Manuscript ID: atmosphere-4519259). See Section 4.1 and Table 6 of the manuscript; the parameter count is verified against Table 5. Run this file directly (`python <filename>.py`) to reproduce the parameter count.
"""

import torch
import torch.nn as nn

PATCH_SIZE = 31
TIME_STEPS = 4
TARGET_SHAPE = (2, PATCH_SIZE, PATCH_SIZE)

EXPECTED_PARAMETERS = 531618

class CNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 256), nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(256, 2 * PATCH_SIZE * PATCH_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return self.head(self.encoder(x.reshape(batch, 8, PATCH_SIZE, PATCH_SIZE))).reshape(batch, *TARGET_SHAPE)


if __name__ == "__main__":
    model = CNNBaseline()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with torch.no_grad():
        output = model(torch.zeros(1, TIME_STEPS, 2, PATCH_SIZE, PATCH_SIZE))
    assert count == EXPECTED_PARAMETERS, f"{count:,} != {EXPECTED_PARAMETERS:,}"
    assert tuple(output.shape) == (1, *TARGET_SHAPE)
    print(f"trainable parameters: {count:,}  (Table 5: 0.532 M)")
    print(f"forward check: (1, 4, 2, 31, 31) -> {tuple(output.shape)}")
