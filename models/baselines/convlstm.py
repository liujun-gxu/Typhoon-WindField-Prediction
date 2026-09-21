"""ConvLSTM baseline: two-layer ConvLSTM retaining its spatial hidden field for prediction.

Extracted from the authors' unified fixed-t0 benchmark code used to produce the results reported in: Liu, J.; Cui, J.; Liu, Y. Multi-Horizon Typhoon Wind Field Prediction via a Lightweight CNN-LSTM Network: Error Growth and Cross-Year Robustness at 6-24 h Lead Times. Atmosphere, submitted (Manuscript ID: atmosphere-4519259). See Section 4.1 and Table 6 of the manuscript; the parameter count is verified against Table 5. Run this file directly (`python <filename>.py`) to reproduce the parameter count.
"""

import torch
import torch.nn as nn

PATCH_SIZE = 31
TIME_STEPS = 4
TARGET_SHAPE = (2, PATCH_SIZE, PATCH_SIZE)

EXPECTED_PARAMETERS = 113218

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor, state):
        hidden, cell = state
        gates = self.conv(torch.cat((x, hidden), dim=1))
        input_gate, forget_gate, output_gate, candidate = torch.chunk(gates, 4, dim=1)
        cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
        hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
        return hidden, cell


class ConvLSTMBaseline(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.cells = nn.ModuleList((ConvLSTMCell(2, hidden_dim), ConvLSTMCell(hidden_dim, hidden_dim)))
        self.head = nn.Conv2d(hidden_dim, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, _, height, width = x.shape
        states = [
            (x.new_zeros(batch, cell.hidden_dim, height, width), x.new_zeros(batch, cell.hidden_dim, height, width))
            for cell in self.cells
        ]
        for step in range(steps):
            value = x[:, step]
            next_states = []
            for cell, state in zip(self.cells, states):
                hidden, memory = cell(value, state)
                next_states.append((hidden, memory))
                value = hidden
            states = next_states
        return self.head(states[-1][0])


if __name__ == "__main__":
    model = ConvLSTMBaseline()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with torch.no_grad():
        output = model(torch.zeros(1, TIME_STEPS, 2, PATCH_SIZE, PATCH_SIZE))
    assert count == EXPECTED_PARAMETERS, f"{count:,} != {EXPECTED_PARAMETERS:,}"
    assert tuple(output.shape) == (1, *TARGET_SHAPE)
    print(f"trainable parameters: {count:,}  (Table 5: 0.113 M)")
    print(f"forward check: (1, 4, 2, 31, 31) -> {tuple(output.shape)}")
