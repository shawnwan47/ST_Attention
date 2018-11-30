from math import sqrt, round
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1):
        hidden_size = round(sqrt(input_size * output_size))
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, input):
        return self.sequential(input)
