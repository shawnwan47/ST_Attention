from math import sqrt
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        hidden_size = round(sqrt(input_size * output_size))
        self.sequential = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, input):
        return self.sequential(input)


class ResMLP(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(size),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(size, size),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        return input + self.sequential(input)


def bias(*sizes):
    assert len(sizes) > 0
    if len(sizes) == 1:
        sizes = (1, sizes[0])
    bias = nn.Parameter(torch.empty(*sizes))
    nn.init.xavier_normal_(bias.data)
    return bias
