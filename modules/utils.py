import math
import torch
import torch.nn as nn


def bias(model_dim):
    assert model_dim > 0
    bias = nn.Parameter(torch.Tensor(model_dim))
    bound = 1 / math.sqrt(model_dim)
    nn.init.uniform_(bias, -bound, bound)
    return bias


class MLP(nn.Module):
    def __init__(self, input_size, output_size, dropout, bias=True):
        super().__init__()
        hidden_size = round(math.sqrt(input_size * output_size))
        self.sequential = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=bias),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size, bias=bias)
        )

    def forward(self, input):
        return self.sequential(input)


class ResMLP(nn.Module):
    def __init__(self, model_dim, dropout=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(model_dim, model_dim)
        self.fc_2 = nn.Linear(model_dim, model_dim)
        self.ln = nn.LayerNorm(model_dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, input):
        hidden = self.drop(self.relu(self.fc_1(self.ln(input))))
        output = self.drop(self.fc_2(hidden))
        return output + input
