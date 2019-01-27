import math
import torch
import torch.nn as nn


def bias(*sizes):
    assert len(sizes) > 0
    if len(sizes) == 1:
        sizes = (1, sizes[0])
    bias = nn.Parameter(torch.empty(*sizes))
    nn.init.xavier_normal_(bias.data)
    return bias


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


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
    def __init__(self, model_dim, dropout):
        super().__init__()
        self.w_1 = nn.Linear(model_dim, model_dim)
        self.w_2 = nn.Linear(model_dim, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout_1(self.relu(self.w_1(self.layer_norm(input))))
        output = self.dropout_2(self.w_2(hidden))
        return output + input
