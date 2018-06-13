import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, hidden_size, output_size, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, hidden):
        # hidden = self.layer_norm(hidden)
        return self.linear(self.dropout(hidden))
