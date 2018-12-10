from math import sqrt, ceil
import torch.nn as nn


class Perceptron(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_size)
        )

    def forward(self, input):
        return self.sequential(input)


class MLPEncoder(nn.Module):
    def __init__(self, input_size, output_size, num_layers=2, dropout=0.2):
        assert num_layers > 1
        super().__init__()
        self.fc_input = Perceptron(input_size, output_size, dropout)
        self.fc_hidden = nn.Sequential(
            *[Perceptron(output_size, output_size, dropout)
              for _ in range(num_layers - 1)]
        )

    def forward(self, input):
        return self.fc_hidden(self.fc_input(input))


class MLP(nn.Module):
    def __init__(self, input_size, output_size, num_layers=2, dropout=0.2):
        assert num_layers > 1
        hidden_size = ceil(sqrt(input_size * output_size))
        super().__init__()
        self.fc_input = Perceptron(input_size, hidden_size, dropout)
        self.fc_output = nn.Linear(hidden_size, output_size)
        self.fc_hidden = nn.Sequential(
            *[Perceptron(hidden_size, hidden_size)
              for _ in range(num_layers - 2)]
        )

    def forward(self, input):
        return self.fc_output(self.fc_hidden(self.fc_input(input))).transpose(1, 2)


class MLPVec2Vec(nn.Module):
    def __init__(self, embedding, mlp):
        super().__init__()
        self.embedding = embedding
        self.mlp = mlp

    def forward(self, data, time, day):
        return self.mlp(self.embedding(data, time, day))
