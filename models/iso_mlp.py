import torch
import torch.nn as nn
from modules import MLP, ResMLP


class IsoMLP(nn.Module):
    def __init__(self, embedding, model_dim, out_dim, num_layers, dropout):
        super().__init__()
        self.embedding = embedding
        self.encoder = nn.Sequential(
            *(ResMLP(model_dim, dropout) for _ in range(num_layers))
        )
        self.decoder = MLP(model_dim, out_dim, dropout)

    def forward(self, data, time, weekday):
        input = self.embedding(data, time, weekday)
        hidden = self.encoder(input)
        return self.decoder(hidden)
