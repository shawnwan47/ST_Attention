import torch
import torch.nn as nn
from modules import MLP


class IsoMLP(nn.Module):
    def __init__(self, embedding, model_dim, out_dim, dropout):
        super().__init__()
        self.embedding = embedding
        self.decoder = MLP(model_dim, out_dim, dropout)

    def forward(self, data, time, weekday):
        input = self.embedding(data, time, weekday)
        return self.decoder(input)
