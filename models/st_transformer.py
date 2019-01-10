import numpy as np
import torch
import torch.nn as nn

from modules import bias, MLP
from modules import STTransformerLayer, STTransformerDecoderLayer
from lib.io import gen_time


class STTransformerEncoder(nn.Module):
    def __init__(self, model_dim, num_layers, heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            STTransformerLayer(model_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, input, mask):
        for layer in self.layers:
            input = layer(input, input, mask)
        return self.layer_norm(input)


class STTransformerDecoder(nn.Module):
    def __init__(self, model_dim, num_layers, heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            STTransformerDecoderLayer(model_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(model_dim)
        self.mlp = MLP(model_dim, 1, dropout)

    def forward(self, input, bank, mask):
        for layer in self.layers:
            input = layer(input, bank, mask)
        return self.mlp(self.layer_norm(input))


class STTransformer(nn.Module):
    def __init__(self, embedding, model_dim,
                 encoder_layers, decoder_layers, heads,
                 horizon, dropout, mask=None):
        super().__init__()
        self.embedding = embedding
        self.encoder = STTransformerEncoder(
            model_dim=model_dim,
            num_layers=encoder_layers,
            heads=heads,
            dropout=dropout
        )
        self.decoder = STTransformerDecoder(
            model_dim=model_dim,
            num_layers=decoder_layers,
            heads=heads,
            dropout=dropout
        )
        self.horizon = horizon
        self.register_buffer('mask', mask)

    def forward(self, data, time, weekday):
        input = self.embedding(data, time, weekday)
        bank = self.encoder(input, self.mask)
        time = gen_time(time[:, -1], self.horizon)
        input = self.embedding(None, time, weekday)
        output = self.decoder(input, bank, self.mask)
        return output + data[:, [-1]]