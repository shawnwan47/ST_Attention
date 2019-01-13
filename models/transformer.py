import torch
import torch.nn as nn

from modules import TransformerLayer, TransformerDecoderLayer
from modules import MLP
from lib.io import gen_time


class TransformerEncoder(nn.Module):
    def __init__(self, model_dim, num_layers, heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(model_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, input):
        for layer in self.layers:
            input = layer(input, input)
        return self.layer_norm(input)


class STransformer(TransformerEncoder):
    def __init__(self, embedding, model_dim, out_dim, num_layers, heads, dropout):
        super().__init__(model_dim, num_layers, heads, dropout)
        self.embedding = embedding
        self.mlp_out = MLP(model_dim, out_dim, dropout)

    def forward(self, data, time, weekday):
        emb = self.embedding(data, time, weekday)
        input = super().forward(emb)
        output = self.mlp_out(input)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, model_dim, out_dim, num_layers, heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(model_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(model_dim)
        self.mlp = MLP(model_dim, out_dim, dropout)

    def forward(self, input, bank):
        for layer in self.layers:
            input = layer(input, bank)
        return self.mlp(self.layer_norm(input))


class Transformer(nn.Module):
    def __init__(self, embedding, model_dim, out_dim,
                 encoder_layers, decoder_layers, heads,
                 horizon, dropout):
        super().__init__()
        self.embedding = embedding
        self.encoder = TransformerEncoder(
            model_dim=model_dim,
            num_layers=encoder_layers,
            heads=heads,
            dropout=dropout
        )
        self.decoder = TransformerDecoder(
            model_dim=model_dim,
            out_dim=out_dim,
            num_layers=decoder_layers,
            heads=heads,
            dropout=dropout
        )
        self.horizon = horizon

    def forward(self, data, time, weekday):
        bank = self.encoder(self.embedding(data, time, weekday))
        time = gen_time(time[:, -1], self.horizon)
        residual = self.decoder(self.embedding(None, time, weekday), bank)
        return residual + data[:, [-1]]
