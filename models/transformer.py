import torch
import torch.nn as nn

from modules import MLP
from modules import TransformerLayer, TransformerDecoderLayer
from modules import STTransformerLayer, STTransformerDecoderLayer
from lib.io import gen_subsequent_time


class STransformer(nn.Module):
    def __init__(self, embedding, model_dim, out_dim, num_layers, heads, dropout):
        super().__init__()
        self.embedding = embedding
        self.encoder = TransformerEncoder(model_dim, num_layers, heads, dropout)
        self.decoder = MLP(model_dim, out_dim, dropout)

    def forward(self, data, time, weekday):
        input = self.embedding(data, time, weekday)
        hidden = self.encoder(input)
        return self.decoder(hidden)


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
        time = gen_subsequent_time(time[:, -1], self.horizon)
        residual = self.decoder(self.embedding(None, time, weekday), bank)
        return residual + data[:, [-1]]


class STTransformer(nn.Module):
    def __init__(self, embedding, model_dim,
                 encoder_layers, decoder_layers, heads,
                 horizon, dropout):
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

    def forward(self, data, time, weekday):
        input = self.embedding(data, time, weekday)
        bank = self.encoder(input)
        time = gen_subsequent_time(time[:, -1], self.horizon)
        input = self.embedding(None, time, weekday)
        res = self.decoder(input, bank)
        return res + data[:, [-1]]


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
            input = layer(input)
        return self.layer_norm(input)


class TransformerDecoder(nn.Module):
    def __init__(self, model_dim, out_dim, num_layers, heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(model_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.decoder = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            MLP(model_dim, out_dim, dropout)
        )

    def forward(self, query, bank):
        for layer in self.layers:
            query = layer(query, bank)
        return self.decoder(query)


class STTransformerEncoder(nn.Module):
    def __init__(self, model_dim, num_layers, heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            STTransformerLayer(model_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return self.layer_norm(input)


class STTransformerDecoder(nn.Module):
    def __init__(self, model_dim, num_layers, heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            STTransformerDecoderLayer(model_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.decoder = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
            MLP(model_dim, 1, dropout)
        )

    def forward(self, input, bank):
        for layer in self.layers:
            input = layer(input, bank)
        return self.decoder(input)
