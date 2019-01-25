import torch
import torch.nn as nn

from modules import MLP
from modules import TransformerLayer, TransformerDecoderLayer
from modules import STTransformerLayer, STTransformerDecoderLayer
from lib.io import gen_subsequent_time


class STransformer(TransformerEncoder):
    def __init__(self, embedding, model_dim, out_dim, num_layers, heads, dropout, mask):
        super().__init__(model_dim, num_layers, heads, dropout)
        self.embedding = embedding
        self.mlp_out = MLP(model_dim, out_dim, dropout)
        self.register_buffer('mask', mask)

    def forward(self, data, time, weekday):
        emb = self.embedding(data, time, weekday)
        input = super().forward(emb, self.mask)
        output = self.mlp_out(input)
        return output


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
                 horizon, dropout, mask_s):
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
        self.register_buffer('mask_s', mask_s)

    def forward(self, data, time, weekday):
        input = self.embedding(data, time, weekday)
        bank = self.encoder(input, self.mask_s)
        time = gen_subsequent_time(time[:, -1], self.horizon)
        input = self.embedding(None, time, weekday)
        res = self.decoder(input, bank, self.mask_s)
        return res + data[:, [-1]]


class TransformerEncoder(nn.Module):
    def __init__(self, model_dim, num_layers, heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(model_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, input, mask=None):
        for layer in self.layers:
            input = layer(input, input, mask)
        return self.layer_norm(input)


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


class STTransformerEncoder(nn.Module):
    def __init__(self, model_dim, num_layers, heads, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            STTransformerLayer(model_dim, heads, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, input, mask_s):
        for layer in self.layers:
            input = layer(input, input, mask_s=mask_s)
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

    def forward(self, input, bank, mask_s):
        for layer in self.layers:
            input = layer(input, bank, mask_s)
        return self.mlp(self.layer_norm(input))
