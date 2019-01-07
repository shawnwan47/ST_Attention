import numpy as np
import torch
import torch.nn as nn

from modules import MultiHeadedAttention
from modules import bias, MLP, ResMLP


class STTransformer(nn.Module):
    def __init__(self, embedding, encoder, decoder, length, mask=None):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.length = length
        self.register_buffer('mask', mask)

    def forward(self, data, time, weekday):
        input = self.embedding(data, time, weekday)
        bank = self.encoder(input, mask)
        time = self._gen_time(time[:, -1])
        input = self.embedding(None, time, weekday)
        output = self.decoder(input, bank, mask)
        return output

    def _gen_time(self, time):
        return torch.stack(time + i + 1 for i in range(self.length), -1)


class STTransformerEncoder(nn.Module):
    def __init__(self, d_model, num_layers, heads, dropout):
        super().__init__()
        self.layers = nn.Modules([
            STTransformerEncoderLayer(d_model, heads, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input, mask):
        for layer in self.layers:
            input = layer(input, mask)
        return self.layer_norm(input)


class STTransformerDecoder(nn.module):
    def __init__(self, d_model, num_layers, heads, dropout):
        super().__init__()
        self.layers = nn.Modules([
            STTransformerDecoderLayer(d_model, heads, dropout)
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, 1, dropout)

    def forward(self, input, bank, mask):
        for layer in self.layers:
            input = layer(input, bank, mask)
        return self.mlp(self.layer_norm(input))


class TransformerLayer(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.attn = MultiHeadedAttention(d_model, heads, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.mlp = ResMLP(d_model, dropout)

    def forward(self, bank, input, mask):
        query = self.layer_norm(input)
        context = self.attn(query, bank, mask)
        output = input + self.drop(context)
        return self.mlp(output)


class STTransformerLayer(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.layer_t = TransformerLayer(d_model, heads, dropout)
        self.layer_s = TransformerLayer(d_model, heads, dropout)

    def forward(self, input, bank, mask=None):
        input_t, bank_t = input.transpose(1, 2), bank.transpose(1, 2)
        output_t = self.layer_t(input_t, bank_t)
        input_s = output_t.transpose(1, 2)
        output = self.layer_s(input_s, input_s, mask)
        return output


class STTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.layer_bank = STTransformerLayer(d_model, heads, dropout)
        self.layer_self = STTransformerLayer(d_model, heads, dropout)

    def forward(self, input, bank, mask):
        output = self.layer_bank(input, bank, mask)
        output = self.layer_self(input, input, mask)
        return output


def gen_temporal_mask(size=24):
    mask = np.triu(np.ones((size, size)), k=1).astype('uint8')
    return torch.from_numpy(mask)
