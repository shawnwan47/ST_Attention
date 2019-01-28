from random import random

import torch
import torch.nn as nn

from modules import MLP
from modules import TransformerLayer


class RNN(nn.Module):
    def __init__(self, rnn_type, model_dim, num_layers, dropout):
        assert rnn_type in ('RNN', 'GRU', 'LSTM')
        super().__init__()
        kwargs = {
            'input_size': model_dim,
            'hidden_size': model_dim,
            'num_layers': num_layers,
            'batch_first': True,
            'dropout': dropout
        }
        self.rnn = getattr(nn, rnn_type)(**kwargs)

    def forward(self, input, hidden=None):
        return self.rnn(input, hidden)


class RNNSeq2Seq(nn.Module):
    def __init__(self, embedding, rnn_type, model_dim, num_layers, out_dim,
                 dropout, horizon):
        super().__init__()
        self.embedding = embedding
        self.horizon = horizon
        self.encoder = RNN(rnn_type, model_dim, num_layers, dropout)
        self.decoder = RNN(rnn_type, model_dim, num_layers, dropout)
        self.ln = nn.LayerNorm(model_dim)
        self.mlp = MLP(model_dim, out_dim, dropout)

    def forward(self, data, time, weekday):
        # encoding
        input = self.embedding(data, time, weekday)
        _, hidden = self.encoder(input)
        # init

        # decoding
        data_i = data[:, [-1]]
        time_i = time[:, [-1]] + 1
        out = []
        for _ in range(self.horizon):
            input_i = self.embedding(data_i.detach(), time_i, weekday)
            inter, hidden = self.decoder(input_i, hidden)
            res = self.mlp(self.ln(inter))
            data_i = data_i + res
            time_i = time_i + 1
            out.append(data_i)
        return torch.cat(out, 1)


class RNNAttnSeq2Seq(RNNSeq2Seq):
    def __init__(self, embedding, rnn_type, model_dim, num_layers, out_dim, heads, dropout, horizon):
        super().__init__(embedding, rnn_type, model_dim, num_layers, out_dim, dropout, horizon)
        self.attn = TransformerLayer(model_dim, heads, dropout)
        self.ln_bank = nn.LayerNorm(model_dim)

    def forward(self, data, time, weekday):
        # encoding
        input = self.embedding(data[:, :-1], time[:, :-1], weekday)
        bank, hidden = self.encoder(input)
        bank = self.ln_bank(bank)
        # decoding
        data_i = data[:, [-1]]
        time_i = time[:, [-1]]
        out = []
        for _ in range(self.horizon):
            input_i = self.embedding(data_i.detach(), time_i, weekday)
            query, hidden = self.decoder(input_i, hidden)
            inter = self.attn(query, bank)
            res = self.mlp(self.ln(inter))
            data_i = data_i + res
            time_i = time_i + 1
            out.append(data_i)
        return torch.cat(out, 1)
