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
        self.mlp = MLP(model_dim, 1)

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
            input_i = self.embedding(data_i, time_i, weekday)
            res, hidden = self.decoder(input_i, hidden)
            out.append(data_i + res)
            data_i = data_i + res.detach()
            time_i = time_i + 1
        return torch.cat(out, 1)


class RNNAttnSeq2Seq(RNNSeq2Seq):
    def __init__(self, embedding, rnn_type, model_dim, num_layers, out_dim, heads, dropout, horizon):
        super().__init__(embedding, rnn_type, model_dim, num_layers, out_dim, dropout, horizon)
        self.attn = MultiHeadedAttention(model_dim, heads, dropout)

    def forward(self, data, time, weekday):
        # encoding
        input = self.embedding(data[:, :-1], time[:, :-1], weekday)
        bank, hidden = self.encoder(input)
        # decoding
        data_i = data[:, [-1]]
        time_i = time[:, [-1]]
        out = []
        for _ in range(self.horizon):
            input_i = self.embedding(data_i, time_i, weekday)
            res, hidden = self.decoder(input_i, hidden, bank)
            out.append(data_i + res)
            data_i = out[-1].detach()
            time_i = time_i + 1
        return torch.cat(out, 1)
