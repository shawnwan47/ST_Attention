from random import random

import torch
import torch.nn as nn

from modules import bias
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


class RNNDecoder(RNN):
    def __init__(self, rnn_type, model_dim, num_layers, out_dim, dropout):
        super().__init__(rnn_type, model_dim, num_layers, dropout)
        self.linear = nn.Linear(model_dim, out_dim)

    def forward(self, input, hidden):
        out, hidden = self.rnn(input, hidden)
        return self.linear(out), hidden


class RNNAttnDecoder(RNNDecoder):
    def __init__(self, rnn_type, model_dim, heads, out_dim, num_layers, dropout):
        super().__init__(rnn_type, model_dim, num_layers, dropout)
        self.attn = TransformerLayer(model_dim, heads, dropout)

    def forward(self, input, hidden, bank):
        query, hidden = self.rnn(input, hidden)
        out = self.attn(query, bank)
        return self.linear(out), hidden


class RNNSeq2Seq(nn.Module):
    def __init__(self, embedding, rnn_type, model_dim, num_layers, out_dim,
                 dropout, horizon):
        super().__init__()
        self.embedding = embedding
        self.horizon = horizon
        self.encoder = RNN(
            rnn_type=rnn_type,
            model_dim=model_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.decoder = RNNDecoder(
            rnn_type=rnn_type,
            model_dim=model_dim,
            num_layers=num_layers,
            out_dim=out_dim,
            dropout=dropout
        )

    def forward(self, data, time, weekday):
        # encoding
        input = self.embedding(data, time, weekday)
        _, hidden = self.encoder(input)
        # decoding
        data = data[:, [-1]]
        time = time[:, [-1]]
        out = []
        for idx in range(self.horizon):
            input = self.embedding(data.detach(), time, weekday)
            res, hidden = self.decoder(input, hidden)
            out.append(res + data)
            data = data + res
            time = time + 1
        return torch.cat(out, 1)


class RNNAttnSeq2Seq(RNNSeq2Seq):
    def __init__(self, embedding, rnn_type, model_dim, num_layers, heads, dropout, horizon):
        super().__init__(self, embedding, rnn_type, model_dim, num_layers, dropout, horizon)
        self.decoder = RNNAttnDecoder(
            rnn_type=rnn_type,
            model_dim=model_dim,
            heads=heads,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, data, time, weekday):
        # encoding
        input = self.embedding(data, time, weekday)
        bank, hidden = self.encoder(input)
        # decoding
        data = data[:, [-1]]
        time = time[:, [-1]]
        out = []
        for idx in range(self.horizon):
            input = self.embedding(data, time, weekday)
            res, hidden = self.decoder(input, hidden, bank)
            data += res
            time += 1
            out.append(data)
        return torch.cat(out, 1)
