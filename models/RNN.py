import random
import torch
import torch.nn as nn

import models.Attention as Attention


class RNN(nn.Module):
    def __init__(self, rnn_type, size, num_layers, dropout=0):
        assert rnn_type in ('RNN', 'RNNReLU', 'GRU', 'LSTM')
        super().__init__()
        kwargs = {
            'input_size': size,
            'hidden_size': size,
            'num_layers': num_layers,
            'batch_first': True,
            'dropout': dropout
        }
        if rnn_type == 'RNNReLU':
            kwargs['nonlinearity'] = 'relu'
        self.rnn = getattr(nn, rnn_type)(**kwargs)

    def forward(self, input, hidden=None):
        return self.rnn(input, hidden)


class RNNDecoder(RNN):
    def __init__(self, rnn_type, size, out_size, num_layers, dropout=0):
        super().__init__(rnn_type, size, num_layers, dropout)
        self.linear_out = nn.Linear(size, out_size)
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, input, hidden):
        output, hidden = super().forward(input, hidden)
        output = self.layer_norm(output)
        output = self.linear_out(output)
        return output, hidden

def build_RNN(args):
    return RNN(
        rnn_type=args.rnn_type,
        size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )


def build_RNNDecoder(args):
    return RNN(
        rnn_type=args.rnn_type,
        size=args.hidden_size,
        out_size=args.num_nodes,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
