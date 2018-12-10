from random import random

import torch
import torch.nn as nn

from models import Framework


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
    def __init__(self, rnn_type, size, output_size, num_layers, dropout=0):
        super().__init__(rnn_type, size, num_layers, dropout)
        self.fc = nn.Linear(size, output_size)

    def forward(self, input, hidden):
        output, hidden = super().forward(input, hidden)
        return self.fc(output), hidden


class RNNSeq2Seq(Framework.Seq2Seq):
    pass
