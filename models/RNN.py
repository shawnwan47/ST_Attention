import random
import torch
import torch.nn as nn

import models.Attention as Attention


class RNN(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size, num_layers, dropout=0):
        assert rnn_type in ('RNN', 'RNNReLU', 'GRU', 'LSTM')
        super().__init__()
        kwargs = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'batch_first': True,
            'dropout': dropout
        }
        if rnn_type == 'RNNReLU':
            kwargs['nonlinearity'] = 'relu'
        self.rnn = getattr(nn, rnn_type)(**kwargs)
        self.hidden_size = hidden_size
        self.dropout_prob = dropout

    def forward(self, input, hidden=None):
        return self.rnn(input, hidden)
