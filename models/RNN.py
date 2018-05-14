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

    def forward(self, input, hidden=None):
        return self.rnn(input, hidden)


class RNNDecoder(RNN):
    def __init__(self, rnn_type, input_size, output_size, hidden_size, num_layers, dropout=0):
        super().__init__(rnn_type, input_size, hidden_size, num_layers, dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.linear(self.dropout(output))
        return output, hidden


class RNNAttnDecoder(RNNDecoder):
    def __init__(self, rnn_type, attn_type, input_size, output_size, hidden_size, num_layers, dropout=0):
        super().__init__(rnn_type, input_size, output_size, hidden_size, num_layers, dropout)
        self.attention = Attention.GlobalAttention(attn_type, hidden_size)

    def forward(self, input, hidden, context):
        output, hidden = self.rnn(input, hidden)
        output, attn = self.attention(output, context)
        output = self.linear(self.dropout(output))
        return output, hidden, attn
