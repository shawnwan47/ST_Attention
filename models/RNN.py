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


class RNNSeq2Seq(nn.Module):
    def __init__(self, embedding, encoder, decoder, history, horizon):
        super().__init__()
        self.hidden_size = embedding.features
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.history = history
        self.horizon = horizon
        self.start = self._init_start()

    def _init_start(self):
        start = nn.Parameter(torch.FloatTensor(self.hidden_size))
        nn.init.xavier_normal_(start.data)
        return start

    def start_decoding(self, input):
        return self.start.expand_as(input[:, [0]])

    def forward(self, data, time, day, teach=0):
        self._check_args(data, time, day)
        his = self.history
        # encoding
        input = self.embedding(data[:, :his], time[:, :his], day[:, :his])
        encoder_output, hidden = self.encoder(input)
        # decoding
        input_i = self.start_decoding(input)
        output_i, hidden = self.decoder(input_i, hidden)
        output = [output_i]
        for idx in range(his, his + self.horizon - 1):
            data_i = data[:, [idx]] if random() < teach else output_i.detach()
            input_i = self.embedding(data_i, time[:, [idx]], day[:, [idx]])
            output_i, hidden = self.decoder(input_i, hidden)
            output.append(output_i)
        output = torch.cat(output, 1)
        return output
