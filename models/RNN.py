import random
import torch
import torch.nn as nn

from models.Framework import Seq2SeqBase, Seq2VecBase


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
        self.linear_out = nn.Linear(size, output_size)
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, input, hidden):
        output, hidden = super().forward(input, hidden)
        output = self.layer_norm(output)
        output = self.linear_out(output)
        return output, hidden


class RNNSeq2Seq(Seq2SeqBase):
    def forward(self, data, time, day, teach=0):
        self._check_args(data, time, day)
        his = self.history
        # encoding
        input = self.embedding(data[:, :his], time[:, :his], day[:, :his])
        encoder_output, hidden = self.encoder(input)
        # decoding
        input = self._expand_input0(input)
        output_i, hidden = self.decoder(input, hidden)
        output = [output_i]
        for idx in range(his, his + self.horizon - 1):
            # data_i = data[:, [idx]] if random() < teach else output_i.detach()
            data_i = output_i.detach()
            input = self.embedding(data_i, time[:, [idx]], day[:, [idx]])
            output_i, hidden = self.decoder(input, hidden)
            output.append(output_i)
        return torch.cat(output, 1)
