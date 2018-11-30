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


class RNNSeq2Vec(Seq2VecBase):
    def forward(self, data, time, day):
        input = self.embedding(data, time, day)
        output, _ = self.encoder(input)
        return self.decoder(output)


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
            output_i, hidden = self.encoder(input, hidden)
            output_i = decoder(output_i)
            output.append(output_i)
        return torch.cat(output, 1)
