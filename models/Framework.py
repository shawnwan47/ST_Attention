import math
from random import random

import torch
import torch.nn as nn

from lib.utils import aeq


class Vec2Vec(nn.Module):
    def __init__(self, embedding, encoder, decoder):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data, time, day):
        return self.decoder(self.encoder(self.embedding(data, time, day)))


class Seq2Vec(nn.Module):
    def __init__(self, embedding, encoder, decoder):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data, time, day):
        input = self.embedding(data, time, day)
        output, _ = self.encoder(input)
        output = self.decoder(output[:, -1])
        return output.transpose(-1, -2)

class Seq2Seq(nn.Module):
    def __init__(self, embedding, encoder, decoder, history, horizon):
        super().__init__()
        self.hidden_size = embedding.output_size
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.history = history
        self.horizon = horizon
        self.start = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self._reset_start()

    def _reset_start(self):
        stdv = 1. / math.sqrt(self.hidden_size)
        self.start.data.uniform_(-stdv, stdv)

    def _check_args(self, data, time, day):
        data_length = self.history + self.horizon - 1
        aeq(data_length, data.size(1), time.size(1), day.size(1))

    def expand_start(self, input):
        return self.start.expand_as(input[:, [0]])

    def forward(self, data, time, day, teach=0):
        self._check_args(data, time, day)
        his = self.history
        # encoding
        input = self.embedding(data[:, :his], time[:, :his], day[:, :his])
        encoder_output, hidden = self.encoder(input)
        # decoding
        input_i = self.expand_start(input)
        output_i, hidden = self.decoder(input_i, hidden)
        output = [output_i]
        for idx in range(his, his + self.horizon - 1):
            data_i = data[:, [idx]] if random() < teach else output_i
            input_i = self.embedding(data_i, time[:, [idx]], day[:, [idx]])
            output_i, hidden = self.decoder(input_i, hidden)
            output.append(output_i)
        output = torch.cat(output, 1)
        return output
