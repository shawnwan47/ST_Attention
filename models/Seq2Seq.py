import math
from random import random

import torch
import torch.nn as nn

from lib.utils import aeq


class Seq2SeqBase(nn.Module):
    def __init__(self, embedding, encoder, decoder, history, horizon):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.history = history
        self.horizon = horizon
        self.input0 = nn.Parameter(torch.FloatTensor(embedding.size))
        self._reset_input0()

    def _reset_input0(self):
        stdv = 1. / math.sqrt(self.embedding.size)
        self.input0.data.uniform_(-stdv, stdv)

    def _expand_input0(self, input):
        return self.input0.expand_as(input[:, [0]])

    def _check_args(self, data, time, day):
        data_length = self.history + self.horizon - 1
        aeq(data_length, data.size(1), time.size(1), day.size(1))


class Seq2SeqRNN(Seq2SeqBase):
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


class Seq2SeqDCRNN(Seq2SeqRNN):
    def forward(self, data, time, day, teach=0):
        # embedding
        data = data.unsqueeze(-1)
        output = super().forward(data, time, day, teach)
        return output.squeeze(-1)


class Seq2SeqGARNN(Seq2SeqBase):
    def forward(self, data, time, day, teach=0):
        data = data.unsqueeze(-1)
        his = self.history
        # encoding
        input = self.embedding(data[:, :his], time[:, :his], day[:, :his])
        encoder_output, hidden = self.encoder(input)
        # decoding
        input = self._expand_input0(input)
        output_i, hidden, attn = self.decoder(input, hidden)
        output = [output_i]
        for idx in range(his, his + self.horizon - 1):
            # data_i = data[:, [idx]] if random() < teach else output_i.detach()
            data_i = output_i.detach()
            input = self.embedding(data_i, time[:, [idx]], day[:, [idx]])
            output_i, hidden, _ = self.decoder(input, hidden)
            output.append(output_i)
        output = torch.cat(output, 1).squeeze(-1)
        return output, attn


class Seq2SeqTransformer(Seq2SeqBase):
    def forward(self, data, time, day, teach=0):
        his = self.history
        # encoding
        input = self.embedding(data[:, :his], time[:, :his], day[:, :his])
        encoder_output, bank, _ = self.encoder(input)
        # decoding
        input = self._expand_input0(input)
        output_i, _, _ = self.decoder(input, bank)
        output = [output_i]
        for idx in range(his, his + self.horizon - 1):
            data_i = data[:, [idx]] if random() < teach else output_i.detach()
            input = self.embedding(data_i, time[:, [idx]], day[:, [idx]])
            output_i, bank, attn = self.encoder(input, bank)
            output.append(output_i)
        output = torch.cat(output, 1)
        return output, attn


class Seq2SeqSTTransformer(Seq2SeqBase):
    def forward(self, data, time, day, teach=0):
        data = data.unsqueeze(-1)
        his = self.history
        # encoding
        input = self.embedding(data[:, :his], time[:, :his], day[:, :his])
        encoder_output, bank, _, _ = self.encoder(input)
        # decoding
        output = [self.decoder(encoder_output[:, [-1]])]
        for idx in range(self.horizon - 1):
            idx += his
            input = data[:, [idx]] if random() < teach else output[-1].detach()
            input = self.embedding(input, time[:, [idx]], day[:, [idx]])
            output_i, bank, attn_s, attn_t = self.encoder(input, bank)
            output.append(self.decoder(output_i))
        output = torch.cat(output, 1).squeeze(-1)
        return output, attn_s, attn_t
