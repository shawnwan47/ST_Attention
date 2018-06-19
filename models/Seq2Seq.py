from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import aeq


class Seq2SeqBase(nn.Module):
    def __init__(self, embedding, encoder, decoder, history, horizon):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.history = history
        self.horizon = horizon

    def _check_args(self, data, time, day):
        data_length = self.history + self.horizon - 1
        aeq(data_length, data.size(1), time.size(1), day.size(1))

    def super_forward(self, data, time, day):
        self._check_args(data, time, day)
        input = self.embedding(data, time, day)
        output = self.encoder(input)
        if isinstance(output, tuple):
            output = output[0]
        output = self.decoder(output)
        if isinstance(output, tuple):
            output = output[0]
        return output[:, -self.horizon:]

class Seq2SeqRNN(Seq2SeqBase):
    def forward(self, data, time, day, teach=0):
        self._check_args(data, time, day)
        his = self.history
        # encoding
        input = self.embedding(data[:, :his], time[:, :his], day[:, :his])
        encoder_output, hidden = self.encoder(input)
        # decoding
        output = [self.decoder(encoder_output[:, [-1]])]
        for idx in range(his, his + self.horizon - 1):
            input = data[:, [idx]] if random() < teach else output[-1].detach()
            input = self.embedding(input, time[:, [idx]], day[:, [idx]])
            encoder_output, hidden = self.encoder(input, hidden)
            output.append(self.decoder(encoder_output))
        return torch.cat(output, 1)


class Seq2SeqDCRNN(Seq2SeqRNN):
    def forward(self, data, time, day, teach):
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
        encoder_output, hidden, _, _ = self.encoder(input)
        # decoding
        output = [self.decoder(encoder_output[:, [-1]])]
        for idx in range(his, his + self.horizon - 1):
            data_i = data[:, [idx]] if random() < teach else output[-1].detach()
            input = self.embedding(data_i, time[:, [idx]], day[:, [idx]])
            output_i, hidden, attn_i, attn_h = self.encoder(input, hidden)
            output.append(self.decoder(output_i))
        output = torch.cat(output, 1).squeeze(-1)
        return output, attn_i, attn_h


class Seq2SeqTransformer(Seq2SeqBase):
    def forward(self, data, time, day, teach=0):
        his = self.history
        # encoding
        input = self.embedding(data[:, :his], time[:, :his], day[:, :his])
        encoder_output, bank, _ = self.encoder(input)
        # decoding
        output = [self.decoder(encoder_output[:, [-1]])]
        for idx in range(self.horizon - 1):
            idx += his
            input = data[:, [idx]] if random() < teach else output[-1].detach()
            input = self.embedding(input, time[:, [idx]], day[:, [idx]])
            output_i, bank, attn = self.encoder(input, bank)
            output.append(self.decoder(output_i))
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
