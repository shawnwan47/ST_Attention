import random

import torch
import torch.nn as nn
import torch.nn.functional as F



class Seq2SeqBase(nn.Module):
    def __init__(self, embedding, encoder, decoder, history, horizon):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.history = history
        self.horizon = horizon

    def _decode(self, input):
        raise NotImplementedError


class Seq2SeqRNN(Seq2SeqBase):
    def _decode(self, input):
        output = self.decoder(input)
        return output[0] if isinstance(output, tuple) else output

    def forward(self, data, time, weekday, teach=0):
        his = self.history
        # encoding
        input = self.embedding(data[:, :his], time[:, :his], weekday[:, :his])
        encoder_output, hidden = self.encoder(input)
        # decoding
        output = [self._decode(encoder_output[:, [-1]])]
        for idx in range(self.horizon - 1):
            idx += his
            input = data[:, [idx]] if random.random() < teach else output[-1]
            input = self.embedding(input, time[:, [idx]], weekday[:, [idx]])
            encoder_output, hidden = self.encoder(input, hidden)
            output.append(self._decode(encoder_output))
        return torch.cat(output, 1)


class Seq2SeqGCRNN(Seq2SeqRNN):
    def forward(self, data, time, weekday, teach):
        # embedding
        data = data.unsqueeze(-1)
        output = super().forward(data, time, weekday, teach)
        return output.squeeze(-1)
