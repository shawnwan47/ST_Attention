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


class Seq2SeqRNN(Seq2SeqBase):
    def forward(self, data, time, weekday, teach=0):
        his = self.history
        # encoding
        input = self.embedding(data[:, :his], time[:, :his], weekday[:, :his])
        encoder_output, hidden = self.encoder(input)
        # decoding
        output = [self.decoder(encoder_output[:, [-1]])]
        for idx in range(self.horizon - 1):
            idx += his
            input = data[:, [idx]] if random.random() < teach else output[-1].detach()
            input = self.embedding(input, time[:, [idx]], weekday[:, [idx]])
            encoder_output, hidden = self.encoder(input, hidden)
            output.append(self.decoder(encoder_output))
        return torch.cat(output, 1)


class Seq2SeqDCRNN(Seq2SeqRNN):
    def forward(self, data, time, weekday, teach):
        # embedding
        data = data.unsqueeze(-1)
        output = super().forward(data, time, weekday, teach)
        return output.squeeze(-1)


class Seq2SeqGARNN(Seq2SeqBase):
    def forward(self, data, time, weekday, teach=0):
        data = data.unsqueeze(-1)
        his = self.history
        # encoding
        input = self.embedding(data[:, :his], time[:, :his], weekday[:, :his])
        encoder_output, hidden, attn_i, attn_h = self.encoder(input)
        # decoding
        output = [self.decoder(encoder_output[:, [-1]])]
        attn_input, attn_hidden = [], []
        for idx in range(self.horizon - 1):
            idx += his
            input = data[:, [idx]] if random.random() < teach else output[-1].detach()
            input = self.embedding(input, time[:, [idx]], weekday[:, [idx]])
            encoder_output, hidden, attn_i, attn_h = self.encoder(input, hidden)
            output.append(self.decoder(encoder_output))
            attn_input.append(attn_i)
            attn_hidden.append(attn_h)
        output = torch.cat(output, 1).squeeze(-1)
        attn_input = torch.stack(attn_input, 1)
        attn_hidden = torch.stack(attn_hidden, 1)
        return output, attn_input, attn_hidden
