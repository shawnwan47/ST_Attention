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

    def _encode(self, data, embedded):
        raise NotImplementedError

    def _decode(self, data, embedded, input):
        raise NotImplementedError

    def _merge_input(self, data, embedded):
        raise NotImplementedError


class Seq2SeqRNN(Seq2SeqBase):
    def _encode(self, data, embedded):
        input = self._merge_input(data, embedded)
        output, hidden = self.encoder(input)
        return output, hidden

    def _decode(self, data, embedded, output):
        output, hidden = output
        output_i = self.decoder(output[:, [-1]])
        output = [output_i]
        for idx in range(self.horizon - 1):
            idx += self.history
            # scheduled sampling
            if random.random() < teach:
                input_num_i = input_num[:, [idx]]
            else:
                input_num_i = output_i
            input_i = torch.cat((input_num_i, embedded[:, [idx]]), -1)
            output_i, hidden = self.encoder(input_i, hidden)
            output_i = self.decoder(output_i)
            output.append(output_i)
        return torch.cat(output, 1)


    def forward(self, data, time, weekday, teach=0):
        # embedding
        embedded = self.embedding(time, weekday)
        # encoding
        input = torch.cat((input_num[:, :self.history],
                                   embedded[:, :self.history]), dim=-1)
        encoder_output, hidden = self.encoder(input)
        # decoding
        output_i = self.decoder(encoder_output[:, [-1]])
        output = [output_i]
        for idx in range(self.horizon - 1):
            idx += self.history
            # scheduled sampling
            if random.random() < teach:
                input_num_i = input_num[:, [idx]]
            else:
                input_num_i = output_i
            input_i = torch.cat((input_num_i, embedded[:, [idx]]), -1)
            output_i, hidden = self.encoder(input_i, hidden)
            output_i = self.decoder(output_i)
            output.append(output_i)
        return torch.cat(output, 1)


class Seq2SeqGCRNN(Seq2SeqRNN):
    def forward(self, data, time, weekday, teach):
        # embedding
        input_num = input_num.unsqueeze(-1)
        output = super().forward(data, time, weekday, teach)
        return output.squeeze(-1)


class Seq2SeqGARNN(Seq2SeqRNN):
    def forward(self, data, time, weekday, teach):
