import random

import torch
import torch.nn as nn
import torch.nn.functional as F



class Seq2SeqBase(nn.Module):
    def __init__(self, embedding, encoder, decoder, seq_len_in, seq_len_out):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.seq_len_in = seq_len_in
        self.seq_len_out = seq_len_out


class Seq2SeqRNN(Seq2SeqBase):
    def forward(self, input_num, input_cat, teach=0):
        # embedding
        embedded = self.embedding(input_cat)
        # encoding
        encoder_input = torch.cat((input_num[:, :self.seq_len_in],
                                   embedded[:, :self.seq_len_in]), dim=-1)
        encoder_output, hidden = self.encoder(encoder_input)
        # decoding
        output_i = self.decoder(encoder_output[:, [-1]])
        output = [output_i]
        for idx in range(self.seq_len_out - 1):
            idx += self.seq_len_in
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
    def forward(self, input_num, input_cat, teach):
        # embedding
        input_num = input_num.unsqueeze(-1)
        output = super().forward(input_num, input_cat, teach)
        return output.squeeze(-1)
