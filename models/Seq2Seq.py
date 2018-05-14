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
    def __init__(self, model, embedding, encoder, decoder,
                 seq_len_in, seq_len_out):
        assert model in ['RNN', 'RNNAttn']
        super().__init__(embedding, encoder, decoder, seq_len_in, seq_len_out)
        self.model = model

    def pre_forward(self, input_num, input_cat):
        # embedding
        embedded = self.embedding(input_cat)
        return input_num, embedded

    def forward(self, input_num, input_cat, teach=0):
        input_num, embedded = self.pre_forward(input_num, input_cat)
        # encoding
        encoder_input = torch.cat((input_num[:, :self.seq_len_in],
                                   embedded[:, :self.seq_len_in]), dim=-1)
        encoder_output, hidden = self.encoder(encoder_input)
        # decoding
        decoder_output = input_num[:, [self.seq_len_in - 1]]
        output, attns = [], []
        for i in range(self.seq_len_out):
            if random.random() < teach:
                decoder_input_num = input_num[:, [self.seq_len_in - 1 + i]]
            else:
                decoder_input_num = decoder_output.detach()
            decoder_input = torch.cat((decoder_input_num, embedded[:, [i]]), -1)
            if self.model == 'RNN':
                decoder_output, hidden = self.decoder(decoder_input, hidden)
            else:
                decoder_output, hidden, attn = self.decoder(
                    decoder_input, hidden, enc_output)
                attns.append(attn)
            output.append(decoder_output)
        if self.model == 'RNN':
            return torch.cat(output, 1)
        else:
            return torch.cat(output, 1), torch.cat(attns, 1)


class Seq2SeqGCRNN(Seq2SeqRNN):
    def pre_forward(self, input_num, input_cat, teach):
        # embedding
        embedded = self.embedding(input_cat)
        input_num = input_num.unsqueeze(-1)
        return input_num, embedded
