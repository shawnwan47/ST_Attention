import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2Seq(nn.Module):
    def __init__(self, embedding, encoder, decoder, enc_len, dec_len):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.enc_len = enc_len
        self.dec_len = dec_len

    def forward(self, input_num, input_cat):
        assert input_num.size(1) == self.enc_len + self.dec_len - 1
        embedded = self.embedding(input_cat)
        enc_input = torch.cat((input_num[:, :self.enc_len], embedded[:, :self.enc_len]), -1)
        return enc_input, embedded


class Seq2SeqRNN(Seq2Seq):
    def forward(self, input_num, input_cat, teach=0.5):
        enc_input, embedded = super().forward(input_num, input_cat)
        # encoding
        hidden = self.encoder.initHidden(input_num.size(0))
        enc_output, hidden = self.encoder(enc_input, hidden)
        # decoding
        dec_output = input_num[:, [self.enc_len - 1]]
        output = []
        for i in range(self.dec_len):
            if random.random() < teach:
                dec_input_num = dec_output.detach()
            else:
                dec_input_num = input_num[:, [self.enc_len + i - 1]]
            dec_input = torch.cat(dec_input_num, embedded[:, [i]], -1)
            dec_output, hidden = self.decoder(dec_input, hidden)
            output.append(dec_output)
        return torch.cat(output, -1)


class Seq2SeqRNNAttn(Seq2Seq):
    def forward(self, input_num, input_cat, teach):
        enc_input, embedded = super().forward(input_num, input_cat)
        # encoding
        hidden = self.encoder.initHidden(input_num.size(0))
        enc_output, hidden = self.encoder(enc_input, hidden)
        # decoding
        de_output = input_num[:, [self.enc_len - 1]]
        output, attn = [], []
        for i in range(self.dec_len):
            if random.random() < teach:
                dec_input_num = dec_output.detach()
            else:
                dec_input_num = input_num[:, [self.enc_len + i - 1]]
            dec_input = torch.cat(dec_input_num, embedded[:, [i]], -1)
            dec_output, hidden, attention = self.decoder(dec_input, hidden, enc_output)
            output.append(dec_output)
            attn.append(attention)
        return torch.cat(output, -1), torch.cat(attn, -2)


class Seq2SeqAttn(Seq2Seq):
    def forward(self, input_num, input_cat, teach):
        enc_input, embedded = super().forward(input_num, input_cat)
        # encoding
        hidden = self.encoder.initHidden(input_num.size(0))
        enc_output, hidden = self.encoder(enc_input, hidden)
        # decoding
        de_output = input_num[:, [self.enc_len - 1]]
        output, attn = [], []
        for i in range(self.dec_len):
            if random.random() < teach:
                dec_input_num = dec_output.detach()
            else:
                dec_input_num = input_num[:, [self.enc_len + i - 1]]
            dec_input = torch.cat(dec_input_num, embedded[:, [i]], -1)
            dec_output, hidden, attention = self.decoder(dec_input, hidden, enc_output)
            output.append(dec_output)
            attn.append(attention)
        return torch.cat(output, -1), torch.cat(attn, -2)
