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
        assert 'RNN' in model
        super().__init__(embedding, encoder, decoder, seq_len_in, seq_len_out)
        self.model = model

    def forward(self, input_num, input_cat, teach=0):
        # embedding
        embedded = self.embedding(input_cat)
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
            if 'Attn' in self.model:
                decoder_output, hidden, attn = self.decoder(
                    decoder_input, hidden, enc_output)
                attns.append(attn)
            else:
                decoder_output, hidden = self.decoder(decoder_input, hidden)
            output.append(decoder_output)
        if self.model == 'RNN':
            return torch.cat(output, 1)
        else:
            return torch.cat(output, 1), torch.cat(attns, 1)


class Seq2SeqGCRNN(Seq2SeqRNN):
    def forward(self, input_num, input_cat, teach):
        # embedding
        input_num = input_num.unsqueeze(-1)
        output = super().forward(input_num, input_cat, teach)
        if self.model == 'GCRNN':
            return output.squeeze(-1)
        else:
            return output[0].squeeze(-1), output[1]
