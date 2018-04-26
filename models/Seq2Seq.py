import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2Seq(nn.Module):
    def __init__(self, embedding, encoder, decoder, past, future):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.past = past
        self.future = future

    def forward(self, input_num, input_cat, teach=0.5):
        '''
        input: batch x length x dimension
        '''
        past, future = self.past, self.future
        embedded = self.embedding(input_cat)
        hidden = self.encoder.initHidden(input_num.size(0))
        enc_input = torch.cat(input_num[:, :past], embedded[:, :past], dim=-1)
        enc_output, hidden = self.encoder(enc_input, hidden)

        dec_output = input_num[:, past - 1]
        output = []
        for i in range(past - 1, past + future-1):
            if random.random() < teach:
                dec_input_num = dec_output.detach()
            else:
                dec_input_num = input_num[:, i]
            dec_input = torch.cat(dec_input_num, embedded[:, i, :], dim=-1)
            dec_output, hidden = self.decoder(
                dec_input, embedded[:, i], hidden)
            output.append(dec_output)
        return torch.cat(output, -1)
