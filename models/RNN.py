import random
import torch
import torch.nn as nn

import models.Attention as Attention


class RNN(nn.Module):
    def __init__(self, rnn_type, nin, nhid, nlayers, pdrop=0):
        assert rnn_type in ('RNN', 'GRU', 'LSTM')
        super().__init__()
        self.rnn = getattr(nn, rnn_type)(
            input_size=nin,
            hidden_size=nhid,
            num_layers=nlayers,
            batch_first=True,
            dropout=pdrop)

    def forward(self, input, hidden=None):
        return self.rnn(input, hidden)


class RNNDecoder(RNN):
    def __init__(self, rnn_type, nin, nout, nhid, nlayers, pdrop=0):
        super().__init__(rnn_type, nin, nhid, nlayers, pdrop)
        self.linear = nn.Linear(nhid, nout)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.linear(self.dropout(output))
        return output, hidden


class RNNAttnDecoder(RNNDecoder):
    def __init__(self, rnn_type, attn_type, nin, nout, nhid, nlayers, pdrop=0):
        super().__init__(rnn_type, nin, nout, nhid, nlayers, pdrop)
        self.attention = Attention.GlobalAttention(attn_type, nhid)

    def forward(self, input, hidden, context):
        output, hidden = self.rnn(input, hidden)
        output, attn = self.attention(output, context)
        output = self.linear(self.dropout(output))
        return output, hidden, attn


class Seq2Seq(nn.Module):
    def __init__(self, model, embedding, encoder, decoder, enc_len, dec_len):
        super().__init__()
        assert model in ['RNN', 'RNNAttn', 'GCRNN']
        self.model = model
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
        self.enc_len = enc_len
        self.dec_len = dec_len

    def forward(self, input_num, input_cat, teach):
        # embedding
        embedded = self.embedding(input_cat)
        enc_input = torch.cat((input_num[:, :self.enc_len],
                               embedded[:, :self.enc_len]), dim=-1)
        # encoding
        enc_output, hidden = self.encoder(enc_input)
        # decoding
        dec_output = input_num[:, [self.enc_len - 1]]
        output, attns = [], []
        for i in range(self.dec_len):
            if random.random() < teach:
                dec_input_num = input_num[:, [self.enc_len + i - 1]]
            else:
                dec_input_num = dec_output.detach()
            dec_input = torch.cat((dec_input_num, embedded[:, [i]]), -1)
            if self.model == 'RNN':
                dec_output, hidden = self.decoder(dec_input, hidden)
            else:
                dec_output, hidden, attn = self.decoder(
                    dec_input, hidden, enc_output)
                attns.append(attn)
            output.append(dec_output)
        if self.model == 'RNN':
            return torch.cat(output, 1)
        else:
            return torch.cat(output, 1), torch.cat(attns, 1)
