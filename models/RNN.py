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
