import torch
import torch.nn as nn

import models.Attention as Attention


class RNN(nn.Module):
    def __init__(self, rnn_type, nin, nhid, nlayers, activation='tanh', pdrop=0):
        assert rnn_type in ('RNN', 'GRU', 'LSTM')
        super().__init__()
        self.rnn_type = rnn_type
        self.nlayers = nlayers
        self.nhid = nhid
        self.rnn = getattr(nn, rnn_type)(
            input_size=nin,
            hidden_size=nhid,
            num_layers=nlayers,
            nonlinearity=activation,
            batch_first=True,
            dropout=pdrop)

    def initHidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def forward(self, input, hidden):
        print(input.size(), hidden.size())
        return self.rnn(input, hidden)


class RNNDecoder(RNN):
    def __init__(self, rnn_type, nin, nout, nhid, nlayers,
                 activation='tanh', pdrop=0):
        super().__init__(rnn_type, nin, nhid, nlayers, activation, pdrop)
        self.linear = nn.Linear(nhid, nout)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.linear(self.dropout(output))
        return output, hidden


class RNNAttnDecoder(RNNDecoder):
    def __init__(self, rnn_type, attn_type, nin, nout, nhid, nlayers,
                 activation='tanh', pdrop=0):
        super().__init__(rnn_type, nin, nout, nhid, nlayers, activation, pdrop)
        self.attn = Attention.GlobalAttention(attn_type, nhid)

    def forward(self, input, hidden, context):
        output, hidden = self.rnn(input, hidden)
        output, attention = self.attn(output, context)
        output = self.linear(self.dropout(output))
        return output, hidden, attention
