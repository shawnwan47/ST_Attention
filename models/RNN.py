import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, rnn_type, nin, nhid, nlayers, nonlinear='tanh', pdrop=0):
        assert rnn_type in ('RNN', 'GRU', 'LSTM')
        super().__init__()
        self.rnn = getattr(nn, rnn_type)(
            input_size=nin,
            hidden_size=nhid,
            num_layers=nlayers,
            nonlinearity=nonlinear,
            batch_first=True,
            dropout=pdrop)

    def initHidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(bsz, self.nlayers, self.nhid),
                    weight.new_zeros(bsz, self.nlayers, self.nhid))
        else:
            return weight.new_zeros(bsz, self.nlayers, self.nhid)

    def forward(self, input, hidden):
        output, hidden = self.rnn(data, hidden)
        return output, hidden


class RNNDecoder(RNN):
    def __init__(self, rnn_type, nin, nout, nhid, nlayers,
                 nonlinear='tanh', dropout=0):
        super().__init__(rnn_type, input_size, hidden_size, num_layers,
                         nonlinear, dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output = self.linear(self.dropout(output))
        return output, hidden
