import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class GraphConvolution(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, graph=None):
        if graph is None: return input
        assert input.size(-2) == len(graph)
        output = input.transpose(-1, -2).matmul(graph).transpose(-1, -2)
        return output


class GCRNNCell(nn.Module):
    def __init__(self, mode, nin, nout):
        assert mode in ['RNN', 'GRU', 'LSTM']
        super().init()
        self.mode = mode
        self.nout = nout
        self.gc = GraphConvolution(nin, nout)
        self.ngate = self.nout
        if self.mode == 'GRU': self.ngate *= 3
        elif self.mode == 'LSTM': self.ngate *= 4
        self.linear_i = nn.Linear(nin, self.ngate)
        self.linear_h = nn.Linear(nout, self.ngate)

    def forward(self, input, hidden, graph):
        if self.mode == 'RNN':
            output = F.tanh(input, hidden)
        elif self.mode == 'GRU':
            output = self.gru(input, hidden)
        else:
            output = self.lstm(input, hidden)
        return output

    def rnn(self, input, hidden, graph):
        fusion = self.linear_i(input) + self.linear_h(hidden)
        return F.tanh(self.gc(fusion, graph))

    def gru(self, input, hidden, graph):
        gi = self.linear_i(input)
        gh = self.linear_h(hidden)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = F.sigmoid(self.gc(i_r + h_r, graph))
        inputgate = F.sigmoid(self.gc(i_i + h_i, graph))
        newgate = F.tanh(self.gc(i_n + resetgate * h_n, graph))
        output = newgate + inputgate * (hidden - newgate)
        return output

    def lstm(self, input, hidden, graph):
        hx, cx = hidden
        gates = self.gc(self.linear_i(input) + self.linear_h(hx), graph)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)
        return hy, cy


class GCRNN(nn.Module):
    def __init__(self, mode, nnode, nin, nhid, nlayers, pdrop=0):
        super().__init__()
        self.mode = mode
        self.nnode = nnode
        self.nhid = nhid
        self.nlayers = nlayers
        self.layers = nn.ModuleList((GCRNNCell(mode, nin, nhid)))
        self.layers.extend((GCRNNCell(mode, nhid, nhid)
                            for i in range(nlayers - 1)))
        self.dropout = nn.Dropout(pdrop)

    def forward(self, input, hidden=None, graph=None):
        seq_len, bsz, nnode, nin = input.size()
        if hidden is None:
            hidden = self.init_hidden(bsz)
        output = []
        for idx in range(seq_len):
            out_i = input[idx]
            for lay in range(self.nlayers):
                if self.mode == 'LSTM':
                    hidden[0][lay], hidden[1][lay] = self.layers[lay](
                        out_i, (hidden[0][lay], hidden[1][lay]), graph)
                    out_i = hidden[0][lay]
                else:
                    hidden[lay] = self.layers[lay](out_i, hidden[lay], graph)
                    out_i = hidden[lay]
                out_i = self.dropout(out_i)
            output.append(out_i)
        output = torch.stack(output)
        return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        size = (self.nlayers, bsz, self.nnode, self.nhid)
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(size), weight.new_zeros(size))
        else:
            return weight.new_zeros(size)
