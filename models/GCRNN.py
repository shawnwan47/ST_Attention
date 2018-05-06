import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, nin, nout, graph):
        super().__init__()
        self.graph = graph
        self.linear = nn.Linear(nin, nout)

    def forward(self, input):
        assert input.size(-2) == len(self.graph)
        return self.graph.matmul(self.linear(input))


class DiffusionConvolution(nn.Module):
    def __init__(self, nin, nout, graph, nhop=1, reversed=False):
        super().__init__()
        self.gc_kernels = nn.ModuleList()
        graph_t = graph.t()
        graph /= graph.sum(0)
        graph_t /= graph_t.sum(0)
        graph_k = graph[:]
        for hop in range(nhop):
            graph_k.matmul(graph)
            self.gc_kernels.append(GraphConvolution(nin, nout, graph_k))
        if reversed:
            graph_k = graph_t[:]
            for hop in range(nhop):
                graph_k.matmul(graph_t)
                self.gc_kernels.append(GraphConvolution(nin, nout, graph_k))

    def forward(self, input):
        return torch.sum((gc(input) for gc in self.gc_kernels), -1)


class GCRNNCell(nn.Module):
    def __init__(self, rnn_type, nin, nout, graph):
        assert rnn_type in ['RNN', 'GRU', 'LSTM']
        super().init()
        self.rnn_type = rnn_type
        self.nout = nout
        self.gc = GraphConvolution()
        self.ngate = self.nout
        if self.rnn_type == 'GRU': self.ngate *= 3
        elif self.rnn_type == 'LSTM': self.ngate *= 4
        self.gc_i = GraphConvolution(nin, self.ngate, graph)
        self.gc_h = GraphConvolution(nout, self.ngate, graph)

    def forward(self, input, hidden):
        if self.rnn_type == 'RNN':
            output = F.tanh(input, hidden)
        elif self.rnn_type == 'GRU':
            output = self.gru(input, hidden)
        else:
            output = self.lstm(input, hidden)
        return output

    def rnn(self, input, hidden):
        fusion = self.gc_i(input) + self.gc_h(hidden)
        output = F.tanh(self.gc(fusion))
        return output

    def gru(self, input, hidden):
        gi = self.gc_i(input)
        gh = self.gc_h(hidden)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)

        resetgate = F.sigmoid(self.gc(i_r + h_r))
        inputgate = F.sigmoid(self.gc(i_i + h_i))
        newgate = F.tanh(self.gc(i_n + resetgate * h_n))
        output = newgate + inputgate * (hidden - newgate)
        return output

    def lstm(self, input, hidden):
        hx, cx = hidden
        gates = self.gc(self.gc_i(input) + self.gc_h(hx))

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * F.tanh(cy)
        return hy, cy


class GCRNN(nn.Module):
    def __init__(self, rnn_type, nnode, nin, nhid, nlayers, graph, pdrop=0):
        super().__init__()
        self.rnn_type = rnn_type
        self.nnode = nnode
        self.nhid = nhid
        self.nlayers = nlayers
        self.layers = nn.ModuleList((GCRNNCell(rnn_type, nin, nhid, graph)))
        self.layers.extend((GCRNNCell(rnn_type, nhid, nhid, graph)
                            for i in range(nlayers - 1)))
        self.dropout = nn.Dropout(pdrop)

    def forward(self, input, hidden=None):
        bsz, seq_len, nnode, nin = input.size()
        if hidden is None:
            hidden = self.init_hidden(bsz)
        output = []
        for idx in range(seq_len):
            out_i = input[idx]
            for lay in range(self.nlayers):
                if self.rnn_type == 'LSTM':
                    hidden[0][:, lay], hidden[1][:, lay] = self.layers[lay](
                        out_i, (hidden[0][:, lay], hidden[1][:, lay]))
                    out_i = hidden[0][:, lay]
                else:
                    hidden[:, lay] = self.layers[lay](out_i, hidden[:, lay])
                    out_i = hidden[:, lay]
                out_i = self.dropout(out_i)
            output.append(out_i)
        output = torch.stack(output)
        return output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        size = (bsz, self.nlayers, self.nnode, self.nhid)
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(size), weight.new_zeros(size))
        else:
            return weight.new_zeros(size)
