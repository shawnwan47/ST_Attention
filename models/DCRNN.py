from random import random

import torch
import torch.nn as nn

from models import SpatialRNN
from models import Framework


class DCRNN(SpatialRNN.SpatialRNN):
    def __init__(self, rnn_type, size, num_layers, num_nodes, adj, hops):
        super().__init__(rnn_type, size, num_layers, num_nodes,
                         func=_DiffusionConvolution,
                         adj=adj,
                         hops=hops)


class DCRNNDecoder(DCRNN):
    def __init__(self, *args, **kw_args):
        super().__init__(*args, **kw_args)
        self.fc = nn.Linear(kw_args['size'], 1)

    def forward(self, *args, **kw_args):
        output, hidden = super().forward(*args, **kw_args)
        return self.fc(output), hidden


class DCRNNSeq2Seq(SpatialRNN.SpatialRNNSeq2Seq):
    pass


class _DiffusionConvolution(nn.Module):
    def __init__(self, input_size, output_size, adj, hops):
        super().__init__()
        self.filters = self._gen_adj_hops(adj, hops)
        self.filters += self._gen_adj_hops(adj.t(), hops)
        self.linear = nn.Linear(input_size, output_size)
        self.linears = nn.ModuleList([
            nn.Linear(input_size, output_size, bias=False)
            for _ in self.filters
        ])

    @staticmethod
    def _gen_adj_hops(adj, hops):
        adj_norm = adj.div(adj.sum(1).unsqueeze(-1))
        adjs = [adj_norm]
        for _ in range(hops - 1):
            adjs.append(adjs[-1].matmul(adj_norm))
        return adjs

    def forward(self, input):
        output = self.linear(input)
        for linear, filter in zip(self.linears, self.filters):
            output = output + filter.matmul(linear(input))
        return output
