from random import random

import torch
import torch.nn as nn

from modules import GraphRNN, GraphRNNCell


class DCRNN(GraphRNN):
    def __init__(self, rnn_type, model_dim, num_layers, num_nodes, adj, hops):
        super().__init__(rnn_type, model_dim, num_layers, num_nodes)


class DCRNNCell(GraphRNNCell):
    def build_func(self, input_size, output_size):
        return DiffusionConvolution(input_size, output_size)


class DiffusionConvolution(nn.Module):
    def __init__(self, in_features, out_features, adj):
        super().__init__()
        self.filters = self._gen_adj_hops(adj)
        self.filters += self._gen_adj_hops(adj.t())
        self.linear = nn.Linear(in_features, out_features)
        self.linears = nn.ModuleList([
            nn.Linear(in_features, out_features, bias=False)
            for _ in self.filters
        ])

    @staticmethod
    def _gen_adj_hops(adj, hops=3):
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