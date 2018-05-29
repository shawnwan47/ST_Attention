import torch
import torch.nn as nn

from lib import graph


class DiffusionConvolution(nn.Module):
    def __init__(self, input_size, output_size, adj, hops, uni=False):
        super().__init__()
        # calculate adjs
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)
        self.filters = self._gen_adj_hops(adj, hops)
        if not uni:
            self.filters.extend(self._gen_adj_hops(adj.t(), hops))
        self.num_filters = len(self.filters)
        # multi-head graph convolution
        self.linears = nn.ModuleList([
            nn.Linear(input_size, output_size, bias=False)
            for _ in range(self.num_filters)
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
            output += filter.matmul(linear(input))
        return output
