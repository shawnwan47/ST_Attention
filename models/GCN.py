import torch
import torch.nn as nn

from lib import graph


class DiffusionConvolution(nn.Module):
    def __init__(self, input_size, output_size, adj, hops, uni=False):
        super().__init__()
        # calculate adjs
        self.filters = [adj.new_tensor(torch.eye(len(adj)))]
        self.filters.extend(self._gen_adj_hops(adj, hops))
        if not uni:
            self.filters.extend(self._gen_adj_hops(adj.t(), hops))
        self.num_filters = len(self.filters)
        self.filters = torch.stack(self.filters, dim=0)
        # multi-head graph convolution
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size * self.num_filters, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_size))

    @staticmethod
    def _gen_adj_hops(adj, hops):
        adj_norm = adj.div(adj.sum(1).unsqueeze(-1))
        adjs = [adj_norm]
        for _ in range(hops - 1):
            adjs.append(adjs[-1].matmul(adj_norm))
        return adjs

    def forward(self, input):
        size_hidden = list(input.size())[:-1]
        size_hidden.extend([self.num_filters, self.output_size])
        output = self.linear(input).view(size_hidden).transpose(-2, -3)
        output = self.filters.matmul(output).sum(-3) + self.bias
        return output
