import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size, adj):
        super().__init__()
        self.adj = adj
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input):
        return self.adj.matmul(self.linear(input))


class DiffusionConvolution(nn.Module):
    def __init__(self, input_size, output_size, adj, hops, uni=False):
        super().__init__()
        # calculate adjs
        self.adj = self._gen_adj_hops(adj, hops)
        if not uni:
            self.adj.extend(self._gen_adj_hops(adj.t(), hops))
        self.adj = torch.cat(self.adj, dim=1)
        # multi-head graph convolution
        self.num_channels = hops * (1 + (not uni))
        self.output_size = output_size
        hidden_size = output_size * self.num_channels
        self.linear = nn.Linear(input_size, hidden_size)

    @staticmethod
    def _gen_adj_hops(adj, hops):
        adj_norm = adj.div(adj.sum(1).unsqueeze(-1))
        adjs = [adj_norm]
        for _ in range(hops - 1):
            adjs.append(adjs[-1].matmul(adj_norm))
        return adjs

    def forward(self, input):
        size_hidden = list(input.size())[:-1]
        size_hidden[-1] *= self.num_channels
        size_hidden.append(self.output_size)
        output = self.linear(input).view(size_hidden)
        output = self.adj.matmul(output)
        return output
