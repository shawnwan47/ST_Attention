import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size, adj):
        super().__init__()
        self.adj = adj
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input):
        assert input.size(-2) == len(self.adj)
        return self.adj.matmul(self.linear(input))


class DiffusionConvolution(nn.Module):
    def __init__(self, input_size, output_size, adj, hops, uni=False):
        super().__init__()
        self.gc = nn.ModuleList()
        adj_t = adj.t()
        adj /= adj.sum(0)
        adj_t /= adj_t.sum(0)
        # directed
        adj_k = adj[:]
        for hop in range(hops):
            adj_k = adj_k.matmul(adj)
            self.gc.append(GraphConvolution(input_size, output_size, adj_k))
        # reversed
        if not uni:
            adj_k = adj_t[:]
            for hop in range(hops):
                adj_k = adj_k.matmul(adj_t)
                self.gc.append(GraphConvolution(input_size, output_size, adj_k))

    def forward(self, input):
        return sum(gc(input) for gc in self.gc)
