import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import GCRNN


class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size, graph):
        super().__init__()
        self.graph = graph
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input):
        assert input.size(-2) == len(self.graph)
        return self.graph.matmul(self.linear(input))


class DiffusionConvolution(nn.Module):
    def __init__(self, input_size, output_size, graph, hops=1, reversed=False):
        super().__init__()
        self.gc_kernels = nn.ModuleList()
        graph_t = graph.t()
        graph /= graph.sum(0)
        graph_t /= graph_t.sum(0)
        graph_k = graph[:]
        for hop in range(hops):
            graph_k.matmul(graph)
            self.gc_kernels.append(GraphConvolution(input_size, output_size, graph_k))
        if reversed:
            graph_k = graph_t[:]
            for hop in range(hops):
                graph_k.matmul(graph_t)
                self.gc_kernels.append(GraphConvolution(input_size, output_size, graph_k))

    def forward(self, input):
        return torch.sum((gc(input) for gc in self.gc_kernels), -1)


class DCRNN(GCRNN):
    def __init__(self, rnn_type, node_count,
                 input_size, hidden_size, num_layers, p_dropout,
                 graph, hops=1, reversed=False):
        super().__init__(rnn_type, node_count,
                         input_size, hidden_size, num_layers, p_dropout,
                         DiffusionConvolution, graph, hops, reversed)
