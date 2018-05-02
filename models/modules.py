import math

import torch
import torch.nn as nn


class DayTimeEmbedding(nn.Module):
    def __init__(self, num_time, time_size, day_size, pdrop=0):
        super().__init__()
        self.embedding_day = nn.Embedding(7, day_size)
        self.embedding_time = nn.Embedding(num_time, time_size)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, data_cat):
        embedded_day = self.embedding_day(data_cat[:, :, 0])
        embedded_time = self.embedding_time(data_cat[:, :, 1])
        return self.dropout(torch.cat((embedded_time, embedded_day), dim=-1))


class SparseMM(torch.autograd.Function):
    def __init__(self, sparse):
        super().__init__()
        self.sparse = sparse

    def forward(self, dense):
        return torch.mm(self.sparse, dense)

    def backward(self, grad_output):
        grad_input = None
        if self.needs_input_grad[0]:
            grad_input = torch.mm(self.sparse.t(), grad_output)
        return grad_input


class GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = SparseMM(adj)(support)
        if self.bias is not None:
            return output + self.bias
        else:
            return outpu
