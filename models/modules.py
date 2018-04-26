import math

import torch
import torch.nn as nn


class DayTimeEmbedding(nn.Module):
    def __init__(self, num_time, time_size, day_size, pdrop=0):
        self.embedding_time = nn.Embedding(num_time, time_size)
        self.embedding_day = nn.Embedding(7, day_size)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, data_cat):
        embedded_time = self.embedding_time(data_cat[:, :, 0])
        embedded_day = self.embedding_day(data_cat[:, :, 1])
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
