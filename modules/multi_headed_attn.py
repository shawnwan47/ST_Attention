import math

import torch
import torch.nn as nn


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.2, d_output=None):
        assert d_model % heads == 0
        if d_output is None:
            d_output = d_model
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_size = d_model // heads
        self.score_scale = math.sqrt(self.head_size)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_output = nn.Linear(d_model, d_output)
        self.softmax = nn.Softmax(-1)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, bank, mask=None):

        def shape(self, x):
            y = x.view(*x.size()[:-1], self.heads, self.head_size)
            return y.transpose(-2, -3).contiguous()

        def unshape(self, x):
            y = x.transpose(-2, -3).contiguous()
            return y.view(*y.size()[:-2], self.size)

        query = shape(self.linear_query(query))
        key = shape(self.linear_key(bank))
        value = shape(self.linear_value(bank))

        scores = torch.matmul(query, key.transpose(-1, -2)) / self.score_scale
        if mask is not None:
            scores.masked_fill_(mask, -float('inf'))
        attn = self.softmax(scores)
        output = torch.matmul(self.drop(attn), value)
        output = self.linear_output(unshape(output))
        return output
