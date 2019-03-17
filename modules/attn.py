import math
import torch
import torch.nn as nn


class MultiHeadedAttention(nn.Module):
    def __init__(self, model_dim, heads, dropout=0.2, out_dim=None):
        assert model_dim % heads == 0
        if out_dim is None:
            out_dim = model_dim
        super().__init__()
        self.model_dim = model_dim
        self.heads = heads
        self.head_dim = model_dim // heads
        self.score_scale = math.sqrt(self.head_dim)
        self.linear_key = nn.Linear(model_dim, model_dim)
        self.linear_value = nn.Linear(model_dim, model_dim)
        self.linear_query = nn.Linear(model_dim, model_dim)
        self.linear_out = nn.Linear(model_dim, out_dim)
        self.softmax = nn.Softmax(-1)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, bank, mask=None):
        def shape(x):
            y = x.view(*x.size()[:-1], self.heads, self.head_dim)
            return y.transpose(-2, -3).contiguous()

        def unshape(x):
            y = x.transpose(-2, -3).contiguous()
            return y.view(*y.size()[:-2], self.model_dim)

        query = shape(self.linear_query(query))
        key = shape(self.linear_key(bank)).transpose(-1, -2).contiguous()
        value = shape(self.linear_value(bank))

        query = query / self.score_scale
        scores = torch.matmul(query, key)
        if mask is not None:
            scores.masked_fill_(mask, -float('inf'))
        attn = self.softmax(scores)

        out = torch.matmul(self.drop(attn), value)
        return self.linear_out(unshape(out))
