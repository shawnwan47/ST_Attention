import math
import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    def __init__(self, model_dim, heads, dropout=0.2, out_dim=None):
        assert model_dim % heads == 0
        if out_dim is None:
            out_dim = model_dim
        super().__init__()
        self.model_dim = model_dim
        self.heads = heads
        self.head_dim = model_dim // heads
        self.score_scale = math.sqrt(self.head_dim)
        self.fc_k = nn.Linear(model_dim, model_dim)
        self.fc_v = nn.Linear(model_dim, model_dim)
        self.fc_q = nn.Linear(model_dim, model_dim)
        self.fc_out = nn.Linear(model_dim, out_dim)
        self.softmax = nn.Softmax(-1)
        self.drop = nn.Dropout(dropout)

    def shape(self, x):
        y = x.view(*x.size()[:-1], self.heads, self.head_dim)
        return y.transpose(-2, -3).contiguous()

    def unshape(self, x):
        y = x.transpose(-2, -3).contiguous()
        return y.view(*y.size()[:-2], self.model_dim)

    def attend(self, query, bank, mask):
        query = self.shape(self.fc_q(query))
        key = self.shape(self.fc_k(bank)).transpose(-1, -2).contiguous()
        query = query / self.score_scale
        scores = torch.matmul(query, key)
        if mask is not None:
            scores.masked_fill_(mask, -float('inf'))
        return self.softmax(scores)

    def forward(self, query, bank, mask=None):
        attn = self.attend(query, bank, mask)
        value = self.shape(self.fc_v(bank))
        context = self.unshape(torch.matmul(self.drop(attn), value))
        output = self.fc_out(context)
        return output


class HeadAttendedAttention(MultiheadAttention):
    def __init__(self, model_dim, heads, dropout=0.1, out_dim=None):
        super().__init__(model_dim, heads, dropout, out_dim)
        self.fc_q_head = nn.Linear(model_dim, heads)
        self.fc_c_head = nn.Linear(model_dim, heads)

    def forward(self, query, bank, mask=None):
        attn = self.attend(query, bank, mask)
        # attn: batch_size x num_heads x num_query x num_bank
        value = self.shape(self.fc_v(bank))
        context = self.unshape(torch.matmul(self.drop(attn), value))
        # context: batch_size x num_query x model_dim
        scores_head = self.fc_q_head(query) + self.fc_c_head(context)
        attn_head = self.softmax(scores_head).unsqueeze(1)
        # attn_head: batch_size x num_query x heads
        context = self.shape(context).transpose(1, -1).mul(attn_head)
        context = self.unshape(context.transpose(1, -1))
        context = self.fc_out(self.heads * context)
        return context
