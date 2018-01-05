import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils import aeq
from UtilClass import *


class Attn(nn.Module):
    def __init__(self, dim, attn_type, dropout=0.1):
        super(Attn, self).__init__()
        assert attn_type in ['dot', 'add', 'general', 'mlp']
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(2)
        self.attn_type = attn_type
        if attn_type == 'general':
            self.w = BottleLinear(dim, dim, bias=False)
        elif attn_type == 'add':
            self.u = BottleLinear(dim, 1, bias=False)
            self.v = BottleLinear(dim, 1, bias=False)
        elif attn_type == 'mlp':
            self.u = BottleLinear(dim, dim, bias=False)
            self.v = BottleLinear(dim, dim, bias=False)
            self.a = BottleLinear(dim, 1, bias=False)

    def forward(self, query, context, mask=None):
        score = self.score(query, context)
        if mask is not None:
            score.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(score)
        out = torch.bmm(attn, context)
        return out, attn

    def score(self, query, context):
        batch, length, dim = query.size()
        query = query.contiguous()
        if self.attn_type in ['dot', 'general']:
            if self.attn_type == 'general':
                context = self.w(context)
            score = torch.bmm(query, context.transpose(1, 2))
            score /= math.sqrt(self.dim)
        else:
            query = self.u(query).unsqueeze(1).expand(batch, length, length, -1)
            context = self.v(context).unsqueeze(2).expand(batch, length, length, -1)
            score = query + context
            if self.attn_type == 'mlp':
                score = self.a(F.tanh(score).view(batch, length, length))
        return score.view(batch, length, length)


class MixAttn(nn.Module):
    def __init__(self, dim, attn_types, dropout=0.1):
        super(MixAttn, self).__init__()
        self.attn = nn.ModuleList([Attn(dim, attn_type, dropout)
                                   for attn_type in attn_types])
        self.softmax = nn.Softmax(2)

    def forward(self, query, context, mask=None):
        score = [attn.score(query, context) for attn in self.attn]
        score = torch.sum(torch.stack(score), 0)
        if mask is not None:
            score.masked_fill_(mask, -float('inf'))
        return self.softmax(score)


class MergeAttn(Attn):
    def __init__(self, dim, attn_type, merge_type, dropout=0.1):
        super(MergeAttn, self).__init__(dim, attn_type, dropout)
        self.merge_type = merge_type
        if merge_type == 'cat':
            self.linear = BottleLinear(2 * dim, dim)

    def forward(self, query, context, mask):
        out, attn = super(MergeAttn, self).forward(query, context, mask)
        if self.merge_type == 'add':
            out = self.dropout(out) + query
        elif self.merge_type == 'cat':
            out = torch.cat((query, out), -1)
            out = self.linear(self.dropout(out))
        return out, attn


class LinearAttn(Attn):
    def __init__(self, in_features, out_features, attn_type='dot', dropout=0.1):
        super(LinearAttn, self).__init__(out_features, attn_type, dropout)
        self.linear_q = BottleLinear(in_features, out_features, bias=False)
        self.linear_k = BottleLinear(in_features, out_features, bias=False)
        self.linear_v = BottleLinear(in_features, out_features, bias=False)

    def forward(self, query, context, mask=None):
        query = self.linear_q(query)
        key = self.linear_k(context)
        val = self.linear_v(context)
        score = self.score(query, key)
        if mask is not None:
            score.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(score)
        out = torch.bmm(attn, val)
        return out, attn


class MultiHeadAttn(nn.Module):
    def __init__(self, dim, head=4, dropout=0.1):
        '''
        Args:
            head(int): number of parallel heads.
            dim(int): the dimension of keys/values/queries in this
                MultiHeadAttn, must be divisible by head.
        '''
        assert dim % head == 0
        super(MultiHeadAttn, self).__init__()
        self.head = head
        self.dim = dim
        self.w_k = BottleLinear(self.dim, self.dim, bias=False)
        self.w_v = BottleLinear(self.dim, self.dim, bias=False)
        self.w_q = BottleLinear(self.dim, self.dim, bias=False)
        self.softmax = nn.Softmax(2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        '''
        query: batch x length x dim
        attn: batch x head x length x dim_head
        '''
        batch, len_query, dim = query.size()
        batch_, len_context, dim_ = key.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        dim_head = dim // self.head

        def shape_projection(x):
            return x.view(batch, -1, self.head, dim_head) \
                .transpose(1, 2).contiguous() \
                .view(batch * self.head, -1, dim_head)

        def unshape_projection(x):
            return x.view(batch, self.head, -1, dim_head) \
                    .transpose(1, 2).contiguous() \
                    .view(batch, -1, dim)

        key = shape_projection(self.w_k(key))
        value = shape_projection(self.w_v(value))
        query = shape_projection(self.w_q(query))

        score = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(dim_head)
        if mask is not None:
            score = score.view(batch, self.head, len_query, len_context)
            score.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(score.view(-1, len_query, len_context))

        out = torch.bmm(self.dropout(attn), value)
        out = self.dropout(unshape_projection(out))
        attn = attn.view(batch, self.head, len_query, len_context)
        return out, attn


class SelfAttn(nn.Module):
    def __init__(self, dim, hop=1, dropout=0.1):
        super(SelfAttn, self).__init__()
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(2)
        self.hop = hop
        hid = int(math.sqrt(dim * hop))
        self.w_1 = BottleLinear(dim, hid)
        self.w_2 = BottleLinear(hid, hop)

    def forward(self, inp):
        '''
        inp: -1 x length x dim
        attn: -1 x hop x length
        out: -1 x hop x dim
        '''
        assert inp.dim() < 4
        hid = self.activation(self.w_1(inp))
        score = self.w_2(self.dropout(hid)).transpose(1, 2)
        attn = self.softmax(score)
        out = torch.bmm(attn, inp)
        return out, attn
