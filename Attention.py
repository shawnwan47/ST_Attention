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
                context = self.w(context).transpose(1, 2)
            score = torch.bmm(query, context) / math.sqrt(self.dim)
        else:
            query = self.u(query).unsqueeze(1).expand(batch, length, length, -1)
            context = self.v(context).unsqueeze(2).expand(batch, length, length, -1)
            score = query + context
            if self.attn_type == 'mlp':
                score = self.a(self.dropout(F.tanh(score)).view(-1, dim))
        return score.view(batch, length, length)


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


class HeadAttn(nn.Module):
    def __init__(self, dim, head=4, dropout=0.1):
        '''
        Args:
            head(int): number of parallel heads.
            dim(int): the dimension of keys/values/queries in this
                HeadAttn, must be divisible by head.
        '''
        pad = head - dim % head
        dim += pad
        super(HeadAttn, self).__init__()
        self.head = head
        self.dim = dim
        self.pad = (pad // 2, pad // 2 + pad % 2)
        self.w_k = BottleLinear(dim, dim, bias=False)
        self.w_v = BottleLinear(dim, dim, bias=False)
        self.w_q = BottleLinear(dim, dim, bias=False)
        self.softmax = nn.Softmax(2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, mask=None):
        '''
        inp: batch x length x dim
        attn: batch x head x length x dim_head
        '''
        if self.pad[1]:
            inp = F.pad(inp, self.pad)
        batch, length, dim = inp.size()
        dim_head = dim // self.head

        def shape_projection(x):
            return x.view(batch, length, self.head, dim_head) \
                .transpose(1, 2).contiguous() \
                .view(batch * self.head, length, dim_head)

        def unshape_projection(x):
            return x.view(batch, self.head, length, dim_head) \
                    .transpose(1, 2).contiguous() \
                    .view(batch, length, dim)

        key = shape_projection(self.w_k(inp))
        value = shape_projection(self.w_v(inp))
        query = shape_projection(self.w_q(inp))

        score = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(dim_head)
        if mask is not None:
            score = score.view(batch, self.head, length, length)
            score.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(score.view(-1, length, length))

        out = torch.bmm(self.dropout(attn), value)
        out = self.dropout(unshape_projection(out))
        attn = attn.view(batch, self.head, length, length)
        if self.pad[1]:
            out = out[:, :, self.pad[0]:-self.pad[1]].contiguous()
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
