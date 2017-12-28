import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils import aeq
from UtilClass import *


class Attn(nn.Module):
    def __init__(self, dim, attn_type, dropout=0.1):
        super(Attn, self).__init__()
        assert attn_type in ['add', 'mul', 'mlp']
        self.dim = dim
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(2)
        self.attn_type = attn_type
        if attn_type == 'add':
            self.u = BottleLinear(dim, 1, bias=False)
            self.v = BottleLinear(dim, 1, bias=False)
        elif attn_type == 'mul':
            self.w = BottleLinear(dim, dim, bias=False)
        elif attn_type == 'mlp':
            self.u = BottleLinear(dim, dim, bias=False)
            self.v = BottleLinear(dim, dim, bias=False)
            self.a = BottleLinear(dim, 1, bias=False)

    def score(self, inp):
        batch, length, dim = inp.size()
        inp = inp.contiguous()
        if self.attn_type == 'mul':
            key = self.w(inp).transpose(1, 2)
            score = torch.bmm(inp, key) / math.sqrt(self.dim)
        else:
            query = self.u(inp).unsqueeze(1).expand(batch, length, length, -1)
            key = self.v(inp).unsqueeze(2).expand(batch, length, length, -1)
            score = query + key
            if self.attn_type == 'mlp':
                score = self.a(self.dropout(self.tanh(score)).view(-1, dim))
        return score.view(batch, length, length)

    def forward(self, inp, mask=None):
        score = self.score(inp)
        if mask is not None:
            score.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(score)
        out = torch.bmm(attn, inp)
        return out, attn


class CatAttn(Attn):
    def __init__(self, dim, attn_type, dropout=0.1):
        super(CatAttn, self).__init__(dim, attn_type, dropout)
        self.linear = BottleLinear(2 * dim, dim)

    def forward(self, inp):
        out, attn = super(CatAttn, self).forward(inp)
        out = torch.cat((inp, out), -1)
        out = self.linear(self.dropout(out))
        return out, attn


class LinearAttn(Attn):
    def __init__(self, in_features, out_features, attn_type, dropout=0.1):
        super(LinearAttn, self).__init__(out_features, attn_type, dropout)
        self.linear_key = BottleLinear(in_features, out_features, bias=False)
        self.linear_val = BottleLinear(in_features, out_features, bias=False)

    def forward(self, inp, mask=None):
        key = self.linear_key(inp)
        val = self.linear_val(inp)
        score = self.score(key)
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
        self.dim = dim
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = BottleLayerNorm(dim)
        self.softmax = nn.Softmax(2)
        self.pad = (pad // 2, pad // 2 + pad % 2)
        self.head = head
        self.w_k = BottleLinear(dim, dim, bias=False)
        self.w_v = BottleLinear(dim, dim, bias=False)
        self.w_q = BottleLinear(dim, dim, bias=False)

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

        score = torch.bmm(query, key.transpose(1, 2))
        score = score / math.sqrt(dim_head)
        if mask is not None:
            score = score.view(batch, self.head, length, length)
            score.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(score.view(-1, length, length))

        out = torch.bmm(self.dropout(attn), value)
        out = unshape_projection(out)
        out = self.dropout(out) + inp
        out = self.layer_norm(out)
        attn = attn.view(batch, self.head, length, length)
        if self.pad[1]:
            out = out[:, :, self.pad[0]:-self.pad[1]]
        return out.contiguous(), attn


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
