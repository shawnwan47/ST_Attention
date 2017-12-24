import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from Utils import aeq
from UtilClass import *


class AttentionBase(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(AttentionBase, self).__init__()
        self.dim = dim
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = BottleLayerNorm(dim)
        self.softmax = nn.Softmax(2)


class AttentionInterface(nn.Module):
    def __init__(self, dim, attn_type, head=1, dropout=0.1):
        super(AttentionInterface, self).__init__()
        if attn_type == 'dot':
            attn = DotAttention(dim, dropout)
        elif attn_type == 'general':
            attn = GeneralAttention(dim, dropout)
        elif attn_type == 'mlp':
            attn = MLPAttention(dim, dropout)
        elif attn_type == 'context':
            attn = ContextAttention(dim, dropout)
        elif attn_type == 'self':
            attn = SelfAttention(dim, dropout)
        elif attn_type == 'head':
            attn = MultiHeadedAttention(dim, head, dropout)
        self.attn = attn

    def forward(self, inp, mask=None):
        return self.attn(inp, mask)


class SelfAttention(AttentionBase):
    def __init__(self, dim, dropout=0.1):
        super(SelfAttention, self).__init__(dim, dropout)
        self.w_1 = BottleLinear(self.dim, self.dim)
        self.w_2 = BottleLinear(self.dim, 1)

    def forward(self, inp):
        '''
        inp: batch x length x dim
        attn: batch x 1 x length
        out: batch x dim
        '''
        batch, length, dim = inp.size()
        aeq(dim, self.dim)
        hid = self.activation(self.w_1(inp))
        score = self.w_2(self.dropout(hid)).transpose(1, 2)
        attn = self.softmax(score)
        out = torch.bmm(attn, inp).view(batch, dim)
        return out, attn


class MultiHeadedAttention(AttentionBase):
    def __init__(self, dim, head=1, dropout=0.1):
        '''
        Args:
            head(int): number of parallel heads.
            dim(int): the dimension of keys/values/queries in this
                MultiHeadedAttention, must be divisible by head.
        '''
        mod = dim % head
        dim += mod
        super(MultiHeadedAttention, self).__init__(dim, dropout)
        self.pad = (mod // 2, mod // 2 + mod % 2)
        self.head = head
        self.dim_head = dim // head
        self.w_k = BottleLinear(dim, dim, bias=False)
        self.w_v = BottleLinear(dim, dim, bias=False)
        self.w_q = BottleLinear(dim, dim, bias=False)

    def forward(self, inp, mask=None):
        '''
        inp: batch x length x dim
        attn: batch x head x length x dim_head
        '''
        if self.pad:
            inp = F.pad(inp, self.pad)
        residual = inp
        batch, length, dim = inp.size()

        def shape_projection(x):
            return x.view(batch, length, self.head, self.dim_head) \
                .transpose(1, 2).contiguous() \
                .view(batch * self.head, length, self.dim_head)

        def unshape_projection(x):
            return x.view(batch, self.head, length, self.dim_head) \
                    .transpose(1, 2).contiguous() \
                    .view(batch, length, self.dim)


        key = shape_projection(self.w_k(inp))
        value = shape_projection(self.w_v(inp))
        query = shape_projection(self.w_q(inp))

        score = torch.bmm(query, key.transpose(1, 2))
        score = score / math.sqrt(self.dim_head)
        if mask is not None:
            score = score.view(batch, self.head, length, length)
            score.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(score.view(-1, length, length))

        # values : (batch * 8) x qlen x dim
        out = torch.bmm(self.dropout(attn), value)
        out = unshape_projection(out)
        out = self.layer_norm(self.dropout(out) + residual)
        out = out[:, :, self.pad[0]:-self.pad[1]]
        attn = attn.view(batch, self.head, length, length)
        return out, attn


class DotAttention(AttentionBase):
    def forward(self, inp, mask):
        attn = torch.bmm(inp, inp.transpose(1, 2)) / math.sqrt(self.dim)
        if mask is not None:
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(attn)
        out = torch.bmm(attn, inp)
        return out, attn


class GeneralAttention(AttentionBase):
    def __init__(self, dim, dropout=0.1):
        super(GeneralAttention, self).__init__(dim, dropout)
        self.w = BottleLinear(dim, dim, bias=False)

    def forward(self, inp, mask=None):
        context = self.w(inp)
        attn = torch.bmm(inp, self.dropout(context.transpose(1, 2)))
        attn /= math.sqrt(self.dim)
        if mask is not None:
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(attn)
        out = torch.bmm(attn, inp)
        return out, attn


class ContextAttention(AttentionBase):
    def __init__(self, dim, dropout=0.1):
        super(ContextAttention, self).__init__(dim, dropout)
        self.w_k = BottleLinear(dim, dim, bias=False)
        self.w_v = BottleLinear(dim, dim, bias=False)

    def forward(self, inp, mask=None):
        key = self.w_k(inp)
        val = self.w_v(inp)
        score = torch.bmm(inp, key.transpose(1, 2))
        score /= math.sqrt(self.dim)
        if mask is not None:
            score.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(score)
        out = torch.bmm(attn, val)
        return out, attn


class MLPAttention(AttentionBase):
    def __init__(self, dim, dropout=0.1):
        super(MLPAttention, self).__init__(dim, dropout)
        self.w_v = BottleLinear(dim, dim, bias=False)
        self.w_k = BottleLinear(dim, dim, bias=False)
        self.w_q = BottleLinear(dim, dim, bias=True)
        self.v = BottleLinear(dim, 1, bias=False)

    def forward(self, inp, mask=None):
        batch, length, dim = inp.size()
        inp = inp.contiguous()
        dim = self.dim
        query = self.w_q(inp).unsqueeze(1)
        key = self.w_k(inp).unsqueeze(2)
        value = self.w_v(inp)
        query = query.expand(batch, length, length, dim)
        key = key.expand(batch, length, length, dim)
        hid = self.dropout(self.activation(query + key)).view(-1, dim)
        score = self.v(hid).view(batch, length, length)
        if mask is not None:
            score.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(score)
        out = torch.bmm(attn, inp)
        return out, attn
