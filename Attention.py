import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils import aeq
from UtilClass import *


class AttentionBase(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super(AttentionBase, self).__init__()
        self.dim = dim
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = BottleLayerNorm(dim)
        self.softmax = nn.Softmax(2)


class AttentionInterface(nn.Module):
    def __init__(self, dim, attn_type, value_proj=False, head=1, dropout=0.2):
        super(AttentionInterface, self).__init__()
        self.attn_type = attn_type
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
        self.value_proj = value_proj
        if value_proj:
            self.w = BottleLinear(dim, dim)

    def forward(self, inp, mask=None):
        inp = inp.contiguous()
        batch, length, dim = inp.size()
        if self.attn_type == 'head':
            return self.attn(inp, mask)
        elif self.attn_type == 'self':
            attn = self.attn(inp)
        else:
            attn = self.attn(inp, mask)
        out = torch.bmm(attn, inp)
        if self.attn_type == 'self':
            out = out.view(batch, dim)
        if self.value_proj:
            out = self.w(out)
        return out, attn


class SelfAttention(AttentionBase):
    def __init__(self, dim, dropout=0.2):
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
        return attn


class DotAttention(AttentionBase):
    def forward(self, inp, mask):
        attn = torch.bmm(inp, inp.transpose(1, 2)) / math.sqrt(self.dim)
        if mask is not None:
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(attn)
        return attn


class GeneralAttention(AttentionBase):
    def __init__(self, dim, dropout=0.2):
        super(GeneralAttention, self).__init__(dim, dropout)
        self.w = BottleLinear(dim, dim, bias=False)

    def forward(self, inp, mask=None):
        context = self.dropout(self.w(inp))
        attn = torch.bmm(inp, context.transpose(1, 2))
        attn /= math.sqrt(self.dim)
        if mask is not None:
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(attn)
        return attn


class MLPAttention(AttentionBase):
    def __init__(self, dim, dropout=0.2):
        super(MLPAttention, self).__init__(dim, dropout)
        self.w_k = BottleLinear(dim, dim, bias=False)
        self.w_q = BottleLinear(dim, dim, bias=True)
        self.v = BottleLinear(dim, 1, bias=False)

    def forward(self, inp, mask=None):
        batch, length, dim = inp.size()
        inp = inp.contiguous()
        dim = self.dim
        query = self.w_q(inp).unsqueeze(1)
        key = self.w_k(inp).unsqueeze(2)
        query = query.expand(batch, length, length, dim)
        key = key.expand(batch, length, length, dim)
        hid = self.dropout(self.activation(query + key)).view(-1, dim)
        score = self.v(hid).view(batch, length, length)
        if mask is not None:
            score.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(score)
        return attn


class MultiHeadedAttention(AttentionBase):
    def __init__(self, dim, head=1, dropout=0.2):
        '''
        Args:
            head(int): number of parallel heads.
            dim(int): the dimension of keys/values/queries in this
                MultiHeadedAttention, must be divisible by head.
        '''
        pad = head - dim % head
        dim += pad
        super(MultiHeadedAttention, self).__init__(dim, dropout)
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
        # out = self.layer_norm(out)
        out = self.layer_norm(self.dropout(out) + inp)
        # out = self.layer_norm(out)
        attn = attn.view(batch, self.head, length, length)
        if self.pad[1]:
            out = out[:, :, self.pad[0]:-self.pad[1]]
        return out.contiguous(), attn
