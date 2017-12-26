import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils import aeq
from UtilClass import *


class AttnInterface(nn.Module):
    def __init__(self, dim, attn_type, dropout=0.2):
        super(AttnInterface, self).__init__()
        self.attn_type = attn_type
        if attn_type == 'add':
            attn = AddAttention(dim, dropout)
        elif attn_type == 'mul':
            attn = MulAttention(dim, dropout)
        elif attn_type == 'mlp':
            attn = MLPAttention(dim, dropout)
        self.attn = attn

    def forward(self, inp, mask=None):
        return self.attn(inp, mask)


class WAttnInterface(AttnInterface):
    def __init__(self, in_features, out_features, attn_type, dropout=0.2):
        super(WAttnInterface, self).__init__(out_features, attn_type, dropout)
        self.linear = BottleLinear(in_features, out_features, bias=False)

    def forward(self, inp, mask=None):
        out = self.linear(inp)
        out, attn = self.attn(out, mask)
        return out, attn, self.linear.weight


class AddAttention(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super(AddAttention, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(2)
        self.v = BottleLinear(2 * dim, 1, bias=False)

    def forward(self, inp, mask=None):
        batch, length, dim = inp.size()
        inp = inp.contiguous()
        key = inp.unsqueeze(2).expand(batch, length, length, dim)
        query = inp.unsqueeze(1).expand(batch, length, length, dim)
        kq = self.dropout(torch.cat((query, key), -1))
        score = self.v(kq).view(batch, length, length)
        if mask is not None:
            score.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(score)
        out = torch.bmm(attn, inp)
        return out, attn


class MulAttention(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super(MulAttention, self).__init__()
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(2)
        self.w = BottleLinear(dim, dim, bias=False)

    def forward(self, inp, mask=None):
        context = self.dropout(self.w(inp))
        attn = torch.bmm(inp, context.transpose(1, 2))
        attn /= math.sqrt(self.dim)
        if mask is not None:
            attn.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(attn)
        out = torch.bmm(attn, inp)
        return out, attn


class MLPAttention(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super(MLPAttention, self).__init__()
        self.dim = dim
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(2)
        self.w = BottleLinear(dim, dim, bias=False)
        self.u = BottleLinear(dim, dim, bias=False)
        self.v = BottleLinear(dim, 1, bias=False)

    def forward(self, inp, mask=None):
        batch, length, dim = inp.size()
        inp = inp.contiguous()
        dim = self.dim
        qu = self.u(inp).unsqueeze(1).expand(batch, length, length, dim)
        kw = self.w(inp).unsqueeze(2).expand(batch, length, length, dim)
        kq = self.dropout(self.activation(qu + kw)).view(-1, dim)
        score = self.v(kq).view(batch, length, length)
        if mask is not None:
            score.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(score)
        out = torch.bmm(attn, inp)
        return out, attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, dim, head=1, dropout=0.2):
        '''
        Args:
            head(int): number of parallel heads.
            dim(int): the dimension of keys/values/queries in this
                MultiHeadedAttention, must be divisible by head.
        '''
        pad = head - dim % head
        dim += pad
        super(MultiHeadedAttention, self).__init__()
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
        # out = self.layer_norm(self.dropout(out) + inp)
        attn = attn.view(batch, self.head, length, length)
        if self.pad[1]:
            out = out[:, :, self.pad[0]:-self.pad[1]]
        return out.contiguous(), attn


class SelfAttention(nn.Module):
    def __init__(self, dim, hop=1, dropout=0.2):
        super(SelfAttention, self).__init__()
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
