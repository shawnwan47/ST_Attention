import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils import aeq
from UtilClass import *


class Attention(nn.Module):
    def __init__(self, dim, attn_type, dropout=0.1):
        super(Attention, self).__init__()
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
        att = self.softmax(score)
        out = torch.bmm(att, context)
        return out, att

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


class MixAttention(nn.Module):
    def __init__(self, dim, attn_types, dropout=0.1):
        super(MixAttention, self).__init__()
        self.att = nn.ModuleList([Attention(dim, attn_type, dropout)
                                   for attn_type in attn_types])
        self.softmax = nn.Softmax(2)

    def forward(self, query, context, mask=None):
        score = [att.score(query, context) for att in self.att]
        score = torch.sum(torch.stack(score), 0)
        if mask is not None:
            score.masked_fill_(mask, -float('inf'))
        return self.softmax(score)


class MergeAttention(Attention):
    def __init__(self, dim, attn_type, merge_type, dropout=0.1):
        super(MergeAttention, self).__init__(dim, attn_type, dropout)
        self.merge_type = merge_type
        if merge_type == 'cat':
            self.linear = BottleLinear(2 * dim, dim)

    def forward(self, query, context, mask):
        out, att = super(MergeAttention, self).forward(query, context, mask)
        if self.merge_type == 'add':
            out = self.dropout(out) + query
        elif self.merge_type == 'cat':
            out = torch.cat((query, out), -1)
            out = self.linear(self.dropout(out))
        return out, att


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_key, dim_val, head=4, dropout=0.1):
        '''
        Args:
            head(int): number of parallel heads.
            dim(int): the dimension of keys/values/queries in this
                MultiHeadAttention, must be divisible by head.
        '''
        assert dim_key % head == 0 and dim_val % head == 0
        super(MultiHeadAttention, self).__init__()
        self.head = head
        self.dim_key = dim_key
        self.dim_val = dim_val
        self.w_q = BottleLinear(self.dim_key, self.dim_key, bias=False)
        self.w_k = BottleLinear(self.dim_key, self.dim_key, bias=False)
        self.w_v = BottleLinear(self.dim_val, self.dim_val, bias=False)
        self.softmax = nn.Softmax(2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, qry, key, val, mask=None):
        '''
        qry: batch x length_q x dim_key
        key: batch x length_c x dim_key
        val: batch x length_c x dim_val
        att: batch x head x length x dim_head
        '''
        batch, len_qry, dim_key = qry.size()
        batch_key, len_ctx, dim_key_ = key.size()
        batch_val, len_ctx_, dim_val = val.size()

        aeq(batch, batch_key, batch_val)
        aeq(dim_key, dim_key_, self.dim_key)
        aeq(dim_val, self.dim_val)
        aeq(len_ctx, len_ctx_)
        dim_key = dim_key // self.head
        dim_val = dim_val // self.head

        def shape_projection(x, dim_head):
            return x.view(batch, -1, self.head, dim_head) \
                .transpose(1, 2).contiguous() \
                .view(batch * self.head, -1, dim_head)

        def unshape_projection(x, dim_head):
            return x.view(batch, self.head, -1, dim_head) \
                    .transpose(1, 2).contiguous() \
                    .view(batch, -1, self.head * dim_head)

        qry = shape_projection(self.w_q(qry), dim_key)
        key = shape_projection(self.w_k(key), dim_key)
        val = shape_projection(self.w_v(val), dim_val)

        score = torch.bmm(qry, key.transpose(1, 2)) / math.sqrt(dim_key)
        if mask is not None:
            score = score.view(batch, self.head, len_qry, len_ctx)
            score.data.masked_fill_(mask, -float('inf'))
        att = self.softmax(score.view(-1, len_qry, len_ctx))

        out = torch.bmm(self.dropout(att), val)
        out = self.dropout(unshape_projection(out, dim_val))
        att = att.view(batch, self.head, len_qry, len_ctx)
        return out, att


class SelfAtt(nn.Module):
    def __init__(self, dim, hop=1, dropout=0.1):
        super(SelfAtt, self).__init__()
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
        att: -1 x hop x length
        out: -1 x hop x dim
        '''
        assert inp.dim() < 4
        hid = self.activation(self.w_1(inp))
        score = self.w_2(self.dropout(hid)).transpose(1, 2)
        att = self.softmax(score)
        out = torch.bmm(att, inp)
        return out, att
