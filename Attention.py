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
        if attn_type == 'add':
            self.u = BottleLinear(dim, 1, bias=False)
            self.v = BottleLinear(dim, 1, bias=False)
        elif attn_type == 'mlp':
            self.u = BottleLinear(dim, dim, bias=False)
            self.v = BottleLinear(dim, dim, bias=False)
            self.a = BottleLinear(dim, 1, bias=False)
        elif attn_type == 'general':
            self.w = BottleLinear(dim, dim, bias=False)

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


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, dim, head=4, dropout=0.1):
        '''
        Args:
            head(int): number of parallel heads.
            dim(int): the dimension of keys/values/queries in this
                MultiHeadAttention, must be divisible by head.
        '''
        assert dim % head == 0 and dim % head == 0
        super(MultiHeadAttention, self).__init__()
        self.head = head
        self.dim = dim
        self.w_q = BottleLinear(dim, dim, bias=False)
        self.w_k = BottleLinear(dim, dim, bias=False)
        self.w_v = BottleLinear(dim, dim, bias=False)
        self.softmax = nn.Softmax(2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, qry, key, val, mask=None):
        '''
        qry: batch x length_q x dim
        key: batch x length_c x dim
        val: batch x length_c x dim
        att: batch x head x length x dim_head
        '''
        batch, len_qry, dim = qry.size()
        batch_key, len_ctx, dim_key = key.size()
        batch_val, len_ctx_, dim_val = val.size()
        aeq(batch, batch_key, batch_val)
        aeq(dim, dim_key, dim_val, self.dim)
        aeq(len_ctx, len_ctx_)
        dim_head = dim // self.head

        def shape_projection(x):
            return x.view(batch, -1, self.head, dim_head) \
                .transpose(1, 2).contiguous() \
                .view(batch * self.head, -1, dim_head)

        def unshape_projection(x):
            return x.view(batch, self.head, -1, dim_head) \
                    .transpose(1, 2).contiguous() \
                    .view(batch, -1, self.dim)

        qry = shape_projection(self.w_q(qry))
        key = shape_projection(self.w_k(key))
        val = shape_projection(self.w_v(val))

        score = torch.bmm(qry, key.transpose(1, 2)) / math.sqrt(dim_head)
        if mask is not None:
            score = score.view(batch, self.head, len_qry, len_ctx)
            score.data.masked_fill_(mask, -float('inf'))
        att = self.softmax(score.view(-1, len_qry, len_ctx))

        out = torch.bmm(self.dropout(att), val)
        out = self.dropout(unshape_projection(out))
        att = att.view(batch, self.head, len_qry, len_ctx)
        return out, att
