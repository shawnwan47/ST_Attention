import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from Utils import aeq
from UtilClass import *


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, dim, head=1, dropout=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        self.attention = MultiHeadAttention(dim, head, dropout)
        self.layer_norm = BottleLayerNorm(dim)
        self.mlp = PointwiseMLP(dim)

    def forward(self, query, context, mask=None):
        out, att = self.attention(query, context, context, mask)
        out = self.layer_norm(query + out)
        out = self.mlp(out)
        return out, att


class Attention(nn.Module):
    def __init__(self, dim, attn_type, dropout=0.1):
        super(Attention, self).__init__()
        assert attn_type in ['dot', 'add', 'general', 'mlp']
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(-1)
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
    def __init__(self, dim, head=1, dropout=0.1):
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


class LogisticMixtures(nn.Module):
    def __init__(self, components, categories):
        super(LogisticMixtures, self).__init__()
        self.components = components
        self.categories = categories
        self.logsoftmax = nn.LogSoftmax(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, params):
        '''params: batch x length x num_params'''
        batch, length, num_params = params.size()
        assert num_params % 3 == 0
        params = params.view(-1, self.components, 3)
        log_pi = self.logsoftmax(params[:, :, [0]])
        mu = params[:, :, [1]]
        inv_std = torch.exp(params[:, :, [2]])
        xs = torch.arange(self.categories) / self.categories * 2 - 1
        xs = xs.expand(batch * length, self.components, self.categories)
        x1 = xs - 1. / self.categories
        x2 = xs + 1. / self.categories
        x1[:, :, 0] = -float('inf')
        x2[:, :, -1] = float('inf')
        x1, x2 = Variable(x1).cuda(), Variable(x2).cuda()
        pdf = self.sigmoid((x2 - mu) * inv_std) - self.sigmoid((x1 - mu) * inv_std)
        print(pdf.sum(-1))
        log_pdf = torch.log(torch.exp(log_pi + torch.log(pdf)).sum(1))
        print(log_pdf)
        return log_pdf.view(batch, length, self.categories)
