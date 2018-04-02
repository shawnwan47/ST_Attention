import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from Utils import aeq
from UtilClass import *


class Attention(nn.Module):
    def __init__(self, dim, att_type, dropout=0.1):
        super(Attention, self).__init__()
        assert att_type in ['dot', 'add', 'general', 'mlp']
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(-1)
        self.att_type = att_type
        if att_type == 'add':
            self.map_q = BottleLinear(dim, 1, bias=False)
            self.map_k = BottleLinear(dim, 1, bias=False)
        elif att_type == 'mlp':
            self.map_q = BottleLinear(dim, dim, bias=False)
            self.map_k = BottleLinear(dim, dim, bias=False)
            self.map_qk = BottleLinear(dim, 1, bias=False)
        elif att_type == 'general':
            self.map_qk = BottleLinear(dim, dim, bias=False)

    def forward(self, qry, key, val, mask=None):
        '''
        qry, key, val: batch x num x features
        att: batch x num_qry x num_key
        '''
        score = self.score(qry, key)
        if mask is not None:
            score.data.masked_fill_(mask, -float('inf'))
        att = self.softmax(score)
        return torch.bmm(self.dropout(att), val), att.cpu()

    def score(self, qry, key):
        batch, len_q, dim = qry.size()
        batch_, len_k, dim_ = key.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        if self.att_type in ['dot', 'general']:
            if self.att_type == 'general':
                key = self.relu(self.map_qk(key))
            score = torch.bmm(qry, key.transpose(1, 2)) / math.sqrt(dim)
        else:
            qry = self.map_q(qry).unsqueeze(2).expand(batch, len_q, len_k, -1)
            key = self.map_k(key).unsqueeze(1).expand(batch, len_q, len_k, -1)
            score = qry + key
            if self.att_type == 'mlp':
                score = self.map_qk(F.relu(self.dropout(qry)))
        return score.view(batch, len_q, len_k)


class AttentionFusion(nn.Module):
    def __init__(self, dim, head, att_type, dropout):
        super().__init__()
        self.head = head
        self.attention_head = nn.ModuleList([Attention(dim, att_type, dropout)
                                             for _ in range(head)])
        self.attention_fusion = Attention(dim, att_type, dropout)

    def forward(self, qry, key, val, mask):
        '''
        qry, key, val: batch x num x features
        contexts: batch x num_qry x head x features
        att_head: batch x num_qry x num_head x num_key
        att_fusion: batch x num_qry x num_head
        '''
        batch, num, features = qry.size()
        contexts = torch.stack([self.attention_head[i](qry, key, val, mask)
                                for i in range(self.head)], 1)
        contexts, att_head = [], []
        for attention in self.attention_head:
            ctx, att = attention(qry, key, val, mask)
            contexts.append(ctx)
            att_head.append(att)
        att_head = torch.stack(att_head, -2)
        contexts = torch.stack(contexts, -2).view(-1, self.head, features)
        context, att_fusion = self.attention_fusion(qry.view(-1, 1, features),
                                                    contexts, contexts)
        context = context.view(batch, num, features)
        att_fusion = att_fusion.view(batch, num, self.head)
        return context, att_fusion, att_head


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, head=4, dropout=0.1):
        '''
        Args:
            head(int): number of parallel heads.
            dim(int): the dimension of keys/values/queries in this
                MultiHeadAttention, must be divisible by head.
        '''
        assert dim % head == 0
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
        batch, len_q, dim = qry.size()
        batch_key, len_c, dim_key = key.size()
        batch_val, len_c_, dim_val = val.size()
        aeq(batch, batch_key, batch_val)
        aeq(dim, dim_key, dim_val, self.dim)
        aeq(len_c, len_c_)
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
            score = score.view(batch, self.head, len_q, len_c)
            score.data.masked_fill_(mask, -float('inf'))
        att = self.softmax(score.view(-1, len_q, len_c))

        out = torch.bmm(self.dropout(att), val)
        out = self.dropout(unshape_projection(out))
        att = att.view(batch, self.head, len_q, len_c)
        return out, att
