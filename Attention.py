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
            sc1 = self.map_q(qry).unsqueeze(2).expand(batch, len_q, len_k, -1)
            sc2 = self.map_k(key).unsqueeze(1).expand(batch, len_q, len_k, -1)
            score = sc1 + sc2
            if self.att_type == 'mlp':
                score = self.map_qk(F.relu(self.dropout(qry)))
        return score.view(batch, len_q, len_k)


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, dim, dropout=0.1):
        assert dim % head == 0
        self.head = head
        self.dim = dim
        super().__init__()
        self.linear_key = nn.Linear(dim, dim)
        self.linear_value = nn.Linear(dim, dim)
        self.linear_query = nn.Linear(dim, dim)
        self.sm = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(dim, dim)

    def forward(self, key, value, query, mask=None):
        batch, k_len, dim = key.size()
        q_len = query.size(1)
        head = self.head
        dim_head = dim // head

        def shape(x):
            return x.view(batch, -1, head, dim_head).transpose(1, 2)

        def unshape(x):
            return x.transpose(1, 2).contiguous().view(batch, -1, dim)

        # 1) Project key, value, and query.
        key_up = shape(self.linear_key(key))
        value_up = shape(self.linear_value(value))
        query_up = shape(self.linear_query(query))

        # 2) Calculate and scale scores.
        query_up = query_up / math.sqrt(dim_head)
        scores = torch.matmul(query_up, key_up.transpose(2, 3))
        if mask is not None:
            scores.masked_fill_(mask, -1e8)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.sm(scores)
        context = unshape(torch.matmul(self.dropout(attn), value_up))
        out = self.final_linear(context)
        attn = attn.view(batch, head, q_len, k_len)
        return out, attn
