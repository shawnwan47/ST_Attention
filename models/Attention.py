import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import aeq


class GlobalAttention(nn.Module):
    def __init__(self, dim, att_type, dropout=0.1):
        super().__init__()
        assert att_type in ['dot', 'general', 'mlp']
        self.att_type = att_type
        if att_type == 'mlp':
            self.linear_query = nn.Linear(dim, dim, bias=False)
            self.linear_context = nn.Linear(dim, dim)
            self.linear_score = nn.Linear(dim, 1)
        elif att_type == 'general':
            self.linear_context = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(-1)
        self.tanh = nn.Tanh()
        self.linear_out_query = nn.Linear(dim, dim, False)
        self.linear_out_context = nn.Linear(dim, dim, bias=att_type=='mlp')

    def forward(self, query, bank, mask=None):
        '''
        query, bank: batch x num x features
        att: batch x num_qry x num_key
        '''
        score = self.score(query, bank)
        if mask is not None:
            score.data.masked_fill_(mask, -float('inf'))
        attn = self.softmax(score)
        context = torch.matmul(attn, bank)
        output = self.linear_out_context(context) + self.linear_out_query(query)
        return output, attn

    def score(self, query, context):
        batch, len_q, dim = query.size()
        batch_, len_c, dim_ = context.size()
        aeq(batch, batch_)
        aeq(dim, dim_)
        if self.att_type in ['dot', 'general']:
            if self.att_type == 'general':
                context = self.linear_context(context)
            score = torch.bmm(query, context.transpose(1, 2)) / math.sqrt(dim)
        else:
            score_size = (batch, len_q, len_c, dim)
            sc1 = self.linear_query(query).unsqueeze(2).expand(score_size)
            sc2 = self.linear_context(context).unsqueeze(1).expand(score_size)
            score = self.linear_score(F.tanh(sc1 + sc2))
        return score.view(batch, len_q, len_c)


class MultiHeadedAttention(nn.Module):
    def __init__(self, head, dim, p_dropout=0):
        assert dim % head == 0
        super().__init__()
        self.head = head
        self.dim = dim
        self.linear_key = nn.Linear(dim, dim)
        self.linear_value = nn.Linear(dim, dim)
        self.linear_query = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p_dropout)
        self.linear_out = nn.Linear(dim, dim)

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
        attn = self.softmax(scores)
        context = unshape(torch.matmul(self.dropout(attn), value_up))
        out = self.linear_out(context)
        attn = attn.view(batch, head, q_len, k_len)
        return out, attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, input_size, output_size, head_count, p_dropout=0):
        assert output_size % head_count == 0
        super().__init__()
        self.head_count = head_count
        self.head_size = output_size // head_count
        self.linear_key = nn.Linear(input_size, output_size)
        self.linear_value = nn.Linear(input_size, output_size)
        self.linear_query = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p_dropout)
        self.linear_out = nn.Linear(dim, dim)

    def forward(self, key, value, query, mask=None):
        batch, k_len, dim = key.size()
        q_len = query.size(1)
        head_count = self.head_count
        dim_head = dim // head_count

        def shape(x):
            return x.view(batch, -1, head_count, dim_head).transpose(1, 2)

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
        attn = self.softmax(scores)
        context = unshape(torch.matmul(self.dropout(attn), value_up))
        out = self.linear_out(context)
        attn = attn.view(batch, head_count, q_len, k_len)
        return out, attn
