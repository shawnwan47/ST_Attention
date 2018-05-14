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
    def __init__(self, input_size, output_size, head_count, p_dropout=0,
                 return_head=False):
        assert output_size % head_count == 0
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.head_count = head_count
        self.head_size = output_size // head_count
        self.linear_key = nn.Linear(input_size, output_size)
        self.linear_value = nn.Linear(input_size, output_size)
        self.linear_query = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p_dropout)
        self.return_head = return_head

    def forward(self, key, value, query, mask=None):
        batch, len_key, key_size = key.size()
        batch_query, len_query, query_size = query.size()
        batch_value, len_value, value_size = values.size()
        aeq(batch, batch_query, batch_value)
        aeq(len_key, len_value)
        aeq(self.input_size, key_size, value_size, query_size)

        def shape(x):
            y = x.view(batch, -1, self.head_count, self.head_size)
            return y.transpose(-1, -2).contiguous()

        def unshape(x):
            y = x.transpose(-1, -2).contiguous()
            if self.return_head:
                return y.view(batch, -1, self.output_size)
            else:
                return y

        # 1) Project key, value, and query.
        key = shape(self.linear_key(key))
        value = shape(self.linear_value(value))
        query = shape(self.linear_query(query))

        # 2) Calculate and scale scores.
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores /= math.sqrt(self.head_size)
        if mask is not None:
            scores.masked_fill_(mask, -1e8)

        # 3) Apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        output = unshape(torch.matmul(self.dropout(attention), value))
        attention = attention.view(batch, self.head_count, len_query, len_key)
        return output, attention
