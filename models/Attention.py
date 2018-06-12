import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import aeq


class GlobalAttention(nn.Module):
    def __init__(self, size, att_type, dropout=0.1):
        super().__init__()
        assert att_type in ['dot', 'general', 'mlp']
        self.att_type = att_type
        if att_type == 'mlp':
            self.linear_query = nn.Linear(size, size, bias=False)
            self.linear_context = nn.Linear(size, size)
            self.linear_score = nn.Linear(size, 1)
        elif att_type == 'general':
            self.linear_context = nn.Linear(size, size, bias=False)
        self.softmax = nn.Softmax(-1)
        self.tanh = nn.Tanh()
        self.linear_out_query = nn.Linear(size, size, False)
        self.linear_out_context = nn.Linear(size, size, bias=att_type=='mlp')

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
        batch, len_q, size = query.size()
        batch_, len_c, size_ = context.size()
        aeq(batch, batch_)
        aeq(size, size_)
        if self.att_type in ['dot', 'general']:
            if self.att_type == 'general':
                context = self.linear_context(context)
            score = torch.bmm(query, context.transpose(1, 2)) / math.sqrt(size)
        else:
            score_size = (batch, len_q, len_c, size)
            sc1 = self.linear_query(query).unsqueeze(2).expand(score_size)
            sc2 = self.linear_context(context).unsqueeze(1).expand(score_size)
            score = self.linear_score(F.tanh(sc1 + sc2))
        return score.view(batch, len_q, len_c)


class MultiAttention(nn.Module):
    def __init__(self, input_size, output_size, head_count, dropout):
        assert output_size % head_count == 0
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.head_count = head_count
        self.head_size = output_size // head_count
        self.linear_key = nn.Linear(input_size, output_size)
        self.linear_value = nn.Linear(input_size, output_size)
        self.linear_query = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)

    def _check_args(self, key, value, query, dist=None, mask=None):
        batch, len_key, key_size = key.size()
        batch_query, len_query, query_size = query.size()
        batch_value, len_value, value_size = value.size()
        aeq(batch, batch_query, batch_value)
        aeq(len_key, len_value)
        aeq(self.input_size, key_size, value_size, query_size)
        if dist is not None:
            aeq(len_query, dist.size(0))
            aeq(len_key, dist.size(1))
        if mask is not None:
            aeq(len_query, mask.size(0))
            aeq(len_key, mask.size(1))

    def _shape(self, x):
        y = x.view(x.size(0), -1, self.head_count, self.head_size)
        return y.transpose(-2, -3).contiguous()

    def _unshape(self, x):
        y = x.transpose(-2, -3).contiguous()
        return y.view(y.size(0), -1, self.output_size)

    def _score(self, key, query):
        scale = math.sqrt(self.head_size)
        scores = torch.matmul(query, key.transpose(-1, -2)) / scale
        return scores

    def _pool(self, attn, value):
        # drop_attn = self.dropout(attn)
        output = torch.matmul(attn, value)
        output = self._unshape(output)
        return output

    def forward(self, key, value, query, mask=None):
        self._check_args(key, value, query, mask=mask)

        key = self._shape(self.linear_key(key))
        value = self._shape(self.linear_value(value))
        query = self._shape(self.linear_query(query))

        scores = self._score(key, query)
        if mask is not None:
            scores.masked_fill_(mask, -1e8)
        attn = self.softmax(scores)
        output = self._pool(attn, value)
        return output


class MultiRelativeAttention(MultiAttention):
    def __init__(self, input_size, output_size, head_count, dropout,
                 num_dists, dist):
        super().init(input_size, output_size, head_count, dropout)
        self.num_dists = num_dists
        dist_dim = embedding_dist.embedding_dim
        self.embedding_dist_key = nn.Embedding(num_dists, self.head_size)
        self.embedding_dist_value = nn.Embedding(num_dists, self.head_size)
        self.register_buffer('dist', dist)

    def _dist_range(self):
        return self.dist.new_tensor(torch.arange(self.num_dists))

    def _select_dist(self, n_row, n_col):
        return self.dist[-n_row:, -n_col:]

    def _score(self, key, query):
        len_query, len_key = query.size(-2), key.size(-2)
        score = query.matmul(key.transpose(-1, -2))
        # compute dist score
        dist_key = self.embedding_dist_key(self._dist_range())
        score_dist = query.matmul(dist_key.transpose(0, 1))
        score_dist = score_dist.view(*score_dist.size()[:-2], -1)
        # index dist key
        dist = self._select_dist(len_query, len_key)
        dist += self._dist_range().unsqueeze(-1) * self.num_dists
        # add and div
        score += score_dist[..., dist]
        score /= math.sqrt(self.head_size)
        return score

    def _pool(self, attn, value):
        output = attn.matmul(value)
        dist = self._select_dist(attn.size(-2), attn.size(-1))
        dist_value = self.embedding_dist_value(self._dist_range())
        output += attn.matmul(dist_value[dist])
        return self._unshape(output)
