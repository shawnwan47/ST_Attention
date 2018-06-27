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
    def __init__(self, size, head_count, dropout=0.2, output_size=None):
        assert size % head_count == 0
        if output_size is None:
            output_size = size
        super().__init__()
        self.size = size
        self.head_count = head_count
        self.head_size = size // head_count
        self.linear_key = nn.Linear(size, size)
        self.linear_value = nn.Linear(size, size)
        self.linear_query = nn.Linear(size, size)
        self.linear_out = nn.Linear(size, output_size)
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)

    def _check_args(self, key, value, query, dist=None, mask=None):
        aeq(key.size(), value.size())
        aeq(key.dim(), query.dim())
        aeq(key.size(0), query.size(0))
        aeq(self.size, key.size(-1), query.size(-1))
        if dist is not None:
            aeq(query.size(-2), dist.size(0))
            aeq(key.size(-2), dist.size(1))
        if mask is not None:
            aeq(query.size(-2), mask.size(0))
            aeq(key.size(-2), mask.size(1))

    def _shape(self, x):
        y = x.view(*x.size()[:-1], self.head_count, self.head_size)
        return y.transpose(-2, -3).contiguous()

    def _unshape(self, x):
        y = x.transpose(-2, -3).contiguous()
        return y.view(*y.size()[:-2], self.size)

    def _score(self, key, query):
        scale = math.sqrt(self.head_size)
        scores = torch.matmul(query, key.transpose(-1, -2)) / scale
        return scores

    def _pool(self, attn, value):
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
            scores.masked_fill_(mask, -float('inf'))
        attn = self.softmax(scores)
        output = self._pool(attn, value)
        output = self.linear_out(output)
        return output, attn


class MultiRelativeAttention(MultiAttention):
    def __init__(self, size, head_count, dist, dropout=0.2, output_size=None):
        super().__init__(size, head_count, dropout, output_size)
        num_dists = dist.max().item() + 1
        self.embedding_dist = nn.Embedding(num_dists, self.head_size)
        dist_index = dist.new_tensor(torch.arange(dist.size(0)))
        dist_index = dist_index.unsqueeze(-1) * num_dists
        self.register_buffer('dist', dist)
        self.register_buffer('dist_index', dist_index)

    def _score(self, key, query):
        len_query, len_key = query.size(-2), key.size(-2)
        score = query.matmul(key.transpose(-1, -2))
        # compute and flatten dist score
        score_dist = query.matmul(self.embedding_dist.weight.transpose(0, 1))
        score_dist = score_dist.view(*score_dist.size()[:-2], -1)
        # index dist
        dist = self.dist[-len_query:, -len_key:] + self.dist_index[:len_query]
        score += score_dist[..., dist]
        score /= math.sqrt(self.head_size)
        return score
