import torch
import torch.nn as nn

from lib.utils import aeq
from lib.pt_utils import torch_to_numpy


class GAT(nn.Module):
    def __init__(self, input_size, output_size, head_count, dropout, mask=None):
        super().__init__()
        self.attention = MultiAttention(
            size=input_size,
            head_count=head_count,
            dropout=dropout,
            output_size=output_size
        )
        self.linear_query = nn.Linear(input_size, output_size, bias=False)
        self.register_buffer('mask', mask)

    def forward(self, input):
        '''
        input: batch_size x ... x node_count x input_size
        '''
        context, attn = self.attention(input, input, input, self.mask)
        output = self.linear_query(input) + context
        return output, attn


class HighwayGAT(GAT):
    def __init__(self, input_size, output_size, head_count, dropout, mask=None):
        super().__init__(input_size, output_size, head_count, dropout, mask)
        self.score_query = nn.Linear(input_size, 1)
        self.score_context = nn.Linear(output_size, 1)
        self.gate = nn.Sigmoid()
        self.register_buffer('mask', mask)

    def forward(self, input):
        context, attn = self.attention(input, input, input, self.mask)
        gate = self.gate(self.score_query(input), self.score_context(context))
        output = self.linear_query(query) * gate + context * (1 - gate)
        return output, gate, attn


class GRAT(GAT):
    def __init__(self, input_size, output_size, head_count, dropout, adj, mask=None):
        super().__init__(input_size, output_size, head_count, dropout, mask)
        self.attention = MultiRelativeAttention(
            size=input_size,
            head_count=head_count,
            dropout=dropout,
            adj=adj,
            output_size=output_size
        )


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
        self.linear_output = nn.Linear(size, output_size)
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(dropout)

    def _check_args(self, key, value, query, adj=None, mask=None):
        aeq(key.size(), value.size())
        aeq(key.dim(), query.dim())
        aeq(key.size(0), query.size(0))
        aeq(self.size, key.size(-1), query.size(-1))
        if adj is not None:
            aeq(query.size(-2), adj.size(0))
            aeq(key.size(-2), adj.size(1))
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
        output = self.linear_output(output)
        return output, torch_to_numpy(attn)


class MultiRelativeAttention(MultiAttention):
    def __init__(self, size, head_count, adj, dropout=0.2, output_size=None):
        super().__init__(size, head_count, dropout, output_size)
        num_adjs = adj.max().item() + 1
        self.embedding_adj_key = nn.Embedding(num_adjs, self.head_size)
        self.embedding_adj_value = nn.Embedding(num_adjs, self.head_size)
        adj_index = adj.new_tensor(torch.arange(adj.size(0)))
        adj_index = adj_index.unsqueeze(-1) * num_adjs
        self.register_buffer('adj', adj)
        self.register_buffer('adj_index', adj_index)

    def _score(self, key, query):
        '''
        query, key: batch x head x length x head_size
        '''
        len_query, len_key = query.size(-2), key.size(-2)
        score = query.matmul(key.transpose(-1, -2))
        # elegant but inefficient way
        # adj = self.adj[-len_query:, -len_key:]
        # adj_key = self.embedding_adj_key(self.adj).transpose(-1, -2)
        # score_adj = query.unsqueeze(-2).matmul(adj_key).squeeze(-2)
        # compute and flatten adj score
        adj_key = self.embedding_adj_key.weight
        score_adj = query.matmul(adj_key.transpose(0, 1))
        score_adj = score_adj.view(*score_adj.size()[:-2], -1)
        adj = self.adj[-len_query:, -len_key:]
        adj_index = adj + self.adj_index[:len_query]
        score_adj = score_adj[..., adj_index]
        # merge
        score += score_adj
        score /= math.sqrt(self.head_size)
        return score

    def _pool(self, attn, value):
        output = torch.matmul(attn, value)
        len_query, len_bank = attn.size(-2), attn.size(-1)
        adj = self.adj[-len_query:, -len_bank:]
        adj_index = adj + self.adj_index[:len_query]
        adj_value = self.embedding_adj_value(adj)
        output_adj = attn.unsqueeze(-2).matmul(adj_value).squeeze(-2)
        output = self._unshape(output + output_adj)
        return output
