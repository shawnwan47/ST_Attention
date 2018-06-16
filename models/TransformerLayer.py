import torch
import torch.nn as nn

from models import Attention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, size // 2)
        self.relu = nn.ReLU(inplace=True)
        self.w_2 = nn.Linear(size // 2, size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(size)

    def forward(self, input):
        output = self.dropout(self.relu(self.w_1(input)))
        output = self.dropout(self.w_2(output)) + input
        return self.layer_norm(output)


class TransformerLayer(nn.Module):
    def __init__(self, size, head_count, dropout):
        super().__init__()
        self.attention = Attention.MultiAttention(size, head_count, dropout)
        self.layer_norm = nn.LayerNorm(size)
        self.feed_forward = PositionwiseFeedForward(size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, mask):
        '''
        output: batch x lenq x size
        attn: batch x lenq x lenc
        '''
        context, attn = self.attention(context, context, query, mask)
        output = self.layer_norm(self.dropout(context) + query)
        output = self.feed_forward(output)
        return output, attn


class RelativeTransformerLayer(TransformerLayer):
    def __init__(self, size, head_count, dropout, dist):
        super().__init__(size, head_count, dropout)
        self.attention = Attention.MultiRelativeAttention(
            size=size,
            head_count=head_count,
            dropout=dropout,
            dist=dist)


class STTransformerLayer(TransformerLayer):
    def __init__(self, size, head_count, dropout):
        super().__init__(size, head_count, dropout)
        self.attention_s = Attention.MultiAttention(size, head_count, dropout)

    def forward(self, query, context, mask=None):
        '''
        output: batch x lenq x size
        attn_s: batch x lenq x num_nodes x num_nodes
        attn_t: batch x num_nodes x lenq x lenc
        '''
        context_s, attn_s = self.attention_s(query, query, query)
        query_t, context_t = query.transpose(-2, -3), context.transpose(-2, -3)
        context_t, attn_t = self.attention(context_t, context_t, query_t, mask)
        context_t = context_t.transpose(-2, -3)
        output = query + self.dropout(context_s) + self.dropout(context_t)
        output = self.layer_norm(output)
        output = self.feed_forward(output)
        return output, attn_s, attn_t


class RelativeSTTransformerLayer(RelativeTransformerLayer):
    def __init__(self, size, head_count, dropout, temporal_dist, spatial_dist):
        super().__init__(size, head_count, dropout, temporal_dist)
        self.attention_s = Attention.MultiRelativeAttention(
            size=size,
            head_count=head_count,
            dropout=dropout,
            dist=spatial_dist)

    def forward(self, query, context, mask=None):
        '''
        output: batch x lenq x size
        attn_s: batch x lenq x num_nodes x num_nodes
        attn_t: batch x num_nodes x lenq x lenc
        '''
        context_s, attn_s = self.attention_s(query, query, query)
        query_t, context_t = query.transpose(-2, -3), context.transpose(-2, -3)
        context_t, attn_t = self.attention(context_t, context_t, query_t, mask)
        context_t = context_t.transpose(-2, -3)
        output = query + self.dropout(context_s) + self.dropout(context_t)
        output = self.layer_norm(output)
        output = self.feed_forward(output)
        return output, attn_s, attn_t
