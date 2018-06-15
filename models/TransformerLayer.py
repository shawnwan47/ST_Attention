import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(size, size)
        self.relu = nn.ReLU(inplace=True)
        self.w_2 = nn.Linear(size, size)
        self.dropout = nn.Dropout(dropout, inplace=True)
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
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, query, context, mask):
        context, attn = self.attention(context, context, query, mask)
        output = self.layer_norm(self.dropout(context) + query)
        output = self.feed_forward(output)
        return output, attn


class STTransformerLayer(nn.Module):
    def __init__(self, size, head_count, dropout):
        self.attention_s = Attention.MultiAttention(size, head_count, dropout)
        self.attention_t = Attention.MultiAttention(size, head_count, dropout)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.layer_norm = nn.LayerNorm(size)
        self.feed_forward = PositionwiseFeedForward(size, dropout)

    def forward(self, query, context, mask=None):
        context_s, attn_s = self.attention_s(query, context)
        query_t, context_t = query.transpose(-2, -3), context.transpose(-2, -3)
        context_t, attn_t = self.attention_t(query_t, context_t)
        context_t = context_t.transpose(-2, -3)
        output = query + self.dropout(context_s) + self.dropout(context_t)
        output = self.layer_norm(output)
        output = self.feed_forward(output)
        return output, attn_s, attn_t
