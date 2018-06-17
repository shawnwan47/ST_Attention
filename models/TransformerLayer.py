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

    def forward(self, query, memory, mask=None):
        '''
        output: batch x lenq x size
        attn: batch x lenq x lenc
        '''
        memory, attn = self.attention(memory, memory, query, mask)
        output = self.layer_norm(self.dropout(memory) + query)
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
