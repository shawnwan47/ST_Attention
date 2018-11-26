import torch
import torch.nn as nn

from models import Attention


class GraphAttention(nn.Module):
    def __init__(self, input_size, output_size, head_count, dropout=0.1, mask=None):
        super().__init__()
        self.attention = Attention.MultiAttention(
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


class GraphAttentionHighway(nn.Module):
    def __init__(self, input_size, output_size, head_count, dropout=0.1, mask=None):
        super().__init__()
        self.attention = Attention.MultiAttention(
            size=input_size,
            head_count=head_count,
            dropout=dropout,
            output_size=output_size
        )
        self.linear_query = nn.Linear(input_size, output_size, bias=False)
        self.input_gate = nn.Linear(input_size, 1)
        self.output_gate = nn.Linear(output_size, 1)
        self.highway = nn.Sigmoid()
        self.register_buffer('mask', mask)

    def forward(self, input):
        context, attn = self.attention(input, input, input, self.mask)
        gate = self.highway(self.input_gate(input), self.output_gate(context))
        output =


class GraphRelativeAttention(GraphAttention):
    def __init__(self, input_size, output_size, head_count, dropout, adj, mask=None):
        super().__init__(input_size, output_size, head_count, dropout, mask)
        self.attention = Attention.MultiRelativeAttention(
            size=input_size,
            head_count=head_count,
            dropout=dropout,
            adj=adj,
            output_size=output_size
        )
