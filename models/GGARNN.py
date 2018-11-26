import torch
import torch.nn as nn

from models import Attention
from models import GRNNBase


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
        self.query_score = nn.Linear(input_size, 1)
        self.context_score = nn.Linear(output_size, 1)
        self.gate = nn.Sigmoid()
        self.register_buffer('mask', mask)

    def forward(self, input):
        context, attn = self.attention(input, input, input, self.mask)
        gate = self.gate(self.query_score(input), self.context_score(context))
        output = self.linear_query(query) * gate + context * (1 - gate)
        return output, gate, attn
