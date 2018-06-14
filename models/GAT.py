import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Attention


class GraphAttention(nn.Module):
    def __init__(self, input_size, output_size, head_count, dropout):
        super().__init__()
        self.attention = Attention.MultiAttention(
            size=input_size,
            head_count=head_count,
            dropout=dropout
        )
        self.linear_query = nn.Linear(input_size, output_size)
        self.linear_context = nn.Linear(input_size, output_size, bias=False)

    def forward(self, input):
        '''
        input: batch_size x ... x node_count x input_size
        '''
        context, attn = self.attention(input, input, input)
        output = self.linear_query(input) + self.linear_context(context)
        return output, attn


class GraphRelativeAttention(GraphAttention):
    def __init__(self, input_size, output_size, head_count, dropout,
                 num_dists, dist):
        super().__init__(input_size, output_size, head_count, dropout)
        self.attention = Attention.MultiRelativeAttention(
            size=input_size,
            head_count=head_count,
            dropout=dropout,
            num_dists=num_dists,
            dist=dist,
        )


class GatedGraphAttention(GraphAttention):
    def __init__(self, input_size, output_size, head_count, dropout):
        super().__init__(input_size, output_size, head_count, dropout)
        self.attention = Attention.MultiGatedAttention(
            input_size, output_size, head_count, dropout
        )
