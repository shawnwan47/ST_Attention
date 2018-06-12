import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Attention


class GraphAttention(nn.Module):
    def __init__(self, input_size, output_size, head_count, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.attention = Attention.MultiAttention(
            input_size, input_size, head_count, dropout
        )
        self.linear_query = nn.Linear(input_size, output_size)
        self.linear_context = nn.Linear(input_size, output_size, bias=False)

    def forward(self, input):
        '''
        input: batch_size x ... x node_count x input_size
        '''
        input_norm = self.layer_norm(input)
        context, attention = self.attention(input_norm, input_norm, input_norm)
        output = self.linear_query(input_norm) + self.linear_context(context)
        return output, attention


class GraphRelativeAttention(nn.Module):
    def __init__(self, input_size, output_size, head_count, dropout,
                 num_dists, dist):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.attention = Attention.MultiRelativeAttention(
            input_size=input_size,
            output_size=input_size,
            head_count=head_count,
            dropout=dropout,
            num_dists=num_dists,
            dist=dist,
        )
        self.linear_query = nn.Linear(input_size, output_size)
        self.linear_context = nn.Linear(input_size, output_size, bias=False)


class GatedGAT(GAT):
    def __init__(self, input_size, output_size, head_count, dropout):
        super().__init__()
        self.attention = Attention.MultiGatedAttention(
            input_size, output_size, head_count, dropout
        )
        self.layer_norm = nn.LayerNorm(output_size)
        self.linear_query = nn.Linear(input_size, output_size)
        self.linear_context = nn.Linear(output_size, output_size, bias=False)
        self.relu = nn.ReLU()

    def forward(self, input):
        '''
        input: batch_size x ... x node_count x input_size
        '''
        input = self.layer_norm(input)
        context, attention = self.attention(input, input, input)
        output = self.linear_query(input) + self.linear_context(context)
        output = self.relu(output)
        return output
