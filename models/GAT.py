import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Attention


class GAT(nn.Module):
    def __init__(self, input_size, output_size, head_count, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.attn = Attention.MultiAttention(
            input_size, input_size, head_count, dropout
        )
        self.linear_query = nn.Linear(input_size, output_size)
        self.linear_context = nn.Linear(input_size, output_size, bias=False)

    def forward(self, input):
        '''
        input: batch_size x ... x node_count x input_size
        '''
        input_norm = self.layer_norm(input)
        context = self.attn(input_norm, input_norm, input_norm)
        output = self.linear_query(input_norm) + self.linear_context(context)
        return output


class GatedGAT(GAT):
    def __init__(self, input_size, output_size, head_count, dropout):
        super().__init__()
        self.attn = Attention.MultiGatedAttention(
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
        context, attention = self.attn(input, input, input)
        output = self.linear_query(input) + self.linear_context(context)
        output = self.relu(output)
        return output
