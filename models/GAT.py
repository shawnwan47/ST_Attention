import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Attention


class GAT(nn.Module):
    def __init__(self, input_size, output_size, head_count, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.attn = Attention.MultiHeadedAttention(
            input_size, output_size, head_count, dropout
        )
        self.linear_context = nn.Linear(output_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        '''
        input: batch_size x ... x node_count x input_size
        '''
        context = self.attn(input, input, input)
        output = input + self.dropout(self.linear_context(context))
        return output


class GatedGAT(GAT):
    def __init__(self, input_size, output_size, head_count, dropout):
        super().__init__()
        self.attn = Attention.MultiHeadedAttention(
            input_size, output_size, head_count, dropout, return_head=True
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
