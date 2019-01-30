from random import random

import torch
import torch.nn as nn

from modules import MultiHeadedAttention
from modules import GraphGRUModel


class GATRNN(GraphGRUModel):
    def __init__(self, embedding, framework, rnn_attn, horizon,
                 model_dim, num_layers, dropout, adj, hops):
        super().__init__(
            embedding=embedding,
            framework=framework,
            rnn_attn=rnn_attn,
            model_dim=model_dim,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
            func=GAT,
            func_kwargs={'heads': heads, 'dropout': dropout}
        )


class GAT(nn.Module):
    def __init__(self, input_dim, output_dim, heads, dropout):
        super().__init__()
        self.attn = MultiHeadedAttention(
            model_dim=input_dim,
            out_dim=output_dim,
            heads=heads,
            dropout=dropout
        )
        self.register_buffer('mask', mask)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        return self.attn(input, input, self.mask) + self.linear(input)
