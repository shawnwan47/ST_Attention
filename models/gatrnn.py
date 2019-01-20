from random import random

import torch
import torch.nn as nn

from modules import MultiHeadedAttention
from modules import GraphGRUSeq2Seq, GraphGRUAttnSeq2Seq
from modules import GraphGRUSeq2Vec, GraphGRUAttnSeq2Vec


class GATRNNSeq2Vec(GraphGRUSeq2Vec):
    def __init__(self, embedding, model_dim, num_layers, dropout, horizon, heads, mask=None):
        super().__init__(
            embedding=embedding,
            model_dim=model_dim,
            horizon=horizon,
            num_layers=num_layers,
            dropout=dropout,
            func=GAT,
            func_kwargs={'heads': heads, 'dropout': dropout, 'mask': mask}
        )


class GATRNNAttnSeq2Vec(GraphGRUAttnSeq2Vec):
    def __init__(self, embedding, model_dim, num_layers, heads, dropout, horizon, mask=None):
        super().__init__(
            embedding=embedding,
            model_dim=model_dim,
            horizon=horizon,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            func=GAT,
            func_kwargs={'heads': heads, 'dropout': dropout, 'mask': mask}
        )


class GATRNNSeq2Seq(GraphGRUSeq2Seq):
    def __init__(self, embedding, model_dim, num_layers, horizon, dropout, heads, mask=None):
        super().__init__(
            embedding=embedding,
            model_dim=model_dim,
            num_layers=num_layers,
            horizon=horizon,
            dropout=dropout,
            func=GAT,
            func_kwargs={'heads': heads, 'dropout': dropout, 'mask': mask}
        )


class GATRNNAttnSeq2Seq(GraphGRUSeq2Seq):
    def __init__(self, embedding, model_dim, num_layers, horizon, dropout, heads, mask=None):
        super().__init__(
            embedding=embedding,
            model_dim=model_dim,
            num_layers=num_layers,
            heads=heads,
            horizon=horizon,
            dropout=dropout,
            func=GAT,
            func_kwargs={'heads': heads, 'dropout': dropout, 'mask': mask}
        )



class GAT(nn.Module):
    def __init__(self, input_dim, output_dim, heads, dropout, mask=None):
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
