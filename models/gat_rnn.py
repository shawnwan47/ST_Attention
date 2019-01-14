import torch
import torch.nn as nn

from modules import GraphGRU, MultiHeadedAttention


class GATRNN(GraphRNN):
    def __init__(self, rnn_type, num_nodes, size, num_layers,
                 heads, dropout, mask=None):
        super().__init__(rnn_type, num_nodes, size, num_layers, dropout,
                         func=MultiHeadedAttention,
                         heads=heads,
                         dropout=dropout,
                         mask=mask)
