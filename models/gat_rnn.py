import torch
import torch.nn as nn

from modules import GraphRNN, MultiHeadedAttention


class GATRNN(GraphRNN):
    def __init__(self, rnn_type, num_nodes, size, num_layers,
                 head_count, dropout, mask=None):
        super().__init__(rnn_type, num_nodes, size, num_layers, dropout,
                         func=MultiHeadedAttention,
                         head_count=head_count,
                         dropout=dropout,
                         mask=mask)
