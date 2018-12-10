import torch
import torch.nn as nn

from models.SpatialRNN import SpatialRNN
from models.Attention import MultiAttention


class SpatialAttentionRNN(SpatialRNN):
    def __init__(self, rnn_type, num_nodes, size, num_layers, dropout, gc_kwargs):
        super().__init__(rnn_type, num_nodes, size, num_layers, dropout,
                         func=MultiAttention,
                         func_args=gc_kwargs)
