import torch
import torch.nn as nn

from models import SpatialRNN
from models import SpatialAttn


class ResSpatialAttnRNN(SpatialRNN):
    def __init__(self, rnn_type, num_nodes, size, num_layers,
                 head_count, dropout, mask=None):
        super().__init__(rnn_type, num_nodes, size, num_layers, dropout,
                         func=SpatialAttn.ResSpatialAttn,
                         head_count=head_count,
                         dropout=dropout,
                         mask=mask)
